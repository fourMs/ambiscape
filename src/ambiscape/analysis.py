"""Session-level descriptors, event detection, and reverberation estimation.

Descriptor conventions follow the Intercontinental-database report
(2026-07-10): fast level = 125 ms RMS on W; events = fast level exceeding a
running background (10th percentile in a sliding 60 s window) by >= 8 dB for
>= 0.25 s; diffuseness/DOA from per-second pseudo-intensity vectors.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import percentile_filter, median_filter

EPS = 1e-20


def db(x, eps=1e-12):
    return 10 * np.log10(np.maximum(x, eps))


def running_background(fast_db: np.ndarray, fast_dt: float, win_s=60.0, pct=10):
    n = max(3, int(round(win_s / fast_dt)) | 1)
    return percentile_filter(fast_db, pct, size=n, mode="nearest")


def detect_events(fast_db, fast_dt, thresh_db=8.0, min_dur=0.25):
    """Return list of dicts (onset index, length, peak index, exceedance)."""
    bg = running_background(fast_db, fast_dt)
    above = fast_db > bg + thresh_db
    events = []
    i, n = 0, len(above)
    min_len = max(1, int(round(min_dur / fast_dt)))
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                k = i + int(np.argmax(fast_db[i:j + 1]))
                events.append(dict(i0=i, i1=j, ipk=k,
                                   exceed=float(fast_db[k] - bg[k])))
            i = j + 1
        else:
            i += 1
    return events, bg


def circular_stats(az_deg, weights=None):
    """Energy-weighted circular mean (deg) and resultant length R."""
    a = np.radians(np.asarray(az_deg, float))
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    C = (w * np.cos(a)).sum() / (w.sum() + EPS)
    S = (w * np.sin(a)).sum() / (w.sum() + EPS)
    return float(np.degrees(np.arctan2(S, C))), float(np.hypot(C, S))


def summarize(F: dict) -> dict:
    """Session descriptor dict from concatenated features (see features.load_features)."""
    fast, fasta = F["fast_db"], F["fast_dba"]
    dt = float(np.median(np.diff(F["t_fast"]))) if len(F["t_fast"]) > 1 else 0.125
    leq = db(np.mean(10 ** (fast.astype(np.float64) / 10)))
    laeq = db(np.mean(10 ** (fasta.astype(np.float64) / 10)))
    l10, l50, l90 = (float(np.percentile(fast, q)) for q in (90, 50, 10))
    events, bg = detect_events(fast, dt)
    dur = float(len(F["t"]))  # 1 s per feature frame; robust across take gaps

    p = F["rms_w"].astype(np.float64) ** 2
    e_fg = p >= np.percentile(p, 75)
    e_bg = p <= np.percentile(p, 25)
    az_mean, R = circular_stats(F["az"], weights=p)
    az_fg, R_fg = circular_stats(F["az"][e_fg], weights=p[e_fg])
    el_fg = float(np.median(F["el"][e_fg]))
    psi = F["diffuse"]

    return {
        "duration_min": round(dur / 60, 1),
        "leq_dbfs": round(float(leq), 1),
        "laeq_dbfs": round(float(laeq), 1),
        "leq_minus_laeq_db": round(float(leq - laeq), 1),
        "L10": round(l10, 1), "L50": round(l50, 1), "L90": round(l90, 1),
        "dynamics_L10_L90": round(l10 - l90, 1),
        "events_per_min": round(len(events) / max(dur / 60, 1e-9), 1),
        "event_median_dur_s": round(float(np.median(
            [(e["i1"] - e["i0"] + 1) * dt for e in events])), 2) if events else None,
        "centroid_median_hz": int(np.median(F["centroid"])),
        "flatness_median": round(float(np.median(F["flatness"])), 3),
        "diffuseness_median": round(float(np.median(psi)), 2),
        "diffuseness_iqr": round(float(np.percentile(psi, 75)
                                       - np.percentile(psi, 25)), 2),
        "azimuth_mean_deg": round(az_mean, 0),
        "azimuth_R": round(R, 2),
        "azimuth_fg_deg": round(az_fg, 0),
        "elevation_fg_median_deg": round(el_fg, 0),
        "n_events": len(events),
    }


def decay_time(x: np.ndarray, fs: int, bands=((250, 500), (500, 1000),
               (1000, 2000), (2000, 4000), (4000, 8000))) -> dict:
    """T60 estimates from an impulse via truncated Schroeder integration.

    The decay is truncated at the first re-attack (envelope rising >= 8 dB
    above its running minimum) and at the noise floor; a linear fit of
    -5 dB .. max(-35 dB, floor + 8 dB) is extrapolated to 60 dB.
    Returns {band: (T60, dynamic_range_db)}.
    """
    from scipy import signal as sg
    pk_i = int(np.abs(x).argmax())
    env_bb = sg.convolve(x ** 2, np.ones(480) / 480, "same")
    tail = 10 * np.log10(env_bb[pk_i:pk_i + 3 * fs] + 1e-15)
    run_min = np.minimum.accumulate(tail)
    re = np.flatnonzero((tail - run_min > 8) & (np.arange(len(tail)) > fs // 10))
    cut = int(re[0]) if len(re) else 2 * fs
    out = {}
    for lo, hi in bands:
        sos = sg.butter(4, [lo, hi], "bandpass", fs=fs, output="sos")
        y = sg.sosfilt(sos, x)
        env = sg.convolve(y ** 2, np.ones(240) / 240, "same")
        pk = int(env[max(0, pk_i - 2400):pk_i + 2400].argmax()) + max(0, pk_i - 2400)
        if pk < fs // 4:
            continue
        noise = float(np.median(env[:pk - fs // 8]))
        dr = 10 * np.log10(env[pk] / (noise + EPS))
        if dr < 20:
            continue
        seg = np.maximum(y[pk:pk + cut] ** 2 - noise, 0)
        sch = np.cumsum(seg[::-1])[::-1]
        sch_db = 10 * np.log10(sch / (sch[0] + EPS) + 1e-15)
        tax = np.arange(len(sch_db)) / fs
        lo_db = max(-35.0, -dr + 8)
        m = (sch_db <= -5) & (sch_db >= lo_db)
        if m.sum() < 150:
            continue
        A = np.vstack([tax[m], np.ones(int(m.sum()))]).T
        slope, _ = np.linalg.lstsq(A, sch_db[m], rcond=None)[0]
        if slope < 0:
            out[f"{lo}-{hi}"] = (round(-60.0 / slope, 2), round(float(dr), 0))
    return out


def pick_segments(F: dict, n=4, seg_s=600.0) -> list[dict]:
    """Suggest representative windows: quietest, most active, median-typical,
    and (if present) the strongest state transition."""
    t, fast = F["t_fast"], F["fast_db"]
    dt = float(np.median(np.diff(t)))
    win = max(1, int(seg_s / dt))
    if len(fast) < win:
        return [dict(kind="whole", t0=float(t[0]), dur=float(t[-1] - t[0]))]
    k = np.ones(win) / win
    m_lvl = np.convolve(10 ** (fast.astype(np.float64) / 10), k, "valid")
    var = np.convolve((fast - fast.mean()) ** 2, k, "valid")
    picks = []
    for kind, idx in (("quietest", int(np.argmin(m_lvl))),
                      ("most_active", int(np.argmax(var))),
                      ("typical", int(np.argmin(np.abs(db(m_lvl) - np.median(db(m_lvl))))))):
        picks.append(dict(kind=kind, t0=float(t[idx]), dur=seg_s))
    smooth = median_filter(fast, size=max(3, int(30 / dt)) | 1)
    jump = np.abs(np.diff(smooth))
    if jump.max() > 6:
        picks.append(dict(kind="transition",
                          t0=float(max(t[0], t[int(np.argmax(jump))] - seg_s / 2)),
                          dur=seg_s))
    return picks[:n]
