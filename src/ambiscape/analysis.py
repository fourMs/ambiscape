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


def detect_cessations(fast_db, fast_dt, drop_db=6.0, min_before=60.0,
                      min_after=5.0):
    """Moments when a sustained level stops: figure-by-absence.

    A level-threshold detector finds sounds that start. It cannot find the
    event that happens when a sound *ends*, and indoors that is often the
    louder event of the two: a ventilation plant runs for nine hours and
    nobody attends to it, then it switches off and the birds, the clock and
    the water come back. Nothing rose; the ground fell, and the room
    changed character in two seconds.

    A cessation is recorded where a level held steady for at least
    ``min_before`` seconds, fell by at least ``drop_db``, and stayed down
    for at least ``min_after``. Returns a list of dicts with the index, the
    size of the drop, and the level either side.

    The asymmetry is the point. Attention is captured by change, not by
    level, and half the changes in a continuously occupied room are
    departures rather than arrivals.
    """
    x = np.asarray(fast_db, float)
    n = len(x)
    nb = max(1, int(round(min_before / fast_dt)))
    na = max(1, int(round(min_after / fast_dt)))
    if n < nb + na + 1:
        return []
    out, i = [], nb
    while i < n - na:
        before = x[i - nb:i]
        after = x[i:i + na]
        if not (np.isfinite(before).all() and np.isfinite(after).all()):
            i += 1
            continue
        lo, hi = float(np.median(before)), float(np.median(after))
        # steady before (a running machine holds a level), and clearly down
        if lo - hi >= drop_db and float(np.std(before)) <= drop_db / 2:
            out.append(dict(i=i, t_s=float(i * fast_dt),
                            drop_db=round(lo - hi, 1),
                            before_db=round(lo, 1), after_db=round(hi, 1)))
            i += nb            # one cessation per steady stretch
        else:
            i += 1
    return out


def trimmed_leq(level_db: np.ndarray, trim_pct: float = 5.0) -> float:
    """Energy mean in dB with the loudest ``trim_pct`` of frames discarded.

    An energy average is a mean of squared pressure, so it is decided by
    the loudest frames it contains: in a quiet room a handful of them can
    outweigh every other frame together. The trimmed level answers the
    companion question — what the average would be without that handful —
    and a large gap between the two says the plain average describes a few
    moments rather than the span. Reported next to ``laeq_dbfs`` in every
    session summary; see the descriptor guide's "Reading energy averages".
    """
    x = np.asarray(level_db, np.float64)
    if x.size == 0:
        return float("nan")
    keep = x <= np.percentile(x, 100.0 - trim_pct)
    if not keep.any():                       # every frame at one level
        keep = np.ones_like(x, bool)
    return float(db(np.mean(10 ** (x[keep] / 10))))


def intermittency_ratio(level_db: np.ndarray, dt: float,
                        k_db: float = 3.0) -> float:
    """Intermittency ratio IR (Wunderli et al. 2016), in percent.

    The share of total sound energy carried by "events": frames whose
    level exceeds the whole-period Leq by ``k_db`` (3 dB per the original
    definition, there on 1 s LAeq frames — here on the fast frames, which
    is equivalent for events longer than the frame). IR ≈ 0 for steady
    scenes (drones, dense traffic), high for scenes whose energy arrives
    in distinct events (rail, church bells, sparse traffic).
    """
    p = 10 ** (np.asarray(level_db, np.float64) / 10)
    leq = db(p.mean())
    mask = level_db >= leq + k_db
    return float(100.0 * p[mask].sum() / (p.sum() + EPS))


def decay_metrics(x: np.ndarray, fs: int, bands=((250, 500), (500, 1000),
                  (1000, 2000), (2000, 4000), (4000, 8000)),
                  pre_roll: bool = True) -> dict:
    """T60, EDT, C50, C80 (dB) and D50 per octave band from an impulse.

    ``pre_roll=False`` says the samples before the peak are not a recording
    of the room but silence prepended by a caller so this estimator can run
    on a trimmed IR. The noise floor is then taken from the quietest part of
    the decay itself rather than from that silence. Read off the padding it
    comes out near 200 dB, which silently satisfies every dynamic-range
    guard below and lets T20/T30 be fitted over noise.

    Same truncated-Schroeder machinery as :func:`decay_time` (which is
    kept unchanged — its output feeds frozen corpus reports), plus the
    standard companions: EDT from the 0…−10 dB fit (perceived
    reverberance), clarity C50/C80 = 10·log10 of the early/late energy
    ratio at 50/80 ms, and definition D50 = early fraction at 50 ms.
    When the dynamic range allows (ISO 3382: floor at least 10 dB below
    the fit end) *and* the decay was observed that far before the signal
    ends, the fixed-range extrapolations T20 (−5…−25 dB) and T30
    (−5…−35 dB) are reported alongside the adaptive-range T60. The second
    condition matters for trimmed impulse responses, whose absent noise
    floor leaves the range guard unable to fire.
    Returns ``{band: {"T60", "T20", "T30", "EDT", "C50", "C80", "D50",
    "dr_db"}}`` (T20/T30 present only when supported by the range).
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
        pk = int(env[max(0, pk_i - 2400):pk_i + 2400].argmax()) \
            + max(0, pk_i - 2400)
        if pk < fs // 4:
            continue
        if pre_roll:
            noise = float(np.median(env[:pk - fs // 8]))
        else:
            # Nothing of the room was recorded before the peak, so the only
            # floor available is the quietest part of what the IR contains.
            after = env[pk:]
            noise = float(np.percentile(after, 10)) if len(after) else 0.0
        dr = 10 * np.log10(env[pk] / (noise + EPS))
        if dr < 20:
            continue
        seg = np.maximum(y[pk:pk + cut] ** 2 - noise, 0)
        sch = np.cumsum(seg[::-1])[::-1]
        sch_db = 10 * np.log10(sch / (sch[0] + EPS) + 1e-15)
        tax = np.arange(len(sch_db)) / fs
        # An impulse response that has been trimmed (archive material, or
        # any IR cut before its decay finished) ends while still well above
        # the fixed fits' lower limit: the dynamic-range guard cannot fire,
        # because the truncated file has no noise floor to measure. Level
        # of the last 20 ms re the peak says how far the decay was actually
        # observed; below that, T20/T30 would extrapolate off the end.
        tail = env[pk:pk + cut][-max(1, int(0.02 * fs)):]
        obs_db = 10 * np.log10(float(tail.mean()) / (env[pk] + EPS) + EPS)
        res = {"dr_db": round(float(dr), 0)}
        for key, hi_db, lo_db, need_dr in (
                ("T60", -5.0, max(-35.0, -dr + 8), 0.0),
                ("T20", -5.0, -25.0, 35.0),
                ("T30", -5.0, -35.0, 45.0),
                ("EDT", 0.0, -10.0, 0.0)):
            if dr < need_dr:
                continue
            if key in ("T20", "T30") and obs_db > lo_db:
                continue                    # range not present in the file
            m = (sch_db <= hi_db) & (sch_db >= lo_db)
            if m.sum() < 150:
                continue
            A = np.vstack([tax[m], np.ones(int(m.sum()))]).T
            slope, _ = np.linalg.lstsq(A, sch_db[m], rcond=None)[0]
            if slope < 0:
                res[key] = round(-60.0 / slope, 2)
        for key, ms in (("C50", 50), ("C80", 80)):
            i = int(ms * fs / 1000)
            if i < len(sch) and sch[i] > 0:
                res[key] = round(float(10 * np.log10(
                    (sch[0] - sch[i]) / (sch[i] + EPS) + EPS)), 1)
        i50 = int(0.05 * fs)
        if i50 < len(sch):
            res["D50"] = round(float((sch[0] - sch[i50]) / (sch[0] + EPS)), 2)
        if "T60" in res:
            out[f"{lo}-{hi}"] = res
    return out


def quietest_channel(x, fs: int, pct: float = 5.0, frame_s: float = 1.0):
    """Which capsule of a multi-microphone node has the most room to measure in.

    A node's capsules share a housing, a preamp and a gain setting, so they
    should agree. When one does not — a blocked port, a damaged capsule — it
    reaches the same peaks as its siblings but on a raised floor, which costs
    dynamic range on every descriptor computed from it. Reading channel 0 by
    convention is then a coin toss.

    Measured in the SINS network: node 9's four capsules reached the same
    98th-percentile level within 0.7 dB, while channel 0's floor sat 5.5 dB
    higher than the other three. Analysing channel 0 halved that node's
    apparent dynamics and pushed the fraction of time it spent at its own
    floor from about 75 % to 96 %, which is most of what made it look broken.

    Run over that corpus it is decisive where it matters and indifferent
    where it does not: on node 9 it picks the same channel on every minute
    tried, away from the raised one; on a node whose capsules agree within
    2 dB it picks whichever is marginally lower, which is of no consequence.

    Returns ``(index, floors_db)`` — the channel with the lowest floor, and
    every channel's floor, so the caller can see how much the choice matters.
    A mono signal returns ``(0, [floor])``.
    """
    a = np.asarray(x, float)
    if a.ndim == 1:
        a = a[:, None]
    n = max(1, int(frame_s * fs))
    floors = []
    for c in range(a.shape[1]):
        y = a[:, c] - a[:, c].mean()
        lv = np.array([10 * np.log10((y[i:i + n] ** 2).mean() + EPS)
                       for i in range(0, max(len(y) - n, 1), n)])
        floors.append(float(np.percentile(lv, pct)) if len(lv) else float("nan"))
    return int(np.nanargmin(floors)), floors


FLOOR_SPREAD_THRESH_DB = 1.5

#: How close to its own floor a second must sit to count as "at the floor".
AT_FLOOR_WITHIN_DB = 3.0


def floor_occupancy(F: dict, within_db: float = AT_FLOOR_WITHIN_DB,
                    pct: float = 5.0) -> dict:
    """How much of a session sits at its own noise floor.

    :func:`floor_suspicion` asks whether a *band's* floor is self-noise. It
    is the right question and it has a limit: across a whole sensor network
    it can fire for every node, because every recorder's top octaves are its
    own hiss during quiet hours. It therefore cannot separate "this band is
    the instrument" from "this room was empty all week".

    This asks the second question. Each second's broadband level is compared
    against the session's own ``pct``-percentile floor, and the fraction
    within ``within_db`` of it is returned. A living room in use spends
    little time there; a bedroom occupied only to sleep spends most of the
    week there. That is a description of how a room is used, not a fault in
    the recorder — a distinction this corpus cost several hours to learn,
    when a microphone in a mostly-empty bedroom was diagnosed as dead.

    Being each session against its own floor, the measure does not move with
    recording gain, and so is comparable across uncalibrated instruments in
    a way an absolute level is not.

    Measured across the SINS network, one week each: the living-room and
    kitchen nodes sit at their own floor 28-56 % of the time (median 2.4-9.8
    dB above it), while the bedroom node sits there **96 %** of the time,
    a median of 0.4 dB above. One number, and it names the room.

    Returns ``{"at_floor_fraction", "floor_db", "median_above_floor_db"}``.
    """
    op = np.asarray(F["oct_pow"], float)
    if op.ndim != 2 or not len(op):
        return {"at_floor_fraction": None, "floor_db": None,
                "median_above_floor_db": None}
    lvl = 10 * np.log10(op.sum(axis=1) + EPS)
    floor = float(np.percentile(lvl, pct))
    above = lvl - floor
    return {"at_floor_fraction": round(float((above <= within_db).mean()), 3),
            "floor_db": round(floor, 1),
            "median_above_floor_db": round(float(np.median(above)), 1)}


def floor_suspicion(F: dict, chunk_s: float = 300.0, pct: float = 10.0,
                    spread_thresh_db: float = FLOOR_SPREAD_THRESH_DB,
                    min_chunks: int = 6, hf_min_hz: float = 2000.0) -> dict:
    """Flag high-frequency band floors that look like recorder self-noise.

    A genuine room background breathes: its low-percentile level moves with
    the day, the weather and the building. A microphone's self-noise floor
    does not — it is abnormally flat over time (and typically spectrally
    smooth). In the SINS sensor-network corpus the 4–8 kHz floor of a
    living room is flat to 0.8 dB across a full week (0.56 dB between six
    separate nights), while every band below 1 kHz varies by 2.4–5.3 dB
    over the same nights: the top of the spectrum is the instrument, and
    any L90-derived descriptor weighted towards it (LA90 in particular)
    measures the recorder rather than the room.

    The check works on the cached 1 s octave-band powers: the session is
    cut into ``chunk_s`` chunks, each chunk's ``pct``-percentile band level
    is that chunk's floor, and the temporal spread of the floor is taken as
    the median minus the 5th-percentile chunk floor — a low-tail statistic,
    so chunks whose floor is raised by activity (television, dishes) do not
    hide a pinned quiet-time floor. A band centred at or above
    ``hf_min_hz`` whose spread is below ``spread_thresh_db`` is suspect.
    The 1.5 dB default sits between the SINS self-noise band (≤ 0.8 dB
    over a week) and the quietest genuinely acoustic bands there
    (≥ 2.4 dB), with at least 0.7 dB of margin to each side. Bands with no
    content below the Nyquist frequency, and sessions shorter than
    ``min_chunks`` chunks (30 min at the defaults), are never flagged.

    This is an annotation, not a correction: no descriptor value changes.
    Returns ``floor_suspect`` (bool), the affected band range
    ``floor_suspect_lo_hz``/``floor_suspect_hi_hz`` (band edges, Hz), and
    ``floor_spread_db`` (the smallest spread among the flagged bands);
    the last three are None when nothing is flagged.
    """
    from .features import OCT_CENTERS
    out = {"floor_suspect": False, "floor_suspect_lo_hz": None,
           "floor_suspect_hi_hz": None, "floor_spread_db": None}
    op = F.get("oct_pow")
    if op is None or len(op) == 0:
        return out
    rows = max(1, int(round(chunk_s)))          # 1 s frames per chunk
    nchunk = len(op) // rows
    if nchunk < min_chunks:
        return out
    lvl = db(np.asarray(op[:nchunk * rows], np.float64))
    floors = np.percentile(lvl.reshape(nchunk, rows, lvl.shape[1]),
                           pct, axis=1)         # (nchunk, nband)
    spread = (np.percentile(floors, 50, axis=0)
              - np.percentile(floors, 5, axis=0))
    centers = np.asarray(OCT_CENTERS, float)[:lvl.shape[1]]
    nyq = float(F.get("fs", 48000)) / 2
    med = np.median(floors, axis=0)
    flagged = ((centers >= hf_min_hz) & (centers / np.sqrt(2) < nyq)
               & (spread < spread_thresh_db) & (med > -119.0))
    if flagged.any():
        idx = np.flatnonzero(flagged)
        out.update({
            "floor_suspect": True,
            "floor_suspect_lo_hz": int(round(centers[idx[0]] / np.sqrt(2))),
            "floor_suspect_hi_hz": int(round(min(centers[idx[-1]]
                                                 * np.sqrt(2), nyq))),
            "floor_spread_db": round(float(spread[idx].min()), 2),
        })
    return out


def circular_stats(az_deg, weights=None):
    """Energy-weighted circular mean (deg) and resultant length R."""
    from .circstats import mean_resultant
    mu, R = mean_resultant(np.radians(np.asarray(az_deg, float)), weights)
    return float(np.degrees(mu)), R


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
    # direction is full 3-D (ambix), lateral-only (stereo), or absent (mono);
    # emit None for whatever this recording's channel layout cannot support
    az = np.asarray(F["az"], float)
    el = np.asarray(F["el"], float)
    psi = np.asarray(F["diffuse"], float)
    fin_az = np.isfinite(az)
    if fin_az.any():
        az_mean, R = circular_stats(az[fin_az], weights=p[fin_az])
        fg_az = e_fg & fin_az
        az_fg = (circular_stats(az[fg_az], weights=p[fg_az])[0]
                 if fg_az.any() else az_mean)
    else:
        az_mean = R = az_fg = None
    el_fg = (float(np.nanmedian(el[e_fg])) if np.isfinite(el[e_fg]).any()
             else None)
    has_psi = np.isfinite(psi).any()

    return {
        "duration_min": round(dur / 60, 1),
        "leq_dbfs": round(float(leq), 1),
        "laeq_dbfs": round(float(laeq), 1),
        "laeq_trim5_dbfs": round(trimmed_leq(fasta, 5.0), 1),
        "leq_minus_laeq_db": round(float(leq - laeq), 1),
        "L10": round(l10, 1), "L50": round(l50, 1), "L90": round(l90, 1),
        "dynamics_L10_L90": round(l10 - l90, 1),
        "events_per_min": round(len(events) / max(dur / 60, 1e-9), 1),
        "event_median_dur_s": round(float(np.median(
            [(e["i1"] - e["i0"] + 1) * dt for e in events])), 2) if events else None,
        "centroid_median_hz": int(np.median(F["centroid"])),
        "flatness_median": round(float(np.median(F["flatness"])), 3),
        "diffuseness_median": round(float(np.nanmedian(psi)), 2) if has_psi else None,
        "diffuseness_iqr": round(float(np.nanpercentile(psi, 75)
                                       - np.nanpercentile(psi, 25)), 2)
        if has_psi else None,
        "azimuth_mean_deg": round(az_mean, 0) if az_mean is not None else None,
        "azimuth_R": round(R, 2) if R is not None else None,
        "azimuth_fg_deg": round(az_fg, 0) if az_fg is not None else None,
        "elevation_fg_median_deg": round(el_fg, 0) if el_fg is not None else None,
        "n_events": len(events),
        "emergence_db": round(float(laeq - np.percentile(fasta, 10)), 1),
        "intermittency_ratio_pct": round(intermittency_ratio(fasta, dt), 1),
        **floor_suspicion(F),
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
    and (if present) the strongest state transition.

    Kinds can coincide: a session barely longer than one window has only
    one window to offer, and a stationary room has no most-active minute
    to distinguish from its quietest one. Coincident kinds are returned
    once, the window keeping the first kind's name and listing the others
    under ``also`` — so the degeneracy is visible rather than presented as
    several identical "representative" segments.
    """
    t, fast = F["t_fast"], F["fast_db"]
    dt = float(np.median(np.diff(t)))
    win = max(1, int(seg_s / dt))
    if len(fast) < win:
        return [dict(kind="whole", t0=float(t[0]), dur=float(t[-1] - t[0]))]
    k = np.ones(win) / win
    m_lvl = np.convolve(10 ** (fast.astype(np.float64) / 10), k, "valid")
    var = np.convolve((fast - fast.mean()) ** 2, k, "valid")
    cands = [("quietest", float(t[int(np.argmin(m_lvl))])),
             ("most_active", float(t[int(np.argmax(var))])),
             ("typical", float(t[int(np.argmin(np.abs(
                 db(m_lvl) - np.median(db(m_lvl)))))]))]
    smooth = median_filter(fast, size=max(3, int(30 / dt)) | 1)
    jump = np.abs(np.diff(smooth))
    if jump.max() > 6:
        cands.append(("transition",
                      float(max(t[0], t[int(np.argmax(jump))] - seg_s / 2))))
    picks: list[dict] = []
    for kind, t0 in cands:
        same = next((p for p in picks if abs(p["t0"] - t0) <= dt), None)
        if same is None:
            picks.append(dict(kind=kind, t0=t0, dur=seg_s))
        else:
            same.setdefault("also", []).append(kind)
    return picks[:n]
