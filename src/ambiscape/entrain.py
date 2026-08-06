"""Sound–motion entrainment: joining the soundscape with a body-motion series.

The AMBIENT project records rooms as audio-visual *and bodily* subjects: an
ambisonic recorder on the desk and an accelerometer on the body. This module
performs the missing sound–motion join, following the crossmodal method of
Guo, Riaz & Jensenius (CMMR 2025): the audio session and a body-motion time
series are resampled onto one common clock and three entrainment measures are
computed —

1. **temporal correlation** between the 125 ms fast level and the quantity
   of motion (QoM), with a permutation p-value from *circular time-shift*
   surrogates (shuffling would destroy autocorrelation and overstate
   significance);
2. **directional correlation** between the audio's azimuthal energy and the
   direction of horizontal micromotion — the Jammalamadaka–SenGupta circular
   correlation from :mod:`ambiscape.circstats`, rotation-invariant so the
   mic frame and the sensor frame need not be aligned;
3. **phase-locking value** (PLV) between the audio's envelope modulation and
   the motion oscillation, per modulation band from 0.1 to 4 Hz, again with
   circular-shift surrogates.

Motion input is a device-agnostic CSV/TSV: one timestamp column (ISO 8601,
or plain seconds) plus accelerometer x/y/z columns, any consistent unit
(g or m/s²) — the constant gravity component is removed internally and every
measure is scale-free, so the unit never enters a result. ISO timestamps are
placed on the session's absolute clock (both devices are assumed set to the
same local time; use ``calibration.json`` clock offsets for a drifting
recorder); a plain seconds column that does not overlap the audio span is
taken as *relative* and aligned to the start of the audio. QoM is computed
as jerk magnitude after gravity removal, per Riaz's micromotion method
(gravity = a 0.25 Hz low-pass of the acceleration), and the horizontal plane
is defined by the gravity estimate itself, so the sensor can sit at any
orientation. Sway direction is the per-frame principal axis of horizontal
micromotion — an *axial* quantity (period 180°), so it is angle-doubled
before circular correlation with the (full-circle) audio azimuth; that
convention, and the use of dB level vs log-QoM for the Pearson correlation
(both compressive, per the paper), are the two design choices most worth a
reviewer's eye.

``analyze_entrainment`` writes ``entrain.json`` + ``entrain.png`` and folds
``ent_``-prefixed descriptors into an existing ``summary.json`` (the same
multimodal-join move as :mod:`ambiscape.vision`).
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
from scipy import signal

from .circstats import circ_corr, mean_resultant

EPS = 1e-20
GRAV_FC = 0.25         # gravity low-pass cutoff (Hz), per the micromotion method
BANDS = ((0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0))
_TIME_NAMES = ("time", "timestamp", "t", "datetime", "date_time", "seconds",
               "sec", "epoch")


def _clean(name: str) -> str:
    """Lower-case a header cell and strip a trailing unit, e.g. 'X (m/s^2)'."""
    return name.split("(")[0].strip().strip('"').lower()


def load_motion(path: str | Path):
    """Parse a motion CSV/TSV into ``(t_seconds, accel[n, 3], meta)``.

    The delimiter is sniffed (tab, semicolon, or comma); the timestamp column
    is the first column named like ``time``/``timestamp``/``t``, else column
    0; the x/y/z columns are the first columns whose cleaned names end in
    ``x``/``y``/``z`` (``acc_x``, ``ax``, ``X (g)`` all match), else the
    three columns after the timestamp. ISO 8601 timestamps come back as
    seconds since midnight of the first sample's date, with
    ``meta["date0"]`` set so callers can move them onto a session clock;
    numeric timestamps pass through with ``meta["date0"] = None``.
    """
    import csv
    path = Path(path)
    text = path.read_text().strip().splitlines()
    head = text[0]
    delim = "\t" if "\t" in head else (";" if ";" in head else ",")
    rows = list(csv.reader(text, delimiter=delim))
    names = [_clean(c) for c in rows[0]]
    ti = next((i for i, nm in enumerate(names) if nm in _TIME_NAMES), 0)
    axes = []
    for ax in "xyz":
        i = next((i for i, nm in enumerate(names)
                  if i != ti and nm.endswith(ax)), None)
        axes.append(i)
    if any(i is None for i in axes):
        axes = [ti + 1, ti + 2, ti + 3]
    data = [r for r in rows[1:] if len(r) > max(ti, *axes) and r[ti].strip()]
    if len(data) < 8:
        raise ValueError(f"{path.name}: fewer than 8 motion samples")
    date0 = None
    first = data[0][ti].strip()
    try:
        float(first)
        t = np.array([float(r[ti]) for r in data])
    except ValueError:
        stamps = [_dt.datetime.fromisoformat(r[ti].strip().replace("Z", ""))
                  for r in data]
        date0 = stamps[0].date()
        t = np.array([(s.date() - date0).days * 86400.0
                      + s.hour * 3600 + s.minute * 60 + s.second
                      + s.microsecond / 1e6 for s in stamps])
    acc = np.array([[float(r[i]) for i in axes] for r in data])
    keep = np.argsort(t, kind="stable")
    return t[keep], acc[keep], {"date0": date0,
                                "columns": [rows[0][i] for i in axes]}


def _micromotion(t_m: np.ndarray, acc: np.ndarray):
    """Gravity-free micromotion on a uniform grid.

    Returns ``(t_grid, dt_m, qom_raw, h1, h2)``: jerk magnitude (the QoM
    integrand) and the two horizontal components of linear acceleration in a
    gravity-referenced frame (vertical = the mean low-passed acceleration,
    so the device may sit at any orientation).
    """
    dt_m = float(np.median(np.diff(t_m)))
    tg = np.arange(t_m[0], t_m[-1], dt_m)
    a = np.stack([np.interp(tg, t_m, acc[:, i]) for i in range(3)], axis=1)
    fs_m = 1.0 / dt_m
    sos = signal.butter(2, min(GRAV_FC, 0.4 * fs_m / 2), "low", fs=fs_m,
                        output="sos")
    grav = signal.sosfiltfilt(sos, a, axis=0)
    lin = a - grav
    qom_raw = np.linalg.norm(np.gradient(lin, dt_m, axis=0), axis=1)
    g = grav.mean(axis=0)
    g = g / (np.linalg.norm(g) + EPS)
    e1 = np.array([1.0, 0.0, 0.0]) - g[0] * g
    if np.linalg.norm(e1) < 1e-6:            # gravity along device x
        e1 = np.array([0.0, 1.0, 0.0]) - g[1] * g
    e1 = e1 / (np.linalg.norm(e1) + EPS)
    e2 = np.cross(g, e1)
    return tg, dt_m, qom_raw, lin @ e1, lin @ e2


def join(sess, motion_path, F=None) -> dict:
    """Resample audio features and motion onto one common 8 Hz clock.

    ``F`` is a loaded feature cache (:func:`ambiscape.features.load_features`);
    if omitted it is loaded from ``<session>/analysis/features`` (run
    ``ambiscape analyze`` first). Returns a dict of aligned series over the
    overlapping span, at the fast-level rate (125 ms): ``t`` (absolute
    session seconds), ``dt``, ``level_db`` (125 ms fast level), ``qom``
    (mean jerk magnitude per frame), ``sway_deg`` (principal axis of
    horizontal micromotion, axial, −90..90), ``sway_pow`` (horizontal
    micromotion energy per frame), and ``az_deg`` (audio azimuth
    interpolated to the common clock; all-NaN for mono input).
    """
    if F is None:
        from .features import load_features
        paths = sorted((Path(sess.folder) / "analysis" / "features")
                       .glob("*.npz"))
        if not paths:
            raise FileNotFoundError(
                f"no cached features under {sess.folder}/analysis — run "
                "'ambiscape analyze' first")
        F = load_features(paths)
    t_m, acc, meta = load_motion(motion_path)
    dt = float(np.median(np.diff(F["t_fast"])))
    ta0, ta1 = float(F["t_fast"][0]), float(F["t_fast"][-1]) + dt
    if meta["date0"] is not None and sess.day0 is not None:
        t_m = t_m + (meta["date0"] - sess.day0).days * 86400.0
    if t_m[-1] <= ta0 or t_m[0] >= ta1:      # disjoint: a relative clock
        t_m = t_m - t_m[0] + ta0
    tg, dt_m, qom_raw, h1, h2 = _micromotion(t_m, acc)

    t0, t1 = max(ta0, float(tg[0])), min(ta1, float(tg[-1]))
    n = int((t1 - t0) / dt)
    if n < 16:
        raise ValueError(
            f"audio and motion overlap for only {max(t1 - t0, 0):.1f} s — "
            "check the motion file's timestamps")
    tc = t0 + dt * np.arange(n)
    level = np.interp(tc + dt / 2, F["t_fast"] + dt / 2,
                      np.asarray(F["fast_db"], float))
    idx = np.floor((tg - t0) / dt).astype(int)
    ok = (idx >= 0) & (idx < n)
    idx = idx[ok]
    cnt = np.maximum(np.bincount(idx, minlength=n), 1)
    qom = np.bincount(idx, weights=qom_raw[ok], minlength=n) / cnt
    sxx = np.bincount(idx, weights=h1[ok] ** 2, minlength=n)
    syy = np.bincount(idx, weights=h2[ok] ** 2, minlength=n)
    sxy = np.bincount(idx, weights=h1[ok] * h2[ok], minlength=n)
    sway = 0.5 * np.degrees(np.arctan2(2 * sxy, sxx - syy))
    sway_pow = (sxx + syy) / cnt
    gaps = np.bincount(idx, minlength=n) == 0
    if gaps.any():                            # bridge motion drop-outs
        qom[gaps] = np.interp(tc[gaps], tc[~gaps], qom[~gaps])
        sway_pow[gaps] = 0.0
    az_sec = np.asarray(F["az"], float)
    az = np.full(n, np.nan)
    if np.isfinite(az_sec).any():
        fin = np.isfinite(az_sec)
        ts = np.asarray(F["t"], float)[fin] + 0.5
        rad = np.radians(az_sec[fin])         # interpolate on the circle
        az = np.degrees(np.arctan2(np.interp(tc + dt / 2, ts, np.sin(rad)),
                                   np.interp(tc + dt / 2, ts, np.cos(rad))))
    return {"t": tc, "dt": dt, "level_db": level.astype(float), "qom": qom,
            "sway_deg": sway, "sway_pow": sway_pow, "az_deg": az,
            "motion_fs_hz": round(1.0 / dt_m, 2), "n": n}


def _shift_p(obs: float, stat, n: int, n_surrogates: int, min_shift: int,
             rng) -> tuple[float, float]:
    """Two-sided p and null 95th percentile from circular-shift surrogates.

    ``stat(k)`` recomputes the statistic with one series rolled by ``k``
    frames. Circular time shifts keep each series' autocorrelation intact —
    the correct null for slow, self-similar signals, where naive shuffling
    would wildly overstate significance.
    """
    lo = min(min_shift, max(n // 4, 1))
    ks = rng.integers(lo, n - lo, size=n_surrogates)
    null = np.array([stat(int(k)) for k in ks])
    p = (1 + int((np.abs(null) >= abs(obs)).sum())) / (1 + n_surrogates)
    return float(p), float(np.percentile(np.abs(null), 95))


def temporal_correlation(level_db, qom, dt, n_surrogates=200,
                         min_shift_s=10.0, seed=0) -> dict:
    """Pearson r between fast level (dB) and log-QoM, surrogate-tested.

    Both series are compressed (dB and log) so a single loud event or jolt
    does not dominate the correlation. The p-value is two-sided against
    circular time-shift surrogates of the motion series.
    """
    x = np.asarray(level_db, float)
    y = np.log10(np.asarray(qom, float) + EPS)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x ** 2).sum() * (y ** 2).sum()) + EPS
    r = float((x * y).sum() / denom)
    p, null95 = _shift_p(
        r, lambda k: float((x * np.roll(y, k)).sum() / denom), len(x),
        n_surrogates, int(min_shift_s / dt), np.random.default_rng(seed))
    return {"r": round(r, 4), "p": round(p, 4), "null95": round(null95, 4),
            "n": len(x), "n_surrogates": n_surrogates}


def directional_correlation(az_deg, sway_deg, mask=None, dt=0.125,
                            n_surrogates=200, min_shift_s=10.0,
                            seed=0) -> dict:
    """Circular correlation between audio azimuth and sway direction.

    The sway direction is axial (a principal axis, period 180°), so it is
    angle-doubled to the full circle before the Jammalamadaka–SenGupta
    coefficient is taken; rotation invariance means the two frames need no
    alignment (the sign of rho is still frame-handedness dependent — judge
    coupling by |rho| and p). ``mask`` selects the frames that enter the
    statistic (e.g. frames where both streams carry energy); surrogates
    roll the full sway series before masking, preserving its rhythm.
    """
    a = np.radians(np.asarray(az_deg, float))
    b = np.radians(2.0 * np.asarray(sway_deg, float))
    if mask is None:
        mask = np.ones(len(a), bool)
    mask = mask & np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 16:
        return {"rho": None, "p": None, "n": int(mask.sum())}
    rho = circ_corr(a[mask], b[mask])
    p, null95 = _shift_p(
        rho, lambda k: circ_corr(a[mask], np.roll(b, k)[mask]),
        len(a), n_surrogates, int(min_shift_s / dt),
        np.random.default_rng(seed))
    return {"rho": round(rho, 4), "p": round(p, 4),
            "null95": round(null95, 4), "n": int(mask.sum()),
            "n_surrogates": n_surrogates}


def plv(audio_env, motion, dt, bands=BANDS, n_surrogates=200,
        min_shift_s=10.0, seed=0) -> list[dict]:
    """Phase-locking value per modulation band between envelope and motion.

    Both series are z-scored, band-passed (zero-phase Butterworth), and
    Hilbert-transformed; PLV = |mean e^{i(φ_audio − φ_motion)}| in [0, 1].
    Bands the record is too short for (fewer than four cycles of the low
    edge) or too slow for (above 0.4/dt) are skipped. Each band carries a
    surrogate p (circular shift of the motion phase series) and the null's
    95th percentile — the significance floor drawn in the figure.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(audio_env, float)
    y = np.asarray(motion, float)
    x = (x - x.mean()) / (x.std() + EPS)
    y = (y - y.mean()) / (y.std() + EPS)
    n = len(x)
    out = []
    for lo, hi in bands:
        if hi > 0.4 / dt or n * dt < 4.0 / lo:
            continue
        sos = signal.butter(3, (lo, hi), "band", fs=1.0 / dt, output="sos")
        ph_x = np.angle(signal.hilbert(signal.sosfiltfilt(sos, x)))
        ph_y = np.angle(signal.hilbert(signal.sosfiltfilt(sos, y)))
        v = float(np.abs(np.exp(1j * (ph_x - ph_y)).mean()))
        p, null95 = _shift_p(
            v, lambda k: float(np.abs(
                np.exp(1j * (ph_x - np.roll(ph_y, k))).mean())),
            n, n_surrogates, int(min_shift_s / dt), rng)
        out.append({"band_hz": [lo, hi], "plv": round(v, 4),
                    "p": round(p, 4), "null95": round(null95, 4)})
    return out


def render(J: dict, doc: dict, out_path, title="", clock=None):
    """Combined figure: aligned timelines, azimuth-vs-sway rose, PLV bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12.8, 7.2), dpi=130)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    tt = J["t"] - J["t"][0]

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(tt, J["level_db"], color="#2a78d6", lw=0.8, label="fast level")
    ax0.set(ylabel="level (dBFS)",
            title=f"{title} — sound and motion on one clock "
                  f"(r {doc['temporal']['r']}, p {doc['temporal']['p']})")
    axq = ax0.twinx()
    axq.plot(tt, 10 * np.log10(J["qom"] + EPS), color="#d66a2a", lw=0.8,
             alpha=0.8, label="QoM")
    axq.set_ylabel("QoM (dB)", color="#d66a2a")
    ax0.set_xlabel("time (s)")
    if clock is not None:
        xt = ax0.get_xticks()
        ax0.set_xticks(xt)
        ax0.set_xticklabels([clock(J["t"][0] + x)[7:] for x in xt],
                            fontsize=8)
        ax0.set_xlim(tt[0], tt[-1])
        ax0.set_xlabel("")
    ax0.grid(alpha=0.2)

    ax1 = fig.add_subplot(gs[1, 0], projection="polar")
    dc = doc.get("directional") or {}
    if np.isfinite(J["az_deg"]).any():
        pw = 10 ** (J["level_db"] / 10)
        edges = np.radians(np.linspace(-180, 180, 25))
        ha, _ = np.histogram(np.radians(J["az_deg"]), bins=edges, weights=pw)
        hs, _ = np.histogram(np.radians(np.concatenate(
            [J["sway_deg"], J["sway_deg"] + 180])), bins=edges,
            weights=np.concatenate([J["sway_pow"]] * 2))
        cent = 0.5 * (edges[:-1] + edges[1:])
        w = edges[1] - edges[0]
        ax1.bar(cent, ha / (ha.max() + EPS), width=w, color="#2a78d6",
                alpha=0.55, label="audio azimuth energy")
        ax1.bar(cent, hs / (hs.max() + EPS), width=w, color="#d66a2a",
                alpha=0.55, label="sway direction (axial)")
        ax1.set_title("azimuth vs sway"
                      + (f" (rho {dc.get('rho')}, p {dc.get('p')})"
                         if dc.get("rho") is not None else ""), fontsize=10)
        ax1.set_theta_zero_location("N")
        ax1.set_theta_direction(1)
        ax1.set_thetagrids([0, 90, 180, 270],
                           ["front", "left", "rear", "right"], fontsize=8.5)
        ax1.set_rticks([])
        ax1.legend(fontsize=7, loc="lower left",
                   bbox_to_anchor=(-0.12, -0.12))
    else:
        ax1.text(0, 0, "no directional audio\n(mono input)",
                 ha="center", va="center")
        ax1.set_axis_off()

    ax2 = fig.add_subplot(gs[1, 1])
    pb = doc["plv"]
    if pb:
        xs = np.arange(len(pb))
        cols = ["#3d9970" if b["p"] < 0.05 else "0.7" for b in pb]
        ax2.bar(xs, [b["plv"] for b in pb], color=cols, width=0.7)
        ax2.plot(xs, [b["null95"] for b in pb], "_", ms=22, color="0.2",
                 label="surrogate 95%")
        ax2.set_xticks(xs, [f"{b['band_hz'][0]:g}–{b['band_hz'][1]:g}"
                            for b in pb], fontsize=8)
        ax2.legend(fontsize=8)
    ax2.set(xlabel="modulation band (Hz)", ylabel="PLV", ylim=(0, 1),
            title="phase locking by band (green = p < 0.05)")
    ax2.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def analyze_entrainment(sess, motion_path, out_dir=None, F=None, bands=BANDS,
                        n_surrogates=200, seed=0) -> dict:
    """Full sound–motion entrainment analysis of one session.

    Joins the session's cached features with the motion file, computes the
    three measures, writes ``entrain.png`` + ``entrain.json`` under
    ``out_dir`` (default ``<session>/analysis``), folds the ``ent_``
    summary rows into an existing ``summary.json`` there, and returns the
    document.
    """
    import json
    out_dir = Path(out_dir) if out_dir else Path(sess.folder) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    J = join(sess, motion_path, F=F)
    dt = J["dt"]
    tc = temporal_correlation(J["level_db"], J["qom"], dt,
                              n_surrogates=n_surrogates, seed=seed)
    pw = 10 ** (J["level_db"] / 10)
    pl = plv(pw, J["qom"], dt, bands=bands, n_surrogates=n_surrogates,
             seed=seed)
    dc = None
    if np.isfinite(J["az_deg"]).any():
        active = ((pw >= np.median(pw))
                  & (J["sway_pow"] >= np.median(J["sway_pow"])))
        dc = directional_correlation(J["az_deg"], J["sway_deg"], mask=active,
                                     dt=dt, n_surrogates=n_surrogates,
                                     seed=seed)
    best = max(pl, key=lambda b: b["plv"]) if pl else None
    summary = {
        "ent_overlap_min": round(J["n"] * dt / 60.0, 2),
        "ent_r_level_qom": tc["r"], "ent_r_p": tc["p"],
        "ent_az_sway_rho": dc["rho"] if dc else None,
        "ent_az_sway_p": dc["p"] if dc else None,
        "ent_plv_max": best["plv"] if best else None,
        "ent_plv_max_band_hz": (round(float(np.sqrt(
            best["band_hz"][0] * best["band_hz"][1])), 3) if best else None),
        "ent_plv_max_p": best["p"] if best else None,
    }
    doc = {
        "motion_file": str(Path(motion_path).name),
        "motion_fs_hz": J["motion_fs_hz"],
        "overlap_s": round(J["n"] * dt, 1),
        "temporal": tc, "directional": dc, "plv": pl,
        "summary": summary,
        "_method_note": (
            "Guo–Riaz–Jensenius crossmodal method: 125 ms fast level vs "
            "quantity of motion (jerk magnitude after 0.25 Hz gravity "
            "removal) on a common 8 Hz clock; Pearson r on dB/log series; "
            "Jammalamadaka–SenGupta circular correlation between audio "
            "azimuth and the angle-doubled principal axis of horizontal "
            "micromotion, on frames above median energy in both streams; "
            "per-band Hilbert PLV of envelope vs QoM. All p-values are "
            "two-sided against circular time-shift surrogates."),
    }
    render(J, doc, out_dir / "entrain.png", title=sess.name,
           clock=sess.clock)
    (out_dir / "entrain.json").write_text(
        json.dumps(doc, indent=2, default=float))
    sp = out_dir / "summary.json"
    if sp.exists():                          # the multimodal join
        s = json.loads(sp.read_text())
        s.update(summary)
        sp.write_text(json.dumps(s, indent=2))
    return doc
