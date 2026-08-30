"""Walk mode: segment-first analysis of a moving recording.

A soundwalk breaks the toolbox's core assumption — one microphone, one
place, one soundscape whose statistics converge with time. Here the
non-stationarity is the signal, so the analysis inverts: segment the walk
into sub-soundscape zones first (:mod:`ambiscape.segmentation`), then
describe each zone as a short stationary session, the way ISO/TS 12913-2
soundwalks treat their stop points.

Two things are computed that a stationary session never needs. Step
cadence: the walker's own footsteps are a quasi-periodic impulse train at
roughly 1–3 Hz in the 50 Hz envelope, and their rate is a walking-speed
proxy while their salience says how much of each zone's foreground is
self-noise. And a figure–ground regime per zone: on a walk the walker's
steps, external events, or nothing at all may be the foreground, and the
regime label says which (``own-steps`` / ``external-events`` /
``bed-only`` / ``mixed``).

Session-level ecoacoustic indices assume a fixed sensor and are not
reported per walk; between-zone comparisons within one walk and one rig
are the defensible use of everything here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import detrend, welch

from .analysis import detect_events
from .segmentation import segment

CADENCE_WIN_S = 20.0
CADENCE_HOP_S = 5.0
CADENCE_LO_HZ = 1.0
CADENCE_HI_HZ = 3.2
CLEAR_GAIT_SALIENCE = 6.0   # above: the steps own the zone's foreground
EVENT_RICH_DB = 12.0        # L10-L90 above: external events dominate
BED_ONLY_DB = 6.0           # L10-L90 below: nothing emerges from the bed


def cadence_track(F: dict, win_s: float = CADENCE_WIN_S,
                  hop_s: float = CADENCE_HOP_S) -> dict:
    """Step cadence and salience over the walk, from the 50 Hz envelope.

    Welch spectrum of the detrended log-envelope per window; cadence is the
    peak in the 1.0–3.2 Hz gait band, salience the ratio of that peak to
    the median spectrum over 0.5–5 Hz. Returns ``{"t", "cadence_hz",
    "salience"}`` with one row per hop.
    """
    env = np.asarray(F["env_hi"], float)
    fs = 1.0 / float(F["hi_dt"])
    t0 = float(np.asarray(F["t_hi"])[0]) if "t_hi" in F else 0.0
    n = len(env)
    win, hop = int(win_s * fs), int(hop_s * fs)
    ts, cad, sal = [], [], []
    for a in range(0, n - win, hop):
        seg = detrend(np.log(env[a:a + win] + 1e-10))
        f, P = welch(seg, fs=fs, nperseg=win // 2)
        band = (f >= CADENCE_LO_HZ) & (f <= CADENCE_HI_HZ)
        ref = (f >= 0.5) & (f <= 5.0)
        if not band.any() or not ref.any():
            continue
        pk = int(np.argmax(P[band]))
        ts.append(t0 + (a + win / 2) / fs)
        cad.append(float(f[band][pk]))
        sal.append(float(P[band][pk] / (np.median(P[ref]) + 1e-20)))
    return {"t": np.array(ts), "cadence_hz": np.array(cad),
            "salience": np.array(sal)}


def classify_regime(l10_l90_db: float, step_salience: float) -> str:
    """Which layer is the foreground in a zone.

    ``own-steps`` when the gait stands clear of everything else,
    ``external-events`` when the level dynamics say things keep emerging,
    ``bed-only`` when nothing does, ``mixed`` otherwise.
    """
    if step_salience > CLEAR_GAIT_SALIENCE:
        return "own-steps"
    if l10_l90_db > EVENT_RICH_DB:
        return "external-events"
    if l10_l90_db < BED_ONLY_DB:
        return "bed-only"
    return "mixed"


def _zone_row(F: dict, ev_times: np.ndarray, tr: dict,
              a: float, b: float) -> dict:
    t = np.asarray(F["t"], float)
    sel = (t >= a) & (t < b)
    tf = np.asarray(F["t_fast"], float)
    self_ = (tf >= a) & (tf < b)
    fast = np.asarray(F["fast_db"], float)[self_]
    op = np.asarray(F["oct_pow"], float)[sel]
    tot = float(op.sum()) or 1.0
    l10, l90 = (np.percentile(fast, [90, 10]) if fast.size
                else (np.nan, np.nan))
    m = (tr["t"] >= a) & (tr["t"] < b)
    good = m & (tr["salience"] > 3.0)
    cad = float(np.median(tr["cadence_hz"][good])) if good.any() else np.nan
    sal = float(np.median(tr["salience"][m])) if m.any() else np.nan
    n_ev = int(((ev_times >= a) & (ev_times < b)).sum())
    dur = b - a
    return {
        "t0": float(a), "t1": float(b), "duration_s": round(dur, 1),
        "leq_dbfs": round(float(10 * np.log10(
            np.mean(10 ** (fast / 10)) + 1e-20)), 1) if fast.size else None,
        "l10_l90_db": round(float(l10 - l90), 1),
        "lf_fraction": round(float(op[:, :4].sum() / tot), 2),
        "mf_fraction": round(float(op[:, 4:7].sum() / tot), 2),
        "hf_fraction": round(float(op[:, 7:].sum() / tot), 2),
        "flatness": round(float(np.median(F["flatness"][sel])), 4),
        "centroid_hz": int(np.median(np.asarray(F["centroid"])[sel])),
        "diffuseness": round(float(np.median(np.asarray(F["diffuse"])[sel])), 2),
        "events_per_min": round(n_ev / (dur / 60.0), 1) if dur else None,
        "cadence_hz": round(cad, 2) if cad == cad else None,
        "steps_per_min": int(round(cad * 60)) if cad == cad else None,
        "step_salience": round(sal, 1) if sal == sal else None,
        "regime": classify_regime(float(l10 - l90),
                                  sal if sal == sal else 0.0),
    }


def analyze_walk(F: dict, min_seg_s: float = 30.0,
                 threshold_db: float = 4.0) -> dict:
    """Segment the walk and describe each zone.

    Returns ``{"boundaries", "zones", "cadence"}`` — boundary times on
    ``F["t"]``'s clock, one descriptor row per zone, and the cadence track.
    """
    t = np.asarray(F["t"], float)
    bounds = segment(F, min_seg_s=min_seg_s, threshold_db=threshold_db)
    edges = [float(t[0])] + bounds + [float(t[-1]) + 1.0]
    tr = cadence_track(F)
    fd = float(np.median(np.diff(F["t_fast"]))) if len(F["t_fast"]) > 1 else 0.125
    events, _bg = detect_events(np.asarray(F["fast_db"], float), fd)
    tf = np.asarray(F["t_fast"], float)
    ev_times = np.array([tf[e["ipk"]] for e in events])
    zones = [_zone_row(F, ev_times, tr, a, b)
             for a, b in zip(edges[:-1], edges[1:])]
    return {"boundaries": bounds, "zones": zones, "cadence": tr}


def _fmt_clock(t: float) -> str:
    s = int(t % 86400)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _route_profile(F: dict, r: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(F["t"], float)
    t0 = t[0]
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1, 1]})
    if "logspec" in F:
        sp = 10 * np.log10(np.asarray(F["logspec"], float) + 1e-12).T
        axes[0].imshow(sp, aspect="auto", origin="lower", cmap="Blues",
                       extent=(0, t[-1] - t0, 0, sp.shape[0]))
    axes[0].set_ylabel("log-f bin")
    tf = np.asarray(F["t_fast"], float) - t0
    axes[1].plot(tf, F["fast_db"], lw=0.4, color="#3567b0")
    axes[1].set_ylabel("dBFS")
    tr = r["cadence"]
    good = tr["salience"] > 3.0
    axes[2].plot(tr["t"][good] - t0, tr["cadence_hz"][good] * 60, ".",
                 ms=3, color="#3567b0")
    axes[2].set_ylabel("steps/min")
    axes[2].set_ylim(40, 200)
    axes[3].plot(t - t0, F["diffuse"], lw=0.6, color="#2e7d32")
    axes[3].set_ylabel("ψ")
    axes[3].set_ylim(0, 1)
    axes[3].set_xlabel("time (s)")
    for ax in axes:
        for b in r["boundaries"]:
            ax.axvline(b - t0, color="#b0413e", lw=0.8, alpha=0.7)
    fig.suptitle("route profile — zones from feature novelty")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


_TSV_COLS = ("t0", "t1", "duration_s", "leq_dbfs", "l10_l90_db",
             "lf_fraction", "mf_fraction", "hf_fraction", "flatness",
             "centroid_hz", "diffuseness", "events_per_min", "cadence_hz",
             "steps_per_min", "step_salience", "regime")


def write_walk(F: dict, folder: str | Path, out_dir: str | Path) -> dict:
    """Run :func:`analyze_walk` and write its outputs beside the analysis.

    Writes ``walk_zones.tsv`` (one row per zone), ``walk.md`` (the zone
    table with clock times and regimes), and ``route_profile.png``.
    Returns the :func:`analyze_walk` result.
    """
    out = Path(out_dir)
    r = analyze_walk(F)
    with open(out / "walk_zones.tsv", "w") as fh:
        fh.write("\t".join(_TSV_COLS) + "\n")
        for z in r["zones"]:
            fh.write("\t".join(str(z[c]) for c in _TSV_COLS) + "\n")
    lines = ["# Walk zones", "",
             f"{len(r['zones'])} zones from {len(r['boundaries'])} feature-"
             "novelty boundaries. Between-zone comparison within this walk "
             "and rig is the defensible use; session-level ecoacoustic "
             "indices are not reported for a moving recording.", "",
             "| Zone | Clock | Leq | L10−L90 | Centroid | ψ | Ev/min "
             "| Steps/min | Regime |", "|---|---|---|---|---|---|---|---|---|"]
    for i, z in enumerate(r["zones"], 1):
        lines.append(
            f"| {i} | {_fmt_clock(z['t0'])}–{_fmt_clock(z['t1'])} "
            f"| {z['leq_dbfs']} | {z['l10_l90_db']} | {z['centroid_hz']} "
            f"| {z['diffuseness']} | {z['events_per_min']} "
            f"| {z['steps_per_min'] if z['steps_per_min'] else '—'} "
            f"| {z['regime']} |")
    (out / "walk.md").write_text("\n".join(lines) + "\n")
    _route_profile(F, r, out / "route_profile.png")
    return r
