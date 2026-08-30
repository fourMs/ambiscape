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


GATE_SALIENCE = 5.0        # zone gait salience above which gating engages
STEP_HALO_S = 0.07         # masked around each detected footfall
WIND_SCORE = 0.5           # per-second LF-fraction x diffuseness above: gust


def step_mask(F: dict) -> np.ndarray:
    """True at 8 Hz fast frames that carry the walker's own footfalls.

    Envelope impulses (50 Hz layer) standing 6 dB over their 30 s local
    mean, dilated by ``STEP_HALO_S``, mapped onto the fast-level clock.
    The mask says where the steps are; whether they matter is the zone's
    gait salience, which is why gating is decided per zone, not here.
    """
    from scipy.ndimage import binary_dilation, uniform_filter1d
    env = np.asarray(F["env_hi"], float)
    hi_dt = float(F["hi_dt"])
    base = uniform_filter1d(env, max(1, int(30.0 / hi_dt)))
    hits = env > 2.0 * base
    hits = binary_dilation(hits, iterations=max(1, int(STEP_HALO_S / hi_dt)))
    th = np.asarray(F["t_hi"], float)
    tf = np.asarray(F["t_fast"], float)
    return np.interp(tf, th, hits.astype(float)) > 0.3


def wind_mask(F: dict) -> np.ndarray:
    """True at 1 Hz seconds that read as wind on the capsules.

    Wind is low, broadband and directionless: the per-second low-band
    fraction (below ~200 Hz) times diffuseness, over ``WIND_SCORE``.
    Shares geophony's caveat — HVAC rumble in a diffuse room scores too;
    on an outdoor walk the reading is usually honest.
    """
    op = np.asarray(F["oct_pow"], float)
    lf = op[:, :3].sum(1) / (op.sum(1) + 1e-20)
    score = lf * np.asarray(F["diffuse"], float)
    return score > WIND_SCORE


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
              a: float, b: float, smask=None, wmask=None) -> dict:
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
    row = {
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
    # self-noise: what the zone sounds like without the walker
    wsec = (np.asarray(F["t"], float) >= a) & (np.asarray(F["t"], float) < b)
    row["wind_time_fraction"] = (round(float(wmask[wsec].mean()), 2)
                                 if wmask is not None and wsec.any() else None)
    row["leq_gated_dbfs"] = None
    row["step_time_fraction"] = None
    if (smask is not None and fast.size
            and sal == sal and sal > GATE_SALIENCE):
        zmask = smask[self_]
        if wmask is not None:
            tfz = tf[self_]
            zmask = zmask | (np.interp(tfz, np.asarray(F["t"], float),
                                       wmask.astype(float)) > 0.5)
        keep = fast[~zmask]
        if keep.size:
            row["leq_gated_dbfs"] = round(float(
                10 * np.log10(np.mean(10 ** (keep / 10)) + 1e-20)), 1)
            row["step_time_fraction"] = round(float(zmask.mean()), 2)
    return row


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
    smask = step_mask(F) if "env_hi" in F else None
    wmask = wind_mask(F) if "oct_pow" in F else None
    zones = [_zone_row(F, ev_times, tr, a, b, smask, wmask)
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


_GEO_COLS = ("distance_m", "speed_ms", "lat", "lon")

_TSV_COLS = ("t0", "t1", "duration_s", "leq_dbfs", "leq_gated_dbfs",
             "step_time_fraction", "wind_time_fraction", "l10_l90_db",
             "lf_fraction", "mf_fraction", "hf_fraction", "flatness",
             "centroid_hz", "diffuseness", "events_per_min", "cadence_hz",
             "steps_per_min", "step_salience", "regime")


def _route_map(track: dict, zones: list, epoch0: float,
               out_path: Path) -> None:
    """The walked route, coloured by zone index, one label per zone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("viridis", max(len(zones), 2))
    tt, la, lo = track["t"], track["lat"], track["lon"]
    for i, z in enumerate(zones):
        a, b = z["t0"] + epoch0, z["t1"] + epoch0
        ts = np.arange(max(a, tt[0]), min(b, tt[-1]), 1.0)
        if len(ts) < 2:
            continue
        ax.plot(np.interp(ts, tt, lo), np.interp(ts, tt, la),
                color=cmap(i), lw=2.5)
        if z.get("lat") is not None:
            ax.annotate(str(i + 1), (z["lon"], z["lat"]), fontsize=8,
                        ha="center", va="center",
                        bbox=dict(boxstyle="circle", fc="white",
                                  ec=cmap(i), lw=1.2))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(la))))
    ax.set_title("route, coloured by zone")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_walk(F: dict, folder: str | Path, out_dir: str | Path,
               track: dict | None = None, epoch0: float = 0.0) -> dict:
    """Run :func:`analyze_walk` and write its outputs beside the analysis.

    Writes ``walk_zones.tsv`` (one row per zone), ``walk.md`` (the zone
    table with clock times and regimes), and ``route_profile.png``. With a
    GPS ``track`` (:func:`ambiscape.geo.load_gpx`) and ``epoch0`` (the
    epoch second matching session time zero), each zone also carries
    distance, mean speed and midpoint position, and ``route_map.png`` is
    drawn. Returns the :func:`analyze_walk` result.
    """
    out = Path(out_dir)
    r = analyze_walk(F)
    cols = _TSV_COLS
    if track is not None:
        from .geo import zone_geo
        for z in r["zones"]:
            z.update(zone_geo(track, z["t0"] + epoch0, z["t1"] + epoch0))
        cols = _TSV_COLS + _GEO_COLS
        _route_map(track, r["zones"], epoch0, out / "route_map.png")
    with open(out / "walk_zones.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for z in r["zones"]:
            fh.write("\t".join(str(z[c]) for c in cols) + "\n")
    lines = ["# Walk zones", "",
             f"{len(r['zones'])} zones from {len(r['boundaries'])} feature-"
             "novelty boundaries. Between-zone comparison within this walk "
             "and rig is the defensible use; session-level ecoacoustic "
             "indices are not reported for a moving recording.", "",
             "| Zone | Clock | Leq | Leq w/o self-noise | L10−L90 | Centroid "
             "| ψ | Ev/min | Steps/min | Regime |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, z in enumerate(r["zones"], 1):
        lines.append(
            f"| {i} | {_fmt_clock(z['t0'])}–{_fmt_clock(z['t1'])} "
            f"| {z['leq_dbfs']} "
            f"| {z['leq_gated_dbfs'] if z['leq_gated_dbfs'] is not None else '—'} "
            f"| {z['l10_l90_db']} | {z['centroid_hz']} "
            f"| {z['diffuseness']} | {z['events_per_min']} "
            f"| {z['steps_per_min'] if z['steps_per_min'] else '—'} "
            f"| {z['regime']} |")
    (out / "walk.md").write_text("\n".join(lines) + "\n")
    _route_profile(F, r, out / "route_profile.png")
    return r
