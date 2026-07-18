"""Spatial dynamics at three time scales.

From the cached per-second spatial features (pseudo-intensity per octave,
DOA, diffuseness) — no audio pass:

- ``direct_diffuse_split`` — per-octave directness (1 − diffuseness proxy)
  per second: the spatial analogue of foreground/background;
- ``passby_events`` — level events whose azimuth sweeps monotonically
  through the event: moving sources, with sweep rate and direction;
- ``azimuth_organization`` — windowed, energy-weighted circular
  concentration R(t): how directionally organized the scene is over time.

``run_session`` writes ``spatial.json`` + ``spatial.png``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .circstats import mean_resultant

EPS = 1e-20


def direct_diffuse_split(F: dict):
    """Per-octave directness in [0, 1]: |pseudo-intensity| / band power.

    Uses the cached ``I_band`` (re W*X etc. per octave) and ``oct_pow``.
    A plane wave scores near 1, a diffuse field near 0. Returns
    (directness[nsec, nband], per-band medians).
    """
    I = np.linalg.norm(F["I_band"], axis=2)
    d = np.clip(I / (F["oct_pow"] + EPS), 0, 1)
    return d, np.median(d, axis=0)


def passby_events(F: dict, min_dur_s=4, min_sweep_deg=25.0, min_r2=0.7):
    """Level events whose azimuth sweeps steadily: moving sources.

    Detects events with :func:`ambiscape.analysis.detect_events`, then fits
    a line to the unwrapped per-second azimuth across each event lasting
    >= ``min_dur_s``. A sweep of >= ``min_sweep_deg`` with fit R^2 >=
    ``min_r2`` is a pass-by; the sweep sign gives the direction of travel
    (mic frame). Returns a list of dicts.
    """
    from .analysis import detect_events
    dt = float(np.median(np.diff(F["t_fast"])))
    events, _bg = detect_events(F["fast_db"], dt)
    t0_abs = float(F["t"][0])
    out = []
    for e in events:
        a = F["t_fast"][e["i0"]] - t0_abs
        b = F["t_fast"][e["i1"]] - t0_abs
        i0, i1 = int(a), int(np.ceil(b))
        if i1 - i0 < min_dur_s or i1 >= len(F["az"]):
            continue
        az = np.unwrap(np.radians(F["az"][i0:i1]))
        x = np.arange(len(az), dtype=float)
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A, az, rcond=None)
        tot = ((az - az.mean()) ** 2).sum()
        r2 = 1 - float(res[0]) / (tot + EPS) if len(res) else 0.0
        sweep = float(np.degrees(coef[0]) * len(az))
        if abs(sweep) >= min_sweep_deg and r2 >= min_r2:
            out.append({
                "t0_s": i0, "dur_s": i1 - i0,
                "sweep_deg": round(sweep, 1),
                "rate_deg_s": round(float(np.degrees(coef[0])), 1),
                "direction": "left-to-right" if sweep < 0 else "right-to-left",
                "r2": round(r2, 2),
            })
    return out


def azimuth_organization(F: dict, win_s=60.0, step_s=20.0):
    """Windowed energy-weighted circular concentration of the azimuth.

    Returns (t_centers, R): R near 1 = one dominant direction, near 0 =
    directionally disorganized. Window in seconds (per-second features).
    """
    p = F["rms_w"].astype(np.float64) ** 2
    az = np.radians(F["az"])
    n, w, s = len(az), int(win_s), int(step_s)
    ts, Rs = [], []
    for i0 in range(0, n - w + 1, s):
        _mu, R = mean_resultant(az[i0:i0 + w], weights=p[i0:i0 + w])
        ts.append(float(F["t"][i0] + w / 2 - F["t"][0]))
        Rs.append(R)
    return np.array(ts), np.array(Rs)


def directional_entropy(F: dict, nbins: int = 36) -> float:
    """Normalized Shannon entropy of the energy-weighted azimuth histogram.

    "How many directions does this place sound from": 0 = all energy from
    one bearing, 1 = energy spread evenly around the horizon — the spatial
    analogue of an acoustic diversity index, and something only an
    ambisonic corpus can report.
    """
    p = np.asarray(F["rms_w"], np.float64) ** 2
    h, _ = np.histogram(F["az"], bins=nbins, range=(-180, 180), weights=p)
    q = h / (h.sum() + EPS)
    return float(-(q * np.log(q + EPS)).sum() / np.log(nbins))


def horizon_fractions(F: dict, limit_deg: float = 10.0) -> dict:
    """Energy fractions arriving from above / around / below the horizon.

    Uses the per-second broadband DOA elevation, energy-weighted. A room
    heard from a couch splits mechanics on walls (above) from footsteps
    and structure-borne paths (level/below); outdoors it separates birds
    and building services from ground traffic.
    """
    p = np.asarray(F["rms_w"], np.float64) ** 2
    el = np.asarray(F["el"], float)
    tot = p.sum() + EPS
    return {"above": round(float(p[el > limit_deg].sum() / tot), 2),
            "level": round(float(p[np.abs(el) <= limit_deg].sum() / tot), 2),
            "below": round(float(p[el < -limit_deg].sum() / tot), 2)}


def fg_bg_az_overlap(F: dict, nbins: int = 36) -> float:
    """Bhattacharyya overlap of foreground vs background azimuth energy.

    Foreground = loudest 25 % of seconds, background = quietest 25 % (the
    corpus convention). 1 = the foreground comes from where the background
    hums (one-source rooms), 0 = figure and ground occupy different
    directions (a street heard past a courtyard fountain).
    """
    p = np.asarray(F["rms_w"], np.float64) ** 2
    fg = p >= np.percentile(p, 75)
    bg = p <= np.percentile(p, 25)
    hists = []
    for m in (fg, bg):
        h, _ = np.histogram(F["az"][m], bins=nbins, range=(-180, 180),
                            weights=p[m])
        hists.append(h / (h.sum() + EPS))
    return float(np.sqrt(hists[0] * hists[1]).sum())


def summarize_spatial(F: dict) -> dict:
    """Spatial descriptors for the analyze summary."""
    hf = horizon_fractions(F)
    return {
        "directional_entropy": round(directional_entropy(F), 3),
        "above_horizon_fraction": hf["above"],
        "below_horizon_fraction": hf["below"],
        "fgbg_az_overlap": round(fg_bg_az_overlap(F), 2),
    }


def run_session(sess, out_dir) -> dict:
    """CLI driver: split + pass-bys + R(t), figure + spatial.json."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .features import load_features, OCT_CENTERS

    out_dir = Path(out_dir)
    F = load_features(sorted((out_dir / "features").glob("*.npz")))
    d, dmed = direct_diffuse_split(F)
    pb = passby_events(F)
    ts, Rs = azimuth_organization(F)
    doc = {
        "directness_median_per_octave": {
            str(int(c)): round(float(v), 2)
            for c, v in zip(OCT_CENTERS, dmed)},
        "azimuth_R_median": round(float(np.median(Rs)), 2),
        "azimuth_R_iqr": round(float(np.percentile(Rs, 75)
                                     - np.percentile(Rs, 25)), 2),
        "passbys": pb,
    }
    (out_dir / "spatial.json").write_text(json.dumps(doc, indent=2,
                                                     default=float))

    fig, ax = plt.subplots(2, 1, figsize=(12.8, 6.4), dpi=130, sharex=True)
    tt = F["t"] - F["t"][0]
    ax[0].pcolormesh(tt, np.arange(len(OCT_CENTERS)), d.T, cmap="magma",
                     vmin=0, vmax=1, shading="auto")
    ax[0].set_yticks(range(len(OCT_CENTERS)),
                     [str(int(c)) for c in OCT_CENTERS], fontsize=7)
    ax[0].set(ylabel="octave (Hz)",
              title=f"{sess.name} — directness per octave (1=plane wave, "
                    "0=diffuse)")
    ax[1].plot(ts, Rs, color="#2a78d6", lw=1.2)
    for e in pb:
        ax[1].axvspan(e["t0_s"], e["t0_s"] + e["dur_s"], color="#d66a2a",
                      alpha=0.3)
    ax[1].set(xlabel="time (s)", ylabel="azimuth R (60 s)", ylim=(0, 1),
              title="directional organization; shaded = pass-by events")
    ax[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "spatial.png")
    plt.close(fig)
    return doc
