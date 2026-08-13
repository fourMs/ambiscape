"""Spatial dynamics at three time scales.

From the cached per-second spatial features (pseudo-intensity per octave,
DOA, diffuseness) — no audio pass:

- ``direct_diffuse_split`` — per-octave directness (1 − diffuseness proxy)
  per second: the spatial analogue of foreground/background;
- ``passby_events`` — level events whose azimuth sweeps monotonically
  through the event: moving sources, with sweep rate and direction;
- ``azimuth_organization`` — windowed, energy-weighted circular
  concentration R(t): how directionally organised the scene is over time.

Every azimuth here is in the recorder's own frame, and
``frame_reference_test`` is the check that says whether that matters: a
recorder that travels with its subject reports its own geometry in every
room it visits, and no amount of correct decoding turns that into a
property of the places.

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
    directionally disorganised. Window in seconds (per-second features).

    R is computed in the recorder's frame, which is the only frame the audio
    knows about. On a fixed recorder that is also the room's frame and R
    describes the scene; on a recorder that moves with a person or a vehicle
    it describes the rig, and a high R then says the recorder holds its
    pose, not that the place has a direction. :func:`frame_reference_test`
    separates the two where a heading series exists.
    """
    p = F["rms_w"].astype(np.float64) ** 2
    az = np.radians(F["az"])
    n, w, s = len(az), int(win_s), int(step_s)
    if n < w:            # take shorter than one window: measure the whole
        w = max(n, 1)    # take once rather than returning no data at all
    ts, Rs = [], []
    for i0 in range(0, n - w + 1, s):
        _mu, R = mean_resultant(az[i0:i0 + w], weights=p[i0:i0 + w])
        ts.append(float(F["t"][i0] + w / 2 - F["t"][0]))
        Rs.append(R)
    return np.array(ts), np.array(Rs)


def frame_reference_test(bearing_deg, heading_deg, weights=None,
                         control_deg=None) -> dict:
    """Is a bearing series fixed to the recorder, or to the world?

    ``bearing_deg`` is a direction of arrival as the recorder reports it,
    one value per session or per window; ``heading_deg`` is where the
    recorder's nose pointed in world coordinates at the same moments, from
    a compass, a magnetometer, or a written-down orientation. The world
    bearing is ``bearing + heading``, and the test is simply the circular
    concentration R of each series:

    - concentrated in the recorder's frame and dispersed in the world's —
      the quantity is a property of the rig, and every place it visited
      returns the same answer;
    - concentrated in the world's frame — a real direction out there, a
      motorway or a prevailing wind, and the recorder happened to move;
    - concentrated in neither — no stable bearing at either scale.

    Returns ``R_rig``, ``R_world``, their ratio, ``R_chance``, ``n`` and a
    ``frame`` label of ``"rig"``, ``"world"`` or ``"neither"``. ``R_chance``
    is ``1 / sqrt(n)``, the root-mean-square resultant of ``n`` uniformly
    random angles, which is what an R has to beat before it means anything
    at all; with weights, ``n`` is the effective count
    ``(sum w)^2 / sum w^2``.

    **Pass a positive control.** ``control_deg`` is a second bearing series,
    in the same rig frame, that is known or expected to behave differently —
    the direction of the operator's own sway, a source that was carried
    along, a bearing to something fixed. Its two R values come back under
    ``control``. Without one, a test that answers "rig" cannot be told from
    a method that always answers "rig", and the difference is the whole
    result.

    WHY THIS EXISTS. A first-order recorder worn on a body was carried
    through 300 recording days and seven kinds of place — corridors, living
    rooms, an auditorium, a train — and its loudest bearing sat at R = 0.813
    in the rig's frame against 0.268 in compass coordinates, chance being
    0.058. The location groups' mean bearings spanned 18 degrees between
    them: a corridor, a lecture hall and a moving train were returning the
    same direction, because the direction was the recorder's. The positive
    control, the wearer's own sway axis, gave 0.490 against 0.194 and so
    ruled out a method that answers "rig" whatever it is fed.

    Re-decoding the same audio with the correct channel convention softened
    the numbers to 0.393 against 0.144 and widened the group spread to 45
    degrees, in a physically sensible order — so a wrong decode makes this
    worse but is not the cause of it, and a correct decode does not clear
    the rig from the measurement. Ask the frame question before interpreting
    any bearing; no decode answers it.

    >>> rng = np.random.default_rng(0)
    >>> heading = rng.uniform(-180, 180, 500)
    >>> frame_reference_test(np.zeros(500), heading)["frame"]
    'rig'
    """
    b = np.radians(np.asarray(bearing_deg, float))
    h = np.radians(np.asarray(heading_deg, float))
    if b.shape != h.shape:
        raise ValueError(f"bearing and heading differ in shape: "
                         f"{b.shape} vs {h.shape}")
    w = np.ones_like(b) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(b) & np.isfinite(h) & np.isfinite(w)
    b, h, w = b[ok], h[ok], w[ok]
    if b.size < 2:
        raise ValueError("need at least two finite bearing/heading pairs")
    n_eff = float(w.sum() ** 2 / ((w ** 2).sum() + EPS))
    r_chance = float(1.0 / np.sqrt(n_eff))
    _mu_r, r_rig = mean_resultant(b, weights=w)
    _mu_w, r_world = mean_resultant(b + h, weights=w)
    if max(r_rig, r_world) <= r_chance:
        frame = "neither"
    else:
        frame = "rig" if r_rig >= r_world else "world"
    out = {"R_rig": round(r_rig, 4), "R_world": round(r_world, 4),
           "ratio": round(float(r_rig / (r_world + EPS)), 2),
           "R_chance": round(r_chance, 4), "n": int(b.size),
           "n_effective": round(n_eff, 1), "frame": frame}
    if control_deg is not None:
        c = np.radians(np.asarray(control_deg, float))[ok]
        _mu_c, cr = mean_resultant(c, weights=w)
        _mu_cw, cw = mean_resultant(c + h, weights=w)
        out["control"] = {"R_rig": round(cr, 4), "R_world": round(cw, 4),
                          "ratio": round(float(cr / (cw + EPS)), 2)}
    return out


def _az_span(F: dict) -> tuple[float, float]:
    """Azimuth-histogram range for this input mode.

    Ambix resolves the full circle; the stereo/binaural *lateral* cue only
    ever spans the front half-plane, so histogramming it over +-180 deg would
    leave half the bins empty and bias every directional-spread measure low.
    """
    mode = str(F.get("mode", "")).lower()
    if mode in ("stereo", "binaural"):
        return (-90.0, 90.0)
    if mode in ("ambix", "mono"):
        return (-180.0, 180.0)
    # pre-0.13 caches carry no mode: infer from the observed azimuth range
    az = np.asarray(F.get("az"), float)
    az = az[np.isfinite(az)]
    if az.size and float(np.nanmax(np.abs(az))) <= 90.5:
        return (-90.0, 90.0)
    return (-180.0, 180.0)


def directional_entropy(F: dict, nbins: int = 36) -> float:
    """Normalized Shannon entropy of the energy-weighted azimuth histogram.

    "How many directions does this place sound from": 0 = all energy from
    one bearing, 1 = energy spread evenly around the horizon — the spatial
    analogue of an acoustic diversity index, and something only an
    ambisonic corpus can report.

    "This place" is the claim to be careful with: the histogram is built in
    the recorder's frame, so on a rig that moves with its subject the answer
    describes the rig's habitual pose and is the same in every room.
    :func:`frame_reference_test` is the check.
    """
    p = np.asarray(F["rms_w"], np.float64) ** 2
    h, _ = np.histogram(F["az"], bins=nbins, range=_az_span(F), weights=p)
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
        h, _ = np.histogram(F["az"][m], bins=nbins, range=_az_span(F),
                            weights=p[m])
        hists.append(h / (h.sum() + EPS))
    return float(np.sqrt(hists[0] * hists[1]).sum())


def summarize_spatial(F: dict) -> dict:
    """Spatial descriptors for the analyze summary.

    Azimuth-based measures (directional entropy, fg/bg overlap) are reported
    for ambix and stereo (lateral) but not mono; elevation-based measures
    (horizon fractions) only for ambix, since neither stereo nor mono
    resolves elevation.
    """
    has_az = np.isfinite(np.asarray(F["az"], float)).any()
    has_el = np.isfinite(np.asarray(F["el"], float)).any()
    hf = horizon_fractions(F) if has_el else None
    return {
        "directional_entropy": round(directional_entropy(F), 3) if has_az else None,
        "above_horizon_fraction": hf["above"] if hf else None,
        "below_horizon_fraction": hf["below"] if hf else None,
        "fgbg_az_overlap": round(fg_bg_az_overlap(F), 2) if has_az else None,
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
