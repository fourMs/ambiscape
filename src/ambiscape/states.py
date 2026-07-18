"""Machine states: on/off segmentation, switch points, and duty cycles.

Domestic and mechanical sources (ventilation, fridges, pumps, HVAC) show up
in a soundscape as a *state* — a band-limited floor that is either present or
absent — rather than as events. This module segments a band-level timeline
into those states from the cached features, no audio pass:

- ``band_level`` — per-second dB level in a frequency band from the cached
  log-band spectrogram (the "machine band" of a source, e.g. 250–1000 Hz for
  a ventilation unit);
- ``state_segments`` — two-state (on/off) segmentation of that level with an
  automatic bimodal threshold, hysteresis, and a minimum duration, each
  segment carrying its median level and within-state stability (SD);
- ``switch_points`` — the transitions between segments (the 07:53:55
  switch-off moments);
- ``duty_cycle`` — cycle statistics of a cycling machine (a fridge's ~24 min
  period at ~50 % duty): period, duty fraction, cycle count.

Typical use: ``segs = state_segments(band_level(F, (250, 1000)))`` and mask
other analyses (fingerprints, masking, taxonomy states) by segment.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

EPS = 1e-20


def band_level(F: dict, band=(250.0, 1000.0)) -> np.ndarray:
    """Per-second dB level in ``band`` (Hz) from the cached ``logspec``."""
    logf = np.asarray(F["logf"], float)
    fc = np.sqrt(logf[:-1] * logf[1:])
    m = (fc >= band[0]) & (fc <= band[1])
    return 10 * np.log10(F["logspec"][:, m].sum(1) + EPS)


def bimodal_threshold(level_db: np.ndarray) -> float:
    """Otsu's threshold on the level histogram: the split that best
    separates the two modes of an on/off timeline."""
    lo, hi = np.percentile(level_db, (0.5, 99.5))
    hist, edges = np.histogram(level_db, bins=128, range=(lo, hi))
    p = hist.astype(float) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    var = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0) + EPS)
    k = int(np.argmax(var))
    # between-class variance is flat across an empty inter-mode gap; the
    # midpoint of the two class means splits the gap centrally
    mu0 = mu[k] / (w0[k] + EPS)
    mu1 = (mu_t - mu[k]) / (1 - w0[k] + EPS)
    return float((mu0 + mu1) / 2)


def state_segments(level_db: np.ndarray, thresh_db: float | None = None,
                   smooth_s: float = 11.0, hysteresis_db: float = 1.0,
                   min_dur_s: float = 30.0) -> list[dict]:
    """Two-state segmentation of a 1 Hz band-level timeline.

    The level is median-smoothed over ``smooth_s``; the threshold defaults
    to the bimodal (Otsu) split of the histogram — pass ``thresh_db`` when
    the timeline is not clearly bimodal. Hysteresis of ``hysteresis_db``
    around the threshold suppresses chatter, and segments shorter than
    ``min_dur_s`` are merged into their neighbors. Returns segments in time
    order as dicts: state ('on'/'off'), t0_s/dur_s (seconds into the
    timeline), median_db, and sd_db (within-state stability of the raw
    level — a running machine is *steady*, ambience is not).
    """
    x = np.asarray(level_db, float)
    k = max(3, int(round(smooth_s)) | 1)
    sm = median_filter(x, size=k, mode="nearest")
    th = bimodal_threshold(sm) if thresh_db is None else float(thresh_db)
    on = np.zeros(len(sm), bool)
    cur = sm[0] > th
    for i, v in enumerate(sm):
        if cur and v < th - hysteresis_db / 2:
            cur = False
        elif not cur and v > th + hysteresis_db / 2:
            cur = True
        on[i] = cur

    def bounds(mask):
        edges = np.flatnonzero(np.diff(mask.astype(int))) + 1
        return [0, *edges.tolist(), len(mask)]

    # merge runs shorter than min_dur_s into the surrounding state
    b = bounds(on)
    for i0, i1 in zip(b[:-1], b[1:]):
        if i1 - i0 < min_dur_s and i0 > 0 and i1 < len(on):
            on[i0:i1] = on[i0 - 1]
    b = bounds(on)
    segs = []
    for i0, i1 in zip(b[:-1], b[1:]):
        seg = x[i0:i1]
        segs.append({
            "state": "on" if on[i0] else "off",
            "t0_s": float(i0), "dur_s": float(i1 - i0),
            "median_db": round(float(np.median(seg)), 1),
            "sd_db": round(float(seg.std()), 2),
        })
    return segs


def switch_points(segments: list[dict]) -> list[dict]:
    """Transitions between consecutive segments: time and direction
    ('on' = machine starts, 'off' = machine stops)."""
    out = []
    for a, b in zip(segments[:-1], segments[1:]):
        out.append({"t_s": float(b["t0_s"]),
                    "direction": b["state"],
                    "step_db": round(b["median_db"] - a["median_db"], 1)})
    return out


def duty_cycle(segments: list[dict]) -> dict:
    """Cycle statistics of a cycling machine from its state segments:
    median period (consecutive on-starts), duty fraction (median on-time
    over period), and the number of complete cycles observed."""
    on_starts = np.array([s["t0_s"] for s in segments if s["state"] == "on"])
    on_durs = np.array([s["dur_s"] for s in segments if s["state"] == "on"])
    if len(on_starts) < 2:
        return {"period_s": None, "duty": None,
                "n_cycles": int(len(on_starts))}
    period = float(np.median(np.diff(on_starts)))
    return {"period_s": round(period, 1),
            "duty": round(float(np.median(on_durs)) / period, 3),
            "n_cycles": int(len(on_starts))}
