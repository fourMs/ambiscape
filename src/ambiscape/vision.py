"""Per-frame visual features: the light/vision analogue of the audio deposit.

The AMBIENT project treats a room as an audio-*visual* subject, and a room's
*look* has a diurnal rhythm just as its sound does. This module extracts a
compact descriptor from a single video frame so a camera can log visual
*behaviour* rather than store imagery --- the same "features, not recordings"
privacy stance as :mod:`ambiscape.capture`. It is numpy-only (no camera or
OpenCV dependency); frame *grabbing* lives in the capture rig, the feature
*definitions* live here so they are versioned and tested with ambiscape.

Per frame (:func:`frame_features`):

- **brightness** / **brightness_sd** --- mean and spread of Rec.709 luma
  (0 = dark, 1 = bright): the room's overall light level and its unevenness;
- **r_frac / g_frac / b_frac**, **warm_cool_ratio** --- colour balance and a
  warm/cool proxy (daylight vs incandescent vs the blue of a screen);
- **saturation**, **colourfulness** --- how colourful the scene is
  (Hasler--Susstrunk colourfulness), near zero for a grey room;
- **spatial_entropy** --- entropy of a coarse brightness grid: 1 when the
  room is evenly lit, low when light is concentrated (a lamp, a window) ---
  the visual analogue of acoustic diffuseness;
- **bright_centroid_x / _y** --- the luma-weighted centre of light in the
  frame (0..1): *where* the light comes from, a visual direction-of-arrival.

:func:`frame_delta` is a motion proxy (mean absolute luma change between two
frames); :func:`summarize_vision` rolls a day of per-frame features into a
``vis_``-prefixed summary that joins the audio ``summary.json``.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
_LUMA = (0.2126, 0.7152, 0.0722)   # Rec.709


def _to_unit(rgb):
    x = np.asarray(rgb, float)
    if x.ndim != 3 or x.shape[-1] < 3:
        raise ValueError("expected an (H, W, 3) RGB frame")
    x = x[..., :3]
    if x.max() > 1.5:                # uint8 / 0..255 -> 0..1
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def luma(rgb) -> np.ndarray:
    """Rec.709 luma of an RGB frame, in [0, 1]."""
    x = _to_unit(rgb)
    return x @ np.array(_LUMA)


def _grid_means(l: np.ndarray, g: int) -> np.ndarray:
    h, w = l.shape
    ys = np.linspace(0, h, g + 1).astype(int)
    xs = np.linspace(0, w, g + 1).astype(int)
    return np.array([[l[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()
                      for j in range(g)] for i in range(g)])


def frame_features(rgb, grid: int = 3) -> dict:
    """Visual descriptor of one RGB frame (uint8 0..255 or float 0..1)."""
    x = _to_unit(rgb)
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    lum = x @ np.array(_LUMA)
    mr, mg, mb = float(R.mean()), float(G.mean()), float(B.mean())
    s = mr + mg + mb + EPS

    mx = x.max(-1)
    mn = x.min(-1)
    sat = float(np.where(mx > 0, (mx - mn) / (mx + EPS), 0.0).mean())
    rg = R - G
    yb = 0.5 * (R + G) - B
    colourfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                          + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    gm = _grid_means(lum, grid)
    p = gm.ravel() / (gm.sum() + EPS)
    spatial_entropy = float(-(p * np.log(p + EPS)).sum() / np.log(grid * grid))

    h, w = lum.shape
    ys, xs = np.mgrid[0:h, 0:w]
    tot = lum.sum() + EPS
    cx = float((lum * xs).sum() / tot / max(w - 1, 1))
    cy = float((lum * ys).sum() / tot / max(h - 1, 1))

    return {
        "brightness": round(float(lum.mean()), 4),
        "brightness_sd": round(float(lum.std()), 4),
        "r_frac": round(mr / s, 4), "g_frac": round(mg / s, 4),
        "b_frac": round(mb / s, 4),
        "warm_cool_ratio": round(mr / (mb + EPS), 4),
        "saturation": round(sat, 4),
        "colourfulness": round(colourfulness, 4),
        "spatial_entropy": round(spatial_entropy, 4),
        "bright_centroid_x": round(cx, 4),
        "bright_centroid_y": round(cy, 4),
    }


def frame_delta(prev_rgb, cur_rgb) -> float:
    """Mean absolute luma change between two frames (motion proxy, 0..1)."""
    return round(float(np.abs(luma(cur_rgb) - luma(prev_rgb)).mean()), 5)


def summarize_vision(features, motion=None) -> dict:
    """Roll a day of per-frame feature dicts into a ``vis_`` day summary."""
    rows = list(features)
    if not rows:
        return {"n_frames": 0}
    def col(k):
        return np.array([r[k] for r in rows if k in r], float)
    b = col("brightness")
    out = {
        "n_frames": len(rows),
        "vis_brightness_median": round(float(np.median(b)), 4),
        "vis_brightness_range": round(float(np.percentile(b, 90)
                                            - np.percentile(b, 10)), 4),
        "vis_colourfulness_median": round(float(np.median(col("colourfulness"))), 4),
        "vis_saturation_median": round(float(np.median(col("saturation"))), 4),
        "vis_warm_cool_median": round(float(np.median(col("warm_cool_ratio"))), 4),
        "vis_spatial_entropy_median": round(
            float(np.median(col("spatial_entropy"))), 4),
    }
    if motion is not None and len(motion):
        m = np.asarray(motion, float)
        out["vis_motion_mean"] = round(float(m.mean()), 5)
        out["vis_motion_p90"] = round(float(np.percentile(m, 90)), 5)
    return out
