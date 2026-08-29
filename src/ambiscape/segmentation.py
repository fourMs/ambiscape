"""Multivariate change-point segmentation of a session's 1 Hz features.

A stationary session is one soundscape; a soundwalk is a sequence of them.
This module finds the seams from the feature matrix ``analyze`` already
caches — no audio access, no training data. The method is Foote's (2000)
novelty on the feature trajectory: log-octave powers, log-centroid,
log-flatness and diffuseness are put on comparable dB-like scales,
smoothed, and the means before and after each second compared; peaks in
that contrast above an absolute floor are boundaries.

The scales are physical rather than statistical on purpose. A z-scored
trajectory normalises a stationary session's noise up to unit variance
and a walk's structure down to it, so the same threshold cannot serve
both; in dB units a stationary session's contrast sits near zero and a
real zone change measures several dB, whatever the session. Transitions
on a walk are gradients rather than cuts, so boundaries are best read as
centres of transition zones.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

SMOOTH_S = 15.0       # feature smoothing before contrast
HALF_WIN_S = 20.0     # seconds compared on each side of a candidate seam
THRESHOLD_DB = 4.0    # dB-norm contrast a boundary must reach
MIN_SEG_S = 30.0      # boundaries closer than this to an edge or peer merge


def feature_matrix(F: dict) -> np.ndarray:
    """The per-second feature matrix the segmenter works on, in dB-like units.

    Columns: 10 log-octave powers (dB), 10·log10(centroid Hz) (an octave of
    centroid shift counts ~3), 10·log10(flatness), and diffuseness ×10 (the
    full 0–1 range counts as 10). No per-session standardisation — see the
    module docstring for why.
    """
    lo = 10.0 * np.log10(np.asarray(F["oct_pow"], float) + 1e-12)
    cen = 10.0 * np.log10(np.maximum(np.asarray(F["centroid"], float), 1.0))
    fla = 10.0 * np.log10(np.maximum(np.asarray(F["flatness"], float), 1e-6))
    dif = 10.0 * np.asarray(F["diffuse"], float)
    return np.column_stack([lo, cen[:, None], fla[:, None], dif[:, None]])


def novelty(F: dict, half_win_s: float = HALF_WIN_S,
            smooth_s: float = SMOOTH_S) -> np.ndarray:
    """Per-second dB-norm contrast between the feature means before and after.

    Zero within ``half_win_s`` of the session edges, where one side of the
    comparison would be incomplete.
    """
    z = uniform_filter1d(feature_matrix(F), max(1, int(smooth_s)), axis=0)
    n = len(z)
    w = max(1, int(half_win_s))
    nov = np.zeros(n)
    for i in range(w, n - w):
        nov[i] = float(np.linalg.norm(z[i - w:i].mean(0) - z[i:i + w].mean(0)))
    return nov


def segment(F: dict, min_seg_s: float = MIN_SEG_S,
            threshold_db: float = THRESHOLD_DB,
            half_win_s: float = HALF_WIN_S) -> list[float]:
    """Boundary times (seconds on ``F["t"]``'s clock) between sub-soundscapes.

    An empty list means the features hold one regime — the stationary case
    the rest of the toolbox assumes.
    """
    t = np.asarray(F["t"], float)
    if len(t) < 2 or (t[-1] - t[0]) < 2 * min_seg_s:
        return []
    nov = novelty(F, half_win_s=half_win_s)
    peaks, _ = find_peaks(nov, distance=max(1, int(min_seg_s)),
                          height=threshold_db,
                          prominence=threshold_db / 2.0)
    return sorted(float(t[p]) for p in peaks
                  if (t[p] - t[0]) >= min_seg_s and (t[-1] - t[p]) >= min_seg_s)
