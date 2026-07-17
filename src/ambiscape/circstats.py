"""Circular statistics, shared by spatial (azimuth) and rhythm (phase) code.

Angles are radians throughout; the degree-facing wrapper lives in
:func:`ambiscape.analysis.circular_stats`, and period-facing helpers here
convert times to phases. The resultant length R in [0, 1] measures
concentration; circular SD = sqrt(-2 ln R); the Rayleigh test gives the
probability of R under uniformity (p ~ exp(-n R^2), adequate for n >= 10).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-20


def mean_resultant(angles: np.ndarray, weights=None):
    """Weighted circular mean (rad) and resultant length R."""
    a = np.asarray(angles, float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    z = (w * np.exp(1j * a)).sum() / (w.sum() + EPS)
    return float(np.angle(z)), float(np.abs(z))


def circular_sd(R: float) -> float:
    """Circular standard deviation (rad) from a resultant length."""
    return float(np.sqrt(-2 * np.log(max(R, EPS))))


def rayleigh_p(R: float, n: int) -> float:
    """Rayleigh-test p-value (uniformity null), small-p approximation."""
    z = n * R * R
    return float(np.clip(np.exp(-z) * (1 + (2 * z - z * z) / (4 * n)),
                         0.0, 1.0))


def phase_stats(times: np.ndarray, period: float) -> dict:
    """Circular statistics of event times folded at ``period``.

    Returns mean phase (cycles), R, circular SD in seconds, and the
    Rayleigh p-value — the standard summary for one strike stream.
    """
    ph = 2 * np.pi * (np.asarray(times, float) / period % 1.0)
    mu, R = mean_resultant(ph)
    return {
        "mean_phase": float((mu / (2 * np.pi)) % 1.0),
        "R": round(R, 4),
        "circ_sd_s": round(circular_sd(R) / (2 * np.pi) * period, 4),
        "rayleigh_p": rayleigh_p(R, len(ph)),
        "n": int(len(ph)),
    }


def relative_phase(times: np.ndarray, ref_times: np.ndarray,
                   period: float) -> dict:
    """Phase of each event relative to the preceding reference event.

    The per-event lock between two streams sharing one period: mean offset
    (cycles and seconds), R, and circular SD in seconds. R near 1 means the
    two streams are phase-locked at strike level.
    """
    ref = np.sort(np.asarray(ref_times, float))
    t = np.asarray(times, float)
    i = np.clip(np.searchsorted(ref, t) - 1, 0, len(ref) - 1)
    d = 2 * np.pi * (((t - ref[i]) / period) % 1.0)
    mu, R = mean_resultant(d)
    off = (mu / (2 * np.pi)) % 1.0
    return {
        "mean_offset_cycles": round(off, 4),
        "mean_offset_s": round(off * period, 4),
        "R": round(R, 4),
        "circ_sd_s": round(circular_sd(R) / (2 * np.pi) * period, 4),
        "n": int(len(t)),
    }
