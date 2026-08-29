"""Multivariate change-point segmentation of a session's 1 Hz features.

A walking recording is a sequence of sub-soundscapes; the segmenter's job
is to find the seams from the cached feature matrix alone. These fixtures
are synthetic feature dicts, not audio: the segmenter must work from what
``analyze`` already saves.
"""
import numpy as np
import pytest

from ambiscape.segmentation import segment


def _flat_features(nsec: int, seed: int = 7) -> dict:
    """A stationary session: constant octave spectrum plus small noise."""
    rng = np.random.default_rng(seed)
    oct_pow = np.tile(10.0 ** np.array([-3, -3, -3.5, -4, -4.5, -5,
                                        -5.5, -6, -6.5, -7]), (nsec, 1))
    oct_pow *= rng.lognormal(0, 0.05, size=oct_pow.shape)
    return {
        "t": np.arange(nsec, dtype=float),
        "oct_pow": oct_pow.astype(np.float32),
        "centroid": 300.0 + rng.normal(0, 5, nsec),
        "flatness": 0.01 + rng.normal(0, 0.001, nsec),
        "diffuse": 0.3 + rng.normal(0, 0.02, nsec),
    }


def _two_zone_features(nsec: int = 300, change: int = 150) -> dict:
    """Same level throughout, but the spectrum flips LF-heavy -> HF-heavy."""
    F = _flat_features(nsec)
    hf = 10.0 ** np.array([-7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -3])
    F["oct_pow"][change:] = hf * F["oct_pow"][change:].sum(1, keepdims=True) / hf.sum()
    F["centroid"][change:] += 2000.0
    F["diffuse"][change:] += 0.4
    return F


def test_boundary_found_at_spectral_change():
    F = _two_zone_features(300, change=150)
    bounds = segment(F)
    assert len(bounds) == 1
    assert abs(bounds[0] - 150.0) <= 15.0


def test_stationary_session_has_no_boundaries():
    F = _flat_features(300)
    assert segment(F) == []


def test_min_segment_length_is_respected():
    F = _two_zone_features(300, change=150)
    bounds = segment(F, min_seg_s=200.0)
    assert bounds == []


def test_boundaries_are_sorted_times_within_session():
    F = _two_zone_features(600, change=200)
    hf2 = F["oct_pow"][0] * 3.0
    F["oct_pow"][400:] = hf2
    F["centroid"][400:] -= 1500.0
    bounds = segment(F)
    assert bounds == sorted(bounds)
    assert all(0 < b < 600 for b in bounds)
    assert len(bounds) >= 2


def test_nonstationarity_flags_a_multi_regime_session():
    F = _two_zone_features(600, change=200)
    F["oct_pow"][400:] = F["oct_pow"][0] * 3.0
    F["centroid"][400:] -= 1500.0
    from ambiscape.segmentation import nonstationarity
    r = nonstationarity(F)
    assert r["nonstationary"] is True
    assert r["n_regimes"] >= 3
    assert "why" in r


def test_nonstationarity_passes_a_stationary_session():
    from ambiscape.segmentation import nonstationarity
    r = nonstationarity(_flat_features(300))
    assert r["nonstationary"] is False
    assert r["n_regimes"] == 1


def test_full_summary_carries_the_nonstationarity_block(bell_features):
    from ambiscape import resolve
    _sess, _out, F = bell_features
    s = resolve.full_summary(F)
    assert "nonstationarity" in s
    assert set(s["nonstationarity"]) >= {"nonstationary", "n_regimes"}
