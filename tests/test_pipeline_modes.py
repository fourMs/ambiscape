"""The analyze pipeline must run end-to-end on mono, stereo, and ambisonic
inputs, with direction reported only where the mode can resolve it.

Complements test_binaural_modes.py (which covers the ISO ear-signal path):
here we exercise extract -> load -> full_summary for each channel layout.
"""
import numpy as np
import pytest
import soundfile as sf

import ambiscape as asc
from ambiscape import features
from ambiscape.resolve import full_summary
from ambiscape.spatial import _az_span

FS = 48000
DUR = 12.0


def _write(path, data):
    sf.write(str(path), data.astype(np.float32), FS, subtype="PCM_16")
    return path


def _noise(n, ch, seed):
    rng = np.random.default_rng(seed)
    return 0.05 * rng.standard_normal((n, ch))


def _pipeline(folder, path):
    sess = asc.open_recording(path)
    paths = features.extract_session(sess, folder / "analysis" / "features",
                                     verbose=False)
    return features.load_features(paths)


@pytest.fixture
def n():
    return int(DUR * FS)


def test_mono_pipeline(tmp_path, n):
    F = _pipeline(tmp_path, _write(tmp_path / "mono.wav", _noise(n, 1, 0)))
    assert str(F["mode"]) == "mono"
    s = full_summary(F)
    assert np.isfinite(s["leq_dbfs"])
    assert s["directional_entropy"] is None          # no azimuth for mono
    assert s["above_horizon_fraction"] is None        # no elevation either


def test_stereo_pipeline_uses_lateral_span(tmp_path, n):
    x = _noise(n, 2, 1)
    x[:, 0] *= 1.3                                     # a lateral imbalance
    F = _pipeline(tmp_path, _write(tmp_path / "stereo.wav", x))
    assert str(F["mode"]) == "stereo"
    assert _az_span(F) == (-90.0, 90.0)               # lateral cue, not +-180
    s = full_summary(F)
    assert np.isfinite(s["leq_dbfs"])
    assert s["directional_entropy"] is not None       # lateral az reported
    assert s["above_horizon_fraction"] is None        # but no elevation


def test_ambix_pipeline_full_direction(tmp_path, n):
    rng = np.random.default_rng(2)
    sig = 0.1 * rng.standard_normal(n)
    az = np.radians(45.0)
    ambix = np.stack([sig, sig * np.sin(az), np.zeros(n),
                      sig * np.cos(az)], axis=1) + 0.2 * _noise(n, 4, 3)
    F = _pipeline(tmp_path, _write(tmp_path / "ambix.wav", ambix))
    assert str(F["mode"]) == "ambix"
    assert _az_span(F) == (-180.0, 180.0)             # full circle
    s = full_summary(F)
    assert np.isfinite(s["leq_dbfs"])
    assert s["directional_entropy"] is not None
    assert s["above_horizon_fraction"] is not None    # elevation resolved
