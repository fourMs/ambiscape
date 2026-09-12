"""Frame-wise tagging and device selection.

PANNs is an optional extra, so the model is stubbed; what is under test is the
tiling contract of :func:`ambiscape.ml.tag_frames` (window centres, one row per
window, columns in the caller's order, batches of several windows at once) and
that a device request rebuilds the cached model instead of being ignored.
"""
import sys
import types

import numpy as np
import pytest

from ambiscape import ml

LABELS = ["Speech", "Music", "Silence", "Dog", "Television"]


@pytest.fixture
def stub_panns(monkeypatch):
    built = []

    class _Model:
        def __init__(self, *a, device="cpu", **k):
            built.append(device)

        def inference(self, clip):
            assert clip.ndim == 2
            # each window answers with its own mean level in the Music column, so rows are distinguishable
            b = clip.shape[0]
            p = np.tile(np.array([[0.8, 0.0, 0.1, 0.0, 0.5]], np.float32), (b, 1))
            p[:, 1] = clip.mean(axis=1)
            return p, None

    mod = types.SimpleNamespace(AudioTagging=_Model, labels=LABELS)
    monkeypatch.setitem(sys.modules, "panns_inference", mod)
    monkeypatch.setattr(ml, "_panns_model", None)
    monkeypatch.setattr(ml, "_panns_model_device", None)
    monkeypatch.setattr(ml, "PANNS_DEVICE", "cpu")
    return built


def test_frames_tile_the_recording(stub_panns):
    fs = 32000
    x = np.concatenate([np.full(fs * 4, 0.1, np.float32), np.full(fs * 4, 0.3, np.float32)])
    t, P, names = ml.tag_frames(x, fs, win_s=4.0, hop_s=2.0)
    assert names == LABELS and P.shape == (3, 5)
    np.testing.assert_allclose(t, [2.0, 4.0, 6.0])
    np.testing.assert_allclose(P[:, 1], [0.1, 0.2, 0.3], atol=1e-6)   # first, straddling, last window


def test_wanted_selects_and_orders_columns(stub_panns):
    x = np.zeros(32000 * 10, np.float32)
    t, P, names = ml.tag_frames(x, 32000, wanted=["Television", "Speech"], batch=2)
    assert names == ["Television", "Speech"]
    assert P.shape == (4, 2) and P[0, 0] == pytest.approx(0.5) and P[0, 1] == pytest.approx(0.8)


def test_short_recording_is_one_window(stub_panns):
    t, P, _ = ml.tag_frames(np.zeros(8000, np.float32), 16000, win_s=4.0)
    assert P.shape[0] == 1 and t[0] == pytest.approx(0.25)


def test_unknown_label_raises(stub_panns):
    with pytest.raises(ValueError, match="not AudioSet labels"):
        ml.tag_frames(np.zeros(32000, np.float32), 32000, wanted=["Musik"])


def test_device_request_rebuilds_the_cached_model(stub_panns):
    x = np.zeros(32000, np.float32)
    ml.tag_probabilities(x, 32000)                      # default: cpu
    ml.tag_probabilities(x, 32000, device="cpu")        # same device: reused
    ml.tag_window(x, 32000, device="mps")               # different device: rebuilt
    assert stub_panns == ["cpu", "mps"]


def test_auto_resolves_to_a_concrete_device(stub_panns, monkeypatch):
    assert ml.resolve_device("cpu") == "cpu"
    assert ml.resolve_device(None) == "cpu"
    assert ml.resolve_device("auto") in ("cpu", "cuda")
