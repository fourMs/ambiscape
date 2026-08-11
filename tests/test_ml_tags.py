"""Named-class tagging: ask for a class by name and get its probability.

PANNs itself is an optional extra and is not installed everywhere, so the
model and the AudioSet label list are stubbed. What is under test is the
contract of :func:`ambiscape.ml.tag_probabilities` --- that it returns the
probability of the classes you name whether or not they came top, and that a
label which is not an AudioSet class is an error rather than silence.
"""
import sys
import types

import numpy as np
import pytest

from ambiscape import ml

LABELS = ["Speech", "Music", "Silence", "Dog", "Television"]
PROBS = np.array([[0.81, 0.42, 0.01, 0.03, 0.55]], np.float32)


@pytest.fixture
def stub_panns(monkeypatch):
    """A panns_inference whose model returns one known probability vector."""
    class _Model:
        def __init__(self, *a, **k):
            pass

        def inference(self, clip):
            assert clip.ndim == 2 and clip.shape[0] == 1
            return PROBS, None

    mod = types.SimpleNamespace(AudioTagging=_Model, labels=LABELS)
    monkeypatch.setitem(sys.modules, "panns_inference", mod)
    monkeypatch.setattr(ml, "_panns_model", None)
    return mod


def _signal(n=32000, fs=32000):
    return np.zeros(n, np.float32), fs


def test_returns_named_classes_even_when_not_top(stub_panns):
    x, fs = _signal()
    p = ml.tag_probabilities(x, fs, ["Music", "Dog"])
    assert set(p) == {"Music", "Dog"}
    assert p["Music"] == pytest.approx(0.42, abs=1e-6)
    # Dog is fourth of five and must still come back
    assert p["Dog"] == pytest.approx(0.03, abs=1e-6)


def test_all_classes_when_none_requested(stub_panns):
    x, fs = _signal()
    p = ml.tag_probabilities(x, fs)
    assert list(p) == LABELS
    assert p["Speech"] == pytest.approx(0.81, abs=1e-6)


def test_unknown_label_raises_rather_than_returning_nothing(stub_panns):
    x, fs = _signal()
    with pytest.raises(ValueError, match="not AudioSet labels"):
        ml.tag_probabilities(*_signal(), ["Speech", "Televison"])   # typo


def test_ordering_is_the_caller_s_not_the_model_s(stub_panns):
    """The dict follows the requested order, so callers can zip it to columns."""
    x, fs = _signal()
    p = ml.tag_probabilities(x, fs, ["Television", "Speech", "Music"])
    assert list(p) == ["Television", "Speech", "Music"]
