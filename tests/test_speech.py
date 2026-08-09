"""Speech detection must measure speech, not recording gain."""
import numpy as np
import pytest

from ambiscape import ml


def _speechlike(fs=16000, dur=4.0, seed=0):
    """A harmonic stack with a syllabic envelope — not speech, but it has
    speech's level structure, which is what the normalisation cares about."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fs)) / fs
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
    x = sum(np.sin(2 * np.pi * k * np.cumsum(f0) / fs) / k for k in range(1, 12))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)          # ~3.5 Hz syllables
    return (x * env + 0.001 * rng.standard_normal(len(t))) * 0.2


def test_vad_input_is_level_invariant():
    """silero is applied to whatever level arrives, so an attenuated copy of
    the same recording reads as less speech: on real audio, 0.513 at unity
    became 0.110 at -24 dB and 0.000 at -30 dB. Nodes with different
    sensitivities are then not comparable, which is exactly how a fixed
    detector turns a gain difference into an apparent finding."""
    x = _speechlike()
    ref = ml._vad_input(x)
    for gain_db in (+12, -12, -24, -40):
        scaled = ml._vad_input(x * 10 ** (gain_db / 20))
        assert np.allclose(scaled, ref, atol=1e-6), (
            f"{gain_db:+d} dB changed the signal handed to the detector")


def test_vad_input_never_clips():
    x = _speechlike() * 50          # far above full scale
    y = ml._vad_input(x)
    assert np.abs(y).max() <= 1.0


def test_vad_input_leaves_silence_alone():
    """Digital silence has no level to normalise to; it must not be
    amplified into noise the detector then reports as speech."""
    y = ml._vad_input(np.zeros(16000))
    assert np.abs(y).max() == 0.0
