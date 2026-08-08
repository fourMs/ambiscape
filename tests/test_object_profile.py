"""Morphology of a single sound object, in numbers rather than a type.

The session-scale descriptors need a minute or more, so a folder of
three-second sound actions returns almost nothing from `analyze`. These
are defined on one object and are what let the toolbox say anything at all
about material in the meso band.
"""
import numpy as np

from ambiscape import timescales as ts
from ambiscape.objects import object_profile

DT = 0.02


def _impulse(dur=2.0):
    t = np.arange(0, dur, DT)
    return np.exp(-t * 6)


def _sustained():
    return np.concatenate([np.linspace(0, 1, 20), np.ones(60),
                           np.linspace(1, 0, 20)])


def _iterative(rate_hz=6.0, dur=2.0):
    t = np.arange(0, dur, DT)
    return (0.5 + 0.5 * np.sin(2 * np.pi * rate_hz * t)) * np.exp(-t * 0.3)


def test_temporal_centroid_separates_front_from_centre_loaded():
    """An impulse puts its energy at the start; a held sound spreads it.

    This is the separation the typology makes categorically, measured on a
    continuous axis instead, so two objects of the same facture can still
    be told apart.
    """
    imp = object_profile(_impulse(), DT)["temporal_centroid"]
    sus = object_profile(_sustained(), DT)["temporal_centroid"]
    assert imp < 0.15, f"an impulse should be front-loaded, got {imp}"
    assert 0.4 < sus < 0.6, f"a held sound should sit mid-object, got {sus}"


def test_crest_separates_a_strike_from_a_texture():
    assert object_profile(_impulse(), DT)["crest_db"] > 10
    assert object_profile(_sustained(), DT)["crest_db"] < 4


def test_iteration_rate_is_recovered():
    p = object_profile(_iterative(rate_hz=6.0), DT)
    assert p["iteration_hz"] is not None
    assert abs(p["iteration_hz"] - 6.0) < 1.0
    assert p["iteration_strength"] > 0.2


def test_a_sound_cut_off_has_no_decay():
    """Truncation is reported as absence, not as a decay of zero."""
    cut = np.ones(50)                       # ends at full amplitude
    assert object_profile(cut, DT)["decay_s"] is None
    assert object_profile(_impulse(), DT)["decay_s"] is not None


def test_too_short_or_silent_returns_nothing():
    assert object_profile(np.zeros(50), DT) == {}
    assert object_profile(np.array([1.0, 2.0]), DT) == {}


def test_the_meso_band_is_no_longer_empty():
    """The point of the exercise: descriptors that work on a sound action.

    Before this profile existed, every windowed descriptor was invalid on a
    six-second clip, so the toolbox could say nothing about the material
    the SoundActions corpus is made of.
    """
    valid = [k for k, w in ts.WINDOWS.items() if w.min_s <= 6.0]
    assert valid, "no descriptor is valid on a six-second sound action"
    for key in ("attack_s", "temporal_centroid", "crest_db"):
        assert key in ts.WINDOWS, f"{key} is not registered in timescales"
        assert ts.WINDOWS[key].min_s <= 8.0
