"""A bearing is fixed to something, and the audio cannot say to what.

``spatial.frame_reference_test`` decides between the recorder's frame and
the world's by taking the circular concentration of the same bearings in
both. These tests build the two cases the function has to separate — a
source that stays in front of the recorder while the recorder turns, and a
source that stays in one compass direction while the recorder turns — and
check that each is named correctly, that neither is named when there is
nothing there, and that the chance floor is the one the docstring quotes.

The mixed case is the one worth having: a rig-fixed bearing with a
world-fixed positive control alongside it, which is the arrangement that
distinguishes "this quantity is rig-fixed" from "this method says rig-fixed
whatever it is given".
"""
import numpy as np
import pytest

from ambiscape import spatial


def _headings(rng, n):
    """A recorder that has been pointed everywhere over the record."""
    return rng.uniform(-180.0, 180.0, n)


def test_a_source_in_front_of_the_recorder_is_named_rig():
    """Constant in the rig's frame, uniform once heading is added back."""
    rng = np.random.default_rng(0)
    h = _headings(rng, 400)
    b = rng.normal(0.0, 8.0, 400)          # always ahead, give or take
    out = spatial.frame_reference_test(b, h)
    assert out["frame"] == "rig"
    assert out["R_rig"] > 0.9
    assert out["R_world"] < 0.15
    assert out["ratio"] > 6


def test_a_source_at_a_fixed_compass_bearing_is_named_world():
    """The same arithmetic run the other way: the world holds the source."""
    rng = np.random.default_rng(1)
    h = _headings(rng, 400)
    world = rng.normal(120.0, 8.0, 400)    # a motorway, say
    b = world - h                          # what the recorder would report
    out = spatial.frame_reference_test(b, h)
    assert out["frame"] == "world"
    assert out["R_world"] > 0.9
    assert out["R_rig"] < 0.15


def test_no_bearing_at_either_scale_is_named_neither():
    """Random in both frames must not be talked into one of them."""
    rng = np.random.default_rng(2)
    out = spatial.frame_reference_test(_headings(rng, 600),
                                       _headings(rng, 600))
    assert out["frame"] == "neither"
    assert out["R_rig"] < 0.15 and out["R_world"] < 0.15


def test_chance_floor_is_one_over_root_n():
    """The number an R has to beat, and it must shrink with the record."""
    rng = np.random.default_rng(3)
    for n in (100, 900):
        out = spatial.frame_reference_test(_headings(rng, n),
                                           _headings(rng, n))
        assert out["R_chance"] == pytest.approx(1 / np.sqrt(n), abs=5e-5)
        assert out["n"] == n


def test_weights_set_the_effective_count_and_the_answer():
    """Four windows carrying the energy are four samples, not four hundred.

    The unweighted series looks emphatically rig-fixed, because 396 of its
    400 windows sit at one bearing. They carry a billionth of the energy
    each. Weighted, the record is four loud windows pointing four ways, and
    the honest answer is that it holds no bearing.
    """
    rng = np.random.default_rng(4)
    n = 400
    b = np.zeros(n)
    b[:4] = (0.0, 90.0, 180.0, 270.0)
    w = np.full(n, 1e-9)
    w[:4] = 1.0
    h = _headings(rng, n)
    h[:4] = (0.0, 45.0, 90.0, 135.0)     # world bearings spread as well
    assert spatial.frame_reference_test(b, h)["frame"] == "rig"
    out = spatial.frame_reference_test(b, h, weights=w)
    assert out["n_effective"] == pytest.approx(4.0, abs=0.01)
    assert out["R_chance"] == pytest.approx(0.5, abs=1e-3)
    assert out["frame"] == "neither"


def test_the_positive_control_is_reported_and_can_disagree():
    """A rig-fixed measure beside a world-fixed control, in one call.

    This is the arrangement the test exists for. If the control came back
    rig-fixed as well, the result would be a fact about the function rather
    than about the recording.
    """
    rng = np.random.default_rng(5)
    h = _headings(rng, 400)
    b = rng.normal(0.0, 8.0, 400)                 # rig-fixed
    ctrl = rng.normal(-60.0, 8.0, 400) - h        # world-fixed
    out = spatial.frame_reference_test(b, h, control_deg=ctrl)
    assert out["frame"] == "rig"
    assert out["control"]["R_world"] > 0.9
    assert out["control"]["R_rig"] < 0.15


def test_non_finite_pairs_are_dropped_not_propagated():
    """A day with no compass fix must not poison the year."""
    rng = np.random.default_rng(6)
    h = _headings(rng, 200)
    b = rng.normal(0.0, 8.0, 200)
    h[::10] = np.nan
    out = spatial.frame_reference_test(b, h)
    assert out["n"] == 180
    assert np.isfinite(out["R_rig"]) and out["frame"] == "rig"


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="differ in shape"):
        spatial.frame_reference_test(np.zeros(10), np.zeros(9))


def test_too_few_pairs_raise():
    with pytest.raises(ValueError, match="at least two"):
        spatial.frame_reference_test([np.nan, 1.0], [0.0, np.nan])
