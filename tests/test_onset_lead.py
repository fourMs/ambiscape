"""A sound-producing action begins before its sound.

ARJ's action-sound theory: an intention becomes neural signals, then muscular
ones, then motion in the arm and the object, and only at the end a sound. So a
sound object embeds an action, and the silence before the attack is where the
action already is.

Measured on 180 clips of the Sound Actions corpus, motion leads sound by a
median of 0.72 s and does so in 84 % of them. That is the quantity this
function returns.

The one thing it must not do is compare two differently-defined onsets: the
lead would then partly measure the definitions. Both series get the same rule,
applied to their own floor-to-peak range, so neither modality's units enter.
"""
import numpy as np
import pytest

from ambiscape import analysis


def _ramp(n, start, rise=10, level=1.0, floor=0.01, seed=0):
    """A series that sits at a floor and then rises at `start`."""
    rng = np.random.default_rng(seed)
    x = np.full(n, floor)
    x[start:start + rise] = np.linspace(floor, level, rise)
    x[start + rise:] = level
    return x * (1 + rng.normal(0, 0.02, n))


def test_it_finds_a_known_lead():
    """Motion at frame 20, sound at frame 40, 25 fps: 0.8 s of lead."""
    motion = _ramp(200, 20)
    audio = _ramp(200, 40, seed=1)
    r = analysis.onset_lead(motion, audio, dt=1 / 25)
    assert abs(r["lead_s"] - 0.8) < 0.1
    assert r["leads"] == "first"


def test_it_reports_the_other_direction_too():
    """Sound first is a real case -- a struck object already in motion, or an
    action out of frame -- and must not be forced positive."""
    r = analysis.onset_lead(_ramp(200, 60), _ramp(200, 30, seed=1), dt=1 / 25)
    assert r["lead_s"] < 0
    assert r["leads"] == "second"


def test_units_do_not_enter_the_comparison():
    """Scaling either series must not move the lead: the rule is applied to
    each series' own range. A motion measure in pixels and an energy in
    arbitrary units are otherwise incomparable."""
    m, a = _ramp(200, 20), _ramp(200, 40, seed=1)
    base = analysis.onset_lead(m, a, dt=1 / 25)["lead_s"]
    scaled = analysis.onset_lead(m * 1e4, a * 1e-6, dt=1 / 25)["lead_s"]
    assert abs(base - scaled) < 1e-9


def test_a_flat_series_yields_nothing():
    """No onset in one modality means no lead, not a lead of zero."""
    flat = np.full(200, 0.5)
    assert analysis.onset_lead(flat, _ramp(200, 40), dt=1 / 25)["lead_s"] is None


def test_lengths_may_differ():
    """Video and audio frame counts rarely match exactly."""
    r = analysis.onset_lead(_ramp(200, 20), _ramp(197, 40, seed=1), dt=1 / 25)
    assert r["lead_s"] is not None
