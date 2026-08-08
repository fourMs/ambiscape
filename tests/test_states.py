"""State boundaries as sound objects.

`state_segments` finds the steady states a room passes through. What it
does not describe is the crossing between them, and the crossing is what
anyone in the room notices: a refrigerator does not fade in, it strikes
and subsides into a hum that is then ignored for eleven minutes.
"""
import numpy as np

# ------------------------------------------------- transitions as objects

def _fridge(seed=0):
    """A quiet room, an abrupt machine onset with a clatter, then silence."""
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal(-58, 1.0, 300),
        [-38, -36, -40],                       # the strike
        rng.normal(-44, 0.4, 600),             # the hum it settles into
        rng.normal(-58, 1.0, 300),
    ])


def test_transition_profile_finds_both_directions():
    from ambiscape.states import state_segments, transition_profile
    x = _fridge()
    t = transition_profile(x, state_segments(x, min_dur_s=30.0))
    assert [e["direction"] for e in t] == ["onset", "cessation"]
    assert t[0]["step_db"] > 10 and t[1]["step_db"] < -10


def test_an_abrupt_machine_is_distinguished_from_a_slow_fade():
    """The morphology is the point: a fridge strikes, a fade does not.

    Both end at the same level, so a state summary describes them
    identically. Only the crossing time tells them apart, and the crossing
    is what a listener in the room actually notices.
    """
    from ambiscape.states import state_segments, transition_profile
    rng = np.random.default_rng(1)
    fade = np.concatenate([rng.normal(-58, .5, 300),
                           np.linspace(-58, -44, 600) + rng.normal(0, .3, 600),
                           rng.normal(-44, .4, 300)])
    abrupt = transition_profile(_fridge(), state_segments(_fridge(),
                                                          min_dur_s=30.0))
    gradual = transition_profile(fade, state_segments(fade, min_dur_s=30.0))
    assert gradual, "the fade should still register as a transition"
    assert gradual[0]["crossing_s"] > 5 * abrupt[0]["crossing_s"]


def test_settling_tolerance_adapts_to_a_noisy_state():
    """A tolerance tighter than the state's own noise is the wrong question."""
    from ambiscape.states import state_segments, transition_profile
    x = _fridge()
    t = transition_profile(x, state_segments(x, min_dur_s=30.0),
                           settle_tol_db=0.1)
    assert all(e["settle_tol_db"] >= 0.1 for e in t)
    assert all(e["settle_s"] is not None for e in t), \
        "a settled state must not read as never settling"
