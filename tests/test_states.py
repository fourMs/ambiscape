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


# ------------------------------------------- the two halves of a cycle

def _thermostat(n_cycles=12, on_s=500.0, p_start=1830.0, p_end=2280.0,
                floor_db=-58.0, depth_db=9.0, seed=3):
    """A machine with a fixed on-time and a lengthening interval.

    Which is what a thermostat is: the compressor runs until the cabinet is
    cold, taking about as long each time, and then waits until it warms,
    which takes longer as the room around it cools.
    """
    rng = np.random.default_rng(seed)
    parts, t = [], 0.0
    for i in range(n_cycles):
        period = p_start + (p_end - p_start) * i / max(1, n_cycles - 1)
        parts.append(rng.normal(floor_db + depth_db, 0.4, int(on_s)))
        parts.append(rng.normal(floor_db, 0.4, int(period - on_s)))
        t += period
    return np.concatenate(parts)


def test_cycle_series_separates_a_fixed_on_time_from_a_moving_period():
    """The finding a duty fraction cannot carry."""
    from ambiscape.states import cycle_series, state_segments
    x = _thermostat()
    c = cycle_series(state_segments(x, min_dur_s=120.0))
    assert c["n_cycles"] >= 10
    # the interval moves and the stroke does not
    assert c["period_trend"]["rho"] > 0.9
    assert c["period_trend"]["spread"] > 300
    assert abs(c["on_trend"]["per_cycle"]) < 5


def test_cycle_series_reports_no_trend_for_a_steady_machine():
    from ambiscape.states import cycle_series, state_segments
    x = _thermostat(p_start=1830.0, p_end=1830.0)
    c = cycle_series(state_segments(x, min_dur_s=120.0))
    assert abs(c["period_trend"]["rho"]) < 0.6
    assert c["period_trend"]["spread"] < 200


def test_cycle_series_needs_no_cycles_to_survive():
    from ambiscape.states import cycle_series
    c = cycle_series([{"state": "off", "t0_s": 0.0, "dur_s": 100.0}])
    assert c["n_cycles"] == 0 and c["period_trend"]["rho"] is None


# ------------------------------------- a split that does not mean anything

def test_a_plainly_cycling_machine_is_bimodal():
    from ambiscape.states import bimodal_separation
    s = bimodal_separation(_thermostat())
    assert s["bimodal"] is True
    assert s["separation_db"] > 6


def test_a_faint_machine_is_not_called_bimodal():
    """The Ghent living room: the same fridge, one room away, at ~0.6 dB.

    Otsu returns a threshold regardless, the segmentation divides noise, and
    duty_cycle reports a period for a machine nothing detected. This is the
    guard that says so.
    """
    from ambiscape.states import bimodal_separation
    rng = np.random.default_rng(5)
    faint = rng.normal(-62.0, 0.25, 20000)
    faint[5000:6000] += 0.6                     # the machine, such as it is
    s = bimodal_separation(faint)
    assert s["bimodal"] is False
    assert s["separation_db"] < 2.0


def test_a_flat_series_is_not_bimodal_and_does_not_raise():
    from ambiscape.states import bimodal_separation
    s = bimodal_separation(np.full(500, -60.0))
    assert s["bimodal"] is False
    assert s["separation_db"] == 0.0


def test_a_machine_on_almost_all_the_time_is_not_two_states():
    """One class nearly empty is not a bimodal timeline either."""
    from ambiscape.states import bimodal_separation
    rng = np.random.default_rng(7)
    x = rng.normal(-40.0, 0.5, 10000)
    x[:50] = -58.0                              # 0.5 % of the series
    assert bimodal_separation(x)["bimodal"] is False


def test_a_run_still_on_at_the_end_is_not_counted_as_an_on_time():
    """A recording that stops mid-run measures a short run that never was.

    Left in, the stub inverts the on-time trend: the Ghent kitchen's fridge
    read 7.6 -> 1.9 min and a negative correlation, when the twelfth run was
    simply cut off by the end of the analysis window.
    """
    from ambiscape.states import cycle_series
    segs = [{"state": "on", "t0_s": 0.0, "dur_s": 500.0},
            {"state": "off", "t0_s": 500.0, "dur_s": 1300.0},
            {"state": "on", "t0_s": 1800.0, "dur_s": 505.0},
            {"state": "off", "t0_s": 2305.0, "dur_s": 1300.0},
            {"state": "on", "t0_s": 3605.0, "dur_s": 40.0}]     # cut short
    c = cycle_series(segs)
    assert c["truncated_final_run"] is True
    assert c["on_s"] == [500.0, 505.0]          # the stub is excluded
    assert len(c["period_s"]) == 2              # onsets still close intervals
