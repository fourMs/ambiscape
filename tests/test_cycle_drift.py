"""A machine whose rhythm is changing — the case between spike and cycle.

An event detector sees a fridge start. An outlier detector sees the same
start and calls it anomalous thirty times a day. A cycle finder sees the
period and calls it normal. None of them sees the thing that actually
matters to whoever owns the building: that the period is *drifting*, because
the compressor is struggling.

It is not an anomaly -- nothing here is out of the ordinary from one moment
to the next -- and it is not the rhythm either, because the rhythm is no
longer what it was. It is a change in the rhythm, and it needs its own test.
"""
import numpy as np

from ambiscape import analysis


def _machine(hours, dt_s=10.0, floor_db=-60.0, depth_db=8.0,
             p_start=2700.0, p_end=2700.0, duty=0.6, seed=0):
    """A cycling machine whose period moves linearly from start to end."""
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 / dt_s)
    p = np.full(n, 10 ** (floor_db / 10))
    phase, on = 0.0, np.zeros(n, bool)
    for i in range(n):
        period = p_start + (p_end - p_start) * i / max(1, n - 1)
        phase += dt_s / period
        on[i] = (phase % 1.0) < duty
    p = p + on * 10 ** ((floor_db + depth_db) / 10)
    return 10 * np.log10(p * 10 ** (rng.normal(0, 0.3, n) / 10)), dt_s


def test_a_steady_machine_shows_no_drift():
    db, dt = _machine(60, p_start=2700.0, p_end=2700.0)
    r = analysis.cycle_drift(db, dt, min_period_s=600, max_period_s=2 * 3600)
    assert r["period_s"] is not None
    assert r["drifting"] is False


def test_a_lengthening_period_is_detected_with_its_direction():
    """The compressor takes longer and longer to complete a cycle."""
    db, dt = _machine(60, p_start=2400.0, p_end=3400.0)
    r = analysis.cycle_drift(db, dt, min_period_s=600, max_period_s=2 * 3600)
    assert r["drifting"] is True
    assert r["direction"] == "lengthening"
    assert r["drift_pct"] > 15.0


def test_a_shortening_period_is_detected_too():
    db, dt = _machine(60, p_start=3400.0, p_end=2400.0)
    r = analysis.cycle_drift(db, dt, min_period_s=600, max_period_s=2 * 3600)
    assert r["drifting"] is True
    assert r["direction"] == "shortening"


def test_it_declines_when_there_is_no_cycle_to_track():
    """A stationary series has no rhythm, so it cannot have a changing one."""
    rng = np.random.default_rng(0)
    db = -60 + rng.normal(0, 0.3, 20000)
    r = analysis.cycle_drift(db, 10.0, min_period_s=600, max_period_s=2 * 3600)
    assert r["period_s"] is None
    assert r["drifting"] is False
