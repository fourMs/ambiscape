"""The rhythm is the signal: what never changes is what we cannot use.

A recorder's own hiss is stationary. Everything else in a room eventually
turns over: a fridge cycles in tens of minutes, a ventilation plant in hours,
a household in a day, a heating system in a season. So the question "is this
signal or noise?" becomes a question about *periodicity across timescales* —
find the period and you have named the thing.

The catch, which the SINS corpus supplies, is that the recorder has a
rhythm too: a converter warms and cools with the building, so its self-noise
carries a diurnal cycle. Periodicity alone therefore does not separate room
from recorder. The *period* does, and these tests pin that distinction.
"""
import numpy as np

from ambiscape import analysis


def _levels(hours, dt_s=10.0, floor_db=-60.0, cycles=(), seed=0):
    """A level series with named cycles laid on a stationary floor.

    ``cycles`` is a sequence of ``(period_s, depth_db, duty)``: square-wave
    machinery rather than a sine, because a fridge switches rather than
    swells.
    """
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 / dt_s)
    t = np.arange(n) * dt_s
    p = np.full(n, 10 ** (floor_db / 10))
    for period, depth, duty in cycles:
        on = (t % period) < (period * duty)
        p = p + on * 10 ** ((floor_db + depth) / 10)
    return 10 * np.log10(p * 10 ** (rng.normal(0, 0.3, n) / 10)), dt_s


def test_a_stationary_floor_has_no_cycle():
    """Self-noise with nothing else in it: no period should be reported."""
    db, dt = _levels(48, cycles=())
    found = analysis.dominant_cycles(db, dt, min_period_s=60,
                                     max_period_s=6 * 3600)
    assert found == [] or all(c["strength"] < 0.25 for c in found)


def test_a_fridge_is_found_at_its_own_period():
    """A machine cycling every 45 minutes, 8 dB up, on 60 % of the time."""
    db, dt = _levels(48, cycles=[(2700, 8.0, 0.6)])
    found = analysis.dominant_cycles(db, dt, min_period_s=300,
                                     max_period_s=6 * 3600)
    assert found, "a cycling machine must be found"
    top = found[0]
    assert abs(top["period_s"] - 2700) / 2700 < 0.15
    assert top["strength"] > 0.3


def test_two_cycles_at_different_scales_are_both_found():
    """A fridge inside a day. Both periods, and the band each belongs to.

    This is the multi-timescale case the toolbox exists for: one series
    carrying a machine and a household at once, neither hiding the other.
    """
    db, dt = _levels(24 * 6, dt_s=30.0,
                     cycles=[(2700, 8.0, 0.6), (86400, 12.0, 0.45)])
    found = analysis.dominant_cycles(db, dt, min_period_s=600,
                                     max_period_s=3 * 86400, top=4)
    periods = [c["period_s"] for c in found]
    assert any(abs(p - 2700) / 2700 < 0.2 for p in periods), periods
    assert any(abs(p - 86400) / 86400 < 0.2 for p in periods), periods


def test_each_cycle_is_named_by_the_band_it_falls_in():
    """A period is only useful once it is placed on the ladder."""
    db, dt = _levels(24 * 6, dt_s=30.0, cycles=[(86400, 12.0, 0.45)])
    found = analysis.dominant_cycles(db, dt, min_period_s=600,
                                     max_period_s=3 * 86400)
    assert found[0]["band"] == "circadian"


def test_the_instruments_own_diurnal_cycle_is_not_mistaken_for_a_room():
    """The trap the SINS corpus set.

    A node that sits at its own noise floor for most of a week, whose only
    real variation is thermal, still shows a 24-hour cycle. Periodicity therefore cannot mean "this is the room" on its own —
    a diurnal cycle with nothing faster beneath it is the signature of an
    recorder warming and cooling, not of a household.
    """
    thermal, dt = _levels(24 * 6, dt_s=30.0, cycles=[(86400, 9.0, 0.5)])
    lived, _ = _levels(24 * 6, dt_s=30.0,
                       cycles=[(2700, 8.0, 0.6), (86400, 12.0, 0.45)], seed=1)

    a = analysis.cycle_profile(thermal, dt, max_period_s=3 * 86400)
    b = analysis.cycle_profile(lived, dt, max_period_s=3 * 86400)

    assert a["has_sub_daily_cycle"] is False
    assert b["has_sub_daily_cycle"] is True
    assert a["diurnal_only"] is True
    assert b["diurnal_only"] is False


def test_a_spike_is_not_a_rhythm_and_the_residual_finds_it():
    """Anomaly and rhythm are opposite readings of the same series.

    Once a cycle is known it can be predicted and removed; what is left is
    the part the room did not repeat. A one-off spike has no period, so it
    survives that subtraction, while the fridge -- which an outlier detector
    would flag thirty times a day -- does not.
    """
    db, dt = _levels(48, cycles=[(2700, 8.0, 0.6)])
    spike_at = int(30 * 3600 / dt)
    db = db.copy()
    db[spike_at:spike_at + 3] += 20.0

    r = analysis.cycle_residual(db, dt, min_period_s=300,
                                max_period_s=6 * 3600)
    assert r["period_s"] is not None
    # the machine is explained away, the spike is not
    assert r["residual_std_db"] < np.std(db)
    assert r["anomalies"], "the spike must survive the subtraction"
    worst = max(r["anomalies"], key=lambda a: a["excess_db"])
    assert abs(worst["t_s"] - spike_at * dt) < 300


def test_a_room_with_only_its_rhythm_has_no_anomalies():
    """A fridge on its own is not an anomaly, however often it starts."""
    db, dt = _levels(48, cycles=[(2700, 8.0, 0.6)])
    r = analysis.cycle_residual(db, dt, min_period_s=300,
                               max_period_s=6 * 3600)
    assert r["period_s"] is not None
    assert r["anomalies"] == []
