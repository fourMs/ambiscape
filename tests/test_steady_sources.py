"""Telling a fridge from the recorder: the thing that stops is not the floor.

Minimum statistics over a short window treats anything steady as floor, which
is wrong for exactly the sources this toolbox exists to study. A fridge, a
ventilation plant and a circulation pump are steady for minutes at a time and
are the object of interest, not the noise.

What separates them from the recorder's own contribution is that **they turn
off**. Self-noise does not. So the floor is tracked at two timescales: over
minutes, which absorbs a running machine, and over hours, which does not
because the machine's off-phase falls inside the window. The difference is the
machinery.

A source that never stops within the long window is genuinely inseparable from
self-noise by this or any level-only method, and must be reported as such
rather than silently subtracted.
"""
import numpy as np

from ambiscape import analysis


def _room(n, dt=1.0, floor_db=-60.0, machine_db=None, period_s=2700,
          duty=0.6, events=True, seed=0):
    """A room: self-noise, optionally a cycling machine, optionally activity."""
    rng = np.random.default_rng(seed)
    p = np.full(n, 10 ** (floor_db / 10))
    if machine_db is not None:
        k = period_s / dt
        on = (np.arange(n) % k) < (k * duty)
        p = p + on * 10 ** (machine_db / 10)
    if events:
        for start in rng.integers(0, n, size=max(1, n // 400)):
            w = int(20 / dt)
            p[start:start + w] += 10 ** (-38 / 10)
    return 10 * np.log10(p * 10 ** (rng.normal(0, 0.1, n) / 10)), dt


def test_a_cycling_machine_is_found_and_not_called_floor():
    """A fridge 8 dB over the floor, on 60 % of a 45-minute cycle."""
    db, dt = _room(20000, machine_db=-52.0, floor_db=-60.0)
    r = analysis.steady_sources(db, dt, short_s=120.0, long_s=7200.0)

    assert r["machine_detected"] is True
    # the long-window floor finds the machine's off-phase; the short one does not
    assert abs(r["self_noise_db"] - (-60.0)) < 1.5
    assert r["steady_excess_db"] > 5.0
    assert 0.4 < r["machine_duty"] < 0.8


def test_a_room_without_machinery_reports_none():
    """Self-noise and occasional events only: nothing steady to find."""
    db, dt = _room(20000, machine_db=None, floor_db=-60.0)
    r = analysis.steady_sources(db, dt, short_s=120.0, long_s=7200.0)

    assert r["machine_detected"] is False
    assert abs(r["self_noise_db"] - (-60.0)) < 1.5
    assert r["steady_excess_db"] < 2.0


def test_a_source_that_never_stops_lands_inside_the_floor_and_says_so():
    """The honest failure, stated as what it is.

    A plant that runs continuously cannot be separated from self-noise by
    level alone -- not by this method and not by any other that sees one
    number per frame. So the flag does not claim detection; it says a
    constant source cannot be ruled out, and the reported self-noise
    therefore includes whatever is constant. Here that is 8 dB of machine.
    """
    n = 20000
    db, dt = _room(n, machine_db=-52.0, floor_db=-60.0, duty=1.0)
    r = analysis.steady_sources(db, dt, short_s=120.0, long_s=7200.0)

    assert r["machine_detected"] is False
    assert r["steady_source_unresolved"] is True
    # the machine is inside the reported floor -- which is what the flag warns
    assert r["self_noise_db"] > -54.0


def test_a_long_window_shorter_than_the_cycle_contaminates_the_floor():
    """The one parameter that matters, and how it fails.

    It does not fail crisply. With a long window shorter than the machine's
    cycle, some windows still catch an off-phase and some do not, so the
    machine is partly absorbed: the reported self-noise drifts upward toward
    the machine, and the excess attributed to it shrinks. The harm is a
    biased floor rather than a missed detection, which is harder to notice.
    """
    db, dt = _room(20000, machine_db=-52.0, floor_db=-60.0, period_s=2700)
    good = analysis.steady_sources(db, dt, short_s=120.0, long_s=7200.0)
    short = analysis.steady_sources(db, dt, short_s=120.0, long_s=300.0)

    # an adequate window recovers the true duty cycle, 0.6
    assert good["machine_detected"] is True
    assert 0.45 < good["machine_duty"] < 0.75

    # a window shorter than the cycle understates how long the machine runs,
    # and eventually the floor itself is pulled up to the machine's level
    assert short["machine_duty"] < 0.25
    assert short["self_noise_db"] > good["self_noise_db"] + 5.0
