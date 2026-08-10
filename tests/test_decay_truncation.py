"""Schroeder integration has to stop where the decay meets the noise.

Backward integration sums everything after a point, so if the integral runs
to the end of the file it accumulates the noise energy of the whole tail into
every earlier value. That flattens the decay curve and inflates T60 without
any warning: a 0.6 s decay recorded over two seconds with a 45 dB floor was
measured at 4.67 s.

Truncating the *fit range* is not the same thing and does not help -- the
curve is already wrong by the time it is fitted. The integration itself must
stop at the knee where the decay reaches the floor. This is Lundeby's method
and ISO 3382 asks for it.
"""
import numpy as np
import pytest

from ambiscape.impulse import ir_metrics

FS = 48000


def _ir(t60=0.6, dur=2.0, floor_db=None, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * FS)
    t = np.arange(n) / FS
    ir = rng.normal(0, 1, n) * 10 ** (-3 * t / t60)
    if floor_db is not None:
        ir = ir + rng.normal(0, 10 ** (floor_db / 20), n)
    return ir


@pytest.mark.parametrize("floor_db", [-40, -45, -50, -60])
def test_t60_survives_a_noise_floor(floor_db):
    """The decay is 0.6 s whatever the floor sits at."""
    got = ir_metrics(_ir(floor_db=floor_db), FS)["1000"]["T60"]
    assert got is None or abs(got - 0.6) < 0.15, (floor_db, got)


def test_a_long_silent_tail_does_not_inflate_the_estimate():
    """Two seconds of file for a 0.6 s decay is the archive-IR case."""
    short = ir_metrics(_ir(dur=0.8, floor_db=-45), FS)["1000"]["T60"]
    long = ir_metrics(_ir(dur=4.0, floor_db=-45), FS)["1000"]["T60"]
    assert short is not None and long is not None
    assert abs(long - short) < 0.2, (short, long)


def test_a_clean_ir_is_unchanged():
    """The fix must not disturb the case that already worked."""
    got = ir_metrics(_ir(floor_db=None), FS)["1000"]["T60"]
    assert abs(got - 0.6) < 0.1


def test_a_floor_too_high_to_measure_returns_nothing():
    """With the floor above the range T60 needs, the honest answer is None,
    not a number arrived at by fitting through noise."""
    band = ir_metrics(_ir(t60=1.5, dur=2.0, floor_db=-12), FS).get("1000")
    # the band may be dropped outright, which is the strongest form of
    # declining to answer
    got = None if band is None else band.get("T60")
    assert got is None or got < 3.0


# ------------------------------------------------------------------ fit span
#
# The `dr < 20` guard bounds the dynamic range but nothing bounds the *fit
# span*. T60's lower limit is adaptive, max(-35, -dr + 8), so the fitted
# range is (dr - 13) dB wide: at dr = 20 that is 7 dB, extrapolated 8.5x to
# reach 60. Measured on a 0.6 s decay cut to 0.20 s, that lever reported
# 0.34 s -- 43% low, with no warning. ISO 3382 fixes T20 and T30 at 20 and
# 30 dB of observed decay for exactly this reason.
#
# Under-estimation, note, where the truncation fault above over-estimated.


def test_a_collapsed_fit_span_is_refused_rather_than_extrapolated():
    """An IR cut to a third of its own decay must not report a confident T60.

    0.20 s of a 0.6 s decay leaves ~7 dB to fit and an 8x extrapolation.
    Reporting nothing is correct; reporting 0.34 s is not.
    """
    band = ir_metrics(_ir(dur=0.20, floor_db=-45), FS).get("1000")
    got = None if band is None else band.get("T60")
    assert got is None or abs(got - 0.6) < 0.15, (
        f"T60 extrapolated from a collapsed fit range: {got}")


@pytest.mark.parametrize("dur", [0.20, 0.22, 0.25])
def test_short_archive_irs_never_report_a_wildly_low_t60(dur):
    """The whole marginal band, not just the one duration that was noticed."""
    band = ir_metrics(_ir(dur=dur, floor_db=-45), FS).get("1000")
    got = None if band is None else band.get("T60")
    assert got is None or got > 0.4, (dur, got)


def test_a_healthy_ir_still_reports_t60():
    """The guard must not cost the cases that were always fine."""
    assert ir_metrics(_ir(dur=2.0, floor_db=-45), FS)["1000"]["T60"] is not None
