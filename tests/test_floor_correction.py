"""Measuring the source rather than the source plus the recorder.

Every level a recorder reports is P = S + N. Reading it as S produced four
wrong findings in the SINS sensor-network corpus: a loudest-room rule that
ranked floor depth, a speech comparison that ranked sensitivity, a "dead
channel" verdict on a quiet bedroom, and a diurnal rhythm that was a
converter warming up.

Three behaviours are required, and the third is the one usually missing:
the floor must be tracked over time rather than fixed, the subtraction must
happen in power, and frames with nothing above the floor must be *censored
and counted* rather than clamped to zero and averaged.
"""
import numpy as np
import pytest

from ambiscape import analysis


def _series(sig_db, floor_db, n, dt=0.125, seed=0, duty=0.5, burst_s=20.0):
    """Frames of 10*log10(signal power + floor power), with jitter.

    The source comes in bursts rather than running continuously, because
    that is what a room does and what minimum statistics requires: a source
    present in every frame of every window is indistinguishable from a
    floor, and is correctly absorbed into one. ``duty=1.0`` produces that
    degenerate case for the test that pins it.
    """
    rng = np.random.default_rng(seed)
    f = 10 ** (np.asarray(floor_db, float) / 10)
    if sig_db is None:
        s = np.zeros(n)
    else:
        s = 10 ** (np.asarray(sig_db, float) * np.ones(n) / 10)
        if duty < 1.0:
            k = max(1, int(burst_s / dt))
            on = (np.arange(n) // k) % 2 == 0
            s = np.where(on, s, 0.0)
    p = (s + f) * 10 ** (rng.normal(0, 0.15, n) / 10)
    return 10 * np.log10(p), dt


def test_tracks_a_floor_that_drifts():
    """A fixed per-node floor cannot describe a floor that moves.

    The dead SINS node swings 10.7 dB between night and midday as its
    electronics warm. An estimator that returns one number for the session
    is wrong by that much at both ends.
    """
    n = 8000
    drift = np.linspace(-65, -55, n)            # 10 dB of thermal drift
    db, dt = _series(None, drift, n)
    floor = analysis.track_noise_floor(db, dt, win_s=60.0)
    assert floor[:200].mean() < floor[-200:].mean() - 6
    assert np.abs(floor - drift).mean() < 2.5


def test_subtracts_in_power_not_in_decibels():
    """A source 10 dB over the floor is 10 dB over it, not 10.4.

    Subtracting decibels would leave the source untouched; adding powers
    and subtracting them recovers it. The difference is small here and
    systematic, which is the dangerous kind.
    """
    n = 12000
    db, dt = _series(-50.0, -60.0, n)
    sig, floor, ok = analysis.floor_corrected_level(db, dt, margin_db=3.0)
    assert ok.mean() > 0.4                      # roughly the duty cycle
    assert np.abs(np.nanmedian(sig) - (-50.0)) < 1.0


def test_frames_without_signal_are_censored_not_zeroed():
    """Silence over a floor must read as unmeasurable, never as zero.

    A frame that does not clear the floor carries no information about the
    source. Returning the floor, or zero, invents one.
    """
    n = 6000
    db, dt = _series(None, -60.0, n)
    sig, floor, ok = analysis.floor_corrected_level(db, dt, margin_db=6.0)
    assert ok.mean() < 0.05
    assert np.isnan(sig[~ok]).all()


def test_aggregate_reports_coverage_and_refuses_when_too_thin():
    """A level over 4 % of frames is a different claim from one over 96 %.

    The number alone cannot tell them apart, so coverage travels with it and
    the aggregate declines rather than mislead. This is the SINS node 11
    case: a raw Leq that reads as a plausible room, from 4 % of frames.
    """
    n = 12000
    loud, dt = _series(-45.0, -60.0, n)
    quiet, _ = _series(None, -60.0, n)

    a = analysis.summarize_floor_corrected(*analysis.floor_corrected_level(loud, dt)[::2])
    assert a["coverage"] > 0.4
    assert a["level_db"] is not None

    b = analysis.summarize_floor_corrected(*analysis.floor_corrected_level(quiet, dt)[::2])
    assert b["coverage"] < 0.1
    assert b["level_db"] is None


def test_censoring_bias_is_reported_not_hidden():
    """The trap this exists to close.

    A source that is loud in one half and absent in the other. Averaging the
    frames that survive censoring overstates the quiet half enormously,
    because only its loudest moments clear the floor. The aggregate must not
    present that as the level of the whole span: coverage makes the
    censoring visible.
    """
    n = 16000
    half = n // 2
    db, dt = _series(-45.0, -60.0, n)
    quiet, _ = _series(None, -60.0, n, seed=1)
    db = np.concatenate([db[:half], quiet[half:]])
    sig_db, floor, ok = analysis.floor_corrected_level(db, dt, margin_db=6.0)

    # the quiet half contributes almost nothing, and that is visible
    assert ok[:half].mean() > 0.4
    assert ok[half:].mean() < 0.1
    a = analysis.summarize_floor_corrected(sig_db, ok)
    assert 0.15 < a["coverage"] < 0.45        # part of the span, not all of it
    # and the reported level belongs to the measurable half, not the mean
    assert abs(a["level_db"] - (-45.0)) < 1.5


def test_margin_is_honoured():
    """A larger margin admits fewer frames; it never admits more."""
    n = 6000
    db, dt = _series(-52.0, -60.0, n)
    lo = analysis.floor_corrected_level(db, dt, margin_db=3.0)[2].mean()
    hi = analysis.floor_corrected_level(db, dt, margin_db=12.0)[2].mean()
    assert hi <= lo


def test_a_source_that_never_stops_is_absorbed_into_the_floor():
    """A documented limitation, pinned so it cannot surprise anyone.

    Minimum statistics asks what the quietest moment of each window looks
    like. A sound present in every frame -- ventilation, a fridge, traffic
    hum -- has no quietest moment to find, so it is measured as floor and
    subtracted away. This is right when the constant *is* the recorder and
    wrong when it is the room, and no arithmetic inside the function can
    tell the two apart. It matters in this corpus, where "loud ventilation"
    is a recurring diary entry.
    """
    n = 12000
    db, dt = _series(-50.0, -60.0, n, duty=1.0)   # never silent
    sig, floor, ok = analysis.floor_corrected_level(db, dt, margin_db=6.0)
    assert ok.mean() < 0.05                       # nothing stands above it
    assert floor.mean() > -52.0                   # the source became the floor
