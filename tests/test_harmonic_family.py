"""Harmonic-family tests: a hypothesised family must be able to fail.

The three cases below are the ones that matter for the method. The last is a
regression test on a wrong published claim rather than on a wrong line of
code: a 16⅔ Hz family contains the European mains family inside it, because
16⅔ times three is fifty, so a recording with nothing but mains in it scores
well on "the Nordic railway supply". That is what put a phantom railway line
into a book chapter, and the test pins the shape of the mistake so the fix
cannot quietly regress.
"""
import numpy as np
import pytest

from ambiscape.tonality import (family_percentile, family_prominence,
                                narrow_line_prominence)

FS = 4096
N = 1 << 19


def _spectrum(x, nfft=N, fs=FS):
    """(freqs, dB) for one Hann-windowed periodogram."""
    w = np.hanning(len(x))
    return (np.fft.rfftfreq(len(x), 1 / fs),
            10 * np.log10(np.abs(np.fft.rfft(x * w)) ** 2 + 1e-30))


def _comb(freqs, n=N, fs=FS, amp=0.5, seed=0):
    """White noise plus sinusoids at each of `freqs`."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    x = rng.normal(0, 1, n)
    for f in freqs:
        x = x + amp * np.sin(2 * np.pi * f * t)
    return x


def test_real_family_is_found_and_ranks_at_the_top():
    """A genuine 50/100/150 comb scores high and beats almost every rival."""
    f, db = _spectrum(_comb([50.0, 100.0, 150.0]))
    mean, rungs = family_prominence(f, db, 50.0)
    assert mean > 15.0
    first_three = [v for _n, _hz, v in rungs[:3]]
    assert all(v > 25.0 for v in first_three), first_three
    pct, score, _grid, _scores = family_percentile(f, db, 50.0)
    assert pct > 97.0, pct
    assert score == pytest.approx(mean)


def test_noise_alone_is_not_exceptional():
    """White noise must not rank at the top of its own sweep.

    Its family score is *not* zero --- comparing a max over the peak window
    against a median over the ring is biased upward --- which is exactly why
    the percentile exists and the raw decibels cannot be read alone.
    """
    rng = np.random.default_rng(7)
    f, db = _spectrum(rng.normal(0, 1, N))
    pct, _score, _grid, _scores = family_percentile(f, db, 50.0)
    assert pct < 95.0, pct


def test_a_lone_rung_does_not_make_a_family():
    """One strong harmonic with an empty fundamental must show as empty rungs.

    The mean is allowed to look respectable here; the contract is that the
    per-rung list exposes the hole, because that is the only thing that
    distinguishes this from a real family.
    """
    f, db = _spectrum(_comb([150.0]))
    _mean, rungs = family_prominence(f, db, 50.0)
    by_n = {n: v for n, _hz, v in rungs}
    assert by_n[3] > 25.0, by_n
    assert by_n[1] < 10.0 and by_n[2] < 10.0, by_n


def test_mains_masquerades_as_the_nordic_railway_supply():
    """A pure 50/100 comb scores well on 16⅔ Hz, and the rungs give it away.

    The regression this pins: 16⅔ x 3 = 50 and 16⅔ x 6 = 100, so a family
    built on the railway supply frequency *contains* the mains family. Any
    mains-bearing recording therefore passes a test for a Nordic train,
    including one made nowhere near a train. Rungs 1, 2, 4 and 5 must stay
    far below rungs 3 and 6 --- that gap is the whole evidence that the
    railway line is not there.
    """
    f, db = _spectrum(_comb([50.0, 100.0]))
    mean, rungs = family_prominence(f, db, 50.0 / 3.0)
    by_n = {n: v for n, _hz, v in rungs}
    assert mean > 5.0, mean
    carried = min(by_n[3], by_n[6])
    absent = max(by_n[1], by_n[2], by_n[4], by_n[5])
    assert carried - absent > 20.0, (carried, absent)


def test_narrow_line_prominence_returns_none_off_the_spectrum():
    """Asking beyond the analysed band is an absence, not a zero."""
    f, db = _spectrum(_comb([50.0]))
    assert narrow_line_prominence(f, db, 1e6) is None
