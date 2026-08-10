"""Which harmonic series is this? More than one answer is defensible.

The sieve's parameters do not merely tune precision -- they choose between
answers, and the defaults choose badly for machinery. The fixture is the
measured peak set of a dishwasher's circulation pump: a Welch estimate of a
25-minute window at 0.37 Hz resolution, peaks with prominence >= 6 dB between
80 and 1300 Hz. Real rather than synthetic on purpose, because a clean comb
does not reproduce the ambiguity -- every tolerance finds the same f0 -- and
the ambiguity is the thing being pinned.
"""
import numpy as np

from ambiscape.tonality import harmonic_sieve

FQ = (100.0, 146.9, 158.2, 183.8, 201.8, 231.1, 269.2, 275.8, 299.9, 302.9,
      343.1, 359.6, 367.7, 378.3, 403.9, 425.5, 517.5, 605.7, 643.8, 781.5,
      881.8, 933.8, 980.7, 1064.6, 1127.9, 1175.9, 1224.6)
DB = (-73.4, -72.5, -71.4, -72.0, -76.0, -76.2, -61.2, -60.3, -70.5, -67.9,
      -76.3, -76.8, -71.0, -78.4, -84.4, -81.5, -84.4, -83.5, -83.9, -87.5,
      -89.2, -90.7, -91.2, -90.9, -92.5, -94.1, -96.9)
SHAFT = 45.96          # what the strong low peaks put the shaft at


def _pump():
    fq = np.array(FQ, float)
    return fq, 10 ** (np.array(DB, float) / 10)


def test_the_default_lower_bound_hides_a_machine_shaft():
    """f0_min = 60 Hz is above many shaft rates, so it returns a harmonic."""
    fq, power = _pump()
    f0, _ = harmonic_sieve(fq, power, tol_cents=8.0, max_harm=28)
    assert f0 is not None and abs(f0 - 2 * SHAFT) < 1.0
    lowered, _ = harmonic_sieve(fq, power, f0_min=40.0, tol_cents=8.0,
                                max_harm=28)
    assert abs(lowered - SHAFT) < 0.5


def test_the_default_tolerance_picks_a_different_fundamental():
    """And scores it higher while explaining fewer peaks."""
    fq, power = _pump()
    loose_f0, loose_h = harmonic_sieve(fq, power, f0_min=40.0,
                                       tol_cents=35.0, max_harm=28)
    tight_f0, tight_h = harmonic_sieve(fq, power, f0_min=40.0,
                                       tol_cents=8.0, max_harm=28)
    assert abs(tight_f0 - SHAFT) < 0.5
    assert abs(loose_f0 - SHAFT) > 5.0          # a different answer entirely
    assert loose_h > tight_h                    # and a higher score


def test_the_tight_fit_is_the_one_that_lands_on_the_strongest_peaks():
    """Which is why the higher score is not the better answer.

    The two strongest peaks are 275.8 and 367.7 Hz. The shaft explains both
    at zero cents; the loose winner misses the second by two semitones.
    """
    fq, power = _pump()
    loose_f0, _ = harmonic_sieve(fq, power, f0_min=40.0, tol_cents=35.0,
                                 max_harm=28)

    def cents(f, f0):
        k = max(1, round(f / f0))
        return 1200 * abs(np.log2(f / (k * f0)))

    for peak in (275.8, 367.7):
        assert cents(peak, SHAFT) < 5.0
    assert cents(367.7, loose_f0) > 100.0


def test_an_empty_peak_set_is_not_an_error():
    assert harmonic_sieve(np.array([]), np.array([])) == (None, 0.0)
