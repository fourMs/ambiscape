"""An object's spectrum has a shape, and the shape is what a short clip has.

`object_profile` measures an envelope and says nothing about frequency. The
session-scale centroid and flux both need a minute of audio, which a sound
object of a second or two does not have. These are the same two quantities
defined on one object.
"""
import numpy as np

from ambiscape.objects import object_profile, object_spectrum

LOGF = np.geomspace(25.0, 16000.0, 97)
CENTRES = np.sqrt(LOGF[:-1] * LOGF[1:])
DT = 0.05


def _spec(rows):
    """dB spectrogram from a list of (centre_hz, width_oct, level_db)."""
    out = np.full((len(rows), len(CENTRES)), -90.0)
    for i, (f0, w, lvl) in enumerate(rows):
        d = np.abs(np.log2(CENTRES / f0))
        out[i] = lvl - 40.0 * np.clip(d - w, 0, None)
    return out


def test_a_held_tone_barely_moves():
    x = _spec([(1000.0, 0.1, 0.0)] * 20)
    r = object_spectrum(x, LOGF, DT)
    assert 800 < r["brightness_hz"] < 1250
    assert abs(r["brightness_drift_oct"]) < 0.05
    assert r["flux_per_s"] < 0.5


def test_a_struck_body_grows_duller_as_it_decays():
    """Bright at the strike, dull in the tail: drift is negative."""
    rows = [(4000.0, 0.4, 0.0), (3000.0, 0.4, -6.0), (1800.0, 0.4, -12.0)]
    rows += [(900.0, 0.4, -18.0 - 2 * i) for i in range(12)]
    r = object_spectrum(_spec(rows), LOGF, DT)
    assert r["brightness_drift_oct"] < -1.0


def test_a_kettle_grows_brighter():
    rows = [(400.0, 0.4, -6.0)] + [(400.0 * 2 ** (i / 8), 0.4, -6.0)
                                   for i in range(1, 16)]
    r = object_spectrum(_spec(rows), LOGF, DT)
    assert r["brightness_drift_oct"] > 0.5


def test_a_churning_object_has_more_flux_than_a_steady_one():
    rng = np.random.default_rng(0)
    steady = _spec([(1000.0, 0.3, 0.0)] * 24)
    churn = _spec([(float(rng.uniform(300, 6000)), 0.3, 0.0) for _ in range(24)])
    assert (object_spectrum(churn, LOGF, DT)["flux_per_s"]
            > 5 * object_spectrum(steady, LOGF, DT)["flux_per_s"])


def test_flux_does_not_depend_on_the_hop():
    """Reported per second, so halving dt must not double it."""
    x = _spec([(1000.0, 0.3, 0.0), (2000.0, 0.3, 0.0)] * 12)
    a = object_spectrum(x, LOGF, 0.05)["flux_per_s"]
    b = object_spectrum(x, LOGF, 0.10)["flux_per_s"]
    assert abs(a - 2 * b) < 0.01          # same churn, half the frame rate
                                          # (0.001 apart, which is the rounding)


def test_a_silent_or_one_frame_object_returns_nothing():
    assert object_spectrum(np.full((1, 96), -90.0), LOGF, DT) == {}
    assert object_spectrum(np.full((8, 96), -300.0), LOGF, DT) == {}


def test_object_profile_merges_the_spectrum_when_given_one():
    env = np.concatenate([np.linspace(0, 1, 4), np.linspace(1, 0, 16)])
    plain = object_profile(env, DT)
    withspec = object_profile(env, DT, logspec=_spec([(1000.0, 0.2, 0.0)] * 20),
                              logf=LOGF)
    assert "brightness_hz" not in plain
    assert withspec["brightness_hz"] > 0
    assert withspec["duration_s"] == plain["duration_s"]
