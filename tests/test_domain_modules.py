"""mechanical / anthropophony / geophony detectors and their summary merge."""
import numpy as np

from ambiscape import anthropophony, geophony, mechanical
from ambiscape.resolve import full_summary


def _oct(low=0.0, mid=0.0, high=0.0, n=60):
    # octaves [31.5,63,125, 250,500,1000,2000, 4000,8000,16000]
    op = np.full((n, 10), 1e-6)
    op[:, 0:3] += low       # < 250 Hz
    op[:, 3:7] += mid       # 250-2000 Hz (voice)
    op[:, 7:10] += high     # 4-16 kHz
    return op


def _F(oct_pow, env_hi=None, flatness=None, diffuse=None):
    n = len(oct_pow)
    return {
        "oct_pow": np.asarray(oct_pow, float),
        "env_hi": np.ones(1000) if env_hi is None else np.asarray(env_hi, float),
        "hi_dt": 0.02,
        "flatness": np.full(n, 0.1) if flatness is None else np.asarray(flatness, float),
        "diffuse": np.full(n, 0.5) if diffuse is None else np.asarray(diffuse, float),
    }


def test_mechanical_lowfreq_discriminates():
    assert mechanical.summarize_mechanical(_F(_oct(low=1.0)))["mech_lowfreq_fraction"] > 0.7
    assert mechanical.summarize_mechanical(_F(_oct(high=1.0)))["mech_lowfreq_fraction"] < 0.3


def test_mechanical_periodicity_peak():
    t = np.arange(2000) * 0.02
    env = 1 + 0.6 * np.sin(2 * np.pi * 2.0 * t)     # 2 Hz bogie-like rhythm
    per = mechanical.envelope_periodicity(_F(_oct(low=1.0), env_hi=env))
    assert per["hz"] is not None and abs(per["hz"] - 2.0) < 0.3
    assert per["strength"] > 0.1


def test_anthropophony_syllabic_modulation():
    t = np.arange(2000) * 0.02
    voiced = _F(_oct(mid=1.0), env_hi=1 + 0.6 * np.sin(2 * np.pi * 4.0 * t))
    flat = _F(_oct(mid=1.0), env_hi=np.ones(2000))
    assert anthropophony.syllabic_modulation(voiced) > 0.1
    assert anthropophony.syllabic_modulation(flat) < 0.05


def test_geophony_wind_vs_rain():
    wind = geophony.summarize_geophony(_F(_oct(low=1.0), diffuse=np.full(60, 0.9)))
    rain = geophony.summarize_geophony(_F(_oct(high=1.0), flatness=np.full(60, 0.9)))
    assert wind["geo_wind_index"] > wind["geo_rain_index"]
    assert rain["geo_rain_index"] > rain["geo_wind_index"]


def test_full_summary_merges_domain_indices(bell_features):
    _, _, F = bell_features
    s = full_summary(F)
    for k in ("mechanical_index", "anthropophony_index", "geophony_index",
              "mech_lowfreq_fraction", "anthro_syllabic_mod", "geo_wind_index"):
        assert k in s, f"missing {k}"
