"""Biophony tests: acoustic-structure and spatial measures on synthetic
features that mimic a bird chorus versus broadband noise."""
import numpy as np
import pytest

from ambiscape import biophony


def _F(nsec=600, nmin=10, seed=0):
    logf = np.geomspace(25, 20000, 97)
    fc = np.sqrt(logf[:-1] * logf[1:])
    freqs = np.linspace(0, 24000, 8193)
    rng = np.random.default_rng(seed)
    ls = 1e-8 * (1 + 0.05 * rng.standard_normal((nsec, 96))) ** 2
    ms = 1e-9 * (1 + 0.05 * rng.standard_normal((nmin, len(freqs)))) ** 2
    return {"logspec": ls, "logf": logf.astype(np.float32), "fc": fc,
            "minspec": ms, "freqs": freqs.astype(np.float32),
            "rms_w": np.sqrt(ls.sum(1)).astype(np.float32),
            "az": np.zeros(nsec, np.float32), "el": np.zeros(nsec, np.float32),
            "t": np.arange(nsec, dtype=float)}


def _add_bird_peaks(F, minutes, freqs_hz=(3000, 4200, 5500), gain_db=25):
    for mi in minutes:
        for f0 in freqs_hz:
            m = np.abs(F["freqs"] - f0) < 8
            F["minspec"][mi, m] *= 10 ** (gain_db / 10)


# ---------------------------------------------------------- narrowband index

def test_narrowband_activity_finds_chorus():
    F = _F()
    _add_bird_peaks(F, range(2, 8))                    # 6 of 10 minutes
    na = biophony.narrowband_activity(F)
    assert na["median_peaks_per_min"] >= 3
    assert na["active_minute_fraction"] == pytest.approx(0.6, abs=0.15)
    quiet = biophony.narrowband_activity(_F())
    assert quiet["median_peaks_per_min"] < na["median_peaks_per_min"]


# ------------------------------------------------------------- band entropy

def test_band_temporal_entropy_structured_lower():
    flat = _F()
    ht_flat = biophony.band_temporal_entropy(flat)
    bursty = _F()
    fc = bursty["fc"]
    m = (fc >= 2000) & (fc <= 11000)
    bursty["logspec"][100:120, m] *= 1e4               # a concentrated burst
    ht_burst = biophony.band_temporal_entropy(bursty)
    assert ht_burst < ht_flat
    assert 0.0 <= ht_burst <= 1.0


# ---------------------------------------------------------------- activity

def test_band_activity_counts_events():
    F = _F()
    fc = F["fc"]
    m = (fc >= 2000) & (fc <= 11000)
    for t0 in (60, 200, 380, 500):
        F["logspec"][t0:t0 + 5, m] *= 1e3              # +30 dB, 5 s
    act = biophony.band_activity(F)
    assert act["event_rate_per_min"] > 0
    assert act["active_fraction"] > 0
    assert biophony.band_activity(_F())["active_fraction"] < \
        act["active_fraction"]


# ----------------------------------------------------------- spatial layer

def test_spatial_dispersion_of_chorus():
    F = _F()
    fc = F["fc"]
    m = (fc >= 2000) & (fc <= 11000)
    rng = np.random.default_rng(3)
    ev = np.arange(50, 550, 25)                        # 20 vocalizations
    for i, t0 in enumerate(ev):
        F["logspec"][t0:t0 + 3, m] *= 1e3
        F["az"][t0:t0 + 3] = rng.uniform(-180, 180)    # from all directions
        F["el"][t0:t0 + 3] = rng.uniform(15, 45)       # elevated
    sd = biophony.spatial_dispersion(F)
    assert sd["directional_entropy"] > 0.7
    assert sd["above_horizon_fraction"] > 0.7


def test_spatial_dispersion_point_source_low():
    F = _F()
    fc = F["fc"]
    m = (fc >= 2000) & (fc <= 11000)
    for t0 in range(50, 550, 25):
        F["logspec"][t0:t0 + 3, m] *= 1e3
        F["az"][t0:t0 + 3] = 12.0                      # one bearing
        F["el"][t0:t0 + 3] = 2.0
    sd = biophony.spatial_dispersion(F)
    assert sd["directional_entropy"] < 0.5


# --------------------------------------------------------------- summary

def test_summarize_biophony_keys():
    F = _F()
    _add_bird_peaks(F, range(2, 8))
    s = biophony.summarize_biophony(F)
    for k in ("bird_peaks_per_min", "bird_band_activity_pct",
              "bird_temporal_entropy", "bird_directional_entropy",
              "bird_above_horizon_fraction"):
        assert k in s


# ---------------------------------------------------------------- ml hook

def test_birdnet_available_is_bool():
    from ambiscape import ml
    assert isinstance(ml.birdnet_available(), bool)
