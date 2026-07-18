"""Function-level tests: circstats, background, schedule, timbre,
modulation, tonality — synthetic arrays, no audio."""
import numpy as np
import pytest

from ambiscape import circstats, background, schedule, tonality
from ambiscape.modulation import profile
from ambiscape.timbre import cluster_events


# ---------------------------------------------------------------- circstats

def test_circ_concentrated_vs_uniform():
    rng = np.random.default_rng(0)
    mu, R = circstats.mean_resultant(rng.normal(1.0, 0.05, 500))
    assert R > 0.99 and mu == pytest.approx(1.0, abs=0.01)
    _mu, Ru = circstats.mean_resultant(rng.uniform(0, 2 * np.pi, 2000))
    assert Ru < 0.1
    assert circstats.rayleigh_p(Ru, 2000) > 1e-4
    assert 0.0 <= circstats.rayleigh_p(0.99, 1000) <= 1.0


def test_phase_stats_recovers_period_phase():
    times = 0.7 + 2.5 * np.arange(100) + np.random.default_rng(1).normal(
        0, 0.02, 100)
    st = circstats.phase_stats(times, 2.5)
    assert st["R"] > 0.98
    assert st["mean_phase"] == pytest.approx(0.7 / 2.5, abs=0.02)
    assert st["circ_sd_s"] == pytest.approx(0.02, abs=0.015)


def test_relative_phase_offset():
    a = 3.0 * np.arange(50)
    b = a + 1.2
    rel = circstats.relative_phase(b, a, 3.0)
    assert rel["mean_offset_s"] == pytest.approx(1.2, abs=0.01)
    assert rel["R"] > 0.999


# ---------------------------------------------------------------- background

def _fake_F(nsec=600, nband=96):
    rng = np.random.default_rng(2)
    logspec = 1e-6 * (1 + 0.1 * rng.standard_normal((nsec, nband))) ** 2
    logf = np.geomspace(25, 20000, nband + 1)
    return {"logspec": logspec, "logf": logf.astype(np.float32)}


def test_spectral_events_finds_blob():
    F = _fake_F()
    F["logspec"][300:310, 40:45] *= 100.0          # +20 dB, 10 s, 5 bands
    bg = background.band_background(F["logspec"])
    rise, frac = background.foreground(F["logspec"], bg)
    ev = background.spectral_events(rise, F["logf"])
    assert any(abs(e["t0_s"] - 300) <= 2 and e["peak_rise_db"] > 15
               for e in ev)
    assert frac[305] > 0.2


def test_masking_index_selective():
    F = _fake_F()
    active = np.zeros(600, bool); active[:300] = True
    F["logspec"][:300, 50:52] *= 10 ** 1.2         # +12 dB in two bands
    mi = background.masking_index(F, active, ~active)
    el = np.array(mi["elevation_db_per_band"])
    assert el[50] == pytest.approx(12.0, abs=1.5)
    assert mi["floor_elevation_median_db"] == pytest.approx(0.0, abs=1.0)
    assert mi["floor_elevation_max_db"] > 10


# ---------------------------------------------------------------- schedule

def test_match_periods_quarter_hour():
    rng = np.random.default_rng(3)
    times = 54000 + 900 * np.arange(12) + 120 + rng.normal(0, 3, 12)
    top = schedule.match_periods(times)[0]
    assert top["period_s"] == 900.0
    assert top["phase_s"] == pytest.approx(120, abs=10)
    assert top["n_cycles"] >= 11


def test_clock_offset_helper():
    assert schedule.clock_offset(75469.0, 76134.0) == pytest.approx(665.0)


# ---------------------------------------------------------------- timbre

def test_cluster_events_two_classes():
    rng = np.random.default_rng(4)
    a = np.tile(np.sin(np.linspace(0, 3, 48)), (10, 1))
    b = np.tile(np.cos(np.linspace(0, 5, 48)), (8, 1))
    fps = np.concatenate([a, b]) + 0.05 * rng.standard_normal((18, 48))
    lab = cluster_events(fps)
    assert len(set(lab[:10])) == 1 and len(set(lab[10:])) == 1
    assert lab[0] != lab[-1]


# ---------------------------------------------------------------- modulation

def test_profile_finds_am_rate():
    dur, hi_dt = 600, 0.02
    t_hi = np.arange(0, dur, hi_dt)
    env = (1 + 0.8 * np.sin(2 * np.pi * 2.0 * t_hi)) ** 2   # 2 Hz AM
    t = np.arange(dur, dtype=float)
    tf = np.arange(0, dur, 0.125)
    F = {"t": t, "t_fast": tf,
         "fast_db": -30 + 3 * np.sin(2 * np.pi * tf / 60),  # 60 s meso wave
         "rms_w": 0.1 * np.ones(dur, np.float32),
         "env_hi": env.astype(np.float32), "hi_dt": hi_dt}
    prof = profile(F)
    assert prof["scales"]["micro"]["peak_freq_hz"] == pytest.approx(2.0,
                                                                    rel=0.15)
    assert prof["scales"]["meso"]["peak_period_s"] == pytest.approx(60,
                                                                    rel=0.25)


# ---------------------------------------------------------------- tonality

def _spec_with_peaks(freqs, peaks, width=6.0):
    s = 1e-9 * np.ones_like(freqs)
    for f0, p in peaks:
        s += p * np.exp(-0.5 * ((freqs - f0) / width) ** 2)
    return s


def test_harmonic_sieve_scores():
    freqs = np.linspace(0, 8000, 4001)
    harm = [(220 * k, 1.0 / k) for k in range(1, 7)]
    spec = _spec_with_peaks(freqs, harm)
    fq, _prom, pw = tonality.tonal_peaks(spec, freqs)
    f0, h = tonality.harmonic_sieve(fq, pw)
    assert h > 0.9 and f0 == pytest.approx(220, rel=0.02)
    bell = [(480 * r, 1.0) for r in (1, 2.0, 2.4, 3.0, 4.0, 5.2, 6.05)]
    fqb, _pb, pwb = tonality.tonal_peaks(_spec_with_peaks(freqs, bell), freqs)
    _f0b, hb = tonality.harmonic_sieve(fqb, pwb)
    assert hb < h - 0.15


def test_pitch_class_profile_peak():
    freqs = np.linspace(0, 8000, 4001)
    minspec = np.tile(_spec_with_peaks(freqs, [(440.0, 1.0), (880.0, 0.5)]),
                      (3, 1))
    pcp = tonality.pitch_class_profile(minspec, freqs)
    assert tonality.NOTE[int(np.argmax(pcp))] == "A"


def test_tonal_tracks_duration():
    freqs = np.linspace(0, 8000, 4001)
    rows = [_spec_with_peaks(freqs, [(1000.0, 1.0)]) for _ in range(8)] + \
           [_spec_with_peaks(freqs, []) for _ in range(4)]
    tracks = tonality.tonal_tracks(np.array(rows), freqs)
    assert tracks and tracks[0]["minutes"] == 8
    assert tracks[0]["f_median_hz"] == pytest.approx(1000, abs=10)
