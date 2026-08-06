"""Tests for the psychoacoustics beyond loudness/sharpness/roughness:
fluctuation strength (Fastl & Zwicker approximation) and DIN 45681-style
tonal prominence — synthetic ground truth throughout."""
import sys

import numpy as np
import pytest

from ambiscape import iso

FS = 48000


def _tone(f=1000.0, dur=6.0, amp=0.3, fs=FS):
    t = np.arange(int(dur * fs)) / fs
    return t, amp * np.sin(2 * np.pi * f * t)


def _am(f_mod, m=1.0, dur=6.0, fs=FS):
    t, x = _tone(dur=dur, fs=fs)
    return x * (1 + m * np.sin(2 * np.pi * f_mod * t))


def _meanspec(x, fs=FS, nfft=8192):
    """Mean hann-windowed power spectrum, same estimator as features.py."""
    win = np.hanning(nfft)
    ws2 = (win ** 2).sum()
    hop = nfft // 2
    n = (len(x) - nfft) // hop + 1
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    X = np.fft.rfft(x[idx] * win)
    return ((X.real ** 2 + X.imag ** 2) / ws2).mean(0), \
        np.fft.rfftfreq(nfft, 1 / fs)


def _pink(n, seed):
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    return np.fft.irfft(X / np.sqrt(f), n)


# ------------------------------------------------- fluctuation strength

def test_am_tone_reads_one_vacil():
    """The classic reference — 1 kHz tone, 100 % AM at 4 Hz — anchors the
    approximation at 1 vacil."""
    assert iso.fluctuation_strength(_am(4.0), FS) == pytest.approx(1.0,
                                                                   abs=0.15)


def test_steady_tone_has_no_fluctuation():
    _t, x = _tone()
    assert iso.fluctuation_strength(x, FS) < 0.05


def test_modulation_rate_weighting_peaks_at_4hz():
    """Same depth at 0.5 Hz scores far below 4 Hz (the ~4 Hz weighting)."""
    fs4 = iso.fluctuation_strength(_am(4.0), FS)
    fs_slow = iso.fluctuation_strength(_am(0.5), FS)
    assert fs4 > 3 * fs_slow


def test_modulation_depth_monotone():
    shallow = iso.fluctuation_strength(_am(4.0, m=0.3), FS)
    deep = iso.fluctuation_strength(_am(4.0, m=1.0), FS)
    assert 0.02 < shallow < deep


def test_unmodulated_noise_stays_low():
    """Random envelope fluctuation of steady noise must not read as AM."""
    rng = np.random.default_rng(3)
    x = 0.1 * rng.standard_normal(6 * FS)
    assert iso.fluctuation_strength(x, FS) < 0.3


def test_fluctuation_index_tracks_am():
    """Feature-domain companion on a 20 ms cached power envelope."""
    hfs = int(0.02 * FS)
    env_am = (_am(4.0) ** 2).reshape(-1, hfs).mean(1)
    env_st = (_tone()[1] ** 2).reshape(-1, hfs).mean(1)
    fi_am = iso.fluctuation_index(env_am, 0.02)
    fi_st = iso.fluctuation_index(env_st, 0.02)
    assert fi_am > 0.5
    assert fi_st < 0.05
    assert iso.fluctuation_index(np.ones(8), 0.02) is None   # too short


# ---------------------------------------------------- tonal prominence

def test_steady_tone_in_noise_detected_with_correct_prominence():
    """A 1 kHz tone in white noise: correct frequency and ΔL within a few
    dB of the analytic tone-vs-critical-band-noise level difference."""
    rng = np.random.default_rng(0)
    a, sigma = 0.05, 0.01
    _t, x = _tone(f=1000.3, amp=a)
    spec, freqs = _meanspec(x + sigma * rng.standard_normal(len(x)))
    tones = iso.tone_prominence(spec, freqs)
    assert tones, "tone not detected"
    top = tones[0]
    assert top["f_hz"] == pytest.approx(1000.3, abs=6.0)     # within a bin
    expected = 10 * np.log10(a ** 2 * FS
                             / (4 * sigma ** 2
                                * float(iso.critical_bandwidth(1000.0))))
    assert top["dL_db"] == pytest.approx(expected, abs=3.0)


def test_prominent_tones_aggregate_presence():
    """A hum present in 4 of 5 minutes aggregates into one persistent tone;
    a single-minute blip falls under the presence threshold."""
    rng = np.random.default_rng(1)
    rows = []
    for r in range(5):
        _t, hum = _tone(f=250.0, amp=0.04 if r < 4 else 0.0, dur=6.0)
        _t, blip = _tone(f=3000.0, amp=0.04 if r == 0 else 0.0, dur=6.0)
        spec, freqs = _meanspec(hum + blip
                                + 0.01 * rng.standard_normal(len(hum)))
        rows.append(spec)
    tones = iso.prominent_tones(np.array(rows), freqs, min_fraction=0.5)
    assert len(tones) == 1
    assert tones[0]["f_hz"] == pytest.approx(250.0, abs=6.0)
    assert tones[0]["present_fraction"] == pytest.approx(0.8, abs=0.01)
    assert tones[0]["n_minutes"] == 4


def test_pink_noise_has_no_prominent_tones():
    rows = [_meanspec(_pink(6 * FS, seed))[0] for seed in range(4)]
    _spec, freqs = _meanspec(_pink(6 * FS, 0))
    assert iso.prominent_tones(np.array(rows), freqs) == []


# ----------------------------------------------------- summary wiring

def test_summarize_psycho_keys_and_degradation():
    """Hum + AM envelope produce the summary keys; a cache without
    minspec/env_hi degrades to None/0 instead of crashing."""
    rng = np.random.default_rng(2)
    _t, hum = _tone(f=120.0, amp=0.04)
    rows = [_meanspec(hum + 0.01 * rng.standard_normal(len(hum)))[0]
            for _ in range(3)]
    _spec, freqs = _meanspec(hum)
    hfs = int(0.02 * FS)
    F = {"minspec": np.array(rows), "freqs": freqs,
         "env_hi": (_am(4.0) ** 2).reshape(-1, hfs).mean(1),
         "hi_dt": np.float64(0.02)}
    s = iso.summarize_psycho(F)
    assert s["tonal_prominence_hz"] == pytest.approx(120.0, abs=6.0)
    assert s["tonal_prominence_db"] > 10
    assert s["n_prominent_tones"] >= 1
    assert s["fluctuation_index"] > 0.5

    empty = iso.summarize_psycho({})
    assert empty == {"tonal_prominence_db": None,
                     "tonal_prominence_hz": None,
                     "n_prominent_tones": 0,
                     "fluctuation_index": None}


def test_full_summary_carries_psycho_keys(bell_features):
    from ambiscape.resolve import full_summary
    _sess, _out, F = bell_features
    s = full_summary(F)
    for key in ("tonal_prominence_db", "tonal_prominence_hz",
                "n_prominent_tones", "fluctuation_index"):
        assert key in s
    assert s["fluctuation_index"] is not None


# ----------------------------- optional dependency and segment bookkeeping

def test_indicators_without_mosqito_names_the_extra(monkeypatch):
    """A stock install must say which extra is missing, not raise a bare
    ModuleNotFoundError from inside the loudness call."""
    monkeypatch.setitem(sys.modules, "mosqito", None)
    monkeypatch.setitem(sys.modules, "mosqito.sq_metrics", None)
    assert iso.mosqito_available() is False
    with pytest.raises(ImportError, match=r"ambiscape\[iso\]"):
        iso.indicators(np.zeros(FS, np.float64), FS)


def _stub_indicators(monkeypatch):
    monkeypatch.setattr(iso, "indicators", lambda x_pa, fs, **kw: {
        "N5_sone": 1.0, "N50_sone": 1.0, "sharpness_median_acum": 1.0,
        "roughness_median_asper": 0.0, "fluctuation_strength_vacil": 0.0})


def _flat_features(sess, dur_s, dt=0.125):
    """Steady fast-level frames in session time, covering one take."""
    n = int(round(dur_s / dt))
    return {"t_fast": sess.takes[0].start + np.arange(n) * dt,
            "fast_db": np.full(n, -40.0)}


def test_segment_dur_reports_audio_delivered(tmp_path, monkeypatch):
    """read_span clamps at the end of a take; the JSON must record the
    audio delivered, not the length requested."""
    import ambiscape as asc
    from .conftest import diffuse_noise, write_bwf
    _stub_indicators(monkeypatch)
    write_bwf(tmp_path / "short.wav", diffuse_noise(5 * FS))
    sess = asc.open_session(tmp_path)
    res = iso.segment_indicators(sess, _flat_features(sess, 5.0), tmp_path,
                                 dur=30.0)
    seg = res["segments"]["whole"]
    assert seg["dur_s"] == pytest.approx(5.0, abs=0.01)
    assert seg["dur_requested_s"] == 30.0


def test_coincident_segments_computed_once(tmp_path, monkeypatch):
    """One-window session: quietest/most_active/typical land on the same
    audio, and the degeneracy is stated rather than repeated three times."""
    import ambiscape as asc
    from .conftest import diffuse_noise, write_bwf
    _stub_indicators(monkeypatch)
    write_bwf(tmp_path / "flat.wav", diffuse_noise(30 * FS))
    sess = asc.open_session(tmp_path)
    res = iso.segment_indicators(sess, _flat_features(sess, 30.0), tmp_path,
                                 dur=30.0)
    assert list(res["segments"]) == ["quietest"]
    seg = res["segments"]["quietest"]
    assert set(seg["also_kinds"]) == {"most_active", "typical"}
    assert "dur_requested_s" not in seg
