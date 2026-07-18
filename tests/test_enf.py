"""ENF tests: recover a known off-nominal mains hum from synthetic audio."""
import numpy as np
import pytest

from ambiscape import enf
from ambiscape.io import open_session

from .conftest import FS, diffuse_noise, write_bwf


def _hum_session(tmp_path, f0=50.032, dur=185.0, level=0.02):
    """Synthetic session: diffuse noise + mains hum at f0 with a 2x harmonic."""
    n = int(dur * FS)
    t = np.arange(n) / FS
    hum = level * np.sin(2 * np.pi * f0 * t) \
        + 0.5 * level * np.sin(2 * np.pi * 2 * f0 * t + 0.7)
    data = diffuse_noise(n, level=0.01, seed=3)
    data[:, 0] += hum
    write_bwf(tmp_path / "hum.wav", data, time="10:00:00")
    return open_session(tmp_path)


def test_hum_peak_millihertz_accuracy():
    fs = 48000
    t = np.arange(60 * fs) / fs
    rng = np.random.default_rng(11)
    w = 0.02 * np.sin(2 * np.pi * 50.047 * t) + 0.01 * rng.standard_normal(len(t))
    f, rise = enf.hum_peak(w, fs, nominal=50.0, search_hz=0.2)
    assert f == pytest.approx(50.047, abs=0.003)
    assert rise > 10


def test_enf_track_and_summary(tmp_path):
    sess = _hum_session(tmp_path, f0=50.032)
    tr = enf.enf_track(sess, step_s=60.0, win_s=60.0, harmonics=(1, 2))
    assert len(tr["t"]) >= 2
    f1, f2 = tr["f"][1], tr["f"][2]           # per-harmonic, fundamental-scaled
    assert np.median(f1) == pytest.approx(50.032, abs=0.003)
    assert np.median(f2) == pytest.approx(50.032, abs=0.003)
    s = enf.enf_summary(tr)
    assert s["mean_hz"] == pytest.approx(50.032, abs=0.003)
    assert s["max_dev_mhz"] < 60
    assert s["harmonic_agreement_mhz"] < 5
    assert s["coverage"] == pytest.approx(1.0, abs=0.01)


def test_enf_track_skips_short_reads(tmp_path):
    sess = _hum_session(tmp_path, dur=65.0)
    tr = enf.enf_track(sess, step_s=60.0, win_s=60.0)
    assert len(tr["t"]) == 1                   # second window would be short
