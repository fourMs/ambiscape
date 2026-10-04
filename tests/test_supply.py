"""Supply pickup against hum in the air, and the bearing of a narrow line,
on synthetic AmbiX with known ground truth."""
import numpy as np
import pytest
import soundfile as sf

from ambiscape import enf
from ambiscape.examples import diffuse_noise, plane_wave

FS = 8000


def _write(path, data):
    sf.write(path, data.astype(np.float32), FS, subtype="FLOAT")
    return path


def _hum(n, f0=50.0, level=0.02):
    t = np.arange(n) / FS
    return level * np.sin(2 * np.pi * f0 * t) + 0.5 * level * np.sin(2 * np.pi * 2 * f0 * t + 0.4)


@pytest.fixture
def n():
    return int(200 * FS)


def test_pickup_sits_in_w_alone(tmp_path, n):
    data = diffuse_noise(n, level=0.01, seed=1)
    data[:, 0] += _hum(n)                         # in phase on every capsule: W only
    sig = enf.supply_signature(_write(tmp_path / "p.wav", data), n_windows=3, win_s=40.0)
    f1 = sig["fundamental"]
    assert f1["prom_W_db"] > 15
    assert f1["line_minus_ring_db"] < -10
    assert sig["verdict"] == "pickup"


def test_hum_in_the_air_fills_xyz(tmp_path, n):
    data = diffuse_noise(n, level=0.01, seed=2) + plane_wave(_hum(n), 120.0)
    sig = enf.supply_signature(_write(tmp_path / "a.wav", data), n_windows=3, win_s=40.0)
    assert sig["fundamental"]["prom_W_db"] > 15
    assert sig["fundamental"]["line_minus_ring_db"] > -10
    assert sig["verdict"] == "acoustic"


def test_no_line_is_said_so(tmp_path, n):
    sig = enf.supply_signature(_write(tmp_path / "q.wav", diffuse_noise(n, 0.01, seed=3)),
                               n_windows=3, win_s=40.0)
    assert sig["fundamental"]["prom_W_db"] < enf.SUPPLY_MIN_PROM_DB
    assert sig["verdict"] == "no line"


def test_stereo_is_skipped(tmp_path, n):
    p = tmp_path / "s.wav"
    sf.write(p, np.zeros((FS, 2), np.float32), FS)
    assert enf.supply_signature(p)["verdict"] == "not ambix"


@pytest.mark.parametrize("az", [-150.0, 40.0, 170.0])
def test_line_bearing_recovers_azimuth(tmp_path, n, az):
    data = diffuse_noise(n, level=0.01, seed=4) + plane_wave(_hum(n, f0=100.0), az, el_deg=-15.0)
    b = enf.line_bearing(_write(tmp_path / "b.wav", data), 100.0, n_windows=3, win_s=40.0)
    assert ((b["az_deg"] - az + 180) % 360) - 180 == pytest.approx(0, abs=3)
    assert b["el_deg"] == pytest.approx(-15, abs=3)
    assert len(b["per_window_az_deg"]) == 3
