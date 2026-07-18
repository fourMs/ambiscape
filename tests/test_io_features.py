"""Session scanning, BWF parsing, clock offset, feature extraction."""
import json

import numpy as np
import pytest

import ambiscape as asc
from tests.conftest import FS, diffuse_noise, plane_wave, write_bwf


def test_open_session_parses_bwf(tmp_path):
    sig = 0.1 * np.sin(2 * np.pi * 440 * np.arange(2 * FS) / FS)
    write_bwf(tmp_path / "a.wav", plane_wave(sig, 0.0), time="12:34:56")
    sess = asc.open_session(tmp_path)
    tk = sess.takes[0]
    assert tk.start == 12 * 3600 + 34 * 60 + 56
    assert tk.channels == 4 and tk.samplerate == FS
    assert tk.order == "ambix" and tk.wyzx == (0, 1, 2, 3)
    assert "12:34" in sess.clock(tk.start)


def test_fuma_order_detected(tmp_path):
    sig = 0.1 * np.sin(2 * np.pi * 440 * np.arange(FS) / FS)
    write_bwf(tmp_path / "a.wav", plane_wave(sig, 0.0), order="fuma")
    tk = asc.open_session(tmp_path).takes[0]
    assert tk.order == "fuma" and tk.wyzx == (0, 2, 3, 1)


def test_clock_offset_applied(tmp_path):
    sig = 0.05 * np.ones(FS)
    write_bwf(tmp_path / "a.wav", plane_wave(sig, 0.0), time="10:00:00")
    (tmp_path / "calibration.json").write_text(
        json.dumps({"clock_offset_s": 90.0}))
    tk = asc.open_session(tmp_path).takes[0]
    assert tk.start == 10 * 3600 + 90


def test_extract_features_shapes_and_level(tmp_path):
    from ambiscape import features
    rng = np.random.default_rng(0)
    dur = 10
    sig = 0.1 * rng.standard_normal(dur * FS)
    write_bwf(tmp_path / "a.wav", plane_wave(sig, 45.0)
              + diffuse_noise(dur * FS, level=0.001))
    sess = asc.open_session(tmp_path)
    F0 = features.extract_take(sess.takes[0])
    assert len(F0["rms_w"]) == dur
    assert len(F0["fast_db"]) == dur * 8
    assert len(F0["env_hi"]) == dur * 50
    assert float(F0["hi_dt"]) == pytest.approx(0.02)
    # fast level of 0.1-RMS noise ~ -20 dBFS (16-bit quantized, loose)
    assert np.median(F0["fast_db"]) == pytest.approx(-20.0, abs=1.5)
    # broadband DOA of the plane wave
    paths = features.extract_session(sess, tmp_path / "feat", verbose=False)
    F = features.load_features(paths)
    assert np.median(F["az"]) == pytest.approx(45.0, abs=5.0)
    assert np.median(F["diffuse"]) < 0.4


def test_diffuse_field_scores_high(tmp_path):
    from ambiscape import features
    write_bwf(tmp_path / "a.wav", diffuse_noise(8 * FS, level=0.1))
    sess = asc.open_session(tmp_path)
    F = features.extract_take(sess.takes[0])
    assert np.median(F["diffuse"]) > 0.7


def test_open_recording_single_file(tmp_path):
    """A single WAV opens as its own one-take scene."""
    import numpy as np
    from ambiscape import open_recording
    from tests.conftest import write_bwf, plane_wave, FS
    x = plane_wave(0.1 * np.random.default_rng(0).standard_normal(3 * FS),
                   az_deg=45.0)
    write_bwf(tmp_path / "20240321_x_Oslo_Kitchen.WAV", x, date="2024-03-21",
              time="09:30:00")
    sess = open_recording(tmp_path / "20240321_x_Oslo_Kitchen.WAV")
    assert sess.name == "20240321_x_Oslo_Kitchen"
    assert len(sess.takes) == 1
    assert sess.day0.isoformat() == "2024-03-21"
    assert sess.clock(sess.takes[0].start).endswith("09:30:00")
