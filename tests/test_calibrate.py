"""Level calibration: derive dbfs->dbspl offsets from a field SPL reading."""
import json

import numpy as np
import soundfile as sf

from ambiscape import iso


def _session(tmp_path, fs=16000, dur=120):
    """1 kHz sine at -23 dBFS RMS for 60 s, then 20 dB louder for 60 s."""
    import ambiscape as asc
    from ambiscape import features as afeat
    t = np.arange(dur * fs) / fs
    x = 0.1 * np.sin(2 * np.pi * 1000 * t)          # RMS ~ -23 dBFS
    x[60 * fs:] *= 10.0                              # +20 dB second half
    sf.write(tmp_path / "20260724_120000_take.wav",
             np.clip(x, -1, 1).astype(np.float32), fs)
    sess = asc.open_session(tmp_path)
    F = afeat.load_features(afeat.extract_session(
        sess, tmp_path / "analysis" / "features", verbose=False))
    return sess, F


def test_derive_offset_from_spl_reading(tmp_path):
    """A 1 kHz tone at -23 dBFS said to measure 60 dB(A) -> offset ~ 83."""
    _, F = _session(tmp_path)
    doc = iso.derive_offset(F, laeq_spl=60.0, t0=5.0, dur=50.0)
    assert abs(doc["dbfs_to_dbspl"] - 83.0) < 1.5
    assert abs(doc["laeq_dbfs"] - (-23.0)) < 1.5


def test_derive_offset_uses_span(tmp_path):
    """The same SPL claim over the loud half gives a ~20 dB smaller offset."""
    _, F = _session(tmp_path)
    quiet = iso.derive_offset(F, laeq_spl=60.0, t0=5.0, dur=50.0)
    loud = iso.derive_offset(F, laeq_spl=60.0, t0=65.0, dur=50.0)
    assert abs((quiet["dbfs_to_dbspl"] - loud["dbfs_to_dbspl"]) - 20.0) < 2.0


def test_write_calibration_merges(tmp_path):
    """Writing preserves unrelated keys and supports per-take offsets."""
    (tmp_path / "calibration.json").write_text(
        json.dumps({"clock_offset_s": 3.0}))
    iso.write_calibration(tmp_path, 83.2, method="SPL app on bench")
    iso.write_calibration(tmp_path, 77.0, take="phone.m4a")
    cal = json.loads((tmp_path / "calibration.json").read_text())
    assert cal["clock_offset_s"] == 3.0              # untouched
    assert cal["dbfs_to_dbspl"] == 83.2
    assert cal["method"] == "SPL app on bench"
    assert cal["dbfs_to_dbspl_takes"]["phone.m4a"] == 77.0


def test_take_offset_resolution():
    cal = {"dbfs_to_dbspl": 83.2,
           "dbfs_to_dbspl_takes": {"phone.m4a": 77.0}}
    assert iso.take_offset(cal, "phone.m4a") == 77.0
    assert iso.take_offset(cal, "zoom.WAV") == 83.2
    assert iso.take_offset({}, "zoom.WAV") is None
