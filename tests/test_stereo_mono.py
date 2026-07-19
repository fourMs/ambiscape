"""Stereo and mono support: ingest, feature extraction, and a mode-aware
analyze summary.

Ground truth from synthetic panning: a source with more energy in the left
channel reads a positive (left) lateral azimuth; a coherent centred source
reads low 'diffuseness' (inter-channel coherence near 1) and a decorrelated
pair reads high; mono resolves no direction at all.
"""
import datetime as dt
import json

import numpy as np
import pytest
import soundfile as sf

from ambiscape import io, features, analysis
from ambiscape.spatial import summarize_spatial

FS = 48000


def _write(path, data, fs=FS):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, fs, subtype="PCM_16")
    return path


def _stereo(n, pan=0.0, decorrelate=False, seed=0):
    """A tone panned by `pan` in [-1, 1] (+ = left), plus noise. When
    `decorrelate`, left and right get independent noise (a diffuse field)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    gl, gr = np.sqrt((1 + pan) / 2), np.sqrt((1 - pan) / 2)
    if decorrelate:
        L = 0.2 * rng.standard_normal(n)
        R = 0.2 * rng.standard_normal(n)
    else:
        L = gl * tone + 0.01 * rng.standard_normal(n)
        R = gr * tone + 0.01 * rng.standard_normal(n)
    return np.stack([L, R], axis=1)


# ---------------------------------------------------------------- ingest

def test_channel_mode():
    assert io.channel_mode(1) == "mono"
    assert io.channel_mode(2) == "stereo"
    assert io.channel_mode(4) == "ambix"
    assert io.channel_mode(6) == "ambix"


def test_filename_timestamp_short(tmp_path):
    # YYMMDD_HHMMSS with no BWF chunk -> parsed from the name
    p = _write(tmp_path / "260719_082930.wav", _stereo(FS))
    sess = io.open_recording(p)
    tk = sess.takes[0]
    assert tk.date == "2026-07-19" and tk.clock == "08:29:30"
    assert tk.mode == "stereo" and sess.day0 == dt.date(2026, 7, 19)


def test_filename_timestamp_long(tmp_path):
    p = _write(tmp_path / "20260719_142530_cafe.wav", _stereo(FS))
    tk = io.open_recording(p).takes[0]
    assert tk.date == "2026-07-19" and tk.clock == "14:25:30"


def test_mtime_fallback(tmp_path):
    # no BWF, no timestamp in the name -> file mtime (date must still parse)
    p = _write(tmp_path / "voice-note.wav", _stereo(FS))
    tk = io.open_recording(p).takes[0]
    assert dt.date.fromisoformat(tk.date)


def test_transcode_m4a(tmp_path):
    pytest.importorskip("subprocess")
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    wavp = _write(tmp_path / "260719_090000.wav", _stereo(3 * FS))
    m4a = tmp_path / "260719_090000.m4a"
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wavp),
                    "-c:a", "aac", str(m4a)], check=True)
    wavp.unlink()                       # only the m4a remains
    sess = io.open_recording(m4a)
    tk = sess.takes[0]
    assert tk.mode == "stereo" and tk.channels == 2
    assert tk.audio_path.suffix == ".wav" and tk.audio_path.exists()
    F = features.extract_take(tk)
    assert np.isfinite(F["fast_db"]).all()


# ---------------------------------------------------------------- features

def test_stereo_lateral_azimuth(tmp_path):
    left = features.extract_take(
        io.open_recording(_write(tmp_path / "L.wav", _stereo(8 * FS, pan=0.6))).takes[0])
    right = features.extract_take(
        io.open_recording(_write(tmp_path / "R.wav", _stereo(8 * FS, pan=-0.6))).takes[0])
    assert np.nanmedian(left["az"]) > 15        # energy to the left -> +
    assert np.nanmedian(right["az"]) < -15      # energy to the right -> -
    assert np.isnan(left["el"]).all()           # no elevation from stereo
    assert str(left["mode"]) == "stereo"


def test_stereo_coherence_diffuseness(tmp_path):
    point = features.extract_take(
        io.open_recording(_write(tmp_path / "pt.wav",
                                 _stereo(8 * FS, pan=0.0))).takes[0])
    diffuse = features.extract_take(
        io.open_recording(_write(tmp_path / "df.wav",
                                 _stereo(8 * FS, decorrelate=True))).takes[0])
    assert np.nanmedian(point["diffuse"]) < 0.3      # coherent centred source
    assert np.nanmedian(diffuse["diffuse"]) > 0.7    # decorrelated field


def test_mono_has_no_direction(tmp_path):
    t = np.arange(6 * FS) / FS
    mono = (0.2 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
    F = features.extract_take(
        io.open_recording(_write(tmp_path / "260719_120000.wav", mono)).takes[0])
    assert np.isnan(F["az"]).all()
    assert np.isnan(F["el"]).all()
    assert np.isnan(F["diffuse"]).all()
    assert str(F["mode"]) == "mono"


# ---------------------------------------------------------------- summary

def _analyze_features(folder, data):
    sess = io.open_recording(_write(folder / "260719_120000.wav", data))
    paths = features.extract_session(sess, folder / "analysis" / "features",
                                     verbose=False)
    return features.load_features(paths)


def test_summary_stereo_is_lateral(tmp_path):
    F = _analyze_features(tmp_path / "s", _stereo(90 * FS, pan=0.5))
    s = analysis.summarize(F)
    s.update(summarize_spatial(F))
    assert s["azimuth_mean_deg"] is not None          # lateral az reported
    assert s["diffuseness_median"] is not None
    assert s["directional_entropy"] is not None
    assert s["elevation_fg_median_deg"] is None        # no elevation
    assert s["above_horizon_fraction"] is None
    json.dumps(s, allow_nan=False)                     # strictly valid JSON


def test_music_load_w_reads_mono_from_stereo(tmp_path):
    pytest.importorskip("librosa")
    from ambiscape import music
    p = _write(tmp_path / "260719_120000.wav", _stereo(6 * FS, pan=0.4))
    tk = io.open_recording(p).takes[0]
    y, sr = music.load_w(tk, t0=1.0, dur=3.0)      # mono ref, resampled
    assert y.ndim == 1 and sr == 22050 and np.isfinite(y).all()


def test_rhythm_runs_on_stereo(tmp_path):
    # the ambisonic intensity path is skipped for stereo; must not crash
    from ambiscape import rhythm, features
    p = _write(tmp_path / "260719_120000.wav", _stereo(90 * FS, pan=0.2))
    sess = io.open_recording(p)
    out = tmp_path / "analysis"
    features.extract_session(sess, out / "features", verbose=False)
    summary = rhythm.run_session(sess, out, verbose=False)
    assert isinstance(summary, dict) and "sources" in summary


def test_summary_mono_has_no_direction(tmp_path):
    t = np.arange(90 * FS) / FS
    mono = (0.2 * np.sin(2 * np.pi * 300 * t)
            + 0.03 * np.random.default_rng(0).standard_normal(90 * FS)
            ).reshape(-1, 1)
    F = _analyze_features(tmp_path / "m", mono)
    s = analysis.summarize(F)
    s.update(summarize_spatial(F))
    for k in ("azimuth_mean_deg", "diffuseness_median",
              "elevation_fg_median_deg", "directional_entropy"):
        assert s[k] is None, k
    json.dumps(s, allow_nan=False)
