"""Session scanning, BWF parsing, clock offset, feature extraction."""
import json
import warnings

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


def test_insv_container_opens_via_transcode(tmp_path):
    """Insta360 .insv (an MP4 in disguise) is decoded on ingest."""
    import shutil
    import subprocess
    import pytest
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    import ambiscape as asc
    # a 2 s stereo AAC mp4, renamed to the Insta360 extension
    mp4 = tmp_path / "20260724_142731_00_037.insv"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=2",
         "-ac", "2", "-c:a", "aac", "-f", "mp4", str(mp4)],
        check=True)
    sess = asc.open_session(tmp_path)
    assert len(sess.takes) == 1
    tk = sess.takes[0]
    assert tk.clock == "14:27:31"          # from the filename stamp
    assert 1.5 < tk.duration < 2.5
    assert tk.channels == 2


def test_per_take_clock_offsets(tmp_path):
    n = 48000
    write_bwf(tmp_path / "a.wav", np.zeros((n, 4)), time="00:00:00")
    write_bwf(tmp_path / "b.wav", np.zeros((n, 4)), time="00:01:00")
    (tmp_path / "calibration.json").write_text(json.dumps({
        "clock_offset_s": 2.0,
        "clock_offsets_s": {"b.wav": -5.0},
    }))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sess = asc.open_session(tmp_path)
    takes = {t.path.name: t for t in sess.takes}
    assert takes["a.wav"].start == 2.0            # global only
    assert takes["b.wav"].start == 60.0 + 2.0 - 5.0   # global + per-take


def test_unmatched_clock_offset_key_warns(tmp_path):
    n = 48000
    write_bwf(tmp_path / "a.wav", np.zeros((n, 4)), time="00:00:00")
    (tmp_path / "calibration.json").write_text(json.dumps({
        "clock_offsets_s": {"typo.wav": -5.0},
    }))
    with pytest.warns(UserWarning, match="clock_offsets_s"):
        asc.open_session(tmp_path)


def test_best_audio_stream_selected(tmp_path):
    """A .360 with stereo AAC first and 4ch PCM s32 second ingests the
    4-channel ambisonic stream, cached at 24-bit."""
    import shutil
    import subprocess
    import pytest
    import soundfile as sf_info
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    import ambiscape as asc
    f = tmp_path / "20260724_120000_gopro.360"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-f", "lavfi", "-i", "anoisesrc=d=2:c=pink",
         "-filter_complex", "[1:a]pan=4.0|c0=c0|c1=c0|c2=c0|c3=c0[a4]",
         "-map", "0:a", "-map", "[a4]",
         "-c:a:0", "aac", "-c:a:1", "pcm_s32le",
         "-f", "mov", str(f)], check=True)
    sess = asc.open_session(tmp_path)
    tk = sess.takes[0]
    assert tk.channels == 4
    assert tk.mode == "ambix"
    assert sf_info.info(str(tk.audio_path)).subtype == "PCM_24"


def test_single_stream_ingest_unchanged(tmp_path):
    import shutil
    import subprocess
    import pytest
    import soundfile as sf_info
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    import ambiscape as asc
    f = tmp_path / "20260724_120000_phone.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=2", "-ac", "2", "-c:a", "aac",
         str(f)], check=True)
    sess = asc.open_session(tmp_path)
    tk = sess.takes[0]
    assert tk.channels == 2
    assert sf_info.info(str(tk.audio_path)).subtype == "PCM_16"


def test_stale_pre_selection_cache_not_reused(tmp_path):
    """A pre-0.19 cache (name = ``<stem>.wav``, always decoded from a:0 at
    16-bit) must not be mistaken for a fresh cache of a two-stream file
    whose best stream is a:1/24-bit -- even if the stale file is newer than
    the source, as an old cache always is once the source stops changing."""
    import os
    import shutil
    import subprocess
    import time
    import pytest
    import soundfile as sf_info
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    import ambiscape as asc
    f = tmp_path / "20260724_120000_gopro.360"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-f", "lavfi", "-i", "anoisesrc=d=2:c=pink",
         "-filter_complex", "[1:a]pan=4.0|c0=c0|c1=c0|c2=c0|c3=c0[a4]",
         "-map", "0:a", "-map", "[a4]",
         "-c:a:0", "aac", "-c:a:1", "pcm_s32le",
         "-f", "mov", str(f)], check=True)

    # Pre-seed a stale cache under the OLD naming convention: a mono WAV
    # (simulating a pre-0.19 a:0/16-bit decode), with an mtime newer than
    # the source -- exactly what the old freshness check would happily
    # reuse forever.
    cache_dir = tmp_path / ".ambiscape_decoded"
    cache_dir.mkdir()
    stale = cache_dir / "20260724_120000_gopro.wav"
    sf_info.write(str(stale), np.zeros(FS, dtype=np.float32), FS,
                  subtype="PCM_16")
    future = time.time() + 3600
    os.utime(stale, (future, future))

    sess = asc.open_session(tmp_path)
    tk = sess.takes[0]
    assert tk.channels == 4
    assert "a1s24" in tk.audio_path.name


def test_extract_drops_partial_last_second_fast_frames(tmp_path):
    """A take with a fractional last second must not end in full-scale
    (0 dB) fill frames: the fast/env streams stop at the last whole second."""
    from ambiscape import features
    rng = np.random.default_rng(1)
    n = int(3.4 * FS)                        # 3 whole seconds + 0.4 s tail
    write_bwf(tmp_path / "a.wav", plane_wave(0.05 * rng.standard_normal(n),
                                             0.0))
    sess = asc.open_session(tmp_path)
    F = features.extract_take(sess.takes[0])
    assert len(F["rms_w"]) == 3
    assert len(F["fast_db"]) == len(F["fast_dba"]) == 3 * 8
    assert len(F["env_hi"]) == 3 * 50
    assert float(F["fast_db"].max()) < -10   # no 0 dBFS fill at the tail
