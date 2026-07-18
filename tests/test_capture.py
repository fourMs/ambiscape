"""Capture-daemon tests with an injected synthetic source and clock, so the
orchestration is exercised without any audio hardware (PortAudio)."""
import datetime as dt
import json

import numpy as np
import soundfile as sf

from ambiscape import capture

from .conftest import FS, plane_wave


def _synth_source(seed=0, az=45.0):
    """A source that writes a plane-wave block to out_path (no device)."""
    def source(out_path, seconds, fs, channels):
        n = int(seconds * fs)
        sig = 0.1 * np.random.default_rng(seed).standard_normal(n)
        sf.write(out_path, plane_wave(sig, az), fs, subtype="PCM_24")
    return source


def _clock(times):
    it = iter(times)
    return lambda: next(it)


def test_daemon_writes_hourly_features_and_discards_audio(tmp_path):
    times = [dt.datetime(2024, 3, 21, 10, 0, 0),
             dt.datetime(2024, 3, 21, 10, 0, 6)]
    d = capture.CaptureDaemon(tmp_path, fs=FS, channels=4, block_seconds=6,
                              source=_synth_source(), now=_clock(times),
                              deposit=False)
    d.run(max_blocks=2)
    feat = tmp_path / "2024-03-21" / "features"
    assert len(sorted(feat.glob("*.npz"))) == 2
    assert not list(feat.glob("*.wav"))         # audio discarded


def test_daemon_keep_audio(tmp_path):
    times = [dt.datetime(2024, 3, 21, 10, 0, 0)]
    d = capture.CaptureDaemon(tmp_path, fs=FS, channels=4, block_seconds=6,
                              source=_synth_source(), now=_clock(times),
                              keep_audio=True, deposit=False)
    d.run(max_blocks=1)
    assert list((tmp_path / "2024-03-21" / "features").glob("*.wav"))


def test_daemon_day_rollover_writes_summary(tmp_path):
    times = [dt.datetime(2024, 3, 21, 23, 59, 52),
             dt.datetime(2024, 3, 22, 0, 0, 4),
             dt.datetime(2024, 3, 22, 0, 0, 10)]
    d = capture.CaptureDaemon(tmp_path, fs=FS, channels=4, block_seconds=6,
                              source=_synth_source(), now=_clock(times),
                              deposit=False)
    d.run(max_blocks=3)
    # first day rolled up once the second block crossed midnight
    s = json.loads((tmp_path / "2024-03-21" / "analysis"
                    / "summary.json").read_text())
    assert s["date"] == "2024-03-21" and "leq_dbfs" in s


def test_daemon_stop_rolls_up_current_day(tmp_path):
    times = [dt.datetime(2024, 3, 21, 8, 0, 0),
             dt.datetime(2024, 3, 21, 8, 0, 6)]
    d = capture.CaptureDaemon(tmp_path, fs=FS, channels=4, block_seconds=6,
                              source=_synth_source(), now=_clock(times),
                              deposit=False)
    d.run(max_blocks=2)
    d.finish()                                   # graceful shutdown rollup
    assert (tmp_path / "2024-03-21" / "analysis" / "summary.json").exists()


def test_daemon_survives_source_error(tmp_path):
    calls = {"n": 0}

    def flaky(out_path, seconds, fs, channels):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("device glitch")
        _synth_source()(out_path, seconds, fs, channels)

    times = [dt.datetime(2024, 3, 21, 9, 0, 0),
             dt.datetime(2024, 3, 21, 9, 0, 6)]
    d = capture.CaptureDaemon(tmp_path, fs=FS, channels=4, block_seconds=6,
                              source=flaky, now=_clock(times), deposit=False,
                              retry_wait_s=0)
    d.run(max_blocks=2)                           # first block errors, survives
    assert len(sorted((tmp_path / "2024-03-21" / "features").glob("*.npz"))) == 1


def test_aformat_to_bformat_identity():
    x = np.random.default_rng(1).standard_normal((100, 4))
    assert np.allclose(capture.aformat_to_bformat(x, np.eye(4)), x)


def test_aformat_to_bformat_shape():
    x = np.random.default_rng(2).standard_normal((100, 4))
    m = np.random.default_rng(3).standard_normal((4, 4))
    out = capture.aformat_to_bformat(x, m)
    assert out.shape == (100, 4)


def test_capture_available_is_bool():
    assert isinstance(capture.capture_available(), bool)
