"""Background-bed rendering: foreground capped, floor untouched."""
import numpy as np
import pytest
import soundfile as sf

from ambiscape import render


def _bursty_wav(path, dur=90.0, fs=16000, seed=3):
    """Constant noise floor + six loud 1 kHz bursts (the 'foreground')."""
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal(int(dur * fs))
    t = np.arange(int(2.0 * fs)) / fs
    burst = 0.4 * np.sin(2 * np.pi * 1000 * t) * np.hanning(len(t))
    starts = [s for s in [10, 25, 40, 55, 70, 82]
              if (s + 2.5) * fs < len(x)]
    for s in starts:
        i = int(s * fs)
        x[i:i + len(burst)] += burst
    sf.write(path, x.astype(np.float32), fs)
    return starts


def _band_rms_db(x, fs, t0, t1, lo=900, hi=1100):
    from scipy import signal
    sos = signal.butter(4, [lo, hi], "band", fs=fs, output="sos")
    seg = signal.sosfilt(sos, x)[int(t0 * fs):int(t1 * fs)]
    return 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-12)


def test_background_bed_caps_bursts_keeps_floor(tmp_path):
    import ambiscape as asc
    starts = _bursty_wav(tmp_path / "20260724_120000_take.wav")
    sess = asc.open_session(tmp_path)
    out = tmp_path / "analysis"
    out.mkdir()
    doc = render.run_session(sess, out, margin_db=6.0)
    y, fs = sf.read(doc["out_path"])
    x, _ = sf.read(tmp_path / "20260724_120000_take.wav")
    assert len(y) == pytest.approx(len(x), abs=fs)      # same duration
    # bursts attenuated by at least 12 dB in their band
    for s in starts[:3]:
        before = _band_rms_db(x, fs, s + 0.5, s + 1.5)
        after = _band_rms_db(y, fs, s + 0.5, s + 1.5)
        assert before - after > 12.0
    # the quiet floor is essentially untouched (< 1.5 dB change)
    fb = _band_rms_db(x, fs, 3.0, 8.0, lo=200, hi=6000)
    fa = _band_rms_db(y, fs, 3.0, 8.0, lo=200, hi=6000)
    assert abs(fb - fa) < 1.5
    assert doc["exceed_fraction_before"] > doc["exceed_fraction_after"]


def test_background_bed_short_take(tmp_path):
    """Takes shorter than the minimum-statistics window still render."""
    import ambiscape as asc
    _bursty_wav(tmp_path / "20260724_120000_short.wav", dur=18.0)
    sess = asc.open_session(tmp_path)
    out = tmp_path / "analysis"
    out.mkdir()
    doc = render.run_session(sess, out, margin_db=6.0)
    y, fs = sf.read(doc["out_path"])
    assert len(y) > 15 * fs
