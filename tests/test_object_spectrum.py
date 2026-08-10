"""An object's spectrum has a shape, and the shape is what a short clip has.

`object_profile` measures an envelope and says nothing about frequency. The
session-scale centroid and flux both need a minute of audio, which a sound
object of a second or two does not have. These are the same two quantities
defined on one object.
"""
import numpy as np

from ambiscape.objects import object_profile, object_spectrum

LOGF = np.geomspace(25.0, 16000.0, 97)
CENTRES = np.sqrt(LOGF[:-1] * LOGF[1:])
DT = 0.05


def _spec(rows):
    """dB spectrogram from a list of (centre_hz, width_oct, level_db)."""
    out = np.full((len(rows), len(CENTRES)), -90.0)
    for i, (f0, w, lvl) in enumerate(rows):
        d = np.abs(np.log2(CENTRES / f0))
        out[i] = lvl - 40.0 * np.clip(d - w, 0, None)
    return out


def test_a_held_tone_barely_moves():
    x = _spec([(1000.0, 0.1, 0.0)] * 20)
    r = object_spectrum(x, LOGF, DT)
    assert 800 < r["brightness_hz"] < 1250
    assert abs(r["brightness_drift_oct"]) < 0.05
    assert r["flux_per_s"] < 0.5


def test_a_struck_body_grows_duller_as_it_decays():
    """Bright at the strike, dull in the tail: drift is negative."""
    rows = [(4000.0, 0.4, 0.0), (3000.0, 0.4, -6.0), (1800.0, 0.4, -12.0)]
    rows += [(900.0, 0.4, -18.0 - 2 * i) for i in range(12)]
    r = object_spectrum(_spec(rows), LOGF, DT)
    assert r["brightness_drift_oct"] < -1.0


def test_a_kettle_grows_brighter():
    rows = [(400.0, 0.4, -6.0)] + [(400.0 * 2 ** (i / 8), 0.4, -6.0)
                                   for i in range(1, 16)]
    r = object_spectrum(_spec(rows), LOGF, DT)
    assert r["brightness_drift_oct"] > 0.5


def test_a_churning_object_has_more_flux_than_a_steady_one():
    rng = np.random.default_rng(0)
    steady = _spec([(1000.0, 0.3, 0.0)] * 24)
    churn = _spec([(float(rng.uniform(300, 6000)), 0.3, 0.0) for _ in range(24)])
    assert (object_spectrum(churn, LOGF, DT)["flux_per_s"]
            > 5 * object_spectrum(steady, LOGF, DT)["flux_per_s"])


def test_flux_does_not_depend_on_the_hop():
    """Reported per second, so halving dt must not double it."""
    x = _spec([(1000.0, 0.3, 0.0), (2000.0, 0.3, 0.0)] * 12)
    a = object_spectrum(x, LOGF, 0.05)["flux_per_s"]
    b = object_spectrum(x, LOGF, 0.10)["flux_per_s"]
    assert abs(a - 2 * b) < 0.01          # same churn, half the frame rate
                                          # (0.001 apart, which is the rounding)


def test_a_silent_or_one_frame_object_returns_nothing():
    assert object_spectrum(np.full((1, 96), -90.0), LOGF, DT) == {}
    assert object_spectrum(np.full((8, 96), -300.0), LOGF, DT) == {}


def test_object_profile_merges_the_spectrum_when_given_one():
    env = np.concatenate([np.linspace(0, 1, 4), np.linspace(1, 0, 16)])
    plain = object_profile(env, DT)
    withspec = object_profile(env, DT, logspec=_spec([(1000.0, 0.2, 0.0)] * 20),
                              logf=LOGF)
    assert "brightness_hz" not in plain
    assert withspec["brightness_hz"] > 0
    assert withspec["duration_s"] == plain["duration_s"]


# ----------------------------------------------- a folder of clips at once

def test_profile_clips_reads_a_folder_and_flags_short_spectrograms(tmp_path):
    """The case the session descriptors cannot serve.

    A clip corpus returns almost nothing from `analyze`, because the session
    summary's descriptors need a minute. Each clip here is one object.
    """
    import soundfile as sf
    from ambiscape.objects import profile_clips
    fs = 48000
    rng = np.random.default_rng(0)
    for name, dur, f0 in (("a_tone.wav", 3.0, 700.0),
                          ("b_noise.wav", 5.0, None)):
        t = np.arange(int(dur * fs)) / fs
        x = (np.sin(2 * np.pi * f0 * t) if f0 else rng.normal(0, 0.2, len(t)))
        x *= np.exp(-t / (dur / 3))                 # give each an envelope
        sf.write(tmp_path / name, x.astype(np.float32), fs)

    rows = profile_clips(tmp_path)
    assert [r["clip"] for r in rows] == ["a_tone.wav", "b_noise.wav"]
    for r in rows:
        assert r["duration_s"] > 0 and r["crest_db"] > 0
        assert "brightness_hz" in r and "n_frames" in r
    # the tone sits far below the noise in brightness
    assert rows[0]["brightness_hz"] < rows[1]["brightness_hz"]


def test_one_damaged_clip_does_not_cost_the_others(tmp_path):
    """A corpus folder is a batch. One bad member is named, not fatal."""
    import soundfile as sf
    from ambiscape.objects import profile_clips
    fs = 48000
    t = np.arange(2 * fs) / fs
    sf.write(tmp_path / "a_good.wav",
             (np.sin(2 * np.pi * 500 * t) * np.exp(-t)).astype(np.float32), fs)
    (tmp_path / "b_broken.wav").write_bytes(b"not a wav")
    rows = profile_clips(tmp_path)
    assert [r["clip"] for r in rows] == ["a_good.wav"]


def test_a_folder_with_nothing_readable_raises(tmp_path):
    import pytest
    from ambiscape.objects import profile_clips
    (tmp_path / "broken.wav").write_bytes(b"not a wav")
    with pytest.raises(FileNotFoundError):
        profile_clips(tmp_path)
