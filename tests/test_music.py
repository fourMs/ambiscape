"""Librosa-backed tempogram/chromagram (skipped without the extra)."""
import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from ambiscape import music
from tests.conftest import FS, plane_wave, write_bwf


@pytest.fixture(scope="module")
def click_session(tmp_path_factory):
    """40 s of A4 clicks at 120 BPM from the front."""
    folder = tmp_path_factory.mktemp("clicks")
    dur = 40.0
    t = np.arange(int(dur * FS)) / FS
    env = np.zeros(len(t))
    for s in np.arange(0.0, dur, 0.5):                 # 120 BPM
        i = int(s * FS)
        seg = np.arange(min(len(t) - i, int(0.1 * FS))) / FS
        env[i:i + len(seg)] = np.exp(-seg / 0.03)
    sig = 0.3 * env * np.sin(2 * np.pi * 440.0 * t)
    write_bwf(folder / "clicks.wav", plane_wave(sig, 0.0))
    return folder


def test_music_tempo_and_chroma(click_session):
    import ambiscape as asc
    sess = asc.open_session(click_session)
    out = click_session / "analysis"
    out.mkdir(exist_ok=True)
    doc = music.run_session(sess, out)
    assert doc["tempo_bpm_global"] == pytest.approx(120, rel=0.06)
    assert doc["top_pitch_classes"][0] == "A"
    assert (out / "music.png").exists()


def _plucks(dur, spacing, jitter=0.0, freq=440.0, sr=22050, seed=0):
    """Exponentially decaying sine plucks every ``spacing`` seconds."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros(len(t))
    for s in np.arange(0.2, dur - 0.3, spacing):
        s += rng.uniform(-jitter, jitter)
        i = int(s * sr)
        seg = np.arange(min(len(t) - i, int(0.25 * sr))) / sr
        y[i:i + len(seg)] += np.exp(-seg / 0.05) * np.sin(2 * np.pi * freq * seg)
    return y, sr


def test_pulse_clarity_metric_vs_rubato():
    y, sr = _plucks(30.0, 0.5)                        # metronomic 120 BPM
    steady = music.pulse_clarity(y, sr)
    assert steady["period_bpm"] == pytest.approx(120, rel=0.08)
    y2, sr = _plucks(30.0, 0.5, jitter=0.16, seed=3)  # heavy rubato
    loose = music.pulse_clarity(y2, sr)
    assert steady["R"] > 0.6
    assert loose["R"] < steady["R"] / 2


def test_fifths_center_triad_vs_uniform():
    triad = np.zeros(12)
    triad[[0, 4, 7]] = [1.0, 0.7, 0.9]                # C-E-G
    doc = music.fifths_center(triad)
    assert doc["R"] > 0.5
    assert doc["center_note"] in ("C", "G")
    flatc = music.fifths_center(np.ones(12))
    assert flatc["R"] < 0.05


def test_tonal_center_spread_suite_vs_wandering():
    def key_chroma(root):
        c = np.zeros(12)
        c[[root % 12, (root + 4) % 12, (root + 7) % 12]] = 1.0
        return c
    suite = music.tonal_center_spread([key_chroma(0), key_chroma(7),
                                       key_chroma(5)])      # C, G, F
    apart = music.tonal_center_spread([key_chroma(0), key_chroma(6)])  # C, F#
    assert suite["R"] > 0.8
    assert apart["R"] < 0.2


def test_tartyp_profile_plucks_vs_drone():
    y, sr = _plucks(20.0, 0.25)                       # short tonic objects
    plucked = music.tartyp_profile(y, sr)
    assert plucked["n_objects"] > 20
    n_family = sum(v for k, v in plucked["dist"].items() if k[0] == "N")
    assert n_family > 0.7
    assert plucked["dist"].get("N'", 0) > 0.3         # impulses dominate

    t = np.arange(int(20.0 * sr)) / sr                # one held tonic object
    drone = music.tartyp_profile(0.3 * np.sin(2 * np.pi * 220 * t), sr)
    held = sum(v for k, v in drone["dist"].items() if k in ("N", "X", "Y"))
    assert held > 0.8
