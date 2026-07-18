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
