"""Carillon bell-template MIR (skipped without the [music] extra)."""
import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from ambiscape import carillon
from tests.conftest import FS, plane_wave, write_bwf


def _bell(f0, t, decay=0.9):
    """A struck-bell note: the five principal partials with an exp decay."""
    y = np.zeros_like(t)
    for _, ratio, w in carillon.BELL_PARTIALS:
        y += w * np.sin(2 * np.pi * f0 * ratio * t)
    return y * np.exp(-t / decay)


@pytest.fixture(scope="module")
def carillon_session(tmp_path_factory):
    """40 s of two bells struck alternately: A4 (440) and C#5 (554.4)."""
    folder = tmp_path_factory.mktemp("carillon")
    dur, sig = 40.0, None
    t = np.arange(int(dur * FS)) / FS
    out = np.zeros(len(t))
    note_len = int(1.4 * FS)
    for k, s in enumerate(np.arange(0.0, dur - 1.5, 0.7)):
        i = int(s * FS)
        seg = t[:note_len]
        f0 = 440.0 if k % 2 == 0 else 554.365
        out[i:i + note_len] += 0.2 * _bell(f0, seg)
    write_bwf(folder / "carillon.wav", plane_wave(out, 30.0))
    return folder


def test_note_name():
    assert carillon.note_name(440.0)[0] == "A4"
    assert carillon.note_name(261.63)[0] == "C4"
    assert abs(carillon.note_name(440.0)[2]) < 1.0


def test_detects_struck_bells_and_suppresses_ghosts(carillon_session):
    import ambiscape as asc
    sess = asc.open_session(carillon_session)
    out = carillon_session / "analysis"
    out.mkdir(exist_ok=True)
    doc = carillon.run_session(sess, out, fmin=220.0, n_octaves=4)
    notes = {b["note"] for b in doc["bells"]}
    # both played bells are found ...
    assert "A4" in notes and "C#5" in notes
    # ... and the octave-down hum ghosts (A3, C#4) are suppressed
    assert "A3" not in notes and "C#4" not in notes
    assert (out / "carillon.png").exists()
    assert doc["range"]["semitones"] >= 4
