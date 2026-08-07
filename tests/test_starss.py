"""STARSS support tests: annotation reader, clip ingestion, DOA validation.

Synthetic FOA clips at the STARSS specification (24 kHz, plain WAV without
BWF timestamps) with plane-wave sources at known azimuths, plus matching or
deliberately mismatched annotation CSVs, give ground truth for the whole
``doavalidate`` path: a correct CSV must recover near-zero azimuth error, a
rotated CSV must show the rotation.
"""
import json

import numpy as np
import pytest
import soundfile as sf

from ambiscape import open_clips, starss
from ambiscape.cli import main

from .conftest import diffuse_noise, plane_wave

FS = 24000          # the STARSS sampling rate


# ----------------------------------------------------------- CSV reader

def test_read_annotations_six_columns(tmp_path):
    p = tmp_path / "clip.csv"
    p.write_text("0,8,0,60,-10,150\n1,8,0,61,-10,152\n")
    rows = starss.read_annotations(p)
    assert len(rows) == 2
    assert rows[0] == {"frame": 0, "class_id": 8, "class_name": "music",
                       "source": 0, "azimuth": 60.0, "elevation": -10.0,
                       "distance_cm": 150.0}


def test_read_annotations_five_columns_no_distance(tmp_path):
    p = tmp_path / "clip.csv"
    p.write_text("10,1,2,-120,5\n")
    (r,) = starss.read_annotations(p)
    assert r["class_name"] == "male speech"
    assert r["azimuth"] == -120.0
    assert r["distance_cm"] is None


def test_read_annotations_skips_blank_lines(tmp_path):
    p = tmp_path / "clip.csv"
    p.write_text("0,0,0,0,0\n\n1,0,0,0,0\n")
    assert len(starss.read_annotations(p)) == 2


def test_read_annotations_rejects_wrong_column_count(tmp_path):
    p = tmp_path / "clip.csv"
    p.write_text("0,8,0,60\n")
    with pytest.raises(ValueError, match="expected 5 or 6"):
        starss.read_annotations(p)


def test_read_annotations_rejects_non_numeric(tmp_path):
    p = tmp_path / "clip.csv"
    p.write_text("frame,class,source,azimuth,elevation\n")
    with pytest.raises(ValueError, match="non-numeric"):
        starss.read_annotations(p)


def test_single_source_frames_excludes_overlaps():
    rows = [{"frame": 0, "azimuth": 10.0}, {"frame": 1, "azimuth": 20.0},
            {"frame": 1, "azimuth": -50.0}, {"frame": 2, "azimuth": 30.0}]
    singles, n_multi = starss.single_source_frames(rows)
    assert sorted(singles) == [0, 2]
    assert n_multi == 1


def test_wrap_deg():
    assert starss.wrap_deg(190.0) == pytest.approx(-170.0)
    assert starss.wrap_deg(-190.0) == pytest.approx(170.0)
    assert starss.wrap_deg(180.0) == pytest.approx(180.0)
    assert starss.wrap_deg(0.0) == pytest.approx(0.0)


# ---------------------------------------------------------- clip fixtures

def _write_clip(path, az_deg, dur_s=8.0, seed=0):
    """A STARSS-style FOA WAV: one band-limited noise source at ``az_deg``
    over a weak diffuse bed, 24 kHz PCM16, no BWF chunk."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * FS)
    sig = 0.3 * rng.standard_normal(n)
    data = plane_wave(sig, az_deg) + diffuse_noise(n, level=0.01, seed=seed + 1)
    sf.write(str(path), np.clip(data, -1, 1), FS, subtype="PCM_16")
    return path


def _write_labels(path, az_deg, n_frames, class_id=8, el=0, dist=200):
    lines = [f"{k},{class_id},0,{int(az_deg)},{el},{dist}"
             for k in range(n_frames)]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(scope="module")
def starss_folder(tmp_path_factory):
    """Two clips at known azimuths (+60, -135) with matching CSVs, labelled
    music and footsteps respectively."""
    root = tmp_path_factory.mktemp("starss")
    clips, ann = root / "foa", root / "metadata"
    clips.mkdir(), ann.mkdir()
    _write_clip(clips / "fold9_room1_mix001.wav", 60.0, seed=0)
    _write_clip(clips / "fold9_room1_mix002.wav", -135.0, seed=7)
    _write_labels(ann / "fold9_room1_mix001.csv", 60.0, 80, class_id=8)
    _write_labels(ann / "fold9_room1_mix002.csv", -135.0, 80, class_id=6)
    return root


# ------------------------------------------------------------ open_clips

def test_open_clips_synthetic_clock(starss_folder):
    sess = open_clips(starss_folder / "foa")
    assert [t.path.name for t in sess.takes] == [
        "fold9_room1_mix001.wav", "fold9_room1_mix002.wav"]
    assert sess.day0.isoformat() == "1970-01-01"
    assert sess.takes[0].start == 0.0
    # chained end-to-end: no gap, no overlap, regardless of file mtimes
    assert sess.takes[1].start == pytest.approx(sess.takes[0].end)
    assert sess.takes[0].clock == "00:00:00"
    assert sess.takes[0].samplerate == FS
    assert sess.takes[0].mode == "ambix"


def test_open_clips_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_clips(tmp_path)


# ----------------------------------------------------------- validation

def test_matching_labels_recover_near_zero_error(starss_folder):
    doc = starss.validate_collection(starss_folder / "foa",
                                     starss_folder / "metadata")
    o = doc["overall"]
    assert o["n_frames"] == 160
    assert doc["n_multi_excluded"] == 0
    assert o["median_abs_deg"] < 3.0
    assert abs(o["bias_deg"]) < 3.0
    assert o["within_20deg"] > 0.95
    # per-class breakdown carries both labelled classes
    assert set(o["per_class"]) == {"music", "footsteps"}
    assert o["per_class"]["music"]["median_abs_deg"] < 3.0


def test_mismatched_labels_show_large_error(starss_folder, tmp_path):
    """The same audio against labels rotated by 90 degrees: the error must
    be the rotation, proving the comparison is not vacuously small."""
    ann = tmp_path / "wrong"
    ann.mkdir()
    _write_labels(ann / "fold9_room1_mix001.csv", 60.0 - 90.0, 80)
    _write_labels(ann / "fold9_room1_mix002.csv", -135.0 - 90.0, 80)
    doc = starss.validate_collection(starss_folder / "foa", ann)
    assert doc["overall"]["median_abs_deg"] > 60.0
    assert doc["overall"]["within_20deg"] < 0.05


def test_multi_source_frames_are_excluded(starss_folder, tmp_path):
    """Frames labelled with two simultaneous sources must not be scored."""
    ann = tmp_path / "multi"
    ann.mkdir()
    csv = ann / "fold9_room1_mix001.csv"
    lines = [f"{k},8,0,60,0,200" for k in range(40)]
    lines += [f"{k},0,1,-30,0,300" for k in range(20, 60)]   # overlap 20-39
    csv.write_text("\n".join(lines) + "\n")
    v = starss.validate_clip(
        starss_folder / "foa" / "fold9_room1_mix001.wav", csv)
    assert v["n_multi"] == 20
    scored = {r["frame"] for r in v["records"]}
    assert scored == set(range(20)) | set(range(40, 60))


def test_labels_beyond_audio_are_dropped(starss_folder, tmp_path):
    ann = tmp_path / "long"
    ann.mkdir()
    # 8 s clip = 80 frames; label 200 frames
    _write_labels(ann / "fold9_room1_mix001.csv", 60.0, 200)
    v = starss.validate_clip(
        starss_folder / "foa" / "fold9_room1_mix001.wav",
        ann / "fold9_room1_mix001.csv")
    assert len(v["records"]) == 80


def test_cli_doavalidate(starss_folder, capsys):
    out = starss_folder / "analysis"
    assert main(["doavalidate", str(starss_folder / "foa"),
                 "--annotations", str(starss_folder / "metadata"),
                 "-o", str(out)]) == 0
    doc = json.loads((out / "doavalidate.json").read_text())
    assert doc["overall"]["median_abs_deg"] < 3.0
    assert "records" not in doc              # stats only, not per-frame rows
    assert (out / "doavalidate.png").stat().st_size > 0
    assert "single-source frames" in capsys.readouterr().out


def test_cli_doavalidate_no_matching_csvs(starss_folder, tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    with pytest.warns(UserWarning, match="no annotation CSV"):
        assert main(["doavalidate", str(starss_folder / "foa"),
                     "--annotations", str(empty),
                     "-o", str(tmp_path / "a")]) == 1
