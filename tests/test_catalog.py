"""Catalog tests: aggregate per-session summary.json across a corpus."""
import json

import pytest

from ambiscape import catalog


def _corpus(tmp_path, sessions):
    for name, summary in sessions.items():
        d = tmp_path / name / "analysis"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(summary))
    return tmp_path


SESS = {
    "a-room": {"duration_min": 30.0, "leq_dbfs": -30.0, "diffuseness_median": 0.2,
               "ndsi": -0.5, "azimuth_R": 0.99},
    "b-field": {"duration_min": 11.0, "leq_dbfs": -40.0, "diffuseness_median": 0.9,
                "ndsi": 0.7, "azimuth_R": 0.4},
    "c-hall": {"duration_min": 12.0, "leq_dbfs": -28.0, "diffuseness_median": 0.85,
               "azimuth_R": 0.92},          # no ndsi key
}


def test_collect_finds_all_sessions(tmp_path):
    col = catalog.collect(_corpus(tmp_path, SESS))
    assert set(col) == {"a-room", "b-field", "c-hall"}
    assert col["a-room"]["leq_dbfs"] == -30.0


def test_to_csv_union_of_keys_with_blanks(tmp_path):
    col = catalog.collect(_corpus(tmp_path, SESS))
    out = catalog.to_csv(col, tmp_path / "catalog.csv")
    lines = out.read_text().splitlines()
    header = lines[0].split(",")
    assert header[0] == "session"
    assert "ndsi" in header and "diffuseness_median" in header
    ci = header.index("ndsi")
    row_c = next(r for r in lines[1:] if r.startswith("c-hall"))
    assert row_c.split(",")[ci] == ""          # missing key -> blank


def test_to_markdown_transposed(tmp_path):
    col = catalog.collect(_corpus(tmp_path, SESS))
    md = catalog.to_markdown(col, keys=["duration_min", "diffuseness_median"])
    assert "| Descriptor |" in md
    assert "a-room" in md and "b-field" in md
    # one row per requested key, sessions as columns
    dur_row = next(l for l in md.splitlines() if l.startswith("| duration_min"))
    assert "30.0" in dur_row and "11.0" in dur_row


def test_rank_and_outliers(tmp_path):
    col = catalog.collect(_corpus(tmp_path, SESS))
    ranked = catalog.rank(col, "diffuseness_median")
    assert ranked[0][0] == "b-field" and ranked[-1][0] == "a-room"
    z = catalog.outliers(col, "azimuth_R", z=0.8)
    assert z[0][0] == "b-field" and z[0][1] < 0   # R=0.40 is the low outlier


def test_collect_empty_dir(tmp_path):
    assert catalog.collect(tmp_path) == {}
