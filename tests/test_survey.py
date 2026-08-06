"""Survey tests: ISO 12913-2 responses -> 12913-3 circumplex coordinates.

Synthetic response sets with known character (a calm-quiet courtyard, a
chaotic-loud interchange) must land in the right quadrants, and the
srv_ summary keys must join the catalog next to the acoustic descriptors.
"""
import csv
import json

import numpy as np
import pytest

from ambiscape import catalog, survey

HEADER = ["respondent", "pleasant", "chaotic", "vibrant", "uneventful",
          "calm", "annoying", "eventful", "monotonous"]

# scale ratings per archetype respondent (1..5 Likert)
CALM_QUIET = dict(pleasant=5, chaotic=1, vibrant=2, uneventful=4,
                  calm=5, annoying=1, eventful=1, monotonous=2)
CHAOTIC_LOUD = dict(pleasant=1, chaotic=5, vibrant=3, uneventful=1,
                    calm=1, annoying=5, eventful=5, monotonous=2)
NEUTRAL = dict.fromkeys(HEADER[1:], 3)


def _jitter(base, rng, lo=1, hi=5):
    """One respondent: the archetype nudged by ±1 within the scale."""
    return {k: int(np.clip(v + rng.integers(-1, 2), lo, hi))
            for k, v in base.items()}


def _write_csv(path, rows, header=HEADER, extras=None):
    cols = header + list(extras or [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"respondent": f"p{i + 1}", **r})
    return path


def _panel(base, n=8, seed=0, to100=False):
    rng = np.random.default_rng(seed)
    rows = [_jitter(base, rng) for _ in range(n)]
    if to100:
        rows = [{k: (v - 1) * 25 for k, v in r.items()} for r in rows]
    return rows


# ----------------------------------------------------------- coordinates

def test_neutral_respondent_is_origin():
    assert survey.coordinates(NEUTRAL) == (0.0, 0.0)


def test_pure_pleasant_known_value():
    """p=5, a=1, rest neutral: P = 4 / (4·(1+√2)), E = 0."""
    scales = dict(NEUTRAL, pleasant=5, annoying=1)
    pl, ev = survey.coordinates(scales)
    assert pl == pytest.approx(1 / (1 + np.sqrt(2)))
    assert ev == pytest.approx(0.0)


def test_extremes_reach_unit_corners():
    """All eight scales at their poles must reach |P| = |E| = 1."""
    best = dict(pleasant=5, calm=5, vibrant=5, annoying=1, chaotic=1,
                monotonous=1, eventful=3, uneventful=3)
    pl, _ = survey.coordinates(best)
    assert pl == pytest.approx(1.0)
    wild = dict(eventful=5, chaotic=5, vibrant=5, uneventful=1, calm=1,
                monotonous=1, pleasant=3, annoying=3)
    _, ev = survey.coordinates(wild)
    assert ev == pytest.approx(1.0)


def test_scale_range_100_matches_5_point():
    """The same response coded 1–5 and 0–100 gives the same point."""
    s5 = CALM_QUIET
    s100 = {k: (v - 1) * 25 for k, v in s5.items()}
    assert survey.coordinates(s5, 1, 5) == \
        pytest.approx(survey.coordinates(s100, 0, 100))


# ------------------------------------------------------------- ingestion

def test_missing_scale_column_raises(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("pleasant,chaotic\n5,1\n")
    with pytest.raises(ValueError, match="vibrant"):
        survey.read_responses(p)


def test_case_insensitive_headers_and_extras(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("ID,Pleasant,Chaotic,Vibrant,Uneventful,Calm,Annoying,"
                 "Eventful,Monotonous,loudness,comment\n"
                 "anna,5,1,2,4,5,1,1,2,30,quiet yard\n"
                 "berit,4,1,2,4,4,2,2,2,40,\n")
    r = survey.read_responses(p)
    assert r["scale"] == "5-point" and len(r["respondents"]) == 2
    assert r["respondents"][0]["id"] == "anna"
    assert r["respondents"][0]["extras"] == {"loudness": 30.0,
                                             "comment": "quiet yard"}
    doc = survey.summarize(r)
    assert doc["extras"]["loudness_mean"] == 35.0


def test_incomplete_row_skipped(tmp_path):
    p = _write_csv(tmp_path / "r.csv",
                   [CALM_QUIET, {**CALM_QUIET, "vibrant": ""}])
    r = survey.read_responses(p)
    assert len(r["respondents"]) == 1 and r["n_skipped"] == 1


def test_scale_autodetect_100_point(tmp_path):
    p = _write_csv(tmp_path / "r.csv", _panel(CALM_QUIET, to100=True))
    r = survey.read_responses(p)
    assert r["scale"] == "100-point" and r["range"] == (0.0, 100.0)


# ------------------------------------------------------------- quadrants

def test_calm_quiet_panel_lands_calm_quadrant(tmp_path):
    p = _write_csv(tmp_path / "r.csv", _panel(CALM_QUIET))
    doc = survey.summarize(survey.read_responses(p))
    assert doc["pleasantness_mean"] > 0.2
    assert doc["eventfulness_mean"] < -0.2
    assert doc["quadrant"] == "calm"


def test_chaotic_loud_panel_lands_chaotic_quadrant(tmp_path):
    p = _write_csv(tmp_path / "r.csv", _panel(CHAOTIC_LOUD, to100=True))
    doc = survey.summarize(survey.read_responses(p))
    assert doc["pleasantness_mean"] < -0.2
    assert doc["eventfulness_mean"] > 0.2
    assert doc["quadrant"] == "chaotic"


def test_ellipse_and_dispersion(tmp_path):
    p = _write_csv(tmp_path / "r.csv", _panel(CALM_QUIET, n=12))
    doc = survey.summarize(survey.read_responses(p))
    ell = doc["ellipse_95"]
    assert ell and ell["width"] >= ell["height"] > 0
    assert 0 < doc["dispersion"] < 0.5
    # two identical respondents: degenerate cloud, no ellipse, sd defined
    p2 = _write_csv(tmp_path / "r2.csv", [CALM_QUIET, CALM_QUIET])
    doc2 = survey.summarize(survey.read_responses(p2))
    assert doc2["ellipse_95"] is None and doc2["dispersion"] == 0.0


# ----------------------------------------------------- session integration

def test_run_survey_writes_outputs_and_merges_summary(tmp_path):
    sess = tmp_path / "2026-08-01-Oslo-courtyard"
    (sess / "analysis").mkdir(parents=True)
    (sess / "analysis" / "summary.json").write_text(json.dumps(
        {"laeq_dbfs": -38.2, "L90": -46.0, "events_per_min": 1.2}))
    p = _write_csv(tmp_path / "r.csv", _panel(CALM_QUIET))
    doc = survey.run_survey(sess, p)
    assert (sess / "analysis" / "survey.json").exists()
    assert (sess / "analysis" / "survey.png").stat().st_size > 0
    merged = json.loads((sess / "analysis" / "summary.json").read_text())
    assert merged["laeq_dbfs"] == -38.2            # acoustic keys survive
    assert merged["srv_n"] == 8
    assert merged["srv_pleasantness_mean"] == doc["pleasantness_mean"]
    assert merged["srv_quadrant"] == "calm"
    # perception-vs-measurement table pairs LAeq with pleasantness
    rows = doc["vs_measurement"]
    laeq = next(r for r in rows if r["measured"].startswith("LAeq"))
    assert laeq["value"] == -38.2
    assert laeq["perceived"] == "pleasantness"
    assert survey.vs_table(rows).startswith("| measured |")


def test_run_survey_without_summary_creates_one(tmp_path):
    sess = tmp_path / "sess"
    sess.mkdir()
    p = _write_csv(tmp_path / "r.csv", _panel(CHAOTIC_LOUD, n=5))
    doc = survey.run_survey(sess, p)
    assert "vs_measurement" not in doc             # nothing to compare against
    merged = json.loads((sess / "analysis" / "summary.json").read_text())
    assert merged["srv_quadrant"] == "chaotic"


def test_catalog_ranks_sessions_perceptually(tmp_path):
    for name, base in (("courtyard", CALM_QUIET),
                       ("interchange", CHAOTIC_LOUD)):
        sess = tmp_path / name
        sess.mkdir()
        survey.run_survey(sess, _write_csv(tmp_path / f"{name}.csv",
                                           _panel(base)))
    col = catalog.collect(tmp_path)
    ranked = catalog.rank(col, "srv_pleasantness_mean")
    assert [n for n, _ in ranked] == ["courtyard", "interchange"]


def test_cli_survey_command(tmp_path, capsys):
    from ambiscape.cli import main
    sess = tmp_path / "sess"
    sess.mkdir()
    p = _write_csv(tmp_path / "r.csv", _panel(CALM_QUIET))
    assert main(["survey", str(sess), "--responses", str(p)]) == 0
    out = capsys.readouterr().out
    assert "calm quadrant" in out and "survey.png" in out
