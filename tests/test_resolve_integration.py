"""Integration of state resolution into analyze/report/catalog."""
import json

import numpy as np
import pytest

from ambiscape import catalog, report, resolve


# --------------------------------------------------------------- auto_states

def test_auto_states_gates_single_state(bell_features):
    """A session with no clear machine on/off split yields no auto states."""
    _sess, _out, F = bell_features
    # flat band with no bimodal structure -> no split
    flat = dict(F)
    flat["logspec"] = np.ones_like(F["logspec"]) * 1e-6
    assert resolve.auto_states(flat) == {}


def test_auto_states_finds_two_states(bell_features):
    _sess, _out, F = bell_features
    st = resolve.auto_states(F, band=(1000, 4000), min_dur_s=30,
                             min_step_db=3.0)
    assert set(st) == {"machine_on", "machine_off"}


# ------------------------------------------------------------- report table

def test_state_table_markdown():
    states_doc = {
        "machine_on": {"duration_min": 533.7, "leq_dbfs": -47.5,
                       "diffuseness_median": 0.63, "ndsi": -0.38},
        "machine_off": {"duration_min": 168.0, "leq_dbfs": -55.1,
                        "diffuseness_median": 0.82, "ndsi": 0.22},
    }
    md = report.state_table(states_doc)
    assert "machine_on" in md and "machine_off" in md
    leq_row = next(l for l in md.splitlines() if "Leq" in l)
    assert "-47.5" in leq_row and "-55.1" in leq_row


def test_state_table_empty():
    assert report.state_table({}) == ""


# ----------------------------------------------------------- catalog states

def _corpus_with_states(tmp_path):
    for name, summ, states in (
        ("s1", {"leq_dbfs": -40.0, "ndsi": 0.1},
         {"on": {"leq_dbfs": -35.0, "ndsi": -0.2},
          "off": {"leq_dbfs": -55.0, "ndsi": 0.4}}),
        ("s2", {"leq_dbfs": -30.0, "ndsi": -0.3}, None),
    ):
        d = tmp_path / name / "analysis"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(summ))
        if states:
            (d / "states.json").write_text(json.dumps(
                {"states": {k: v for k, v in states.items()}}))
    return tmp_path


def test_collect_with_states_expands_rows(tmp_path):
    col = catalog.collect(_corpus_with_states(tmp_path), include_states=True)
    assert "s1" in col and "s1::on" in col and "s1::off" in col
    assert "s2" in col and not any(k.startswith("s2::") for k in col)
    assert col["s1::off"]["leq_dbfs"] == -55.0


def test_collect_without_states_default(tmp_path):
    col = catalog.collect(_corpus_with_states(tmp_path))
    assert set(col) == {"s1", "s2"}
