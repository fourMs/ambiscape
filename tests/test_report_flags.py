"""README honesty: soft-flagged descriptors and multi-regime sessions.

A soft low-confidence flag lives in summary.json but the README table is
what people read and quote — a flagged value printed bare there reads as
trustworthy. And a session whose features hold several regimes (a walk, a
machine day) needs its averages caveated where they are printed.
"""
from types import SimpleNamespace

from ambiscape import report


def _sess(tmp_path):
    return SimpleNamespace(folder=tmp_path, name="flags-test", takes=[])


def test_soft_flagged_rows_carry_a_marker_and_footnote(tmp_path):
    out = tmp_path / "analysis"
    out.mkdir()
    summary = {
        "duration_min": 1.3,
        "ndsi": -0.34,
        "fluctuation_index": 0.47,
        "low_confidence": [
            {"key": "ndsi", "needs_s": 360.0, "had_s": 78.0, "kind": "soft",
             "why": "did not converge across 5-360 s"},
        ],
    }
    report.write_readme(_sess(tmp_path), summary, out)
    txt = (tmp_path / "README.md").read_text()
    ndsi_row = next(ln for ln in txt.splitlines() if "NDSI" in ln)
    assert "†" in ndsi_row
    fluct_row = next(ln for ln in txt.splitlines() if "Fluctuation" in ln)
    assert "†" not in fluct_row
    assert "did not converge" in txt


def test_unflagged_summary_has_no_footnote(tmp_path):
    out = tmp_path / "analysis"
    out.mkdir()
    report.write_readme(_sess(tmp_path), {"duration_min": 60.0,
                                          "ndsi": -0.1}, out)
    txt = (tmp_path / "README.md").read_text()
    assert "†" not in txt


def test_nonstationary_session_gets_a_warning_line(tmp_path):
    out = tmp_path / "analysis"
    out.mkdir()
    summary = {
        "duration_min": 12.8,
        "nonstationarity": {
            "nonstationary": True, "n_regimes": 9,
            "boundaries_s": [1.0], "why": "the features hold 9 distinct "
            "regimes; session-level descriptors average over them — for a "
            "walking recording run 'ambiscape walk', for machine states "
            "run 'ambiscape draft'"},
    }
    report.write_readme(_sess(tmp_path), summary, out)
    txt = (tmp_path / "README.md").read_text()
    assert "9 distinct regimes" in txt
    assert "ambiscape walk" in txt


def test_stationary_session_has_no_regime_warning(tmp_path):
    out = tmp_path / "analysis"
    out.mkdir()
    summary = {"duration_min": 60.0,
               "nonstationarity": {"nonstationary": False, "n_regimes": 1}}
    report.write_readme(_sess(tmp_path), summary, out)
    assert "regimes" not in (tmp_path / "README.md").read_text()
