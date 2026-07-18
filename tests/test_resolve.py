"""State-resolution tests on the synthetic bell session (bells active
0–240 s, then quiet to 300 s) and on synthetic multi-axis features."""
import numpy as np
import pytest

from ambiscape import resolve


def test_intervals_from_mask():
    t = np.arange(100, dtype=float)
    mask = np.zeros(100, bool)
    mask[10:20] = True
    mask[40:45] = True
    iv = resolve.intervals_from_mask(t, mask)
    assert iv == [(10.0, 20.0), (40.0, 45.0)]


def test_slice_features_all_axes(bell_features):
    _sess, _out, F = bell_features
    t0 = float(F["t"][0])
    sub = resolve.slice_features(F, [(t0, t0 + 120.0)])
    assert len(sub["t"]) == pytest.approx(120, abs=2)
    assert sub["t_fast"].max() < t0 + 121
    assert sub["logspec"].shape[0] == len(sub["t"])
    assert sub["logspec"].shape[1] == F["logspec"].shape[1]
    assert np.array_equal(sub["freqs"], F["freqs"])       # scalars preserved
    assert sub["minspec"].shape[0] <= 3                    # ~2 minutes


def test_resolve_states_separates_active_quiet(bell_features):
    _sess, _out, F = bell_features
    t0 = float(F["t"][0])
    states = {"active": [(t0, t0 + 235.0)], "quiet": [(t0 + 245.0, t0 + 300.0)]}
    res = resolve.resolve(F, states)
    assert set(res) == {"active", "quiet"}
    assert res["active"]["leq_dbfs"] > res["quiet"]["leq_dbfs"] + 3
    assert res["active"]["duration_min"] == pytest.approx(3.9, abs=0.2)
    # both carry the full descriptor set
    for k in ("diffuseness_median", "ndsi", "directional_entropy",
              "bird_band_activity_pct"):
        assert k in res["active"] and k in res["quiet"]


def test_machine_states_auto_discovers_on_off(bell_features):
    _sess, _out, F = bell_features
    st = resolve.machine_states(F, band=(1000, 4000), min_dur_s=30)
    assert "machine_on" in st
    on_dur = sum(b - a for a, b in st["machine_on"])
    assert on_dur == pytest.approx(240, abs=40)


def test_full_summary_matches_plain_summary(bell_features):
    _sess, _out, F = bell_features
    from ambiscape import analysis
    fs = resolve.full_summary(F)
    plain = analysis.summarize(F)
    assert fs["leq_dbfs"] == plain["leq_dbfs"]
    assert fs["n_events"] == plain["n_events"]
