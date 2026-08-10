"""What a descriptor is evidence about, and the guard that keeps it honest.

Four claims were withdrawn from this project in a month and every one was a
perceptual quantity read off a signal statistic. The registry exists so that
translation is written down rather than assumed; these tests exist so that it
stays written down as the toolbox grows.
"""
import pytest

from ambiscape import grounding as G
from ambiscape import timescales as T


def test_every_known_summary_key_has_a_tier():
    """The coverage guard. A new descriptor must be classified, not defaulted.

    Silently defaulting an unclassified descriptor to "signal only" would be
    the convenient answer and the wrong one -- it is exactly how a perceptual
    claim gets attached to a signal number without anyone deciding to.
    """
    known = set(T.WINDOWS) | set(T.EXEMPT)
    missing = sorted(known - set(G.GROUNDINGS) - G.EXEMPT)
    assert not missing, f"descriptors with no evidence tier: {missing}"


def test_registry_invents_no_descriptors():
    """Every registered key must be a descriptor the toolbox actually emits."""
    known = set(T.WINDOWS) | set(T.EXEMPT)
    invented = sorted(set(G.GROUNDINGS) - known)
    assert not invented, f"registered but never emitted: {invented}"


def test_tiers_are_valid():
    for g in G.GROUNDINGS.values():
        assert g.tier in G.TIERS
        assert g.why, f"{g.key} has no reason recorded"


def test_unknown_tier_is_refused():
    with pytest.raises(ValueError):
        G.Grounding("nonsense", "XX", "why")


def test_check_cautions_on_a_perceptually_defined_quantity():
    """A PD number in a summary must say so; that is the whole point."""
    out, cautions = G.check({"fg_fraction_median": 0.3, "leq_dbfs": -40.0})
    assert any(c.startswith("fg_fraction_median") for c in cautions)
    assert out["grounding_cautions"] == cautions


def test_check_is_quiet_on_signal_only_summaries():
    """No PD, no PM -> nothing to warn about. A warning that always fires is
    a warning nobody reads."""
    _, cautions = G.check({"leq_dbfs": -40.0, "centroid_median_hz": 500})
    assert cautions == []


def test_check_counts_pm_without_itemising_them():
    _, cautions = G.check({"n_events": 3, "emergence_db": 8.0})
    assert len(cautions) == 1 and "perceptually motivated" in cautions[0]


def test_unregistered_reports_an_unclassified_key():
    assert G.unregistered({"leq_dbfs": 1, "brand_new_thing": 2}) == [
        "brand_new_thing"]


def test_foreground_descriptors_are_perceptually_defined():
    """Figure and ground are a relation to a listener, not a signal property.
    This project retracted a claim for forgetting that."""
    for k in ("fg_fraction_median", "fg_fraction_p90", "fgbg_az_overlap",
              "azimuth_fg_deg"):
        assert G.tier_of(k) == "PD", k


def test_a_weighted_levels_are_calibrated_and_unweighted_ones_are_not():
    assert G.tier_of("laeq_dbfs") == "PC"
    assert G.tier_of("leq_dbfs") == "S"


def test_table_covers_the_registry():
    assert len(G.table()) == len(G.GROUNDINGS)
    assert sum(G.counts().values()) == len(G.GROUNDINGS)
