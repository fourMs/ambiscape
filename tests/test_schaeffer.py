"""draft.schaeffer_hint proposes Schaeffer mass/facture from features."""
import numpy as np

from ambiscape.draft import schaeffer_hint


def _F(n, flatness, total=None):
    t = np.arange(total or n, dtype=float)
    return {"t": t, "flatness": np.full(len(t), flatness),
            "rms_w": np.full(len(t), 0.1)}


def test_mass_from_flatness():
    assert schaeffer_hint(_F(60, 0.02), 0, 59)["mass"] == "tonic"
    assert schaeffer_hint(_F(60, 0.10), 0, 59)["mass"] == "tonic-complex"
    assert schaeffer_hint(_F(60, 0.35), 0, 59)["mass"] == "complex"
    assert schaeffer_hint(_F(60, 0.70), 0, 59)["mass"] == "noise"


def test_facture_from_continuity():
    # a ground filling the whole session -> unlimited
    assert schaeffer_hint(_F(60, 0.1), 0, 59)["facture"] == "unlimited"
    # a short steady regime inside a long session -> sustained
    assert schaeffer_hint(_F(30, 0.1, total=300), 0, 30)["facture"] == "sustained"


def test_evidence_surfaced():
    h = schaeffer_hint(_F(60, 0.1), 0, 59)
    assert "_schaeffer" in h and "flatness" in h["_schaeffer"]
    assert h["_schaeffer"]["dynamic"] in ("varied", "unvaried")
