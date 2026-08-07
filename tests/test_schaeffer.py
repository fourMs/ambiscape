"""draft proposes Schaeffer mass/facture from features; long sessions with
many steady regimes must survive the draft → taxonomy path."""
import itertools
import json

import numpy as np

from ambiscape.draft import _labels, draft_annotations, schaeffer_hint


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


def test_regime_labels_unbounded():
    labs = list(itertools.islice(_labels(), 80))
    assert labs[:3] == ["A", "B", "C"]
    assert labs[25] == "Z" and labs[26] == "AA" and labs[27] == "AB"
    assert len(set(labs)) == 80  # no repeats


def _many_regime_F(n_regimes=20, regime_s=300.0, dt=0.5):
    """Synthetic features: level alternates every regime_s seconds, so the
    change-point detector proposes n_regimes steady-state keynotes."""
    tf = np.arange(0.0, n_regimes * regime_s, dt)
    fast = np.where((tf // regime_s) % 2 == 0, -60.0, -40.0)
    t = np.arange(0.0, n_regimes * regime_s, 1.0)
    n = len(t)
    return {"t_fast": tf, "fast_db": fast, "t": t,
            "flatness": np.full(n, 0.3), "rms_w": np.full(n, 0.05),
            "az": np.zeros(n), "el": np.zeros(n),
            "diffuse": np.full(n, 0.5)}


def test_draft_survives_more_than_16_regimes(tmp_path):
    # regression: a 16-letter label iterator raised StopIteration here
    out = draft_annotations(_many_regime_F(), tmp_path)
    doc = json.loads(out.read_text())
    keynotes = [o for o in doc["objects"] if o["kind"] == "keynote"]
    assert len(keynotes) > 16
    names = [o["name"] for o in keynotes]
    assert len(set(names)) == len(names)  # every regime uniquely labelled
    # and the taxonomy figures render from the oversized draft
    from ambiscape.taxonomy import render
    doc["states"] = []  # the human-edit step drops the TODO template
    (tmp_path / "annotations.json").write_text(json.dumps(doc))
    smap, timeline = render(tmp_path)
    assert smap.exists() and timeline.exists()
