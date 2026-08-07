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


def test_draft_clusters_regimes_into_beds(tmp_path):
    # 20 alternating regimes at two levels must draft as two keynote beds
    # (all spans kept), not one object per regime
    from ambiscape.taxonomy import MAX_BED_LANES, render
    out = draft_annotations(_many_regime_F(), tmp_path)
    doc = json.loads(out.read_text())
    keynotes = [o for o in doc["objects"] if o["kind"] == "keynote"]
    assert 1 <= len(keynotes) <= MAX_BED_LANES + 1
    assert sum(len(o["spans"]) for o in keynotes) > 16  # every regime kept
    names = [o["name"] for o in keynotes]
    assert len(set(names)) == len(names)
    assert all("bed" in n and "spans" in n for n in names)
    # no per-object boilerplate label
    assert not any(str(o.get("label", "")).startswith("AUTO")
                   for o in doc["objects"])
    # and the taxonomy figures render from the draft
    doc["states"] = []  # the human-edit step drops the TODO template
    (tmp_path / "annotations.json").write_text(json.dumps(doc))
    smap, timeline = render(tmp_path)
    assert smap.exists() and timeline.exists()


def _auto_objects(n=65, seed=7):
    """Synthetic machine-drafted regimes shaped like a real SINS draft."""
    rng = np.random.default_rng(seed)
    labels = itertools.islice(_labels(), n)
    objs = []
    for i, lab in enumerate(labels):
        lvl = float(rng.uniform(-62, -28))
        objs.append({
            "name": f"steady state {lab} ({lvl:.0f} dBFS)",
            "kind": "keynote", "mass": "complex", "facture": "sustained",
            "label": "AUTO — mass/facture proposed from features; "
                     f"listen to confirm (median {lvl:.0f} dBFS)",
            "spans": [[i * 300.0, i * 300.0 + 280.0]],
        })
    return objs


def test_merge_keynote_beds_bounded():
    from ambiscape.taxonomy import MAX_BED_LANES, merge_keynote_beds
    objs = _auto_objects()
    merged = merge_keynote_beds(objs)
    keynotes = [o for o in merged if o["kind"] == "keynote"]
    assert len(keynotes) <= MAX_BED_LANES + 1          # beds + "other"
    assert sum(len(o["spans"]) for o in keynotes) == 65  # no span lost
    assert all("spans" in o["name"] for o in keynotes)   # count in the label
    # beds respect the ~6 dB banding rule
    import re
    for o in keynotes:
        m = re.match(r".*?(-\d+) to (-\d+) dBFS", o["name"])
        if m:
            assert float(m.group(2)) - float(m.group(1)) <= 6.5


def test_hand_annotations_keep_their_lanes():
    from ambiscape.taxonomy import merge_keynote_beds
    objs = [{"name": "air-pump drone", "kind": "keynote",
             "spans": [[0, 100]]},
            {"name": "church bells", "kind": "soundmark",
             "events": [50]}]
    assert merge_keynote_beds(objs) == objs


def test_many_regime_session_renders_bounded(tmp_path):
    # regression: 60+ regimes gave a ~4000 px staircase timeline and a map
    # smeared with per-point AUTO boilerplate
    import matplotlib.pyplot as plt
    from ambiscape.taxonomy import schaeffer_map, schafer_timeline
    ann = {"objects": _auto_objects()
           + [{"name": "events (unclassified)", "kind": "figure",
               "events": [123.0, 4567.0]}]}
    tl = tmp_path / "schafer_timeline.png"
    schafer_timeline(ann, tl, title="synthetic")
    assert plt.imread(tl).shape[0] < 1600  # bounded height, was ~4000 px
    smap = tmp_path / "schaeffer_map.png"
    schaeffer_map(ann, smap, title="synthetic")
    assert smap.exists()


def test_map_point_labels_never_boilerplate():
    from ambiscape.taxonomy import _point_label
    auto = {"name": "steady state A (-58 dBFS)", "kind": "keynote",
            "label": "AUTO — mass/facture proposed from features; "
                     "listen to confirm (median -58 dBFS)"}
    # crowded map, crowded cell: no per-point text at all
    assert _point_label(auto, cell_n=9, n_placed=60) is None
    # crowded map, cell singleton (outlier): named, but never the boilerplate
    assert _point_label(auto, cell_n=1, n_placed=60) == \
        "steady state A (-58 dBFS)"
    # small map: hand labels drawn as before
    hand = {"name": "bells", "label": "cathedral bells (hourly)"}
    assert _point_label(hand, cell_n=2, n_placed=4) == \
        "cathedral bells (hourly)"


# --- human activity ground truth (SINS-style CSV) -------------------------

ACT_CSV = """Class;Start time;Stop time
absence;2017-01-30 00:00:00.000;2017-01-30 00:30:00.000
cooking;2017-01-30 00:30:00.000;2017-01-30 00:50:00
vacuumcleaner;2017-01-30 00:50:00.000;2017-01-30 01:00:00.000
absence;2017-01-30 01:00:00.000;2017-01-30 06:00:00.000
"""


def _acts(tmp_path, day0=None):
    from ambiscape.taxonomy import load_activities
    p = tmp_path / "living_labels.csv"
    p.write_text(ACT_CSV)
    return load_activities(p, day0=day0)


def test_load_activities_aligns_to_day0(tmp_path):
    import datetime as dt
    acts = _acts(tmp_path)  # day0 defaults to the first row's date
    assert acts[0] == {"class": "absence", "start": 0.0, "stop": 1800.0}
    assert acts[1]["stop"] == 3000.0     # timestamp without fractional part
    # an explicit day0 one day earlier shifts everything by 86400 s
    acts2 = _acts(tmp_path, day0=dt.date(2017, 1, 29))
    assert acts2[0]["start"] == 86400.0


def test_activity_suffix_time_shares():
    from ambiscape.taxonomy import activity_suffix
    acts = [{"class": "absence", "start": 0, "stop": 710},
            {"class": "sleeping", "start": 710, "stop": 1000}]
    assert activity_suffix([(0.0, 1000.0)], acts) == \
        " — during: absence 71%, sleeping 29%"
    assert activity_suffix([(0.0, 1000.0)], []) == ""     # nothing concurrent


def test_dominant_activity_spans_and_events():
    from ambiscape.taxonomy import _dominant_activity
    acts = [{"class": "cooking", "start": 0, "stop": 100},
            {"class": "eating", "start": 100, "stop": 400}]
    assert _dominant_activity({"spans": [[50, 350]]}, acts) == "eating"
    assert _dominant_activity({"events": [40.0]}, acts) == "cooking"
    assert _dominant_activity({"events": [999.0]}, acts) is None


def test_timeline_ribbon_and_bed_activity_labels(tmp_path):
    # synthetic session + synthetic activity CSV: the timeline renders with
    # an activity ribbon and legend (taller figure), beds gain "during:" text
    import matplotlib.pyplot as plt
    from ambiscape.taxonomy import schafer_timeline
    acts = _acts(tmp_path)
    ann = {"objects": _auto_objects(12)}   # > MAX_BED_LANES, so beds merge
    plain = tmp_path / "plain.png"
    schafer_timeline(ann, plain, title="synthetic")
    withact = tmp_path / "withact.png"
    schafer_timeline(ann, withact, title="synthetic", activities=acts)
    assert withact.exists()
    assert plt.imread(withact).shape[0] > plt.imread(plain).shape[0]


def test_schaeffer_map_with_activities(tmp_path):
    from ambiscape.taxonomy import schaeffer_map
    acts = _acts(tmp_path)
    ann = {"objects": [
        {"name": "kettle", "kind": "signal", "mass": "noise",
         "facture": "sustained", "spans": [[1800.0, 2400.0]]},
        {"name": "door", "kind": "figure", "mass": "complex",
         "facture": "impulse", "events": [3100.0]}]}
    out = tmp_path / "map.png"
    schaeffer_map(ann, out, title="synthetic", activities=acts)
    assert out.exists()


def test_render_activities_csv_and_missing_csv(tmp_path):
    # via the library/CLI entry point: with the CSV both figures render; a
    # missing path is identical to not passing one at all
    from ambiscape.taxonomy import render
    ann = {"objects": _auto_objects(12), "states": []}
    (tmp_path / "annotations.json").write_text(json.dumps(ann))
    csv_path = tmp_path / "living_labels.csv"
    csv_path.write_text(ACT_CSV)
    smap, tl = render(tmp_path, activities=csv_path)
    assert smap.exists() and tl.exists()
    import matplotlib.pyplot as plt
    h_with = plt.imread(tl).shape[0]
    smap2, tl2 = render(tmp_path, activities=tmp_path / "nope.csv")
    assert plt.imread(tl2).shape[0] < h_with   # no ribbon, no legend
