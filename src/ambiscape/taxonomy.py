"""Figures for two separate sound taxonomies, from per-session annotations.

Two traditions are represented here and their founders have nearly the same name. They ask different
questions and neither constrains the other, so keep them apart.

Pierre Schaeffer (*Traité des objets musicaux*, 1966) asks what a sound IS in itself, heard through
reduced listening and classified by its internal make-up. Here that is ``mass`` and ``facture``.

R. Murray Schafer (*The Soundscape*, 1977) asks what a sound DOES in a place, classified by the role
it plays for the people living among it. Here that is ``kind`` and ``soundmark``.

A third scheme is present and belongs to neither. ``source`` (biophony, geophony, anthrophony) is
soundscape ecology, after Krause and Pijanowski, and classifies by physical origin.

One object carries all three labels independently. A ventilation drone is ``noise``/``unlimited`` to
Schaeffer, a ``keynote`` to Schafer, and ``anthrophony`` to soundscape ecology.

**The two figures work at different timescales, and this is not incidental.** A Schafer keynote is a
level that persists for minutes or hours — the ground of a place, heard as its condition. A Schaeffer
sound object is an event of roughly half a second to five seconds, short enough to be held whole in
one act of attention. Plotting steady-state regimes on the typo-morphology plane would conflate the
two: it would ask of an eight-hour ventilation bed the question Schaeffer asks of a single closing
door. So the map is built from detected *events* (see :mod:`ambiscape.objects`) and the timeline from
regimes, and neither borrows the other's unit.

The annotation file (``annotations.json`` or ``.yml`` in the session folder) is hand-authored:
instruments detect *when* things sound, but assigning a sound to any of the three schemes is an
interpretive act. This module turns that interpretation into two figures, one per tradition:

- ``schaeffer_map``  — sound objects on the facture x mass plane, which is Schaeffer's question. One
  point per object, extracted from the session's detected events and typed on both axes from its own
  spectral and temporal signature; hand-authored objects of the same scale join them. Keynote
  regimes never appear. Points are coloured by Schafer's ``kind`` only so you can see whether the two
  schemes happen to agree in a given corpus; the colouring carries no classificatory weight;
- ``schafer_timeline`` — the session clock. Two layouts. The acoustic-first layout (the only one
  without activity data) gives one lane per hand-authored object: keynote spans as bars, events as
  markers, lo-fi states shaded. Hi-fi and lo-fi are Schafer's terms too. Machine-drafted
  steady-state regimes are merged into a bounded set of keynote-bed lanes (see
  :func:`merge_keynote_beds`) so the figure height does not grow with the regime count. The
  activity-first layout (the default whenever activities are provided) inverts this: the human
  activities become the organising structure, one lane per activity class, each span's fill
  coloured by its measured fast level in dB re the day median (same palette as the cross-node day
  figures), with the machine keynote-bed structure compacted to a single strip.

Annotation schema (JSON; YAML accepted if PyYAML is installed)::

    {
      "objects": [
        {"name": "air-pump drone",
         "label": "air-pump drone (130 Hz comb, 9 h)",   # optional
         "kind": "keynote",             # keynote|signal|soundmark|figure
         "soundmark": "dwelling",       # optional: community|dwelling
         "source": "anthrophony",       # optional: ...|biophony|geophony
         "mass": "noise",               # tonic|tonic-complex|complex|noise
         "facture": "unlimited",        # impulse|iteration|sustained|unlimited
         "spans": [["23:01:36", "1 07:53:55"]],   # and/or
         "events": ["1 04:42:51"]},
        ...
      ],
      "states": [
        {"label": "LO-FI (drone masks the field)",
         "span": ["23:01:36", "1 07:53:55"]}
      ]
    }

Times are ``"[D ]HH:MM:SS"`` where the optional leading integer D is days
after the session's first day (or plain seconds as a number).

Independently of all three schemes, both figures can overlay **human-annotated
activities** (what the people in the space were doing: cooking, sleeping,
absence...) from a dataset ground-truth CSV in the SINS format
(``Class;Start time;Stop time`` with absolute timestamps, semicolon-separated;
Dekkers et al. 2017). These are data, not machine inference: captions
attribute them to the dataset, and they are never conflated with the
machine-drafted mass/facture judgements. See :func:`load_activities` and the
``activities`` parameter of :func:`render`, :func:`schaeffer_map` and
:func:`schafer_timeline`.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .figures import RC, INK, SEC, MUT, GRID, SURF, BLUE, GREEN, MAGENTA, YELLOW
from .figures import _gap_split

FACTURES = ["impulse", "iteration", "sustained", "unlimited"]
MASSES = ["tonic", "tonic-complex", "complex", "noise"]
FACTURE_LABELS = ["impulse", "iteration", "sustained\n(delimited)",
                  "sustained\n(unlimited / excentric)"]
MASS_LABELS = ["tonic\n(pitched)", "tonic-complex\n(pitch + noise)",
               "complex\n(unpitched)", "noise\n(broadband)"]
KIND_COLOR = {"keynote": BLUE, "signal": GREEN, "soundmark": MAGENTA,
              "figure": MUT}
LOFI = "#f0efec"

BED_BAND_DB = 6.0     # level band that groups steady regimes into one bed
MAX_BED_LANES = 8     # keynote-bed lanes on the timeline; the rest -> "other"
MAX_POINT_LABELS = 12  # above this, only cell-singleton outliers get map text
MAX_SCATTER = 1200    # points drawn on the map; cell counts stay complete
ACT_MIN_SHARE = 0.005  # activity classes under this share of labelled time
                       # are pooled into the "other" lane
LEVEL_CMAP = "magma"   # level colouring, dB re day median (as the cross-node
                       # day figures)
MINUS = "−"       # typographic minus for level text
AUTO_NOTE = "machine-drafted labels; listen to confirm"
ACTIVITY_NOTE = "activities: human annotations, Dekkers et al. 2017"

# Ribbon colours per activity class. "absence" and "other" are deliberately
# muted so that colour on the ribbon means somebody was doing something.
ACTIVITY_FIXED = {"absence": "#dddcd6", "other": "#b7b5ae"}
ACTIVITY_PALETTE = [BLUE, GREEN, MAGENTA, YELLOW, "#7f5bd5", "#b4232f",
                    "#0e9a8f", "#a86a00", "#5b7d16", "#d0568f", "#3f57c6",
                    "#8c6d5a"]


def parse_time(x) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    parts = str(x).strip().split()
    day = int(parts[0]) if len(parts) == 2 else 0
    h, m, s = (int(v) for v in parts[-1].split(":"))
    return day * 86400 + h * 3600 + m * 60 + s


def load_annotations(folder: str | Path) -> dict:
    folder = Path(folder)
    for name in ("annotations.json", "annotations.yml", "annotations.yaml"):
        p = folder / name
        if p.exists():
            if p.suffix == ".json":
                return json.loads(p.read_text())
            import yaml  # optional dependency
            return yaml.safe_load(p.read_text())
    raise FileNotFoundError(f"no annotations.json/yml in {folder}")


def _parse_stamp(s: str) -> _dt.datetime:
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return _dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognised activity timestamp: {s!r}")


def load_activities(path: str | Path, day0: _dt.date | None = None) -> list:
    """Human activity ground truth from a SINS-style CSV, on the session clock.

    The file is semicolon-separated with a ``Class;Start time;Stop time``
    header and absolute timestamps (Dekkers et al. 2017). Each row becomes
    ``{"class": str, "start": float, "stop": float}`` with times in seconds
    since midnight of ``day0`` — pass the session's ``day0`` so the spans land
    on the same clock as the annotation spans; without it the date of the
    first row is used.
    """
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows.append((r["Class"].strip(),
                         _parse_stamp(r["Start time"]),
                         _parse_stamp(r["Stop time"])))
    if not rows:
        return []
    if day0 is None:
        day0 = rows[0][1].date()
    base = _dt.datetime.combine(day0, _dt.time())
    return [{"class": c,
             "start": (a - base).total_seconds(),
             "stop": (b - base).total_seconds()} for c, a, b in rows]


def _overlap_shares(spans, activities) -> list:
    """Per-class share of the spans' total time, ``[(class, frac), ...]`` desc."""
    total = sum(b - a for a, b in spans)
    if total <= 0:
        return []
    acc: dict[str, float] = {}
    for a, b in spans:
        for act in activities:
            ov = min(b, act["stop"]) - max(a, act["start"])
            if ov > 0:
                acc[act["class"]] = acc.get(act["class"], 0.0) + ov
    return sorted(((c, v / total) for c, v in acc.items()),
                  key=lambda kv: -kv[1])


def activity_suffix(spans, activities, max_classes: int = 3,
                    min_share: float = 0.05) -> str:
    """Label suffix naming the dominant concurrent activities by time share,
    e.g. ``" — during: absence 71%, sleeping 22%"``; empty when nothing
    overlaps."""
    top = [(c, s) for c, s in _overlap_shares(spans, activities)[:max_classes]
           if s >= min_share]
    if not top:
        return ""
    return " — during: " + ", ".join(f"{c} {round(100 * s)}%" for c, s in top)


def _dominant_activity(o: dict, activities):
    """The activity class during which the object mostly occurs, or None."""
    spans = _spans_s(o)
    if spans:
        shares = _overlap_shares(spans, activities)
        return shares[0][0] if shares else None
    for e in o.get("events", []):
        t = parse_time(e)
        for act in activities:
            if act["start"] <= t <= act["stop"]:
                return act["class"]
    return None


def _activity_colors(activities) -> dict:
    """Stable class -> colour map: fixed muted greys for absence/other,
    palette colours by sorted class name for the rest."""
    classes = sorted({a["class"] for a in activities})
    colors = dict(ACTIVITY_FIXED)
    i = 0
    for c in classes:
        if c not in colors:
            colors[c] = ACTIVITY_PALETTE[i % len(ACTIVITY_PALETTE)]
            i += 1
    return {c: colors[c] for c in classes}


def _activity_lanes(activities, min_share: float = ACT_MIN_SHARE) -> list:
    """Lanes for the activity-first timeline: ``[(class, [activities]), ...]``.

    One lane per class, ordered by total duration (longest first); classes
    under ``min_share`` of the total labelled time are pooled into an
    ``"other"`` lane (which also absorbs a dataset-provided ``other`` class)
    placed last.
    """
    dur: dict[str, float] = {}
    for a in activities:
        dur[a["class"]] = dur.get(a["class"], 0.0) + a["stop"] - a["start"]
    total = sum(dur.values())
    major = sorted((c for c, d in dur.items()
                    if c != "other" and d >= min_share * total),
                   key=lambda c: -dur[c])
    pooled = [c for c in dur if c not in major]
    lanes = [(c, [a for a in activities if a["class"] == c]) for c in major]
    if pooled:
        lanes.append(("other", [a for a in activities
                                if a["class"] in pooled]))
    return lanes


def _level_context(F):
    """``(day median dBFS, Normalize)`` for level colouring from the session
    cache's fast level stream, or ``(None, None)`` without one.

    The norm spans a robust range of the day's fast levels re their median
    (P5 to P99, at least 6 dB wide, clipped), so span colours are read on the
    same "dB re day median" scale as the cross-node day figures.
    """
    if not F or "fast_db" not in F:
        return None, None
    v = np.asarray(F["fast_db"], float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None, None
    med = float(np.median(v))
    lo = float(np.percentile(v, 5)) - med
    hi = float(np.percentile(v, 99)) - med
    if hi - lo < 6.0:
        hi = lo + 6.0
    from matplotlib.colors import Normalize
    return med, Normalize(vmin=lo, vmax=hi, clip=True)


def _span_level(F, a: float, b: float):
    """Median fast level (dBFS) in ``[a, b)``, or None with no coverage."""
    t = np.asarray(F["t_fast"], float)
    v = np.asarray(F["fast_db"], float)[(t >= a) & (t < b)]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else None


def _fmt_dur(s: float) -> str:
    return f"{s/3600:.1f} h" if s >= 3600 else f"{s/60:.0f} min"


def _lane_label(name: str, acts: list, F=None) -> str:
    """Activity-lane label carrying the acoustic summary, e.g.
    ``"watching tv — 2.1 h, median −41 dBFS"``; without a feature cache the
    level part is omitted."""
    dur = sum(a["stop"] - a["start"] for a in acts)
    txt = f"{name} — {_fmt_dur(dur)}"
    if F is not None and "t_fast" in F:
        t = np.asarray(F["t_fast"], float)
        m = np.zeros(len(t), bool)
        for a in acts:
            m |= (t >= a["start"]) & (t < a["stop"])
        v = np.asarray(F["fast_db"], float)[m]
        v = v[np.isfinite(v)]
        if len(v):
            txt += f", median {np.median(v):.0f} dBFS".replace("-", MINUS)
    return txt


def _point_color(o: dict, activities=None, act_colors=None):
    """Map point colour and the class behind it: ``(colour, class|None)``.

    With activities, a point takes its dominant concurrent activity's colour
    (the same class colours as the timeline); otherwise — or when nothing
    overlaps — it keeps its Schafer ``kind`` colour and returns None.
    """
    if activities:
        dom = _dominant_activity(o, activities)
        if dom and (act_colors or {}).get(dom):
            return act_colors[dom], dom
    return KIND_COLOR[o["kind"]], None


def _marker(obj) -> str:
    if obj.get("source") == "biophony":
        return "^"
    if obj.get("kind") == "soundmark":
        return "D"
    return "o"


def _is_auto(o: dict) -> bool:
    """Machine-drafted object (from ``draft``), as opposed to hand-authored."""
    return bool(o.get("_auto")) \
        or str(o.get("label", "")).startswith("AUTO") \
        or str(o.get("name", "")).startswith("steady state ")


def _level_of(o: dict):
    """The object's median level in dBFS, or None if not recoverable."""
    if "_level_dbfs" in o:
        return float(o["_level_dbfs"])
    for s in (o.get("name", ""), o.get("label", "")):
        m = re.search(r"(-\d+(?:\.\d+)?)\s*dBFS", str(s))
        if m:
            return float(m.group(1))
    return None


def _bed_mid_level(o: dict):
    """Mid level (dBFS) of a keynote bed, from its 'lo to hi dBFS' name, a
    single-level name, or ``_level_dbfs``; None if not recoverable."""
    m = re.search(r"(-?\d+(?:\.\d+)?) to (-?\d+(?:\.\d+)?)\s*dBFS",
                  str(o.get("name", "")))
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    return _level_of(o)


def bed_name(lo: float, hi: float, n_spans: int) -> str:
    """Human name for a keynote bed: 'quiet bed, -60 to -54 dBFS, 23 spans'."""
    mid = (lo + hi) / 2
    desc = ("quiet bed" if mid <= -50 else
            "moderate bed" if mid <= -38 else "loud bed")
    lo_i, hi_i = round(lo), round(hi)
    rng = f"{lo_i} dBFS" if lo_i == hi_i else f"{lo_i} to {hi_i} dBFS"
    return f"{desc}, {rng}, {n_spans} span{'s' if n_spans != 1 else ''}"


def _spans_s(o: dict):
    return [(parse_time(a), parse_time(b)) for a, b in o.get("spans", [])]


def merge_keynote_beds(objects: list, max_beds: int = MAX_BED_LANES) -> list:
    """Cluster machine-drafted keynote regimes into level beds for the timeline.

    A domestic day yields 60+ steady-state regimes; one lane each gave a
    mostly-empty staircase thousands of pixels tall. Here auto-drafted
    keynotes (and only those — hand-authored objects always keep their own
    lane) are grouped into beds of similar level (~``BED_BAND_DB``-wide
    bands), one lane per bed carrying all of its spans, capped at
    ``max_beds`` lanes by total duration with the remainder pooled into
    "other beds". Returns the list unchanged when there is nothing to merge.
    """
    auto = [o for o in objects
            if o.get("kind") == "keynote" and _is_auto(o)
            and o.get("spans") and _level_of(o) is not None]
    if len(auto) <= max_beds:
        return list(objects)
    beds: list[dict] = []
    for o in sorted(auto, key=_level_of):
        lv = _level_of(o)
        if beds and lv - beds[-1]["lo"] <= BED_BAND_DB:
            beds[-1]["objs"].append(o)
            beds[-1]["hi"] = lv
        else:
            beds.append({"lo": lv, "hi": lv, "objs": [o]})

    def dur(b):
        return sum(t1 - t0 for o in b["objs"] for t0, t1 in _spans_s(o))

    beds.sort(key=dur, reverse=True)
    keep, spill = beds[:max_beds], beds[max_beds:]
    merged = []
    for b in sorted(keep, key=lambda b: -(b["lo"] + b["hi"]) / 2):
        spans = [s for o in b["objs"] for s in o.get("spans", [])]
        merged.append({"name": bed_name(b["lo"], b["hi"], len(spans)),
                       "kind": "keynote", "spans": spans, "_auto": True})
    if spill:
        spans = [s for b in spill for o in b["objs"]
                 for s in o.get("spans", [])]
        merged.append({"name": f"other beds ({len(spans)} spans)",
                       "kind": "keynote", "spans": spans, "_auto": True})
    auto_ids = {id(o) for o in auto}
    rest = [o for o in objects if id(o) not in auto_ids]
    return merged + rest


def _point_label(o: dict, cell_n: int, n_placed: int):
    """Text to draw beside a map point, or None to leave it unlabelled.

    Per-point text only where there are few points overall, or where a point
    sits alone in its grid cell (a notable outlier). Machine boilerplate
    ("AUTO — ...") is never drawn; the one ``AUTO_NOTE`` in the title covers
    every drafted point.
    """
    if n_placed > MAX_POINT_LABELS and cell_n > 1:
        return None
    text = str(o.get("label", o["name"]))
    if text.startswith("AUTO"):
        text = str(o["name"])
    return text


def _is_object_scale(o: dict, max_dur: float) -> bool:
    """True when an annotation entry is itself a sound object.

    An entry marked at points in time (``events``) is one by construction; an
    entry with spans is one only while every span stays inside the object
    window. A span of minutes is a keynote regime and belongs on the Schafer
    timeline, not on Schaeffer's plane.
    """
    spans = _spans_s(o)
    if spans:
        return max(b - a for a, b in spans) <= max_dur
    return bool(o.get("events"))


def _expand_events(o: dict) -> list:
    """One dict per event of a hand-authored object; spans are kept as they are.

    A "church bells" entry with twelve event times is twelve sound objects of
    the same type, and the map is a census of objects.
    """
    ev = o.get("events") or []
    if not ev:
        return [o]
    return [dict(o, events=[e], spans=[]) for e in ev]


def map_objects(ann: dict | None = None, F: dict | None = None,
                min_dur: float = None, max_dur: float = None) -> tuple:
    """The sound objects a Schaeffer map plots, and the census behind them.

    Two sources, both at object scale. From ``F`` (a
    :func:`~ambiscape.features.load_features` dict) come the session's
    detected events, filtered to the object duration window and typed on both
    axes by :func:`ambiscape.objects.extract_objects`. From ``ann`` come the
    hand-authored entries that are themselves object-scale — events, or spans
    no longer than the window — expanded one point per event. Machine-drafted
    keynote regimes are counted and set aside: they are Schafer's material.

    Returns ``(objects, stats)``, where ``stats`` carries every count the
    caption needs: ``n_detected``, ``n_short``, ``n_long``, ``n_hand``,
    ``n_regime``, ``n_untyped``, and the window actually used.
    """
    from .objects import OBJECT_MAX_S, OBJECT_MIN_S, extract_objects
    min_dur = OBJECT_MIN_S if min_dur is None else min_dur
    max_dur = OBJECT_MAX_S if max_dur is None else max_dur
    stats = {"n_detected": 0, "n_short": 0, "n_long": 0, "n_hand": 0,
             "n_regime": 0, "n_untyped": 0,
             "min_dur_s": min_dur, "max_dur_s": max_dur}
    objs: list = []
    if F is not None:
        r = extract_objects(F, min_dur=min_dur, max_dur=max_dur)
        for k in ("n_detected", "n_short", "n_long"):
            stats[k] = r[k]
        objs += r["objects"]
    for o in (ann or {}).get("objects", []):
        if o.get("_object"):
            continue                     # already extracted above
        if _is_auto(o) or not _is_object_scale(o, max_dur):
            stats["n_regime"] += 1
            continue
        if o.get("facture") not in FACTURES or o.get("mass") not in MASSES:
            stats["n_untyped"] += 1
            continue
        expanded = _expand_events(o)
        stats["n_hand"] += len(expanded)
        objs += expanded
    stats["n_untyped"] += sum(1 for o in objs
                              if o.get("facture") not in FACTURES
                              or o.get("mass") not in MASSES)
    return objs, stats


def _alpha_scale(objs: list):
    """``level -> alpha`` over the plotted objects' own level range.

    Louder objects are drawn more opaque, so a dense cell still shows where
    its energy sits. Returns a function; without recoverable levels it returns
    a constant.
    """
    lv = [_level_of(o) for o in objs]
    lv = [v for v in lv if v is not None]
    if len(lv) < 2:
        return lambda o: 0.85
    lo = float(np.percentile(lv, 5))
    hi = float(np.percentile(lv, 95))
    if hi - lo < 1.0:
        return lambda o: 0.85
    def alpha(o):
        v = _level_of(o)
        if v is None:
            return 0.5
        return 0.18 + 0.72 * float(np.clip((v - lo) / (hi - lo), 0, 1))
    return alpha


def _subsample(cells: dict, max_points: int, seed: int = 11):
    """Per-cell draw lists for the scatter, proportional to each cell's count.

    Returns ``(draw, sampled)``: ``draw`` maps a cell to the objects actually
    plotted, ``sampled`` is True when anything was left out. Sampling is
    stratified so no occupied cell disappears, and the cell counts printed on
    the figure always come from the full census, not from the sample.
    """
    total = sum(len(v) for v in cells.values())
    if total <= max_points:
        return {k: list(v) for k, v in cells.items()}, False
    rng = np.random.default_rng(seed)
    frac = max_points / total
    draw = {}
    for k, v in cells.items():
        n = max(1, int(round(len(v) * frac)))
        idx = rng.choice(len(v), size=min(n, len(v)), replace=False)
        draw[k] = [v[i] for i in sorted(idx)]
    return draw, True


def schaeffer_map(source, out_path, title="", activities=None,
                  stats=None, max_points: int = MAX_SCATTER):
    """Sound objects on the facture x mass grid — one point per object.

    ``source`` is a list of sound objects (as returned by
    :func:`map_objects`), or an annotation dict, which is passed through
    :func:`map_objects` with no feature cache so that only its hand-authored
    object-scale entries are plotted. Keynote regimes are never plotted: a
    multi-minute level bed is Schafer's unit, not Schaeffer's, and it is on
    the timeline where it belongs.

    Every object is one point, jittered inside its cell so that density is
    visible, with the cell's full count printed at its corner. Points are
    coloured by Schafer function — or, with ``activities`` (from
    :func:`load_activities`), by each point's dominant concurrent activity,
    with the same class colours as the timeline — and their opacity tracks the
    object's level, so the loud objects in a crowded cell stand out from the
    quiet ones. A session of tens of thousands of objects is subsampled for
    the scatter (``max_points``, stratified by cell, stated in the caption)
    while the printed counts stay complete. Objects few enough to name carry
    their labels, and on sparse maps also the activity they occurred during.

    ``stats`` is the census dict from :func:`map_objects`; when omitted it is
    derived from ``source``. The caption keeps the provenances apart:
    mass/facture are machine-drafted listening proposals, the activities are
    dataset ground truth.
    """
    if isinstance(source, dict):
        source, derived = map_objects(source)
        stats = derived if stats is None else stats
    objects = list(source)
    if stats is None:
        _, stats = map_objects({"objects": []})
        stats["n_untyped"] = sum(1 for o in objects
                                 if o.get("facture") not in FACTURES
                                 or o.get("mass") not in MASSES)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=130)
        ax.grid(False)
        for i in range(5):
            ax.axhline(i - 0.5, color=GRID, lw=0.8, zorder=0)
            ax.axvline(i - 0.5, color=GRID, lw=0.8, zorder=0)
        # An object the annotator (or the detector) has not typed on both axes
        # cannot be placed without inventing the very coordinates that are
        # missing, so it is left off the grid and counted in the title instead.
        cells: dict[tuple, list] = {}
        for o in objects:
            if o.get("facture") not in FACTURES or o.get("mass") not in MASSES:
                continue
            key = (FACTURES.index(o["facture"]), MASSES.index(o["mass"]))
            cells.setdefault(key, []).append(o)
        n_placed = sum(len(v) for v in cells.values())
        draw, sampled = _subsample(cells, max_points)
        n_drawn = sum(len(v) for v in draw.values())
        size = 170 if n_placed <= MAX_POINT_LABELS else (
            60 if n_placed <= 60 else (22 if n_placed <= 400 else 9))
        alpha_of = _alpha_scale(objects)
        act_colors = _activity_colors(activities) if activities else {}
        kinds_seen, classes_seen = set(), set()
        bio_seen, ring_seen = False, False
        for (x, y), shown in sorted(draw.items()):
            n = len(cells[(x, y)])
            rng = np.random.default_rng(97 + 13 * x + 5 * y)
            m = len(shown)
            if m <= len(_OFFSETS):
                pos = _OFFSETS[:m]
            else:
                pos = list(zip(rng.uniform(-0.36, 0.36, m),
                               rng.uniform(-0.30, 0.30, m)))
            for o, (dx, dy) in zip(shown, pos):
                ring = "soundmark" in o and o.get("kind") != "soundmark"
                ring_seen |= ring
                bio_seen |= o.get("source") == "biophony"
                c, cls = _point_color(o, activities, act_colors)
                if cls:
                    classes_seen.add(cls)
                else:
                    kinds_seen.add(o.get("kind", "figure"))
                ax.scatter(x + dx, y + dy, s=size, marker=_marker(o),
                           color=c, zorder=3, alpha=alpha_of(o),
                           edgecolors=MAGENTA if ring else "none",
                           linewidths=2.2)
                text = _point_label(o, n, n_placed)
                if text and activities:
                    dom = _dominant_activity(o, activities)
                    if dom:
                        text += f" — during {dom}"
                if text:
                    ax.annotate(text, (x + dx, y + dy),
                                xytext=(0, -15), ha="center",
                                textcoords="offset points", fontsize=8.3,
                                color=INK, zorder=4)
            if n_placed > MAX_POINT_LABELS:
                ax.annotate(f"n={n}", (x + 0.45, y - 0.42), ha="right",
                            va="top", fontsize=8, color=SEC, zorder=5)
        ax.set_xticks(range(4), FACTURE_LABELS)
        ax.set_yticks(range(4), MASS_LABELS)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(3.5, -0.5)
        ax.set_xlabel("facture / temporal sustainment  (Schaeffer typology) →")
        ax.set_ylabel("← mass  (Schaeffer morphology)")
        by = ("coloured by dominant concurrent activity" if activities
              else "coloured by Schafer function")
        head = (f"{title} — {n_placed} sound objects "
                f"({stats['min_dur_s']:g}–{stats['max_dur_s']:g} s) in "
                f"Schaeffer's typo-morphology, {by}, opacity by level")
        line2 = _census_line(stats)
        if line2:
            head += "\n" + line2
        notes = []
        if any(_is_auto(o) for o in objects):
            notes.append("mass/facture: machine-drafted, listen to confirm")
        if activities:
            notes.append(ACTIVITY_NOTE)
        if sampled:
            notes.append(f"scatter shows {n_drawn} of {n_placed} objects "
                         "(stratified by cell); counts are complete")
        if notes:
            head += "\n" + " · ".join(notes)
        ax.set_title(head, loc="left", fontsize=10.5)
        names = {"keynote": "keynote (ground)", "signal": "signal (figure)",
                 "soundmark": "community soundmark",
                 "figure": "incidental figure"}
        handles = [Line2D([], [], marker="o", ls="none", color=KIND_COLOR[k],
                          label=names[k]) for k in names if k in kinds_seen]
        handles += [Line2D([], [], marker="s", ls="none", color=act_colors[c],
                           label=c) for c in sorted(classes_seen)]
        if ring_seen:
            handles.append(Line2D([], [], marker="o", ls="none", color=SURF,
                                  markeredgecolor=MAGENTA, markeredgewidth=2,
                                  label="dwelling soundmark (ring)"))
        if bio_seen:
            handles.append(Line2D([], [], marker="^", ls="none", color=GREEN,
                                  label="biophony (triangle)"))
        # outside the axes: on a dense map every cell carries points, and a
        # legend inside would sit on top of them
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  frameon=False, fontsize=8, ncol=1)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


_OFFSETS = [(0, .1), (-.18, -.12), (.18, -.12), (-.18, .3), (.18, .3)]


def _census_line(stats: dict) -> str:
    """What the map left out, and where it went: the caption's second line."""
    bits = []
    if stats.get("n_detected"):
        out = stats.get("n_short", 0) + stats.get("n_long", 0)
        s = f"from {stats['n_detected']} detected events"
        if out:
            s += (f" ({stats['n_short']} shorter than "
                  f"{stats['min_dur_s']:g} s, {stats['n_long']} longer than "
                  f"{stats['max_dur_s']:g} s — not sound objects)")
        bits.append(s)
    if stats.get("n_hand"):
        bits.append(f"{stats['n_hand']} hand-authored")
    if stats.get("n_regime"):
        bits.append(f"{stats['n_regime']} keynote regime"
                    f"{'s' if stats['n_regime'] != 1 else ''} on the Schafer "
                    "timeline instead")
    if stats.get("n_untyped"):
        bits.append(f"{stats['n_untyped']} not yet typed, omitted")
    return "; ".join(bits)


def _panels(ann: dict, session=None):
    """Panel time ranges: session take-groups, else annotation extent."""
    if session is not None:
        t = np.array([tk.start for tk in session.takes]
                     + [session.takes[-1].end])
        groups = []
        cur = [session.takes[0]]
        for tk in session.takes[1:]:
            if tk.start - cur[-1].end > 600:
                groups.append((cur[0].start, cur[-1].end))
                cur = [tk]
            else:
                cur.append(tk)
        groups.append((cur[0].start, cur[-1].end))
        return groups
    ts = []
    for o in ann["objects"]:
        for a, b in o.get("spans", []):
            ts += [parse_time(a), parse_time(b)]
        ts += [parse_time(e) for e in o.get("events", [])]
    return [(min(ts), max(ts))]


def _xaxis(ax, t0: float, t1: float):
    """Clock ticks and panel cosmetics shared by both timeline layouts."""
    ax.set_xlim(t0, t1)
    span = t1 - t0
    step = 3600 if span > 5400 else (600 if span > 900 else 120)
    ticks = np.arange(np.ceil(t0 / step) * step, t1, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(x % 86400)//3600:02d}:"
                        f"{int(x % 3600)//60:02d}" for x in ticks])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5)


def schafer_timeline(ann: dict, out_path, title="", session=None,
                     activities=None, F=None, layout="auto"):
    """Schafer timeline of the session, in one of two layouts.

    ``layout="acoustic"`` (and any layout without ``activities``) is the
    lane timeline of annotated objects, lo-fi states shaded: machine-drafted
    steady-state regimes are merged into keynote beds by
    :func:`merge_keynote_beds`, so the lane count — and with it the figure
    height — stays bounded however many regimes a long session proposes.
    With ``activities`` it gains a compact ribbon of coloured activity spans
    along the top and each keynote-bed label its dominant concurrent
    activities by time share ("quiet bed, -60 to -54 dBFS, 23 spans —
    during: absence 71%, sleeping 22%").

    Whenever ``activities`` (from :func:`load_activities`) are given and
    ``layout`` is ``"auto"`` (default) or ``"activity"``, the layout inverts:
    the human activities become the organising structure. One lane per
    activity class (longest first, minor classes pooled into "other"), each
    span's fill coloured by its measured fast level in dB re the day median
    (``F``, a :func:`~ambiscape.features.load_features` dict; same palette as
    the cross-node day figures), lane labels carrying the acoustic summary
    ("watching tv — 2.1 h, median −41 dBFS"). Hand-authored objects keep
    their lanes and markers, the machine keynote-bed structure is compacted
    to a single strip coloured by band, and the events lane sits at the foot.
    """
    if layout not in ("auto", "activity", "acoustic"):
        raise ValueError(f"unknown timeline layout {layout!r}")
    if activities and layout != "acoustic":
        return _activity_timeline(ann, out_path, title=title,
                                  session=session, activities=activities,
                                  F=F)
    return _acoustic_timeline(ann, out_path, title=title, session=session,
                              activities=activities)


def _acoustic_timeline(ann: dict, out_path, title="", session=None,
                       activities=None):
    """The acoustic-first lane timeline (see :func:`schafer_timeline`)."""
    objects = merge_keynote_beds(ann["objects"])
    if activities:
        objects = [dict(o, name=o["name"]
                        + activity_suffix(_spans_s(o), activities))
                   if o.get("kind") == "keynote" and _is_auto(o)
                   and o.get("spans") else o
                   for o in objects]
    states = ann.get("states", [])
    lanes = ["state"] + [o["name"] for o in objects] if states else \
            [o["name"] for o in objects]
    act_colors = _activity_colors(activities) if activities else {}
    panels = _panels(ann, session)
    extra = 0.85 if activities else 0.0  # ribbon lane + class legend
    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            1, len(panels),
            figsize=(12.8, 0.52 * len(lanes) + 1.6 + extra), dpi=130,
            sharey=True, squeeze=False,
            gridspec_kw={"width_ratios": [b - a for a, b in panels],
                         "wspace": 0.03})
        axes = axes[0]
        ny = len(lanes)

        def Y(name):
            return ny - 1 - lanes.index(name)

        classes_drawn = set()
        for ax, (t0, t1) in zip(axes, panels):
            ax.grid(False)
            for act in activities or []:
                a, b = max(act["start"], t0), min(act["stop"], t1)
                if a >= b:
                    continue
                classes_drawn.add(act["class"])
                ax.add_patch(Rectangle((a, ny - 0.22), b - a, 0.56,
                                       color=act_colors[act["class"]], lw=0,
                                       zorder=2))
            for st in states:
                a, b = (parse_time(x) for x in st["span"])
                a, b = max(a, t0), min(b, t1)
                if a >= b:
                    continue
                ax.axvspan(a, b, color=LOFI, zorder=0)
                ax.add_patch(Rectangle((a, Y("state") - 0.3), b - a, 0.6,
                                       color=YELLOW, alpha=0.55, lw=0))
                ax.annotate(st.get("label", "lo-fi"), ((a + b) / 2, Y("state")),
                            ha="center", va="center", fontsize=8,
                            color="#6b4a00")
            for o in objects:
                y = Y(o["name"])
                c = KIND_COLOR[o["kind"]]
                for a, b in o.get("spans", []):
                    a, b = parse_time(a), parse_time(b)
                    a, b = max(a, t0), min(b, t1)
                    if a >= b:
                        continue
                    ax.add_patch(Rectangle((a, y - 0.2), max(b - a, (t1-t0)*0.004),
                                           0.4, color=c,
                                           alpha=0.85 if o["kind"] == "keynote"
                                           else 1.0, lw=0))
                ev = [parse_time(e) for e in o.get("events", [])]
                ev = [e for e in ev if t0 <= e <= t1]
                if ev:
                    mk = _marker(o)
                    if o["kind"] == "figure":
                        mk = "|"
                    ax.plot(ev, [y] * len(ev), ls="none", marker=mk,
                            ms=11 if mk == "|" else 7, mew=1.8, color=c)
            ax.set_ylim(-0.7, ny + 0.55 if activities else ny - 0.3)
            _xaxis(ax, t0, t1)
            if session is not None and len(panels) > 1:
                ax.set_title(session.clock(t0)[:6], loc="left",
                             fontsize=8.5, color=SEC)
        if activities:
            axes[0].set_yticks(list(range(ny)) + [ny],
                               lanes[::-1] + ["activities (human)"])
        else:
            axes[0].set_yticks(range(ny), lanes[::-1])
        for lab in axes[0].get_yticklabels():
            o = next((o for o in objects if o["name"] == lab.get_text()), None)
            if o and o["kind"] == "keynote":
                lab.set_color("#1c5cab")
        head = (f"{title} — Schafer soundscape timeline: keynotes (blue "
                "lanes), signals (green), soundmarks (magenta), "
                "incidental figures (grey)")
        notes = []
        if any(_is_auto(o) for o in objects):
            notes.append(AUTO_NOTE)
        if activities:
            notes.append(ACTIVITY_NOTE)
        top = 0.96
        if notes:
            head += "\n" + " · ".join(notes)
            top = 0.93
        fig.suptitle(head, x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout(rect=(0, 0.06 if classes_drawn else 0, 1, top))
        if classes_drawn:
            handles = [Rectangle((0, 0), 1, 1, color=act_colors[c], lw=0,
                                 label=c) for c in sorted(classes_drawn)]
            fig.legend(handles=handles, loc="lower left",
                       bbox_to_anchor=(0.01, 0.0), frameon=False,
                       fontsize=8, ncol=min(6, len(handles)),
                       handlelength=1.2, columnspacing=1.2)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def _activity_timeline(ann: dict, out_path, title="", session=None,
                       activities=None, F=None):
    """The activity-first timeline (see :func:`schafer_timeline`).

    Lanes, top to bottom: one per human activity class (longest first,
    minors pooled into "other"), hand-authored objects, one compact strip
    for all machine-drafted keynote beds, lo-fi states (if any), and the
    events lane at the foot. Activity spans and bed spans are filled by
    level — the median fast level of the span in dB re the day median, on
    the same palette as the cross-node day figures — so loud cooking and
    quiet cooking read differently at a glance. Without a feature cache the
    activity spans fall back to a neutral grey and lane labels carry
    durations only.
    """
    objects = merge_keynote_beds(ann["objects"])
    beds = [o for o in objects if o.get("kind") == "keynote" and _is_auto(o)
            and o.get("spans")]
    bed_ids = {id(o) for o in beds}
    ev_objs = [o for o in objects
               if id(o) not in bed_ids and o.get("kind") == "figure"]
    ev_ids = {id(o) for o in ev_objs}
    hand = [o for o in objects
            if id(o) not in bed_ids and id(o) not in ev_ids]
    states = ann.get("states", [])
    panels = _panels(ann, session)
    # a dataset activity log can cover weeks; lane order, pooling, durations
    # and level stats must all describe this session only
    t_lo, t_hi = panels[0][0], panels[-1][1]
    activities = [dict(a, start=max(a["start"], t_lo),
                       stop=min(a["stop"], t_hi)) for a in activities
                  if min(a["stop"], t_hi) > max(a["start"], t_lo)]
    med, norm = _level_context(F)
    cmap = plt.get_cmap(LEVEL_CMAP)

    def level_color(lv):
        return cmap(norm(lv - med)) if lv is not None and med is not None \
            else "#c9c7c1"

    n_bed_spans = sum(len(o.get("spans", [])) for o in beds)
    bed_lane = (f"keynote beds — machine draft, {n_bed_spans} "
                f"span{'s' if n_bed_spans != 1 else ''}")
    rows = [("act", _lane_label(c, acts, F), acts)
            for c, acts in _activity_lanes(activities)]
    rows += [("obj", o["name"], o) for o in hand]
    if beds:
        rows.append(("beds", bed_lane, beds))
    if states:
        rows.append(("state", "state", states))
    rows += [("obj", o["name"], o) for o in ev_objs]
    ny = len(rows)
    h_in = 0.52 * ny + 1.7
    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            1, len(panels), figsize=(12.8, h_in), dpi=130,
            sharey=True, squeeze=False,
            gridspec_kw={"width_ratios": [b - a for a, b in panels],
                         "wspace": 0.03})
        axes = axes[0]
        for ax, (t0, t1) in zip(axes, panels):
            ax.grid(False)
            minw = (t1 - t0) * 0.004
            for st in states:
                a, b = (parse_time(x) for x in st["span"])
                a, b = max(a, t0), min(b, t1)
                if a < b:
                    ax.axvspan(a, b, color=LOFI, zorder=0)
            for i, (kind, _, payload) in enumerate(rows):
                y = ny - 1 - i
                if kind == "act":
                    for act in payload:
                        a, b = max(act["start"], t0), min(act["stop"], t1)
                        if a >= b:
                            continue
                        lv = _span_level(F, a, b) if F is not None else None
                        ax.add_patch(Rectangle((a, y - 0.27),
                                               max(b - a, minw), 0.54,
                                               color=level_color(lv), lw=0,
                                               zorder=2))
                elif kind == "beds":
                    for o in payload:
                        c = level_color(_bed_mid_level(o))
                        for a, b in o.get("spans", []):
                            a, b = parse_time(a), parse_time(b)
                            a, b = max(a, t0), min(b, t1)
                            if a < b:
                                ax.add_patch(Rectangle((a, y - 0.18),
                                                       max(b - a, minw),
                                                       0.36, color=c, lw=0))
                elif kind == "state":
                    for st in payload:
                        a, b = (parse_time(x) for x in st["span"])
                        a, b = max(a, t0), min(b, t1)
                        if a >= b:
                            continue
                        ax.add_patch(Rectangle((a, y - 0.3), b - a, 0.6,
                                               color=YELLOW, alpha=0.55,
                                               lw=0))
                        ax.annotate(st.get("label", "lo-fi"),
                                    ((a + b) / 2, y), ha="center",
                                    va="center", fontsize=8, color="#6b4a00")
                else:                       # hand-authored object or events
                    o = payload
                    c = KIND_COLOR.get(o.get("kind", "figure"), MUT)
                    for a, b in o.get("spans", []):
                        a, b = parse_time(a), parse_time(b)
                        a, b = max(a, t0), min(b, t1)
                        if a < b:
                            ax.add_patch(Rectangle((a, y - 0.2),
                                                   max(b - a, minw), 0.4,
                                                   color=c, lw=0))
                    ev = [parse_time(e) for e in o.get("events", [])]
                    ev = [e for e in ev if t0 <= e <= t1]
                    if ev:
                        mk = "|" if o.get("kind") == "figure" else _marker(o)
                        ax.plot(ev, [y] * len(ev), ls="none", marker=mk,
                                ms=11 if mk == "|" else 7, mew=1.8, color=c)
            ax.set_ylim(-0.7, ny - 0.3)
            _xaxis(ax, t0, t1)
            if session is not None and len(panels) > 1:
                ax.set_title(session.clock(t0)[:6], loc="left",
                             fontsize=8.5, color=SEC)
        axes[0].set_yticks(range(ny), [r[1] for r in rows][::-1])
        for lab, (kind, _, payload) in zip(axes[0].get_yticklabels(),
                                           rows[::-1]):
            if kind == "beds" or (kind == "obj"
                                  and payload.get("kind") == "keynote"):
                lab.set_color("#1c5cab")
        head = (f"{title} — Schafer soundscape timeline, activity-first: one "
                "lane per human activity, spans coloured by the fast level "
                "re the day median; machine keynote beds as one strip, "
                "events at the foot")
        notes = []
        if beds or any(_is_auto(o) for o in objects):
            notes.append(AUTO_NOTE)
        notes.append(ACTIVITY_NOTE)
        head += "\n" + " · ".join(notes)
        fig.suptitle(head, x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.subplots_adjust(top=1 - 0.66 / h_in, bottom=0.45 / h_in)
        if med is not None:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cb = fig.colorbar(sm, ax=list(axes), fraction=0.03, pad=0.015)
            cb.set_label(("fast level, dB re day median "
                          f"({med:.0f} dBFS)").replace("-", MINUS))
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def render(folder: str | Path, out_dir=None, session=None, activities=None,
           layout="auto", object_window=None):
    """Load annotations from a session folder and write both figures.

    ``activities`` is an optional path to a SINS-style activity CSV
    (``Class;Start time;Stop time``, semicolon-separated, absolute
    timestamps); when given and present, the human-annotated activities are
    aligned to the session clock (via the session's ``day0``) and the
    timeline switches to the activity-first layout (``layout="acoustic"``
    keeps the acoustic-first lane timeline with the activity ribbon), with
    span level colouring and lane level stats drawn from the session's
    cached features when available. A missing file leaves both figures
    exactly as without it.

    The map is built from the session's cached features whenever they are
    present: the detected events are extracted as sound objects and typed on
    Schaeffer's two axes (see :func:`map_objects`). Without a feature cache
    the map falls back to whatever object-scale entries the annotation file
    itself carries. ``object_window`` is an optional ``(min_s, max_s)`` pair
    overriding the 0.2–8 s duration window that decides what counts as a sound
    object.
    """
    lo, hi = object_window if object_window else (None, None)
    folder = Path(folder)
    ann = load_annotations(folder)
    out = Path(out_dir) if out_dir else folder / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    if session is None:
        from .io import open_session
        try:
            session = open_session(folder)
        except (FileNotFoundError, ValueError):
            session = None
    acts = None
    if activities is not None and Path(activities).exists():
        acts = load_activities(
            activities, day0=session.day0 if session else None)
    F = None
    paths = sorted((out / "features").glob("*.npz"))
    if paths:
        from .features import load_features
        try:
            F = load_features(paths)
        except Exception:
            F = None           # figures still render, without the cache
    name = folder.name
    objs, stats = map_objects(ann, F, min_dur=lo, max_dur=hi)
    schaeffer_map(objs, out / "schaeffer_map.png", title=name,
                  activities=acts, stats=stats)
    schafer_timeline(ann, out / "schafer_timeline.png", title=name,
                     session=session, activities=acts, F=F, layout=layout)
    return out / "schaeffer_map.png", out / "schafer_timeline.png"
