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

The annotation file (``annotations.json`` or ``.yml`` in the session folder) is hand-authored:
instruments detect *when* things sound, but assigning a sound to any of the three schemes is an
interpretive act. This module turns that interpretation into two figures, one per tradition:

- ``schaeffer_map``  — objects on the facture x mass plane, which is Schaeffer's question. It is
  coloured by Schafer's ``kind`` only so you can see whether the two schemes happen to agree in a
  given corpus; the colouring carries no classificatory weight;
- ``schafer_timeline`` — one lane per hand-authored object on the session clock, keynote spans as
  bars, events as markers, lo-fi states shaded. Hi-fi and lo-fi are Schafer's terms too.
  Machine-drafted steady-state regimes are merged into a bounded set of keynote-bed lanes
  (see :func:`merge_keynote_beds`) so the figure height does not grow with the regime count.

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
"""
from __future__ import annotations

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
AUTO_NOTE = "machine-drafted labels; listen to confirm"


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


def schaeffer_map(ann: dict, out_path, title=""):
    """Objects on the facture x mass grid, coloured by Schafer function."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=130)
        ax.grid(False)
        for i in range(5):
            ax.axhline(i - 0.5, color=GRID, lw=0.8, zorder=0)
            ax.axvline(i - 0.5, color=GRID, lw=0.8, zorder=0)
        # `draft` deliberately leaves facture and mass as "TODO" — they are listening judgements a
        # detector should not guess — so a half-annotated object is the normal case here, not a
        # broken one. Placing it on the grid would mean inventing the very coordinates the annotator
        # has withheld, so it is left off the map and counted in the title instead. Before this, any
        # such object raised ValueError from .index() and took the whole render down with it.
        cells: dict[tuple, list] = {}
        n_unplaced = 0
        for o in ann["objects"]:
            if o.get("facture") not in FACTURES or o.get("mass") not in MASSES:
                n_unplaced += 1
                continue
            key = (FACTURES.index(o["facture"]), MASSES.index(o["mass"]))
            cells.setdefault(key, []).append(o)
        offsets = [(0, .1), (-.18, -.12), (.18, -.12), (-.18, .3), (.18, .3)]
        n_placed = sum(len(v) for v in cells.values())
        kinds_seen, bio_seen, ring_seen = set(), False, False
        for (x, y), objs in cells.items():
            n = len(objs)
            if n <= len(offsets):
                pos = offsets[:n]
                size, alpha = 170, 1.0
            else:
                # crowded cell: deterministic jitter, points shrunk, count shown
                rng = np.random.default_rng(97 + 13 * x + 5 * y)
                pos = rng.uniform(-0.33, 0.33, size=(n, 2))
                size, alpha = max(40, int(850 / n)), 0.75
                ax.annotate(f"n={n}", (x + 0.42, y - 0.4), ha="right",
                            fontsize=8.5, color=SEC, zorder=4)
            for o, (dx, dy) in zip(objs, pos):
                ring = "soundmark" in o and o["kind"] != "soundmark"
                ring_seen |= ring
                bio_seen |= o.get("source") == "biophony"
                kinds_seen.add(o["kind"])
                ax.scatter(x + dx, y + dy, s=size, marker=_marker(o),
                           color=KIND_COLOR[o["kind"]], zorder=3, alpha=alpha,
                           edgecolors=MAGENTA if ring else "none",
                           linewidths=2.2)
                text = _point_label(o, n, n_placed)
                if text:
                    ax.annotate(text, (x + dx, y + dy),
                                xytext=(0, -15), ha="center",
                                textcoords="offset points", fontsize=8.3,
                                color=INK, zorder=4)
        ax.set_xticks(range(4), FACTURE_LABELS)
        ax.set_yticks(range(4), MASS_LABELS)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(3.5, -0.5)
        ax.set_xlabel("facture / temporal sustainment  (Schaeffer typology) →")
        ax.set_ylabel("← mass  (Schaeffer morphology)")
        note = (f"  ({n_unplaced} object{'s' if n_unplaced > 1 else ''} not yet typed, omitted)"
                if n_unplaced else "")
        head = (f"{title} — sound objects in Schaeffer's typo-morphology,"
                f" colored by Schafer function{note}")
        if any(_is_auto(o) for o in ann["objects"]):
            head += f"\n{AUTO_NOTE}"
        ax.set_title(head, loc="left", fontsize=10.5)
        names = {"keynote": "keynote (ground)", "signal": "signal (figure)",
                 "soundmark": "community soundmark",
                 "figure": "incidental figure"}
        handles = [Line2D([], [], marker="o", ls="none", color=KIND_COLOR[k],
                          label=names[k]) for k in names if k in kinds_seen]
        if ring_seen:
            handles.append(Line2D([], [], marker="o", ls="none", color=SURF,
                                  markeredgecolor=MAGENTA, markeredgewidth=2,
                                  label="dwelling soundmark (ring)"))
        if bio_seen:
            handles.append(Line2D([], [], marker="^", ls="none", color=GREEN,
                                  label="biophony (triangle)"))
        ax.legend(handles=handles, loc="lower left", frameon=False,
                  fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


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


def schafer_timeline(ann: dict, out_path, title="", session=None):
    """Lane timeline of annotated objects; lo-fi states shaded.

    Machine-drafted steady-state regimes are merged into keynote beds by
    :func:`merge_keynote_beds`, so the lane count — and with it the figure
    height — stays bounded however many regimes a long session proposes.
    """
    objects = merge_keynote_beds(ann["objects"])
    states = ann.get("states", [])
    lanes = ["state"] + [o["name"] for o in objects] if states else \
            [o["name"] for o in objects]
    panels = _panels(ann, session)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            1, len(panels), figsize=(12.8, 0.52 * len(lanes) + 1.6), dpi=130,
            sharey=True, squeeze=False,
            gridspec_kw={"width_ratios": [b - a for a, b in panels],
                         "wspace": 0.03})
        axes = axes[0]
        ny = len(lanes)

        def Y(name):
            return ny - 1 - lanes.index(name)

        for ax, (t0, t1) in zip(axes, panels):
            ax.grid(False)
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
            ax.set_ylim(-0.7, ny - 0.3)
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
            if session is not None and len(panels) > 1:
                ax.set_title(session.clock(t0)[:6], loc="left",
                             fontsize=8.5, color=SEC)
        axes[0].set_yticks(range(ny), lanes[::-1])
        for lab in axes[0].get_yticklabels():
            o = next((o for o in objects if o["name"] == lab.get_text()), None)
            if o and o["kind"] == "keynote":
                lab.set_color("#1c5cab")
        head = (f"{title} — Schafer soundscape timeline: keynotes (blue "
                "lanes), signals (green), soundmarks (magenta), "
                "incidental figures (grey)")
        top = 0.96
        if any(_is_auto(o) for o in objects):
            head += f"\n{AUTO_NOTE}"
            top = 0.93
        fig.suptitle(head, x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout(rect=(0, 0, 1, top))
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def render(folder: str | Path, out_dir=None, session=None):
    """Load annotations from a session folder and write both figures."""
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
    name = folder.name
    schaeffer_map(ann, out / "schaeffer_map.png", title=name)
    schafer_timeline(ann, out / "schafer_timeline.png", title=name,
                     session=session)
    return out / "schaeffer_map.png", out / "schafer_timeline.png"
