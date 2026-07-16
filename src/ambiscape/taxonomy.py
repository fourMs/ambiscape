"""Schaeffer / Schafer taxonomy figures from per-session annotations.

The annotation file (``annotations.json`` or ``.yml`` in the session folder)
is hand-authored: instruments detect *when* things sound, but assigning a
sound to Schaeffer's typo-morphology or Schafer's functional categories is an
interpretive act. This module turns that interpretation into two figures:

- ``schaeffer_map``  — objects on the facture x mass plane, colored by
  Schafer function (keynote / signal / soundmark / figure);
- ``schafer_timeline`` — one lane per object on the session clock, keynote
  spans as bars, events as markers, lo-fi states shaded.

Annotation schema (JSON; YAML accepted if PyYAML is installed)::

    {
      "objects": [
        {"name": "air-pump drone",
         "label": "air-pump drone (130 Hz comb, 9 h)",   # optional
         "kind": "keynote",             # keynote|signal|soundmark|figure
         "soundmark": "dwelling",       # optional: community|dwelling
         "source": "anthropophony",     # optional: ...|biophony|geophony
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


def schaeffer_map(ann: dict, out_path, title=""):
    """Objects on the facture x mass grid, colored by Schafer function."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=130)
        ax.grid(False)
        for i in range(5):
            ax.axhline(i - 0.5, color=GRID, lw=0.8, zorder=0)
            ax.axvline(i - 0.5, color=GRID, lw=0.8, zorder=0)
        cells: dict[tuple, list] = {}
        for o in ann["objects"]:
            key = (FACTURES.index(o["facture"]), MASSES.index(o["mass"]))
            cells.setdefault(key, []).append(o)
        offsets = [(0, .1), (-.18, -.12), (.18, -.12), (-.18, .3), (.18, .3)]
        kinds_seen, bio_seen, ring_seen = set(), False, False
        for (x, y), objs in cells.items():
            for o, (dx, dy) in zip(objs, offsets):
                ring = "soundmark" in o and o["kind"] != "soundmark"
                ring_seen |= ring
                bio_seen |= o.get("source") == "biophony"
                kinds_seen.add(o["kind"])
                ax.scatter(x + dx, y + dy, s=170, marker=_marker(o),
                           color=KIND_COLOR[o["kind"]], zorder=3,
                           edgecolors=MAGENTA if ring else "none",
                           linewidths=2.2)
                ax.annotate(o.get("label", o["name"]), (x + dx, y + dy),
                            xytext=(0, -15), ha="center",
                            textcoords="offset points", fontsize=8.3,
                            color=INK, zorder=4)
        ax.set_xticks(range(4), FACTURE_LABELS)
        ax.set_yticks(range(4), MASS_LABELS)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(3.5, -0.5)
        ax.set_xlabel("facture / temporal sustainment  (Schaeffer typology) →")
        ax.set_ylabel("← mass  (Schaeffer morphology)")
        ax.set_title(f"{title} — sound objects in Schaeffer's typo-morphology,"
                     " colored by Schafer function", loc="left", fontsize=10.5)
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
    """Lane timeline of annotated objects; lo-fi states shaded."""
    objects = ann["objects"]
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
        fig.suptitle(f"{title} — Schafer soundscape timeline: keynotes (blue "
                     "lanes), signals (green), soundmarks (magenta), "
                     "incidental figures (grey)", x=0.01, ha="left",
                     fontsize=10.5, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
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
