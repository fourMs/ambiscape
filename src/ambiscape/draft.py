"""Draft-annotation generator.

Pre-fills ``annotations.draft.json`` for a session from its cached features:
steady level regimes become keynote candidates (with spans), detected
transient events become one unclassified "figure" object whose entries carry
listening hints (clock time, exceedance, azimuth/elevation, diffuseness).
``mass`` and ``facture`` — the two Schaeffer axes the taxonomy map plots —
are now pre-proposed from the features (see :func:`schaeffer_hint`), with the
evidence under ``_schaeffer``; the fields that still need a human ear
(``kind``, soundmark status, the object name) stay "TODO". Confirm by ear,
rename, delete, then save as ``annotations.json`` and run ``ambiscape
taxonomy``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from .analysis import detect_events, db
from .figures import _gap_split

MIN_STATE_S = 120.0     # regimes shorter than this are merged away
STATE_STEP_DB = 5.0     # level change that separates regimes
MAX_EVENTS = 120        # cap on listed events


def _fmt(t: float) -> str:
    day, s = int(t // 86400), int(t % 86400)
    hms = f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
    return f"{day} {hms}" if day else hms


def _regimes(t: np.ndarray, fast: np.ndarray, dt: float):
    """Split a contiguous segment into steady-level regimes.

    The reference level is fixed per regime (median of its first two
    minutes); a boundary requires the smoothed level to differ from that
    reference by STATE_STEP_DB *and* to hold that difference for two more
    minutes — so transients don't split regimes, but genuine state changes
    (machine on/off) do.
    """
    n = len(fast)
    k = max(3, int(30 / dt)) | 1
    smooth = median_filter(fast, size=k, mode="nearest")
    hold = max(1, int(MIN_STATE_S / dt))
    skip = max(1, int(5 / dt))
    bounds = [0]
    start = 0
    while True:
        ref = float(np.median(smooth[start:min(start + hold, n)]))
        j = min(start + hold, n)
        nxt = None
        while j < n:
            if abs(smooth[j] - ref) > STATE_STEP_DB:
                if np.median(np.abs(smooth[j:j + hold] - ref)) > STATE_STEP_DB:
                    nxt = j
                    break
                j += skip
            else:
                j += 1
        if nxt is None:
            break
        bounds.append(nxt)
        start = nxt
    bounds.append(n)
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if (b - a) * dt >= MIN_STATE_S:
            out.append((float(t[a]), float(t[min(b, len(t) - 1)]),
                        float(np.median(fast[a:b]))))
    return out


MAX_TAGGED = 40  # PANNs windows per draft (model runs ~1-3 s each on CPU)


def _tagger(session):
    """Returns a tag(t_center) -> list function, or None if unavailable."""
    if session is None:
        return None
    try:
        from . import ml
        if not ml.panns_available():
            return None
        from .io import read_span

        def tag(t_center: float):
            try:
                x, fs = read_span(session, max(t_center - 5, 0), 10)
            except ValueError:
                return None
            return ml.tag_window(x[:, 0], fs)
        return tag
    except Exception:
        return None


def schaeffer_hint(F: dict, a: float, b: float) -> dict:
    """Propose Schaeffer typo-morphology for a [a, b]-second span from features.

    Turns the machine's ``mass`` and ``facture`` guesses (the axes the taxonomy
    map plots) into pre-fills, with the evidence surfaced under ``_schaeffer``:

    - **mass** from median spectral flatness (0 tonal → 1 noisy): tonic /
      tonic-complex / complex / noise;
    - **facture** (sustainment) from continuity: a ground that fills most of
      the session reads as *unlimited*, a bounded steady regime as *sustained*;
    - a **dynamic** hint (unvaried / varied) from the span's level spread.

    These are suggestions for the human annotator to confirm, not decisions.
    """
    t = np.asarray(F["t"], float)
    sel = (t >= a) & (t <= b)
    if not sel.any():
        return {"mass": "TODO", "facture": "sustained"}
    flat = float(np.median(np.asarray(F["flatness"], float)[sel]))
    mass = ("tonic" if flat < 0.05 else "tonic-complex" if flat < 0.2
            else "complex" if flat < 0.5 else "noise")
    span = b - a
    total = float(t.max() - t.min()) if t.size > 1 else span
    facture = "unlimited" if total > 0 and span >= 0.8 * total else "sustained"
    lvl = 20 * np.log10(np.asarray(F["rms_w"], float)[sel] + 1e-9)
    rng = float(np.percentile(lvl, 90) - np.percentile(lvl, 10))
    return {"mass": mass, "facture": facture,
            "_schaeffer": {"flatness": round(flat, 3),
                           "level_range_db": round(rng, 1),
                           "dynamic": "varied" if rng > 6 else "unvaried"}}


def draft_annotations(F: dict, folder: str | Path,
                      out_name="annotations.draft.json",
                      session=None) -> Path:
    folder = Path(folder)
    tf, fast = F["t_fast"], F["fast_db"]
    dt = float(np.median(np.diff(tf))) if len(tf) > 1 else 0.125
    objects = []
    tag = _tagger(session)
    n_tagged = 0

    # --- steady-state keynote candidates, per contiguous take-group
    label = iter("ABCDEFGHIJKLMNOP")
    for i0, i1 in _gap_split(F["t"]):
        m = (tf >= F["t"][i0]) & (tf <= F["t"][i1 - 1] + 1)
        for a, b, lvl in _regimes(tf[m], fast[m], dt):
            hint = schaeffer_hint(F, a, b)
            obj = {
                "name": f"steady state {next(label)} ({lvl:.0f} dBFS)",
                "kind": "keynote",
                "mass": hint["mass"], "facture": hint["facture"],
                "label": "AUTO — mass/facture proposed from features; "
                         f"listen to confirm (median {lvl:.0f} dBFS)",
                "spans": [[_fmt(a), _fmt(b)]],
            }
            if "_schaeffer" in hint:
                obj["_schaeffer"] = hint["_schaeffer"]
            if tag and n_tagged < MAX_TAGGED:
                tags = tag((a + b) / 2)
                n_tagged += 1
                if tags:
                    obj["_tags"] = tags
            objects.append(obj)

    # --- transient events with listening hints
    events, _bg = detect_events(fast, dt)
    events.sort(key=lambda e: -e["exceed"])
    listed = sorted(events[:MAX_EVENTS], key=lambda e: e["ipk"])
    details, times = [], []
    p = F["rms_w"].astype(np.float64) ** 2
    for e in listed:
        te = float(tf[e["ipk"]])
        si = int(np.clip(np.searchsorted(F["t"], te) - 1, 0, len(F["t"]) - 1))
        times.append(_fmt(te))
        hint = {
            "t": _fmt(te),
            "exceed_db": round(e["exceed"], 1),
            "level_dbfs": round(float(fast[e["ipk"]]), 1),
            "az": round(float(F["az"][si]), 0),
            "el": round(float(F["el"][si]), 0),
            "diffuseness": round(float(F["diffuse"][si]), 2),
        }
        if tag and n_tagged < MAX_TAGGED:
            tags = tag(te)
            n_tagged += 1
            if tags:
                hint["tags"] = tags
        details.append(hint)
    if times:
        objects.append({
            "name": "events (unclassified)",
            "kind": "figure", "mass": "TODO", "facture": "impulse",
            "label": "TODO — split into named objects by listening",
            "events": times,
            "_hints": details,
        })

    doc = {
        "_instructions": (
            "DRAFT generated by ambiscape. mass/facture are auto-proposed "
            "from features (evidence under _schaeffer) — confirm or correct "
            "them by ear. For each object also set kind (keynote|signal|"
            "soundmark|figure) — mass is (tonic|tonic-complex|complex|noise), "
            "facture (impulse|iteration|sustained|unlimited); rename it; "
            "split 'events (unclassified)' "
            "into one object per sound type (the _hints give clock time, "
            "level, azimuth/elevation, diffuseness for each event); delete "
            "what you don't want; optionally add soundmark: community|"
            "dwelling, source: biophony, and a states list for lo-fi spans. "
            "Save as annotations.json and run: ambiscape taxonomy <folder>."),
        "objects": objects,
        "states": [{"label": "TODO e.g. LO-FI (drone masks the field)",
                    "span": ["HH:MM:SS", "HH:MM:SS"]}],
    }
    out = folder / out_name
    out.write_text(json.dumps(doc, indent=2))
    return out
