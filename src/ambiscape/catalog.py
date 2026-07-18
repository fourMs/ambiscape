"""Corpus aggregation: one cross-session table from cached summaries.

Every ``ambiscape analyze`` writes ``<session>/analysis/summary.json``.
This module collects them across a corpus folder into one table — CSV for
analysis, a transposed Markdown table (descriptor rows, session columns) for
a consolidated report — plus simple ranking and outlier helpers. No audio,
no features: it reads only the cached summaries, so a whole corpus
aggregates in milliseconds.

The Markdown layout follows the Intercontinental-database
``CONSOLIDATED.md`` convention. Sessions with differing descriptor sets
(older caches, optional modules) are handled by taking the union of keys
and leaving blanks where a session lacks one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def collect(corpus_dir: str | Path,
            pattern: str = "*/analysis/summary.json",
            include_states: bool = False) -> dict:
    """Map session name → summary dict for every summary under ``corpus_dir``.

    The session name is the top-level folder (the parent of ``analysis/``).
    Unreadable or malformed files are skipped. With ``include_states``, each
    session's ``analysis/states.json`` (if present) contributes extra
    ``"<session>::<state>"`` rows right after the pooled session row — the
    state-resolved corpus view.
    """
    corpus_dir = Path(corpus_dir)
    out = {}
    for p in sorted(corpus_dir.glob(pattern)):
        name = p.parent.parent.name
        try:
            out[name] = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if include_states:
            sp = p.parent / "states.json"
            if sp.exists():
                try:
                    states = json.loads(sp.read_text()).get("states", {})
                except (json.JSONDecodeError, OSError):
                    states = {}
                for label, summ in states.items():
                    out[f"{name}::{label}"] = {
                        k: v for k, v in summ.items() if k != "intervals_s"}
    return out


def _all_keys(collected: dict, keys=None) -> list:
    if keys is not None:
        return list(keys)
    seen = []
    for summary in collected.values():
        for k in summary:
            if k not in seen:
                seen.append(k)
    return seen


def to_csv(collected: dict, path: str | Path, keys=None) -> Path:
    """Write a session-per-row CSV (union of keys, blanks for missing)."""
    path = Path(path)
    cols = _all_keys(collected, keys)
    lines = ["session," + ",".join(cols)]
    for name, summary in collected.items():
        row = [name]
        for k in cols:
            v = summary.get(k, "")
            row.append("" if v is None else str(v))
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def to_markdown(collected: dict, keys=None, labels: dict | None = None) -> str:
    """Transposed Markdown table: one row per descriptor, one column per
    session (the consolidated-report layout). ``labels`` optionally maps
    descriptor keys to human labels."""
    names = list(collected)
    cols = _all_keys(collected, keys)
    labels = labels or {}
    head = "| Descriptor | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(names) + 1)
    rows = [head, sep]
    for k in cols:
        cells = []
        for n in names:
            v = collected[n].get(k)
            cells.append("" if v is None else str(v))
        rows.append(f"| {labels.get(k, k)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _values(collected: dict, key: str):
    out = {}
    for name, summary in collected.items():
        v = summary.get(key)
        if isinstance(v, (int, float)):
            out[name] = float(v)
    return out


def rank(collected: dict, key: str, descending: bool = True) -> list:
    """(session, value) pairs sorted by ``key`` (numeric sessions only)."""
    vals = _values(collected, key)
    return sorted(vals.items(), key=lambda kv: kv[1], reverse=descending)


def outliers(collected: dict, key: str, z: float = 1.5) -> list:
    """(session, z-score) for sessions more than ``z`` SDs from the mean
    on ``key``, most extreme first — the cheap "what stands out" query."""
    vals = _values(collected, key)
    if len(vals) < 3:
        return []
    x = np.array(list(vals.values()))
    mu, sd = float(x.mean()), float(x.std())
    if sd == 0:
        return []
    scored = [(n, round((v - mu) / sd, 2)) for n, v in vals.items()]
    return sorted([s for s in scored if abs(s[1]) >= z],
                  key=lambda s: -abs(s[1]))
