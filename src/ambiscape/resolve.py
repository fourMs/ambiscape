"""State-resolved descriptors: summarize each state of a session separately.

A single descriptor row for a multi-state session is a duration-weighted
average of things that never coexisted — the Haarlem loft's row is dominated
by the 9-hour air-pump night and barely reflects the hi-fi afternoon. This
module slices the cached features by time and runs the *full* summary
pipeline on each state, so "vent on" and "vent off" (or day / night, or any
supplied intervals) get their own complete descriptor set.

- ``slice_features`` — a sub-``F`` restricted to a set of time intervals,
  valid across every feature axis (1 s, 125 ms fast, 20 ms envelope,
  per-minute PSD) for all the ``summarize_*`` functions;
- ``full_summary`` — the merged descriptor dict (level/event, spectral
  foreground, ecoacoustic, spatial, biophony) — the same set ``analyze``
  writes, computed on any ``F``;
- ``resolve`` — ``{state: full_summary}`` for a dict of named intervals;
- ``machine_states`` / ``diel_states`` — auto-discover the states from a
  machine band (via :func:`ambiscape.states.state_segments`) or the wall
  clock (day / night).

Event detection and percentiles run per state, so an interval need not be
contiguous; a state shorter than a few frames is skipped.
"""
from __future__ import annotations

import numpy as np

# feature keys carried on each time axis
_SEC_KEYS = ("rms_w", "peak", "oct_pow", "centroid", "flatness", "logspec",
             "I_band", "az", "el", "diffuse")
_FAST_KEYS = ("fast_db", "fast_dba")
_SCALAR_KEYS = ("freqs", "logf", "fs", "fast_dt", "hi_dt")


def intervals_from_mask(t: np.ndarray, mask: np.ndarray) -> list:
    """Contiguous ``[start, stop)`` intervals (in ``t`` units) of a boolean
    mask over the 1 s frames."""
    t = np.asarray(t, float)
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    edges = np.flatnonzero(np.diff(m.astype(int)))
    starts = [0] if m[0] else []
    stops = []
    for e in edges:
        (stops if m[e] else starts).append(e + 1)
    if m[-1]:
        stops.append(len(m))
    return [(float(t[a]), float(t[b - 1]) + 1.0) for a, b in zip(starts, stops)]


def _axis_mask(tvec, intervals):
    m = np.zeros(len(tvec), bool)
    for a, b in intervals:
        m |= (tvec >= a) & (tvec < b)
    return m


def slice_features(F: dict, intervals: list) -> dict:
    """Restrict cached features to ``intervals`` (absolute seconds).

    Returns a sub-``F`` with every time-indexed array masked to the
    intervals and every scalar/axis array copied through — accepted by all
    the ``summarize_*`` functions.
    """
    out = {}
    m1 = _axis_mask(F["t"], intervals)
    out["t"] = F["t"][m1]
    for k in _SEC_KEYS:
        if k in F:
            out[k] = F[k][m1]
    if "t_fast" in F:
        mf = _axis_mask(F["t_fast"], intervals)
        out["t_fast"] = F["t_fast"][mf]
        for k in _FAST_KEYS:
            if k in F:
                out[k] = F[k][mf]
    if "t_hi" in F and "env_hi" in F:
        mh = _axis_mask(F["t_hi"], intervals)
        out["t_hi"] = F["t_hi"][mh]
        out["env_hi"] = F["env_hi"][mh]
    if "min_t" in F:
        mm = _axis_mask(F["min_t"], intervals)
        out["min_t"] = F["min_t"][mm]
        out["minspec"] = F["minspec"][mm]
    for k in _SCALAR_KEYS:
        if k in F:
            out[k] = F[k]
    return out


def full_summary(F: dict) -> dict:
    """The complete analyze descriptor set for any ``F`` (no calibration)."""
    from . import analysis, background, biophony, ecology, spatial
    s = analysis.summarize(F)
    s.update(background.summarize_foreground(F))
    s.update(ecology.summarize_ecology(F))
    s.update(spatial.summarize_spatial(F))
    s.update(biophony.summarize_biophony(F))
    return s


def resolve(F: dict, states: dict, min_frames: int = 30) -> dict:
    """``{state: full_summary}`` for a dict of ``{label: intervals}``.

    States whose sliced 1 s length is below ``min_frames`` are skipped
    (too short for stable percentiles/events).
    """
    out = {}
    for label, intervals in states.items():
        sub = slice_features(F, intervals)
        if len(sub["t"]) >= min_frames:
            out[label] = full_summary(sub)
    return out


def machine_states(F: dict, band=(250.0, 1000.0), min_dur_s: float = 120.0,
                   labels=("machine_on", "machine_off")) -> dict:
    """Auto-discover on/off states from a machine band.

    Segments the band level with :func:`ambiscape.states.state_segments`
    and returns ``{labels[0]: on-intervals, labels[1]: off-intervals}``
    in absolute seconds (empty sides dropped).
    """
    from .states import band_level, state_segments
    segs = state_segments(band_level(F, band), min_dur_s=min_dur_s)
    t = F["t"]
    n = len(t)
    states = {labels[0]: [], labels[1]: []}
    for s in segs:
        i0 = int(s["t0_s"])
        i1 = int(min(s["t0_s"] + s["dur_s"], n))
        if i1 <= i0:
            continue
        key = labels[0] if s["state"] == "on" else labels[1]
        states[key].append((float(t[i0]), float(t[i1 - 1]) + 1.0))
    return {k: v for k, v in states.items() if v}


def diel_states(F: dict, sess, night=(22, 6),
                labels=("night", "day")) -> dict:
    """Split a session into night / day by the wall clock.

    ``night`` is ``(start_hour, end_hour)`` wrapping midnight; uses the
    session's ``day0`` to turn absolute seconds into hour-of-day.
    """
    import datetime as _dt
    base = _dt.datetime.combine(sess.day0, _dt.time())
    hours = np.array([(base + _dt.timedelta(seconds=float(x))).hour
                      for x in F["t"]])
    lo, hi = night
    if lo < hi:
        night_mask = (hours >= lo) & (hours < hi)
    else:
        night_mask = (hours >= lo) | (hours < hi)
    out = {labels[0]: intervals_from_mask(F["t"], night_mask),
           labels[1]: intervals_from_mask(F["t"], ~night_mask)}
    return {k: v for k, v in out.items() if v}
