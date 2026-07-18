"""Schedule matching: test event/strike streams against civic time grids.

Bells, chimes, and sirens follow wall-clock schedules — hourly strikes,
quarter-hour chimes, fixed evening ringing. Given event times on the
session's absolute clock, ``match_periods`` folds them at candidate civic
periods and scores each with circular statistics; ``clock_offset`` turns a
known schedule into a recorder-clock correction (the workflow behind
``clock_offset_s`` in ``calibration.json``).

Times must be *absolute* seconds (session clock, i.e. ``take.start`` +
offset into the take), otherwise grid phases are meaningless.
"""
from __future__ import annotations

import numpy as np

from .circstats import circular_sd, mean_resultant, rayleigh_p

CIVIC_PERIODS = (60.0, 300.0, 900.0, 1800.0, 3600.0, 86400.0)


def match_periods(times_abs: np.ndarray, periods=CIVIC_PERIODS) -> list[dict]:
    """Fold events at each candidate period; score alignment.

    Returns one dict per period — phase of the grid the events cluster on
    (seconds past the grid tick), R, circular SD, Rayleigh p — sorted by R.
    A meaningful match needs both high R and enough events spread over
    several grid cycles (``n_cycles``); R is trivially 1 when all events
    fall inside one cycle.
    """
    t = np.asarray(times_abs, float)
    out = []
    for P in periods:
        ph = 2 * np.pi * (t / P % 1.0)
        mu, R = mean_resultant(ph)
        out.append({
            "period_s": P,
            "phase_s": round(float((mu / (2 * np.pi)) % 1.0 * P), 1),
            "R": round(R, 3),
            "circ_sd_s": round(circular_sd(R) / (2 * np.pi) * P, 1),
            "rayleigh_p": rayleigh_p(R, len(t)),
            "n": int(len(t)),
            "n_cycles": int(np.ptp(t) // P) + 1,
        })
    return sorted(out, key=lambda d: -d["R"])


def grid_scan(F: dict, period_s: float, phase_s: float = 0.0,
              band=(300.0, 1500.0), win_s: float = 120.0,
              min_rise_db: float = 6.0, bg_win_s: float = 300.0) -> list[dict]:
    """Targeted scan of every tick of a civic grid for band-limited strikes.

    The complement of :func:`match_periods`: instead of asking which grid an
    event stream fits, look *at each tick* of a known grid (every quarter
    hour, every hour + ``phase_s``) for energy in a band — a church clock in
    the bell band, whether or not the broadband detector heard it. Uses the
    cached features (band level above a running ``bg_win_s`` low-percentile
    background), so the scan is instant.

    Returns one dict per tick inside the feature timeline: ``t_tick``
    (absolute seconds), ``detected``, ``rise_db`` (peak exceedance within
    ``win_s`` centered on the tick), and ``offset_s`` of that peak from the
    tick — a consistent nonzero offset across ticks is recorder-clock error
    (see :func:`clock_offset`).
    """
    from scipy.ndimage import percentile_filter
    from .states import band_level

    t = np.asarray(F["t"], float)
    lvl = band_level(F, band)
    n = max(3, int(round(bg_win_s)) | 1)
    rise = lvl - percentile_filter(lvl, 10, size=n, mode="nearest")
    first = np.ceil((t[0] - phase_s) / period_s) * period_s + phase_s
    out = []
    for tick in np.arange(first, t[-1], period_s):
        m = np.abs(t - tick) <= win_s / 2
        if not m.any():
            continue
        i = int(np.argmax(rise[m]))
        r = float(rise[m][i])
        out.append({
            "t_tick": float(tick),
            "detected": bool(r >= min_rise_db),
            "rise_db": round(r, 1),
            "offset_s": round(float(t[m][i] - tick), 1),
        })
    return out


def clock_offset(observed_abs: float, true_clock_s: float) -> float:
    """Recorder-clock correction from one event of known wall-clock time.

    ``observed_abs`` is the event's time on the recorder clock (absolute
    session seconds), ``true_clock_s`` the known true time (seconds since
    midnight). Returns the ``clock_offset_s`` value for
    ``calibration.json`` (positive = recorder clock was slow).
    """
    return float(true_clock_s - observed_abs % 86400.0)


def run_session(sess, out_dir) -> dict:
    """CLI driver: match cached event streams against civic periods.

    Uses broadband events (always) and rhythm strikes when a prior
    ``ambiscape rhythm`` run left ``rhythm.json`` phase data; writes
    ``schedule.json``.
    """
    import json
    from pathlib import Path
    from .analysis import detect_events
    from .features import load_features

    out_dir = Path(out_dir)
    F = load_features(sorted((out_dir / "features").glob("*.npz")))
    dt = float(np.median(np.diff(F["t_fast"])))
    events, _bg = detect_events(F["fast_db"], dt)
    t_ev = np.array([float(F["t_fast"][e["ipk"]]) for e in events])
    doc = {"events": match_periods(t_ev)[:4] if len(t_ev) >= 3 else []}
    (out_dir / "schedule.json").write_text(json.dumps(doc, indent=2))
    return doc
