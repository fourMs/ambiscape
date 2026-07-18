"""Longitudinal analysis: how a place sounds across weeks, months, a year.

The unit here is the *dated session summary*, not the audio. A year-long
study is best run as many short sessions---one a day, say, as in the
StillStanding archive---each analyzed to a small ``summary.json``; a year is
then 365 tiny rows, so the longitudinal analysis is inherently out-of-core
however large the underlying audio was. (A single *continuous* multi-month
recording exceeds what the in-memory feature pipeline can hold; the
supported path for year-scale work is to segment it into per-day sessions
first.)

- ``collect_series`` --- read every session's summary under a corpus,
  ordered by date (from a ``date`` field, else parsed from the folder name),
  into per-descriptor time series;
- ``decompose`` --- additive split of one descriptor into a slow **trend**
  (day-windowed rolling median), a repeating **seasonal** component
  (monthly climatology of the detrended series), and the **residual**;
- ``seasonal_climatology`` / ``trend_slope`` --- the two components on their
  own (per-month means; long-term change per year);
- ``summarize_longitudinal`` --- trend per year, seasonal amplitude, peak and
  trough months, span;
- ``render`` --- a figure: the descriptor over time with its trend, plus the
  monthly climatology.

Everything is numpy-only. The motivating example is already in the
StillStanding data: bird mentions peak in July and fall to zero in winter---
not because the birds leave, but because the windows close.
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

import numpy as np

_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_COMPACT = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)")


def parse_date(name: str) -> _dt.date | None:
    """Parse a leading ``YYYY-MM-DD`` or ``YYYYMMDD`` date from a string."""
    for rx in (_ISO, _COMPACT):
        m = rx.search(name)
        if m:
            try:
                return _dt.date(int(m[1]), int(m[2]), int(m[3]))
            except ValueError:
                continue
    return None


def collect_series(corpus_dir: str | Path, keys=None,
                   pattern: str = "*/analysis/summary.json") -> dict:
    """Dated, date-ordered descriptor time series from a corpus of sessions.

    Each session's date comes from the summary's ``date`` field if present,
    otherwise from a date parsed out of the session folder name; sessions
    with no resolvable date are skipped. Returns ``{"dates": [date, ...],
    "sessions": [name, ...], "series": {key: np.array}}`` with all arrays in
    date order. ``keys`` limits the descriptors (default: the union across
    sessions).
    """
    import json
    corpus_dir = Path(corpus_dir)
    rows = []
    for p in sorted(corpus_dir.glob(pattern)):
        name = p.parent.parent.name
        try:
            summ = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        d = None
        if isinstance(summ.get("date"), str):
            d = parse_date(summ["date"])
        if d is None:
            d = parse_date(name)
        if d is not None:
            rows.append((d, name, summ))
    rows.sort(key=lambda r: r[0])
    if keys is None:
        keys, seen = [], set()
        for _d, _n, s in rows:
            for k, v in s.items():
                if k not in seen and isinstance(v, (int, float)):
                    seen.add(k)
                    keys.append(k)
    series = {}
    for k in keys:
        series[k] = np.array([float(s[k]) if isinstance(s.get(k), (int, float))
                              else np.nan for _d, _n, s in rows])
    return {"dates": [r[0] for r in rows],
            "sessions": [r[1] for r in rows], "series": series}


def _ordinals(dates):
    return np.array([d.toordinal() for d in dates], float)


def _finite(dates, values):
    y = np.asarray(values, float)
    m = np.isfinite(y)
    return [d for d, ok in zip(dates, m) if ok], y[m]


def rolling_trend(dates, values, window_days: float = 365.0) -> np.ndarray:
    """Day-windowed rolling median: a slow trend robust to spikes and to
    irregular sampling (each point is the median of all points within
    ``window_days`` of it)."""
    t = _ordinals(dates)
    y = np.asarray(values, float)
    out = np.empty(len(y))
    for i in range(len(y)):
        m = np.abs(t - t[i]) <= window_days / 2
        out[i] = np.median(y[m])
    return out


def seasonal_climatology(dates, values):
    """Per-calendar-month mean and count of a series (month 1..12)."""
    ds, y = _finite(dates, values)
    months = np.array([d.month for d in ds])
    clim, counts = {}, {}
    for m in range(1, 13):
        sel = months == m
        counts[m] = int(sel.sum())
        clim[m] = float(y[sel].mean()) if sel.any() else np.nan
    return clim, counts


def decompose(dates, values, window_days: float = 365.0) -> dict:
    """Additive decomposition: trend + seasonal (monthly) + residual.

    Returns date-ordered arrays plus the monthly ``climatology`` (mean of the
    detrended series per calendar month, mean-centered so the seasonal
    component sums to ~0).
    """
    ds, y = _finite(dates, values)
    order = np.argsort(_ordinals(ds))
    ds = [ds[i] for i in order]
    y = y[order]
    trend = rolling_trend(ds, y, window_days)
    detr = y - trend
    clim, _counts = seasonal_climatology(ds, detr)
    vals = [v for v in clim.values() if np.isfinite(v)]
    center = float(np.mean(vals)) if vals else 0.0
    clim = {m: (v - center if np.isfinite(v) else 0.0) for m, v in clim.items()}
    months = np.array([d.month for d in ds])
    seasonal = np.array([clim[m] for m in months])
    return {"dates": ds, "trend": trend, "seasonal": seasonal,
            "residual": y - trend - seasonal, "values": y,
            "climatology": clim}


def trend_slope(dates, values) -> float:
    """Long-term linear change per year (least-squares slope × 365.25)."""
    ds, y = _finite(dates, values)
    if len(y) < 2:
        return 0.0
    t = _ordinals(ds)
    A = np.vstack([t - t[0], np.ones(len(t))]).T
    slope = np.linalg.lstsq(A, y, rcond=None)[0][0]
    return float(slope * 365.25)


def summarize_longitudinal(dates, values, window_days: float = 365.0) -> dict:
    """Trend/seasonal descriptors of one dated series."""
    ds, y = _finite(dates, values)
    if len(y) == 0:
        return {"n": 0}
    dec = decompose(ds, y, window_days)
    clim = {m: v for m, v in dec["climatology"].items()}
    finite_clim = {m: v for m, v in clim.items()
                   if seasonal_climatology(ds, y)[1][m] > 0}
    t = _ordinals(dec["dates"])
    return {
        "n": int(len(y)),
        "span_days": int(t[-1] - t[0]),
        "trend_per_year": round(trend_slope(ds, y), 3),
        "seasonal_amplitude": round(
            max(finite_clim.values()) - min(finite_clim.values()), 3)
        if finite_clim else 0.0,
        "peak_month": max(finite_clim, key=finite_clim.get)
        if finite_clim else None,
        "trough_month": min(finite_clim, key=finite_clim.get)
        if finite_clim else None,
        "residual_sd": round(float(np.std(dec["residual"])), 3),
    }


def render(dates, values, out_path, key: str = "", window_days: float = 365.0):
    """Two-panel figure: series + rolling trend over time, and the monthly
    climatology."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .figures import RC, BLUE, YELLOW, MUT

    dec = decompose(dates, values, window_days)
    ds, y, trend = dec["dates"], dec["values"], dec["trend"]
    t = _ordinals(ds)
    clim = dec["climatology"]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=130,
                               gridspec_kw=dict(width_ratios=[2.6, 1]))
        ax[0].plot(t, y, ".", ms=3, color=BLUE, alpha=0.5, label=key or "value")
        ax[0].plot(t, trend, color=YELLOW, lw=2,
                   label=f"trend ({window_days:.0f}-day median)")
        yr = np.arange(np.ceil(ds[0].year), ds[-1].year + 1)
        ax[0].set_xticks([_dt.date(int(y_), 1, 1).toordinal() for y_ in yr],
                         [str(int(y_)) for y_ in yr])
        ax[0].set(ylabel=key or "descriptor",
                  title=f"{key} over time")
        ax[0].legend(fontsize=8)
        months = list(range(1, 13))
        ax[1].bar(months, [clim[m] for m in months], color=BLUE, alpha=0.7)
        ax[1].axhline(0, color=MUT, lw=0.7)
        ax[1].set_xticks(months, ["J", "F", "M", "A", "M", "J", "J", "A",
                                  "S", "O", "N", "D"], fontsize=8)
        ax[1].set(title="seasonal (monthly, detrended)")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return out_path


def run_corpus(corpus_dir, out_dir, keys=None, window_days: float = 365.0) -> dict:
    """CLI driver: per-descriptor longitudinal summaries + a figure for each,
    writing ``longitudinal.json``."""
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = collect_series(corpus_dir, keys=keys)
    doc = {"n_sessions": len(s["dates"]),
           "date_range": [s["dates"][0].isoformat(), s["dates"][-1].isoformat()]
           if s["dates"] else None,
           "descriptors": {}}
    for k, y in s["series"].items():
        if np.isfinite(y).sum() >= 3:
            doc["descriptors"][k] = summarize_longitudinal(s["dates"], y,
                                                           window_days)
    (out_dir / "longitudinal.json").write_text(json.dumps(doc, indent=2))
    return doc
