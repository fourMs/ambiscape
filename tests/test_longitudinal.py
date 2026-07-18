"""Longitudinal tests: trend + seasonal decomposition of dated summaries."""
import datetime as dt
import json

import numpy as np
import pytest

from ambiscape import longitudinal as L


def _dates(n, start=dt.date(2024, 1, 1)):
    return [start + dt.timedelta(days=i) for i in range(n)]


def _series(n=730, trend_per_year=2.0, seas_amp=1.0, peak_month=7, seed=0):
    dates = _dates(n)
    t = np.array([d.toordinal() for d in dates], float)
    t0 = t[0]
    rng = np.random.default_rng(seed)
    doy = np.array([d.timetuple().tm_yday for d in dates])
    phase = 2 * np.pi * (doy - (peak_month - 1) * 30.4) / 365.25
    y = (trend_per_year * (t - t0) / 365.25
         + seas_amp * np.cos(phase)
         + 0.05 * rng.standard_normal(n))
    return dates, y


# ---------------------------------------------------------------- collect

def _corpus(tmp_path, key="ndsi"):
    dates, y = _series(120)
    for d, v in zip(dates, y):
        folder = tmp_path / f"{d.isoformat()}_room"
        (folder / "analysis").mkdir(parents=True)
        (folder / "analysis" / "summary.json").write_text(
            json.dumps({"date": d.isoformat(), key: round(float(v), 4),
                        "leq_dbfs": -40.0}))
    return tmp_path


def test_collect_series_orders_by_date(tmp_path):
    s = L.collect_series(_corpus(tmp_path), keys=["ndsi"])
    assert len(s["dates"]) == 120
    assert s["dates"] == sorted(s["dates"])
    assert "ndsi" in s["series"] and len(s["series"]["ndsi"]) == 120


def test_collect_series_folder_name_date_fallback(tmp_path):
    # summary without a date field -> parse from folder name (ISO and compact)
    for name in ("2025-11-30_a", "20260111_b"):
        d = tmp_path / name / "analysis"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"ndsi": 0.1}))
    s = L.collect_series(tmp_path, keys=["ndsi"])
    assert [d.isoformat() for d in s["dates"]] == ["2025-11-30", "2026-01-11"]


# ---------------------------------------------------------------- decompose

def test_decompose_recovers_seasonal_and_trend():
    dates, y = _series(730, trend_per_year=2.0, seas_amp=1.0, peak_month=7)
    dec = L.decompose(dates, y)                  # default ~1-year trend window
    # trend increases across the two years
    assert dec["trend"][-1] > dec["trend"][0] + 2.0
    clim = dec["climatology"]
    peak = max(clim, key=clim.get)
    assert peak in (6, 7, 8)                    # summer peak
    assert max(clim.values()) - min(clim.values()) == pytest.approx(2.0,
                                                                    abs=0.6)
    # residual is small relative to the signal (amplitude ~2, trend ~4)
    assert np.std(dec["residual"]) < 0.4


def test_seasonal_climatology_counts():
    dates, y = _series(365)
    clim, counts = L.seasonal_climatology(dates, y)
    assert set(clim) == set(range(1, 13))
    assert sum(counts.values()) == 365


def test_trend_slope_sign_and_magnitude():
    dates, y = _series(730, trend_per_year=3.0, seas_amp=0.0, seed=1)
    slope = L.trend_slope(dates, y)
    assert slope == pytest.approx(3.0, abs=0.3)


# ---------------------------------------------------------------- summary

def test_summarize_longitudinal_keys():
    dates, y = _series(730, trend_per_year=1.5, seas_amp=2.0, peak_month=7)
    s = L.summarize_longitudinal(dates, y)
    assert s["n"] == 730
    assert s["trend_per_year"] == pytest.approx(1.5, abs=0.4)
    assert s["seasonal_amplitude"] == pytest.approx(4.0, abs=1.0)
    assert s["peak_month"] in (6, 7, 8)
    assert s["span_days"] == pytest.approx(729, abs=1)


def test_decompose_empty_is_safe():
    assert L.summarize_longitudinal([], [])["n"] == 0
