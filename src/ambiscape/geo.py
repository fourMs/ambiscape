"""GPS tracks for walk mode: GPX in, places and speeds out.

A soundwalk's zones live in time; a GPX track turns them into places.
Parsing is standard-library only (GPX is plain XML), distances are
haversine, and a zone's speed is its track distance over its duration —
an independent cross-check on the acoustic step cadence.
"""
from __future__ import annotations

import datetime as _dt
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

EARTH_R_M = 6371000.0


def load_gpx(path: str | Path) -> dict:
    """Track points from a GPX file: ``{"t", "lat", "lon", "ele"}``.

    ``t`` is epoch seconds (GPX times are ISO 8601, usually UTC). Points
    without a timestamp are dropped — an untimed track cannot be joined
    to a recording.
    """
    root = ET.parse(str(path)).getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag[:root.tag.index("}") + 1]
    t, lat, lon, ele = [], [], [], []
    for pt in root.iter(f"{ns}trkpt"):
        tm = pt.find(f"{ns}time")
        if tm is None or tm.text is None:
            continue
        stamp = _dt.datetime.fromisoformat(tm.text.replace("Z", "+00:00"))
        t.append(stamp.timestamp())
        lat.append(float(pt.attrib["lat"]))
        lon.append(float(pt.attrib["lon"]))
        el = pt.find(f"{ns}ele")
        ele.append(float(el.text) if el is not None and el.text else np.nan)
    if not t:
        raise ValueError(f"no timestamped track points in {path}")
    order = np.argsort(t)
    return {"t": np.asarray(t, float)[order],
            "lat": np.asarray(lat, float)[order],
            "lon": np.asarray(lon, float)[order],
            "ele": np.asarray(ele, float)[order]}


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres (vectorised)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return EARTH_R_M * 2 * np.arcsin(np.sqrt(a))


def zone_geo(track: dict, t0: float, t1: float) -> dict:
    """Distance, mean speed and midpoint position for [t0, t1] epoch seconds.

    The track is resampled to 1 Hz inside the span so distance does not
    depend on the logger's point spacing. Returns ``None`` values when the
    span falls outside the track.
    """
    tt = track["t"]
    if t1 <= tt[0] or t0 >= tt[-1]:
        return {"distance_m": None, "speed_ms": None,
                "lat": None, "lon": None}
    a, b = max(t0, tt[0]), min(t1, tt[-1])
    ts = np.arange(a, b + 1e-9, 1.0)
    la = np.interp(ts, tt, track["lat"])
    lo = np.interp(ts, tt, track["lon"])
    d = float(haversine_m(la[:-1], lo[:-1], la[1:], lo[1:]).sum()) if len(ts) > 1 else 0.0
    dur = t1 - t0
    mid = (t0 + t1) / 2
    return {"distance_m": round(d, 1),
            "speed_ms": round(d / dur, 2) if dur > 0 else None,
            "lat": round(float(np.interp(mid, tt, track["lat"])), 6),
            "lon": round(float(np.interp(mid, tt, track["lon"])), 6)}
