"""Geophony measures: wind, rain and water by acoustic structure.

The non-biological, non-human ground of a soundscape. Two structures the
cached features expose (no audio pass):

- **wind** -- low-frequency (below ~200 Hz) energy that is diffuse and
  non-directional: wind on the capsules is incoherent between channels, so
  ambisonic diffuseness rises. Measured as low-band energy weighted by median
  diffuseness (falls back to plain low-band energy for stereo/mono);
- **rain / water** -- broadband high-frequency hiss that is spectrally flat
  and temporally steady (a shower, a fountain, a stream). Measured as
  high-band (2-16 kHz) energy times median spectral flatness.

Caveats: proxies, not detection. Wind and HVAC rumble share the low band; rain,
applause and frying share the flat high-band hiss. Diffuseness disambiguates
wind only for ambisonic input. Treat the indices as candidates to confirm by
ear.
"""
from __future__ import annotations

import numpy as np

from .features import OCT_CENTERS

EPS = 1e-12
LOWBAND_HZ = 200.0
HIGH_BAND = (2000.0, 16000.0)


def _oct_frac(F, lo, hi):
    c = np.asarray(OCT_CENTERS, float)
    m = (c >= lo) & (c <= hi)
    op = np.asarray(F["oct_pow"], float)
    return float(op[:, m].sum() / (op.sum() + EPS))


def _median_diffuse(F):
    d = np.asarray(F.get("diffuse"), float)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else None


def wind_index(F: dict) -> float:
    lf = _oct_frac(F, 0.0, LOWBAND_HZ)
    diff = _median_diffuse(F)
    return float(np.clip(lf * diff if diff is not None else lf, 0.0, 1.0))


def rain_index(F: dict) -> float:
    hf = _oct_frac(F, *HIGH_BAND)
    flat = np.asarray(F.get("flatness"), float)
    flat = flat[np.isfinite(flat)]
    fl = float(np.median(flat)) if flat.size else 0.0
    return float(np.clip(hf * fl, 0.0, 1.0))


def summarize_geophony(F: dict) -> dict:
    """Geophony (wind / rain / water) descriptors for the analyze summary."""
    lf = _oct_frac(F, 0.0, LOWBAND_HZ)
    hf = _oct_frac(F, *HIGH_BAND)
    wind = wind_index(F)
    rain = rain_index(F)
    return {
        "geo_lowfreq_fraction": round(lf, 3),
        "geo_highband_fraction": round(hf, 3),
        "geo_wind_index": round(wind, 3),
        "geo_rain_index": round(rain, 3),
        "geophony_index": round(max(wind, rain), 3),
    }
