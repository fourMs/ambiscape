"""Mechanical & transport measures: engines, machinery, rail and road traffic.

These sources are broadband but low-frequency-weighted, and temporally steady
or quasi-periodic (engine firing, bogies over rail joints, a compressor duty
cycle). No single ambiscape module owned them before -- the pieces were
scattered across rhythm, spatial, states and compare. This module reads the
cached features (no audio pass) and returns a compact mechanical/transport
descriptor for the analyze summary:

- **low-frequency fraction** and **rumble level** from the octave powers
  (traffic and machinery pile energy below ~250 Hz);
- an **envelope periodicity** peak from the 50 Hz broadband envelope -- the
  engine/bogie/duty-cycle rhythm (0.3-12 Hz) and how strongly it stands out.

Caveats: acoustic-structure proxies, not detections. Indoor HVAC rumble and a
passing lorry look alike here, and a steady tonal machine may also light up
:mod:`ambiscape.enf` (mains hum). For directional confirmation use the
pass-by view in :mod:`ambiscape.spatial`; for pitched machinery, the strike
periodicity in :mod:`ambiscape.rhythm`.
"""
from __future__ import annotations

import numpy as np

from .features import OCT_CENTERS

EPS = 1e-12
LOWBAND_HZ = 250.0          # traffic / machinery weight below here
RUMBLE = (31.5, 125.0)


def _oct_frac(F, lo, hi):
    c = np.asarray(OCT_CENTERS, float)
    m = (c >= lo) & (c <= hi)
    op = np.asarray(F["oct_pow"], float)
    return op[:, m].sum(1), op.sum(1)


def low_frequency_fraction(F: dict) -> float:
    band, tot = _oct_frac(F, 0.0, LOWBAND_HZ)
    return float(band.sum() / (tot.sum() + EPS))


def rumble_level_db(F: dict) -> float:
    band, _ = _oct_frac(F, *RUMBLE)
    return float(10 * np.log10(band.mean() + EPS))


def envelope_periodicity(F: dict, band=(0.3, 12.0)) -> dict:
    """Peak frequency and prominence of the broadband envelope modulation."""
    env = np.asarray(F.get("env_hi"), float)
    dt = float(F.get("hi_dt", 0.02))
    if env.size < 16:
        return {"hz": None, "strength": 0.0}
    x = env - env.mean()
    fr = np.fft.rfftfreq(len(x), dt)
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    sel = (fr >= band[0]) & (fr <= band[1])
    if not sel.any() or P[sel].sum() <= 0:
        return {"hz": None, "strength": 0.0}
    i = int(np.argmax(P[sel]))
    strength = float(P[sel][i] / (P[1:].sum() + EPS))   # peak vs total AC power
    return {"hz": round(float(fr[sel][i]), 2), "strength": round(strength, 3)}


def summarize_mechanical(F: dict) -> dict:
    """Mechanical/transport descriptors for the analyze summary."""
    lf = low_frequency_fraction(F)
    per = envelope_periodicity(F)
    index = float(np.clip(lf * (0.5 + per["strength"]), 0.0, 1.0))
    return {
        "mech_lowfreq_fraction": round(lf, 3),
        "mech_rumble_db": round(rumble_level_db(F), 1),
        "mech_periodicity_hz": per["hz"],
        "mech_periodicity_strength": per["strength"],
        "mechanical_index": round(index, 3),
    }
