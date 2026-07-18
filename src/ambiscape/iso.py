"""ISO 12913-3-style psychoacoustic indicators + level calibration.

Calibration
-----------
A session is calibrated by ``<folder>/calibration.json``::

    {"dbfs_to_dbspl": 94.0,
     "method": "SPL app next to mic, air pump running, LAeq 42 dB",
     "date": "2026-07-16"}

``dbfs_to_dbspl`` is the offset O such that a signal at −X dBFS corresponds
to (O − X) dB SPL. With it, dBFS descriptors become dB SPL and waveforms
convert to pascals for psychoacoustic metrics.

The same file may carry ``clock_offset_s`` — seconds added to every take's
start time when the recorder clock was found to be off (positive = clock was
slow; e.g. calibrated against a known external time reference). Applied in
:func:`ambiscape.io.open_session`, so all clock-labeled outputs agree. Both
keys are optional.

Indicators (via MoSQITo, optional dependency)
---------------------------------------------
ISO 532-1 time-varying loudness (N5, N50), DIN 45692 sharpness, and
Daniel & Weber roughness, computed per ear on a binaural render of the
B-format signal. If `ambiviz` (with its HRIR-based binauralizer) is
installed it is used; otherwise a documented fallback renders a
back-to-back cardioid pair at ±90° — a pseudo-binaural approximation
without pinna/ILD spectral cues. Uncalibrated sessions are computed with an
assumed offset and flagged: absolute sone/acum values are then indicative
only (their *ratios* between segments remain meaningful).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

P_REF = 20e-6
ASSUMED_OFFSET = 94.0  # used (and flagged) when no calibration.json exists


def load_calibration(folder: str | Path) -> dict | None:
    p = Path(folder) / "calibration.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def to_pascal(x: np.ndarray, dbfs_to_dbspl: float) -> np.ndarray:
    return x.astype(np.float64) * P_REF * 10 ** (dbfs_to_dbspl / 20)


def apply_calibration(summary: dict, cal: dict) -> dict:
    """Add dB SPL versions of the level descriptors to a summary dict."""
    off = float(cal["dbfs_to_dbspl"])
    out = dict(summary)
    for key in ("leq_dbfs", "laeq_dbfs", "L10", "L50", "L90"):
        if key in summary and summary[key] is not None:
            out[key.replace("_dbfs", "") + "_db_spl"] = round(summary[key] + off, 1)
    out["calibration"] = {"dbfs_to_dbspl": off,
                          "method": cal.get("method", "")}
    return out


def binaural(x4: np.ndarray, fs: int) -> tuple[np.ndarray, str]:
    """FOA (W,Y,Z,X columns) -> stereo ear signals.

    Tries ambiviz's HRIR binauralizer; falls back to a ±90° cardioid pair
    (no pinna cues). Returns (n, 2) array and the method name.
    """
    try:
        from ambiviz.ambisonics.binauralizer import binauralize  # type: ignore
        y = binauralize(x4.T, fs)  # ambiviz convention: channels first
        return np.asarray(y).T[:, :2], "ambiviz-hrir"
    except Exception:
        w, ych = x4[:, 0], x4[:, 1]
        left = 0.5 * (w + ych)
        right = 0.5 * (w - ych)
        return np.stack([left, right], axis=1), "cardioid-pair-fallback"


def indicators(x_pa: np.ndarray, fs: int, rough_dur: float = 10.0) -> dict:
    """ISO 532-1 loudness (N5/N50), DIN 45692 sharpness, D&W roughness
    for one calibrated (pascal) channel.

    MoSQITo runs ~5x slower than realtime, and roughness is the costliest
    metric; it is therefore computed on a central `rough_dur`-second slice
    (roughness is a texture measure and stabilizes within seconds).
    """
    from mosqito.sq_metrics import (loudness_zwtv,
                                    sharpness_din_from_loudness,
                                    roughness_dw)
    N, N_spec, _bark, _t = loudness_zwtv(x_pa, fs, field_type="diffuse")
    S = sharpness_din_from_loudness(N, N_spec)
    n_r = int(rough_dur * fs)
    mid = max(0, (len(x_pa) - n_r) // 2)
    R = np.atleast_1d(roughness_dw(x_pa[mid:mid + n_r], fs)[0])
    return {
        "N5_sone": round(float(np.percentile(N, 95)), 2),
        "N50_sone": round(float(np.percentile(N, 50)), 2),
        "sharpness_median_acum": round(float(np.median(S)), 2),
        "roughness_median_asper": round(float(np.median(R)), 3),
    }


def segment_indicators(sess, F: dict, folder: str | Path,
                       dur: float = 30.0, offset: float | None = None) -> dict:
    """Compute per-ear indicators on representative segments.

    Segments come from analysis.pick_segments (typical / quietest /
    most_active / transition); `dur` seconds from the start of each.
    """
    from .analysis import pick_segments
    from .io import read_span

    cal = load_calibration(folder)
    has_spl = bool(cal and "dbfs_to_dbspl" in cal)
    if offset is None:
        offset = float(cal["dbfs_to_dbspl"]) if has_spl else ASSUMED_OFFSET
    calibrated = has_spl or offset != ASSUMED_OFFSET

    out = {"calibrated": calibrated, "dbfs_to_dbspl": offset,
           "field_type": "diffuse", "segments": {}}
    if not calibrated:
        out["warning"] = (f"no calibration.json — assumed offset "
                          f"{ASSUMED_OFFSET} dB; absolute values indicative only")
    for pick in pick_segments(F, seg_s=dur):
        try:
            x, fs = read_span(sess, pick["t0"], dur)
        except ValueError:
            continue
        ears, method = binaural(x, fs)
        seg = {"t0": sess.clock(pick["t0"]), "dur_s": dur,
               "binaural_method": method}
        for ch, name in ((0, "left"), (1, "right")):
            seg[name] = indicators(to_pascal(ears[:, ch], offset), fs)
        seg["N5_sone_max_ear"] = max(seg["left"]["N5_sone"],
                                     seg["right"]["N5_sone"])
        out["segments"][pick["kind"]] = seg
    return out


# ------------------------------------------------------- room noise criteria

NR_A = {31.5: 55.4, 63: 35.5, 125: 22.0, 250: 12.0, 500: 4.8,
        1000: 0.0, 2000: -3.5, 4000: -6.1, 8000: -8.0}
NR_B = {31.5: 0.681, 63: 0.790, 125: 0.870, 250: 0.930, 500: 0.974,
        1000: 1.000, 2000: 1.015, 4000: 1.025, 8000: 1.030}
# ANSI S12.2 tangent NC curves, octave levels 63 Hz .. 8 kHz per NC value
NC_TABLE = {
    15: (47, 36, 29, 22, 17, 14, 12, 11),
    20: (51, 40, 33, 26, 22, 19, 17, 16),
    25: (54, 44, 37, 31, 27, 24, 22, 21),
    30: (57, 48, 41, 35, 31, 29, 28, 27),
    35: (60, 52, 45, 40, 36, 34, 33, 32),
    40: (64, 56, 50, 45, 41, 39, 38, 37),
    45: (67, 60, 54, 49, 46, 44, 43, 42),
    50: (71, 64, 58, 54, 51, 49, 48, 47),
    55: (74, 67, 62, 58, 56, 54, 53, 52),
    60: (77, 71, 67, 63, 61, 59, 58, 57),
    65: (80, 75, 71, 68, 66, 64, 63, 62),
}
NC_FREQS = (63, 125, 250, 500, 1000, 2000, 4000, 8000)


def room_criteria(oct_spl_db: dict) -> dict:
    """NR, NC, and RC ratings of an octave-band SPL spectrum.

    ``oct_spl_db`` maps octave center frequency (Hz) to band SPL (dB).
    Ratings are only physically meaningful for *calibrated* levels
    (``dbfs_to_dbspl`` in ``calibration.json``); on uncalibrated dBFS they
    are relative numbers, comparable within one recorder+gain setup only.

    - **NR** (ISO/R 1996 Noise Rating): analytic curves ``L = a + b*NR``;
      the rating is the highest per-band NR value and
      ``NR_governing_hz`` names the band that sets it.
    - **NC** (ANSI S12.2 Noise Criterion): tangency against the tabulated
      curves, linearly interpolated per band (63 Hz–8 kHz).
    - **RC** (Blazier Room Criterion, simplified): arithmetic mean of the
      500/1000/2000 Hz levels; the reference line has a −5 dB/octave slope
      through (1 kHz, RC). ``RC_class`` is "R" (rumble) when any
      31.5–250 Hz band exceeds the line by > 5 dB, "H" (hiss) when any
      2–4 kHz band exceeds it by > 3 dB, "RH" for both, "N" (neutral)
      otherwise.
    """
    spec = {float(k): float(v) for k, v in oct_spl_db.items()}

    nr_per = {f: (spec[f] - NR_A[f]) / NR_B[f] for f in NR_A if f in spec}
    f_gov = max(nr_per, key=nr_per.get)
    nr = nr_per[f_gov]

    nc = None
    ncs = sorted(NC_TABLE)
    per_band = []
    for i, f in enumerate(NC_FREQS):
        if f not in spec:
            continue
        levels = np.array([NC_TABLE[n][i] for n in ncs], float)
        per_band.append(float(np.interp(spec[f], levels, ncs)))
    if per_band:
        nc = max(per_band)

    rc = None
    rc_class = None
    if all(f in spec for f in (500.0, 1000.0, 2000.0)):
        rc = (spec[500.0] + spec[1000.0] + spec[2000.0]) / 3
        ref = {f: rc + 5 * np.log2(1000.0 / f) for f in spec}
        rumble = any(spec[f] > ref[f] + 5 for f in (31.5, 63.0, 125.0, 250.0)
                     if f in spec)
        hiss = any(spec[f] > ref[f] + 3 for f in (2000.0, 4000.0)
                   if f in spec)
        rc_class = ("RH" if rumble and hiss else
                    "R" if rumble else "H" if hiss else "N")

    return {"NR": round(nr, 1), "NR_governing_hz": int(f_gov),
            "NC": round(nc, 1) if nc is not None else None,
            "RC": round(rc, 1) if rc is not None else None,
            "RC_class": rc_class}


def background_octaves_db(F: dict, pct: float = 50.0,
                          offset_db: float = 0.0) -> dict:
    """Per-octave percentile level (dB) from cached features, for
    :func:`room_criteria`. ``offset_db`` is the dBFS→dB SPL calibration
    offset (0 keeps uncalibrated dBFS)."""
    from .features import OCT_CENTERS
    lv = 10 * np.log10(np.asarray(F["oct_pow"], float) + 1e-20) + offset_db
    return {c: float(np.percentile(lv[:, i], pct))
            for i, c in enumerate(OCT_CENTERS) if c <= 8000}
