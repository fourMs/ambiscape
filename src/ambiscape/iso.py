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
        cal = json.loads(p.read_text())
        if "dbfs_to_dbspl" not in cal:
            raise ValueError(f"{p}: missing 'dbfs_to_dbspl'")
        return cal
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
    if offset is None:
        offset = float(cal["dbfs_to_dbspl"]) if cal else ASSUMED_OFFSET
    calibrated = cal is not None

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
