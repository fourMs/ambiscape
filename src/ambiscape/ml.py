"""Machine-listening helpers (optional ``[ml]`` extra).

- **PANNs** (CNN14, AudioSet, 527 classes) tags 10-s windows around detected
  events and steady states; used by ``ambiscape draft`` to suggest object
  names in ``annotations.draft.json``.
- **silero-vad** estimates the fraction of speech in a file or span — the
  privacy gate to run on every excerpt before publishing (Freesound etc.).
- **BirdNET** (``birdnetlib``) identifies bird species in 3-s windows — the
  species layer for biophony, best run on the hi-fi windows the drone-free
  soundscape exposes (see :mod:`ambiscape.biophony` for the no-ML
  structural measures it confirms).

All models are trained on 16/32 kHz mono internet audio: the W (omni)
channel is downmixed and resampled, spatial information is not used, and
low-SNR domestic material is out of distribution — treat tags as
*suggestions to confirm by ear*, not ground truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_panns_model = None


def _resample(x: np.ndarray, fs: int, target: int) -> np.ndarray:
    if fs == target:
        return x
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(fs, target)
    return resample_poly(x, target // g, fs // g)


def panns_available() -> bool:
    try:
        import panns_inference  # noqa: F401
        return True
    except ImportError:
        return False


def tag_window(x: np.ndarray, fs: int, top_k: int = 3,
               min_prob: float = 0.10) -> list[dict]:
    """AudioSet tags for one mono window via PANNs CNN14 (32 kHz input)."""
    global _panns_model
    from panns_inference import AudioTagging, labels
    if _panns_model is None:
        _panns_model = AudioTagging(checkpoint_path=None, device="cpu")
    y = _resample(x.astype(np.float32), fs, 32000)
    clip = np.clip(y, -1, 1)[None, :]
    clipwise, _emb = _panns_model.inference(clip)
    probs = np.asarray(clipwise)[0]
    order = np.argsort(probs)[::-1][:top_k]
    return [{"label": labels[i], "p": round(float(probs[i]), 2)}
            for i in order if probs[i] >= min_prob]


def birdnet_available() -> bool:
    try:
        import birdnetlib  # noqa: F401
        return True
    except ImportError:
        return False


_birdnet_analyzer = None


def birdnet_window(x: np.ndarray, fs: int, lat: float | None = None,
                   lon: float | None = None, week: int = -1,
                   min_conf: float = 0.25) -> list[dict]:
    """BirdNET species detections for one mono window (48 kHz input).

    Analyzes the W channel resampled to 48 kHz. ``lat``/``lon`` and
    ``week`` (1–48, ISO-ish) enable BirdNET's location/season species
    filter — pass the session's coordinates to cut false positives.
    Returns ``[{"species", "common_name", "confidence"}]`` above
    ``min_conf``. Requires the ``[ml]`` extra plus ``birdnetlib``.
    """
    global _birdnet_analyzer
    import tempfile
    import soundfile as sf
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer
    if _birdnet_analyzer is None:
        _birdnet_analyzer = Analyzer()
    y = _resample(x.astype(np.float32), fs, 48000)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, np.clip(y, -1, 1), 48000)
        kw = dict(min_conf=min_conf, week_48=week)
        if lat is not None and lon is not None:
            kw.update(lat=lat, lon=lon)
        rec = Recording(_birdnet_analyzer, tmp.name, **kw)
        rec.analyze()
    return [{"species": d["scientific_name"],
             "common_name": d["common_name"],
             "confidence": round(float(d["confidence"]), 2)}
            for d in rec.detections]


def birdnet_session(sess, F=None, windows=None, win_s: float = 9.0,
                    hifi_max_diffuse: float | None = None,
                    lat: float | None = None, lon: float | None = None,
                    min_conf: float = 0.25) -> dict:
    """Run BirdNET across a session, optionally only on hi-fi windows.

    ``windows`` is an explicit list of absolute start seconds; if omitted,
    the session is tiled in ``win_s`` steps. When ``F`` (cached features)
    and ``hifi_max_diffuse`` are given, windows whose median diffuseness
    exceeds the threshold are skipped — a cheap "is the room masked?" gate
    so BirdNET runs where birds are actually legible, not under a drone.
    Returns per-window detections and an aggregated species tally.
    """
    from .io import read_span
    if windows is None:
        windows = []
        for tk in sess.takes:
            t = tk.start + 1.0
            while t + win_s <= tk.end:
                windows.append(t)
                t += win_s
    tally: dict[str, dict] = {}
    per_window = []
    for t0 in windows:
        if F is not None and hifi_max_diffuse is not None:
            i0 = int(np.searchsorted(F["t"], t0))
            i1 = int(np.searchsorted(F["t"], t0 + win_s))
            if i1 > i0 and float(np.median(F["diffuse"][i0:i1])) > \
                    hifi_max_diffuse:
                continue
        x, fs = read_span(sess, t0, win_s)
        dets = birdnet_window(x[:, 0], fs, lat=lat, lon=lon,
                              min_conf=min_conf)
        if dets:
            per_window.append({"t0_s": float(t0), "detections": dets})
            for d in dets:
                e = tally.setdefault(d["species"], {
                    "common_name": d["common_name"], "n": 0, "max_conf": 0.0})
                e["n"] += 1
                e["max_conf"] = max(e["max_conf"], d["confidence"])
    species = sorted(({"species": k, **v} for k, v in tally.items()),
                     key=lambda s: (-s["n"], -s["max_conf"]))
    return {"n_windows_analyzed": len(windows),
            "n_windows_with_birds": len(per_window),
            "n_species": len(species),
            "species": species, "windows": per_window}


def speech_fraction(x: np.ndarray, fs: int) -> dict:
    """silero-vad speech statistics for one mono signal."""
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    y = _resample(x.astype(np.float32), fs, 16000)
    ts = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(y)),
                               model, sampling_rate=16000)
    dur = len(y) / 16000
    speech = sum((t["end"] - t["start"]) for t in ts) / 16000
    return {"duration_s": round(dur, 1),
            "speech_s": round(speech, 1),
            "speech_fraction": round(speech / dur, 4) if dur else 0.0,
            "n_speech_segments": len(ts),
            "first_speech_at_s": round(ts[0]["start"] / 16000, 1) if ts else None}


def speech_gate(path: str | Path, threshold: float = 0.01) -> dict:
    """Privacy gate for a WAV file (any channel count; W/ch0 is analyzed).

    Returns the speech statistics plus a pass/fail verdict against
    `threshold` (default: fail if more than 1 % of the file is speech).
    """
    import soundfile as sf
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    res = speech_fraction(x[:, 0], fs)
    res["file"] = str(path)
    res["passes"] = res["speech_fraction"] <= threshold
    return res
