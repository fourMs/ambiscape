"""Machine-listening helpers (optional ``[ml]`` extra).

- **PANNs** (CNN14, AudioSet, 527 classes) tags 10-s windows around detected
  events and steady states; used by ``ambiscape draft`` to suggest object
  names in ``annotations.draft.json``.
- **silero-vad** estimates the fraction of speech in a file or span — the
  privacy gate to run on every excerpt before publishing (Freesound etc.).

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
