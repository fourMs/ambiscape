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
    """Whether `panns_inference` can be imported, so tagging can be skipped rather than fail.

    The model packages are optional dependencies: the analysis runs without them and simply
    omits the tags.
    """
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


def tag_probabilities(x: np.ndarray, fs: int,
                      wanted: list[str] | tuple[str, ...] | None = None
                      ) -> dict[str, float]:
    """Probability of each named AudioSet class for one mono window.

    :func:`tag_window` returns the few labels that came top and cleared a
    threshold, which is what naming an object wants. Asking a *specific*
    question needs the opposite: the probability of a class you name, whether
    or not it reached the top three. "How much speech and how much music is in
    this window" is that kind of question, and a window can be plainly musical
    while ``Music`` sits fourth behind three instrument labels.

    ``wanted`` is a list of AudioSet label strings; omit it for all 527.
    Unknown labels raise rather than returning silently empty, because a typo
    in a class name is otherwise indistinguishable from a class that never
    fires.

    The caveat on the module applies with force here. These are AudioSet
    posteriors from a model trained on internet video, read off a domestic
    recording at some distance from the source; they order windows usefully
    and they are not calibrated probabilities of anything. Compare them
    against each other, not against 0.5.
    """
    from panns_inference import labels as _labels
    global _panns_model
    from panns_inference import AudioTagging
    if _panns_model is None:
        _panns_model = AudioTagging(checkpoint_path=None, device="cpu")
    if wanted is not None:
        unknown = [w for w in wanted if w not in _labels]
        if unknown:
            raise ValueError(f"not AudioSet labels: {unknown}")
    y = _resample(x.astype(np.float32), fs, 32000)
    clip = np.clip(y, -1, 1)[None, :]
    clipwise, _emb = _panns_model.inference(clip)
    probs = np.asarray(clipwise)[0]
    names = wanted if wanted is not None else _labels
    idx = {lab: i for i, lab in enumerate(_labels)}
    return {lab: float(probs[idx[lab]]) for lab in names}


def birdnet_available() -> bool:
    """Whether `birdnetlib` can be imported. Optional, like `panns_available`."""
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


#: RMS the detector's input is scaled to (about -26 dBFS).
_VAD_TARGET_RMS = 0.05


def _vad_input(x: np.ndarray, target_rms: float = _VAD_TARGET_RMS):
    """Scale a signal to a fixed RMS before voice-activity detection.

    silero applies a fixed probability threshold to whatever level arrives,
    so without this the result is a function of recording gain as much as of
    speech. Measured on one minute of a real recording: 0.513 speech at
    unity, 0.289 at -12 dB, 0.110 at -24 dB, 0.000 at -30 dB — the same
    conversation, four answers. Uncalibrated recorders are then not
    comparable with each other, which is how a gain difference between
    microphones becomes an apparent difference between rooms.

    Digital silence is returned untouched: there is no level to normalise
    to, and amplifying it would manufacture noise for the detector to find.
    """
    x = np.asarray(x, np.float64)
    rms = float(np.sqrt((x ** 2).mean())) if x.size else 0.0
    if rms <= 0:
        return x
    y = x * (target_rms / rms)
    peak = float(np.abs(y).max())
    if peak > 1.0:                       # keep inside the model's range
        y = y / peak
    return y


def speech_fraction(x: np.ndarray, fs: int, normalize: bool = True) -> dict:
    """silero-vad speech statistics for one mono signal.

    ``normalize`` scales the input to a fixed RMS first, so the result
    describes speech rather than recording gain — see :func:`_vad_input`.
    Pass ``normalize=False`` to reproduce numbers computed before 0.29.0.
    """
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    if normalize:
        x = _vad_input(x)
    y = _resample(np.asarray(x, np.float32), fs, 16000)
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
    """Privacy gate for a WAV file (any channel count; W/ch0 is analysed).

    Returns the speech statistics plus a pass/fail verdict against
    `threshold` (default: fail if more than 1 % of the file is speech).
    """
    import soundfile as sf
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    res = speech_fraction(x[:, 0], fs)
    res["file"] = str(path)
    res["passes"] = res["speech_fraction"] <= threshold
    return res
