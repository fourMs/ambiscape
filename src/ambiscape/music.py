"""Librosa-based tempogram and chromagram (optional ``[music]`` extra).

Complements the built-in analyses with the MIR-standard views: the
**tempogram** (onset autocorrelation over time, in BPM — against which the
windowed-ACF tempogram in :mod:`rhythm` can be cross-checked) and the
**chromagram** (12-bin pitch-class energy over time, the time-resolved
counterpart of :func:`ambiscape.tonality.pitch_class_profile`).

Audio is read from the W channel and resampled to 22.05 kHz; long sessions
are fine (a 25 min file takes on the order of a minute). Requires
``pip install "ambiscape[music]"``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def _require_librosa():
    try:
        import librosa
        return librosa
    except ImportError as e:
        raise ImportError(
            "librosa is required: pip install 'ambiscape[music]'") from e


def load_w(take, t0=0.0, dur=None, sr=22050):
    """W channel of a take, resampled to ``sr``."""
    librosa = _require_librosa()
    fs = take.samplerate
    iW = take.wyzx[0]
    with sf.SoundFile(str(take.path)) as f:
        f.seek(int(t0 * fs))
        n = f.frames - int(t0 * fs) if dur is None else int(dur * fs)
        x = f.read(n, dtype="float32", always_2d=True)[:, iW]
    return librosa.resample(x, orig_sr=fs, target_sr=sr), sr


def tempogram(y, sr, hop=512, win_s=8.0):
    """Autocorrelation tempogram of the onset-strength envelope.

    Returns (times, bpm_axis, T, tempo_bpm): the tempogram plus librosa's
    global tempo estimate (which resolves the octave ambiguity a raw
    tempogram argmax suffers from).
    """
    librosa = _require_librosa()
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    win = int(win_s * sr / hop)
    T = librosa.feature.tempogram(onset_envelope=onset, sr=sr,
                                  hop_length=hop, win_length=win)
    times = librosa.frames_to_time(np.arange(T.shape[1]), sr=sr,
                                   hop_length=hop)
    bpm = librosa.tempo_frequencies(T.shape[0], sr=sr, hop_length=hop)
    try:
        from librosa.feature.rhythm import tempo as _tempo
    except ImportError:                      # librosa < 0.10
        _tempo = librosa.beat.tempo
    t_est = float(_tempo(onset_envelope=onset, sr=sr, hop_length=hop)[0])
    return times, bpm, T, t_est


def chromagram(y, sr, hop=512):
    """STFT chromagram (12 pitch classes over time)."""
    librosa = _require_librosa()
    C = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr,
                                   hop_length=hop)
    return times, C


def run_session(sess, out_dir, t0=0.0, dur=None) -> dict:
    """Tempogram + chromagram figure and summary for the first take."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    librosa = _require_librosa()
    from .tonality import NOTE

    out_dir = Path(out_dir)
    y, sr = load_w(sess.takes[0], t0=t0, dur=dur)
    tt, bpm, T, bpm_peak = tempogram(y, sr)
    tc, C = chromagram(y, sr)
    chroma_mean = C.mean(1)
    doc = {
        "tempo_bpm_global": round(bpm_peak, 1),
        "tempo_period_s": round(60.0 / bpm_peak, 3),
        "chroma_mean": {NOTE[i]: round(float(v / chroma_mean.sum()), 3)
                        for i, v in enumerate(chroma_mean)},
        "top_pitch_classes": [NOTE[i]
                              for i in np.argsort(chroma_mean)[::-1][:3]],
    }
    (out_dir / "music.json").write_text(json.dumps(doc, indent=2))

    fig, ax = plt.subplots(2, 1, figsize=(12.8, 7.2), dpi=130, sharex=True)
    bmask = (bpm > 5) & (bpm < 300)
    ax[0].pcolormesh(tt, bpm[bmask], T[bmask], cmap="magma", shading="auto")
    ax[0].set(yscale="log", ylabel="tempo (BPM)",
              title=f"{sess.name} — tempogram (librosa); global peak "
                    f"{bpm_peak:.1f} BPM = {60/bpm_peak:.2f} s")
    ax[1].pcolormesh(tc, np.arange(12), C, cmap="magma", shading="auto")
    ax[1].set_yticks(range(12), NOTE, fontsize=7)
    ax[1].set(xlabel="time (s)", ylabel="pitch class",
              title="chromagram (librosa)")
    fig.tight_layout()
    fig.savefig(out_dir / "music.png")
    plt.close(fig)
    return doc
