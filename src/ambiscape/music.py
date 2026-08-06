"""Librosa-based tempogram and chromagram (optional ``[music]`` extra).

Complements the built-in analyses with the MIR-standard views: the
**tempogram** (onset autocorrelation over time, in BPM — against which the
windowed-ACF tempogram in :mod:`rhythm` can be cross-checked) and the
**chromagram** (12-bin pitch-class energy over time, the time-resolved
counterpart of :func:`ambiscape.tonality.pitch_class_profile`).

Three circular-statistics views build on :mod:`ambiscape.circstats`:
:func:`pulse_clarity` (metric lock of onsets, robust where plain BPM fails
on rubato material), :func:`fifths_center` (tonal centre + focus of a
chroma vector on the circle of fifths), and
:func:`tonal_center_spread` (how tightly a set of recordings clusters in
key space — a between-recording statistic with no linear equivalent).
:func:`tartyp_profile` classifies onset-bounded sound objects on a
simplified Schaeffer typology grid — the object-level counterpart of the
regime-level :func:`ambiscape.draft.schaeffer_hint`.

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
    """Mono reference of a take (W / L-R mean / channel), resampled to ``sr``.

    Reads the take's decoded audio (so a transcoded ``.m4a`` works too) and
    downmixes per the take's mode — MIR runs on the same mono reference the
    rest of the pipeline uses."""
    librosa = _require_librosa()
    fs = take.samplerate
    with sf.SoundFile(str(take.audio_path)) as f:
        f.seek(int(t0 * fs))
        n = f.frames - int(t0 * fs) if dur is None else int(dur * fs)
        x = f.read(n, dtype="float32", always_2d=True)
    return librosa.resample(take.mono_ref(x), orig_sr=fs, target_sr=sr), sr


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


# --------------------------------------------------------------------------
# Circular-statistics views and the object-level Schaeffer profile.
# Developed on a five-album solo-harp catalogue (57 tracks); the TARTYP
# thresholds below are calibrated on that tonal instrumental corpus.

FIFTHS_ANGLE = 2 * np.pi * (7 * np.arange(12) % 12) / 12

# Onset-strength floor (dB-difference units): real attacks measure O(1-10),
# peak-picked numerical noise on steady signals O(0.01).
ONSET_FLOOR = 0.5


def dominant_period(onset_env, sr, hop=512, bpm_range=(40.0, 200.0)):
    """Dominant beat period (s) from the onset-envelope autocorrelation."""
    librosa = _require_librosa()
    ac = librosa.autocorrelate(onset_env, max_size=int(4 * sr / hop))
    lo = max(1, int(60.0 / bpm_range[1] * sr / hop))
    hi = min(len(ac), int(60.0 / bpm_range[0] * sr / hop))
    lag = lo + int(np.argmax(ac[lo:hi]))
    return lag * hop / sr


def pulse_clarity(y, sr, hop=512, bpm_range=(40.0, 200.0)) -> dict:
    """Metric lock of the onsets: circular concentration of onset phases.

    Onsets are folded at the dominant period and their strength-weighted
    resultant length R taken as the score — 0 is free rubato, 1 metronomic.
    Works where a beat tracker's BPM is meaningless (rubato, drones); the
    single global period means slow tempo drift also reads as low R, so
    treat R as "lock to one steady grid", not "has any pulse at all".
    """
    from .circstats import mean_resultant, rayleigh_p
    librosa = _require_librosa()
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    p0 = dominant_period(env, sr, hop, bpm_range)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                    hop_length=hop, units="frames")
    fr = fr[env[fr] >= ONSET_FLOOR]
    t = librosa.frames_to_time(fr, sr=sr, hop_length=hop)
    if len(t) < 8:
        return {"R": 0.0, "period_s": round(p0, 4),
                "period_bpm": round(60.0 / p0, 1), "n_onsets": int(len(t))}
    w = np.interp(t, librosa.times_like(env, sr=sr, hop_length=hop), env)
    # The ACF peak may sit an octave off the felt pulse (folding 120 BPM
    # onsets at a 1 s period cancels the resultant), so let R itself pick
    # among the peak and its metrical neighbours.
    period, R = p0, -1.0
    for p in (p0 / 2, p0, p0 * 2):
        if not (60.0 / bpm_range[1] <= p <= 60.0 / bpm_range[0]):
            continue
        _, r = mean_resultant(2 * np.pi * (t / p % 1.0), weights=w)
        if r > R:
            period, R = p, r
    return {"R": round(R, 4), "period_s": round(period, 4),
            "period_bpm": round(60.0 / period, 1),
            "rayleigh_p": rayleigh_p(R, len(t)), "n_onsets": int(len(t))}


def fifths_center(chroma) -> dict:
    """Tonal centre and focus of a chroma vector on the circle of fifths.

    The 12 pitch classes are placed a fifth apart around the circle and the
    chroma-weighted resultant taken: the mean angle is the tonal centre
    (returned as the nearest note name and as an angle in fifths steps),
    R the tonal focus — diatonic material concentrates on one arc, chromatic
    or inharmonic material smears. Complements
    :func:`ambiscape.tonality.pitch_class_profile`, which keeps the full
    12-bin shape.
    """
    from .circstats import mean_resultant
    from .tonality import NOTE
    w = np.asarray(chroma, float)
    mu, R = mean_resultant(FIFTHS_ANGLE, weights=w)
    k = (mu / (2 * np.pi) * 12) % 12                 # steps along the fifths circle
    note = NOTE[int(7 * int(round(k)) % 12)]
    return {"center_note": note, "center_fifths": round(float(k), 3),
            "R": round(R, 4)}


def tonal_center_spread(chromas) -> dict:
    """Concentration of many recordings' tonal centres on the fifths circle.

    Feed one mean-chroma vector per recording; returns the resultant length
    R of their centres — near 1 when a corpus stays in neighbouring keys
    (a suite), near 0 when it wanders the whole circle. Key centres have no
    meaningful linear mean, so this is inherently a circular statistic.
    """
    from .circstats import circular_sd, mean_resultant
    angles = []
    for c in chromas:
        w = np.asarray(c, float)
        mu, _ = mean_resultant(FIFTHS_ANGLE, weights=w)
        angles.append(mu)
    _, R = mean_resultant(np.asarray(angles))
    return {"R": round(R, 4),
            "circ_sd_fifths": round(circular_sd(R) / (2 * np.pi) * 12, 3),
            "n": len(angles)}


# TARTYP proxy thresholds (median object spectral flatness; std of log2
# centroid in octaves; share of 4–20 Hz envelope-modulation energy)
TARTYP_TONIC = 0.004
TARTYP_COMPLEX = 0.02
TARTYP_DRIFT = 0.35
TARTYP_ITER = 0.45
TARTYP_IMPULSE_S = 0.3


def tartyp_profile(y, sr, hop=512) -> dict:
    """Duration-weighted Schaeffer typology profile of onset-bounded objects.

    Each inter-onset segment is one sound object, classified on a simplified
    TARTYP grid: mass N (tonic) / Y (variable) / X (complex) from spectral
    flatness and centroid drift, facture held / impulse (``'``) / iteration
    (``''``) from duration and 4–20 Hz envelope modulation. Returns the
    share of sounding time per type plus the object count.

    These are signal proxies for aural categories — a complement to reduced
    listening, not a substitute. Mass maps onto the annotation vocabulary of
    :mod:`ambiscape.taxonomy` roughly as N→tonic, Y→tonic-complex,
    X→complex/noise; the regime-level counterpart is
    :func:`ambiscape.draft.schaeffer_hint`.
    """
    librosa = _require_librosa()
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    flat = librosa.feature.spectral_flatness(S=S)[0]
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    fine_hop = hop // 8
    fine = librosa.feature.rms(y=y, frame_length=fine_hop * 4,
                               hop_length=fine_hop)[0]
    fine_rate = sr / fine_hop
    frame_t = hop / sr

    onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr)
    peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                       hop_length=hop, units="frames")
    peaks = peaks[onset_env[peaks] >= ONSET_FLOOR]
    onsets = (librosa.onset.onset_backtrack(peaks, onset_env)
              if len(peaks) else peaks)
    bounds = np.unique(np.concatenate([[0], onsets, [S.shape[1]]]))

    def _iter_ratio(a, b):
        seg = fine[int(a * hop / fine_hop):int(b * hop / fine_hop)]
        seg = seg - seg.mean()
        if len(seg) < 16 or not np.any(seg):
            return 0.0
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        fq = np.fft.rfftfreq(len(seg), 1 / fine_rate)
        tot = spec[fq > 0.2].sum()
        return float(spec[(fq >= 4) & (fq < 20)].sum() / tot) if tot else 0.0

    type_time, n_objects = {}, 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        dur = (b - a) * frame_t
        if dur < 0.05:
            continue
        n_objects += 1
        fmed = float(np.median(flat[a:b]))
        drift = float(np.std(np.log2(cent[a:b] + 1e-6)))
        if dur < TARTYP_IMPULSE_S:
            suffix = "'"
        elif _iter_ratio(a, b) > TARTYP_ITER:
            suffix = "''"
        else:
            suffix = ""
        if fmed >= TARTYP_COMPLEX:
            mass = "X"
        elif fmed >= TARTYP_TONIC or drift > TARTYP_DRIFT:
            mass = "Y"
        else:
            mass = "N"
        key = mass + suffix
        type_time[key] = type_time.get(key, 0.0) + dur

    tot = sum(type_time.values())
    dist = {k: round(v / tot, 4) for k, v in sorted(type_time.items())} if tot else {}
    return {"dist": dist, "n_objects": n_objects}


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
