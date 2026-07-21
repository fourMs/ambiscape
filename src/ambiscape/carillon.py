"""Carillon / bell-instrument MIR: which bells were played (optional
``[music]`` extra).

A carillon is a tuned set of cast bells played from a keyboard. Unlike a
swinging tower bell (see :mod:`ambiscape.rhythm`), a carillon plays *melodies*,
so the musically interesting question is not "how fast does it toll" but
**which bells sounded** — the set of distinct strike notes, their tuning, the
range of the instrument used, and the pitch-class centre of the music.

The obstacle is the bell partial series. A struck bell is inharmonic; its
principal partials sit at fixed ratios to the strike note (prime):

    hum 1/2 · prime 1 · tierce 6/5 (a **minor third**) · quint 3/2 · nominal 2

Naive pitch tracking locks onto the loud tierce and nominal and reports notes a
minor third or an octave above what a listener hears. This module instead does
**bell-template matching**: for every candidate strike note it sums the energy
found at that note's five partial positions, so a true strike note — which has
energy at *all* of them — outscores any single partial. Accumulated over the
onset frames of a whole recital, the peaks of that salience are the bells that
were played.

Runs on the W channel resampled to 22.05 kHz, in chunks, so an hour-long
concert is fine. Requires ``pip install "ambiscape[music]"``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .tonality import NOTE

# principal bell partials as (name, frequency ratio to the strike note, weight).
# The minor-third tierce (6/5) is the signature that separates a bell from a
# harmonic (voiced / wind) source; prime and nominal fix the perceived pitch.
BELL_PARTIALS = (
    ("hum", 0.5, 0.5),
    ("prime", 1.0, 1.0),
    ("tierce", 1.2, 0.8),
    ("quint", 1.5, 0.4),
    ("nominal", 2.0, 1.0),
)


def _require_librosa():
    try:
        import librosa
        return librosa
    except ImportError as e:
        raise ImportError(
            "librosa is required: pip install 'ambiscape[music]'") from e


def note_name(f: float) -> tuple[str, int, float]:
    """(name+octave, MIDI, cents-deviation) for a frequency, A4 = 440 Hz."""
    midi = 69.0 + 12.0 * np.log2(f / 440.0)
    m = int(round(midi))
    cents = round(100.0 * (midi - m), 1)
    return f"{NOTE[m % 12]}{m // 12 - 1}", m, cents


def _gather_shift(C: np.ndarray, shift: float):
    """``C`` (n_bins, n_frames) resampled so row *i* holds the value at bin
    ``i + shift`` (fractional, linear-interpolated). Rows whose source falls
    off either edge are zeroed — a partial outside the transform contributes 0.
    """
    n = C.shape[0]
    lo = int(np.floor(shift))
    frac = shift - lo
    idx = np.arange(n) + lo
    ok_a = (idx >= 0) & (idx < n)
    ok_b = (idx + 1 >= 0) & (idx + 1 < n)
    Ca = np.where(ok_a[:, None], C[np.clip(idx, 0, n - 1)], 0.0)
    Cb = np.where(ok_b[:, None], C[np.clip(idx + 1, 0, n - 1)], 0.0)
    return (1 - frac) * Ca + frac * Cb


def bell_partial_energies(C: np.ndarray, bins_per_oct: int) -> dict:
    """Per-partial energy stacks: for each principal bell partial, the CQT
    power that a strike note at bin *i* would show at that partial's position.
    Returns ``{partial_name: (n_bins, n_frames) array}``."""
    return {name: _gather_shift(C, bins_per_oct * np.log2(ratio))
            for name, ratio, _ in BELL_PARTIALS}


def bell_salience_from_parts(parts: dict) -> np.ndarray:
    """Template-weighted sum of per-partial energies (the bell salience)."""
    wsum = sum(w for _, _, w in BELL_PARTIALS)
    sal = sum(w * parts[name] for name, _, w in BELL_PARTIALS)
    return sal / wsum


def _strike_frames(y, sr, hop):
    """Onset-strength envelope and a boolean mask of its salient frames."""
    librosa = _require_librosa()
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    med = np.median(oenv)
    mad = np.median(np.abs(oenv - med)) + 1e-9
    return oenv, (oenv - med) / mad > 3.0


def analyze_bells(take, t0=0.0, dur=None, sr=22050, fmin=110.0, n_octaves=5,
                  bins_per_oct=36, hop=2048, chunk_s=120.0):
    """Accumulate bell salience over the strike frames of a take.

    Returns dict with the candidate-pitch grid (``fcq``), the accumulated
    salience over strike frames (``sal``), a coarse mean chroma, the strike
    (onset) count, and bookkeeping. Processed in ``chunk_s`` chunks so
    hour-long recitals stay within memory.
    """
    librosa = _require_librosa()
    fs = take.samplerate
    n_bins = int(n_octaves * bins_per_oct)
    fcq = fmin * 2.0 ** (np.arange(n_bins) / bins_per_oct)
    part_acc = {name: np.zeros(n_bins) for name, _, _ in BELL_PARTIALS}
    chroma = np.zeros(12)
    n_strike = 0
    n_frames = 0
    t_end = (take.frames / fs) if dur is None else (t0 + dur)
    hann_pad = int(0.05 * fs)
    with sf.SoundFile(str(take.audio_path)) as f:
        t = t0
        while t < t_end - 0.2:
            n = int(min(chunk_s, t_end - t) * fs) + hann_pad
            f.seek(int(t * fs))
            x = f.read(n, dtype="float32", always_2d=True)
            if x.shape[0] < fs // 2:
                break
            y = librosa.resample(take.mono_ref(x), orig_sr=fs, target_sr=sr)
            C = np.abs(librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins,
                                   bins_per_octave=bins_per_oct,
                                   hop_length=hop)) ** 2
            oenv, mask = _strike_frames(y, sr, hop)
            m = mask[:C.shape[1]]
            if m.any():
                parts = bell_partial_energies(C[:, m], bins_per_oct)
                for name in part_acc:
                    part_acc[name] += parts[name].sum(1)
                n_strike += int(m.sum())
            chroma += librosa.feature.chroma_cqt(
                y=y, sr=sr, hop_length=hop).sum(1)
            n_frames += C.shape[1]
            t += chunk_s
    sal = bell_salience_from_parts(part_acc)
    return {"fcq": fcq, "sal": sal, "part_acc": part_acc,
            "bins_per_oct": bins_per_oct, "chroma": chroma,
            "n_strike_frames": n_strike, "n_frames": n_frames,
            "hop": hop, "sr": sr}


def detect_bells(fcq, sal, bins_per_oct, part_acc=None, prime_lo=123.0,
                 prime_hi=1200.0, rel_floor=0.06, merge_cents=55.0,
                 tierce_floor=0.12, nominal_floor=0.10):
    """Peak-pick the salience into a bell inventory.

    Restricts candidate strike notes to ``[prime_lo, prime_hi]`` Hz, keeps
    local maxima above ``rel_floor`` of the strongest peak, merges peaks closer
    than ``merge_cents``, and refines each to its parabolic-interpolated
    frequency. When ``part_acc`` (per-partial energy stacks) is supplied, a
    candidate is kept only if it has genuine **tierce** and **nominal** support
    relative to its prime — the minor-third tierce is the bell fingerprint that
    an octave-down "hum ghost" or a tierce/quint ghost of a real bell lacks.
    Returns a list of dicts sorted low→high in pitch.
    """
    from scipy.signal import find_peaks
    s = sal / (sal.max() + 1e-12)
    band = (fcq >= prime_lo) & (fcq <= prime_hi)
    dist = max(1, int(bins_per_oct * merge_cents / 1200.0))
    pk, _ = find_peaks(np.where(band, s, 0.0), height=rel_floor, distance=dist)
    if part_acc is not None:
        prime = part_acc["prime"]; tierce = part_acc["tierce"]
        nominal = part_acc["nominal"]
        prime_abs = 0.04 * prime[band].max()      # a real strike note rings
        pk = np.array([i for i in pk
                       if prime[i] > prime_abs
                       and tierce[i] > tierce_floor * prime[i]
                       and nominal[i] > nominal_floor * prime[i]], int)
    bells = []
    for i in pk:
        # parabolic refine on the log-frequency salience
        if 0 < i < len(s) - 1:
            a, b, c = s[i - 1], s[i], s[i + 1]
            denom = (a - 2 * b + c)
            off = 0.5 * (a - c) / denom if denom != 0 else 0.0
        else:
            off = 0.0
        f = float(fcq[i] * 2.0 ** (off / bins_per_oct))
        name, midi, cents = note_name(f)
        bells.append({"freq_hz": round(f, 1), "note": name, "midi": midi,
                      "cents": cents, "salience": round(float(s[i]), 3)})
    # merge any that collapsed onto the same MIDI note (keep the stronger)
    best = {}
    for bdct in bells:
        k = bdct["midi"]
        if k not in best or bdct["salience"] > best[k]["salience"]:
            best[k] = bdct
    return sorted(best.values(), key=lambda d: d["freq_hz"])


def run_session(sess, out_dir, t0=0.0, dur=None, fmin=110.0, n_octaves=5) -> dict:
    """Full carillon MIR for the first take: bell inventory + figures.

    Writes ``carillon.json`` (detected bells, range, pitch-class centre, note
    density) and ``carillon.png`` (bell-salience inventory with note names +
    pitch-class profile). Returns the summary dict."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    librosa = _require_librosa()

    out_dir = Path(out_dir)
    take = sess.takes[0]
    A = analyze_bells(take, t0=t0, dur=dur, fmin=fmin, n_octaves=n_octaves)
    bells = detect_bells(A["fcq"], A["sal"], A["bins_per_oct"],
                         part_acc=A["part_acc"])

    chroma = A["chroma"] / (A["chroma"].sum() + 1e-12)
    span = (dur if dur is not None else take.duration - t0)
    midis = [b["midi"] for b in bells]
    doc = {
        "t0_s": round(t0, 1), "analysed_s": round(span, 1),
        "n_bells_detected": len(bells),
        "range": ({"low": bells[0]["note"], "high": bells[-1]["note"],
                   "semitones": (max(midis) - min(midis))} if bells else None),
        "bells": bells,
        "strike_frames": A["n_strike_frames"],
        "top_pitch_classes": [NOTE[i] for i in np.argsort(chroma)[::-1][:3]],
        "chroma_profile": {NOTE[i]: round(float(chroma[i]), 3)
                           for i in range(12)},
        "_method_note": (
            "bell-template salience (hum/prime/tierce/quint/nominal, 6/5 minor-"
            "third tierce) accumulated over onset frames of the W channel; "
            "peaks are strike notes. Distant/soft bells and bells masked by the "
            "chord above them may be missed; tuning cents are relative to "
            "A4=440 equal temperament."),
    }
    (out_dir / "carillon.json").write_text(
        json.dumps(doc, indent=2, default=float))

    # ---- figure: bell inventory (salience vs pitch) + pitch-class profile ----
    fcq, s = A["fcq"], A["sal"] / (A["sal"].max() + 1e-12)
    fig, ax = plt.subplots(2, 1, figsize=(13, 7.4), dpi=130,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].semilogx(fcq, s, color="0.4", lw=1.0)
    ax[0].fill_between(fcq, s, color="0.85")
    for b in bells:
        ax[0].axvline(b["freq_hz"], color="#d66a2a", lw=0.8, alpha=0.7)
        ax[0].annotate(b["note"], (b["freq_hz"], b["salience"]),
                       textcoords="offset points", xytext=(0, 3),
                       ha="center", fontsize=7, rotation=90, color="#8a3d10")
    ax[0].set(xlim=(fcq[0], fcq[-1]), ylim=(0, 1.12),
              ylabel="bell-template salience (norm.)",
              title=f"{sess.name} — carillon bell inventory: "
                    f"{len(bells)} bells detected"
                    + (f", {doc['range']['low']}–{doc['range']['high']}"
                       if bells else ""))
    ax[0].set_xticks([131, 165, 196, 262, 330, 392, 523, 659, 784, 1047])
    ax[0].set_xticklabels(["C3", "E3", "G3", "C4", "E4", "G4", "C5", "E5",
                           "G5", "C6"], fontsize=7)
    ax[0].grid(True, which="both", axis="x", alpha=0.15)
    ax[1].bar(range(12), chroma, color="#2a78d6")
    ax[1].set_xticks(range(12)); ax[1].set_xticklabels(NOTE, fontsize=8)
    ax[1].set(ylabel="pitch-class\nenergy",
              title=f"pitch-class profile — centre on "
                    f"{', '.join(doc['top_pitch_classes'])}")
    fig.tight_layout()
    fig.savefig(out_dir / "carillon.png")
    plt.close(fig)
    return doc
