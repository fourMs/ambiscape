"""Clockwork escapement: tick-level regularity of fast mechanical clicks.

:mod:`ambiscape.rhythm` targets pitched, ringing strike sources (bells) with
cycles up to 8 s and onsets no closer than 0.3 s. A tower-clock escapement is
the opposite signal: dry broadband clicks with no stable partials, repeating
at 0.5–5 Hz, usually with an audible tick/tock alternation. This module reads
the audio of the first take directly (like :mod:`ambiscape.carillon`): a
band-passed click envelope at 1 ms resolution, onset picking at the beat
rate, an alternation split, and watchmaker-style regularity statistics —
beat period, tick/tock beat error, fast jitter vs slow wander, coverage
(missing beats), and Allan deviation of the beat period across averaging
scales.

Times in the JSON are seconds into the take; ``sess.clock`` turns them into
wall-clock strings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

EPS = 1e-12
ENV_FS = 1000  # click-envelope rate (Hz): 1 ms resolution


def click_envelope(x: np.ndarray, fs: int, band=(2000.0, 9000.0),
                   env_fs: int = ENV_FS) -> np.ndarray:
    """Log-compressed, detrended click envelope of a mono signal.

    Band-passes to the click band (defaults reject rumble, voices and bell
    partials), rectifies, decimates to ``env_fs``, log-compresses relative
    to the median and removes a 2 s running median so only transients stand
    proud of zero.
    """
    hi = min(band[1], 0.45 * fs)
    sos = signal.butter(4, [band[0], hi], btype="band", fs=fs, output="sos")
    y = signal.sosfilt(sos, x.astype(np.float32))
    hop = int(round(fs / env_fs))
    n = (len(y) // hop) * hop
    env = np.abs(y[:n]).reshape(-1, hop).mean(axis=1)
    env = np.log1p(env / (np.median(env) + EPS))
    return env - signal.medfilt(env, 2 * env_fs + 1)


def estimate_beat(env: np.ndarray, env_fs: int = ENV_FS,
                  min_rate: float = 0.3, max_rate: float = 6.0) -> float:
    """Beat period (s) from the envelope autocorrelation.

    The tick/tock alternation puts ACF peaks at both the beat and the full
    cycle; the *earliest* peak that reaches 40 % of the strongest in-range
    peak is taken, so the beat is not confused with its own multiples.
    """
    e = env - env.mean()
    n = int(env_fs / min_rate * 4)
    acf = signal.correlate(e[: 400 * env_fs], e[: 400 * env_fs], mode="full")
    acf = acf[len(acf) // 2:][: n]
    lo, hi = int(env_fs / max_rate), int(env_fs / min_rate)
    seg = acf[lo:hi]
    pk, props = signal.find_peaks(seg, height=0.0)
    if not len(pk):
        return float((lo + np.argmax(seg)) / env_fs)
    heights = props["peak_heights"]
    good = pk[heights >= 0.4 * heights.max()]
    return float((lo + good[0]) / env_fs)


def detect_ticks(env: np.ndarray, env_fs: int, beat_s: float,
                 k: float = 3.0):
    """Pick tick onsets: local maxima above a global ``k``-MAD threshold,
    no closer than 0.55 beats, refined by parabolic interpolation.
    Returns (times_s, amplitudes)."""
    mad = np.median(np.abs(env - np.median(env))) + EPS
    thr = np.median(env) + k * mad
    dist = max(1, int(0.55 * beat_s * env_fs))
    pk, props = signal.find_peaks(env, height=thr, distance=dist)
    t = pk.astype(np.float64)
    for j, i in enumerate(pk):                       # parabolic refine
        if 0 < i < len(env) - 1:
            a, b, c = env[i - 1], env[i], env[i + 1]
            den = a - 2 * b + c
            if den != 0:
                t[j] = i + 0.5 * (a - c) / den
    return t / env_fs, props["peak_heights"]


def _contiguous_runs(t: np.ndarray, beat_s: float):
    """Split tick times into runs of consecutive beats (gap < 1.6 beats)."""
    runs, start = [], 0
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > 1.6 * beat_s:
            if i - start >= 4:
                runs.append((start, i))
            start = i
    if len(t) - start >= 4:
        runs.append((start, len(t)))
    return runs


def allan_deviation(periods: np.ndarray, beat_s: float,
                    max_m: int = 256) -> dict:
    """Overlapping Allan deviation of a beat-period series, in ppm of the
    beat, keyed by averaging time (s)."""
    out = {}
    m = 1
    while m <= min(max_m, len(periods) // 4):
        y = np.convolve(periods, np.ones(m) / m, mode="valid")[::m]
        if len(y) < 3:
            break
        adev = np.sqrt(0.5 * np.mean(np.diff(y) ** 2))
        out[round(m * beat_s, 2)] = round(float(1e6 * adev / beat_s), 1)
        m *= 2
    return out


def beat_stats(t: np.ndarray, amps: np.ndarray, beat_s: float) -> dict:
    """Watchmaker statistics from tick times within contiguous runs."""
    runs = _contiguous_runs(t, beat_s)
    iois, resid_fast, periods_all = [], [], []
    beat_errs, amp_alts, wts = [], [], []
    for s, e in runs:
        ioi = np.diff(t[s:e])
        ok = (ioi > 0.6 * beat_s) & (ioi < 1.6 * beat_s)
        iois.append(ioi[ok])
        # full-cycle periods (tick_k -> tick_k+2) cancel the alternation;
        # fast jitter is judged on these, not the bimodal raw IOIs
        cyc = (t[s + 2:e] - t[s:e - 2]) / 2.0
        periods_all.append(cyc)
        if len(cyc) >= 3:
            med9 = signal.medfilt(cyc, min(9, 2 * (len(cyc) // 2) - 1))
            resid_fast.append(cyc - med9)
        # alternation is judged per run: parity across runs is arbitrary
        ha, hb = ioi[0::2], ioi[1::2]
        aa, ab = amps[s:e][0::2], amps[s:e][1::2]
        if len(ha) >= 2 and len(hb) >= 2:
            beat_errs.append(abs(np.median(ha) - np.median(hb)) / 2.0)
            amp_alts.append(abs(20 * np.log10(
                (np.median(aa) + EPS) / (np.median(ab) + EPS))))
            wts.append(e - s)
    if not iois or not len(np.concatenate(iois)):
        return {"n_valid_iois": 0}
    ioi = np.concatenate(iois)
    fast = np.concatenate(resid_fast) if resid_fast else np.array([0.0])
    periods = np.concatenate(periods_all) if periods_all else ioi
    w = np.array(wts, float) / max(sum(wts), 1)
    beat_err = float(np.dot(w, beat_errs)) if beat_errs else 0.0
    amp_alt = float(np.dot(w, amp_alts)) if amp_alts else 0.0
    lag1 = (float(np.corrcoef(periods[:-1], periods[1:])[0, 1])
            if len(periods) > 3 else 0.0)
    # slow wander: sd of the running-median beat across the recording
    med_track = signal.medfilt(periods, min(31, 2 * (len(periods) // 2) - 1)
                               if len(periods) >= 3 else 1)
    # rate change first half vs second half, in ppm
    h = len(periods) // 2
    drift_ppm = (1e6 * (np.median(periods[h:]) - np.median(periods[:h]))
                 / beat_s) if h >= 4 else 0.0
    return {
        "n_valid_iois": int(len(ioi)),
        "sd_ms": round(float(1000 * ioi.std()), 2),
        "cv": round(float(ioi.std() / ioi.mean()), 5),
        "fast_sd_ms": round(float(1000 * fast.std()), 2),
        "slow_sd_ms": round(float(1000 * med_track.std()), 2),
        "lag1": round(lag1, 3),
        "beat_error_ms": round(float(1000 * beat_err), 2),
        "amp_alternation_db": round(amp_alt, 2),
        "rate_drift_ppm": round(float(drift_ppm), 1),
        "allan_ppm": allan_deviation(periods, beat_s),
    }


def render(env, env_fs, t, amps, beat_s, stats, out_path, title="",
           clock=None, t0=0.0):
    """Figure: beat period timeline, IOI histogram, Allan deviation,
    envelope close-up with detections."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13.2, 7.6), dpi=130)
    ioi = np.diff(t)
    ok = (ioi > 0.6 * beat_s) & (ioi < 1.6 * beat_s)
    tm = t0 + t[1:]
    ax[0, 0].plot(tm[ok], 1000 * ioi[ok], ".", ms=2.5, color="#2a78d6",
                  alpha=0.6)
    ax[0, 0].axhline(1000 * beat_s, color="0.4", lw=0.8)
    ax[0, 0].set(ylabel="beat interval (ms)",
                 title=f"{title} — beat {beat_s:.3f} s "
                       f"({3600 / beat_s:.0f} beats/h), "
                       f"jitter {stats.get('fast_sd_ms', float('nan'))} ms")
    if clock is not None:
        xt = ax[0, 0].get_xticks()
        ax[0, 0].set_xticks(xt)
        ax[0, 0].set_xticklabels([clock(v)[7:] for v in xt], fontsize=7)
        ax[0, 0].set_xlim(tm[0], tm[-1])
    ax[0, 1].hist(1000 * ioi[ok & (np.arange(len(ioi)) % 2 == 0)], bins=60,
                  alpha=0.65, color="#2a78d6", label="even")
    ax[0, 1].hist(1000 * ioi[ok & (np.arange(len(ioi)) % 2 == 1)], bins=60,
                  alpha=0.65, color="#d66a2a", label="odd")
    ax[0, 1].set(xlabel="beat interval (ms)", title="tick/tock intervals")
    ax[0, 1].legend(fontsize=8)
    al = stats.get("allan_ppm", {})
    if al:
        taus = list(al.keys())
        ax[1, 0].loglog(taus, list(al.values()), "o-", color="#3d9970")
        ax[1, 0].set(xlabel="averaging time (s)", ylabel="Allan dev (ppm)",
                     title="beat-period stability vs averaging time")
        ax[1, 0].grid(alpha=0.25, which="both")
    n0 = int(min(20.0, len(env) / env_fs) * env_fs)
    te = np.arange(n0) / env_fs
    ax[1, 1].plot(te, env[:n0], lw=0.5, color="0.3")
    sel = t[t < n0 / env_fs]
    ax[1, 1].plot(sel, np.interp(sel, np.arange(len(env)) / env_fs, env),
                  "v", ms=4, color="#d66a2a")
    ax[1, 1].set(xlabel="s into take", title="click envelope + detections "
                                             "(first 20 s)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run_session(sess, out_dir, t0=0.0, dur=None, band=(2000.0, 9000.0),
                k=3.0, min_rate=0.3, max_rate=6.0) -> dict:
    """CLI driver: escapement statistics for the first take.

    Writes ``escapement.json`` (beat period, jitter/wander split, beat
    error, coverage, Allan deviation, tick times) and ``escapement.png``.
    """
    import json
    out_dir = Path(out_dir)
    take = sess.takes[0]
    fs = take.samplerate
    with sf.SoundFile(str(take.audio_path)) as f:
        f.seek(int(t0 * fs))
        n = f.frames - int(t0 * fs) if dur is None else int(dur * fs)
        x = f.read(n, dtype="float32", always_2d=True)
    env = click_envelope(take.mono_ref(x), fs, band=band)
    beat = estimate_beat(env, ENV_FS, min_rate=min_rate, max_rate=max_rate)
    t, amps = detect_ticks(env, ENV_FS, beat, k=k)
    stats = beat_stats(t, amps, beat)
    span = len(env) / ENV_FS
    ioi = np.diff(t)
    valid = ioi[(ioi > 0.6 * beat) & (ioi < 1.6 * beat)]
    # mean, not median: tick/tock alternation makes the IOIs bimodal and
    # the mean of the two modes is the true beat
    bp = float(valid.mean()) if len(valid) else beat
    doc = {
        "t0_s": round(t0, 2), "analysed_s": round(span, 1),
        "band_hz": [float(band[0]), float(band[1])],
        "beat_period_s": round(bp, 4),
        "rate_bph": None, "n_ticks": int(len(t)),
        "coverage": round(float(len(t) * bp / span), 3),
        **stats,
        "tick_times_s": [round(float(v) + t0, 3) for v in t],
        "_method_note": (
            "band-passed 1 ms click envelope; onsets above a k-MAD "
            "threshold at least 0.55 beats apart; statistics on contiguous "
            "runs only, so pauses (chimes, voices) count against coverage "
            "but do not bias the jitter. beat_error is the tick/tock "
            "asymmetry; Allan deviation is over full cycles (alternation "
            "cancelled)."),
    }
    doc["rate_bph"] = round(3600.0 / doc["beat_period_s"], 1)
    render(env, ENV_FS, t, amps, doc["beat_period_s"], stats,
           out_dir / "escapement.png", title=sess.name, clock=sess.clock,
           t0=take.start + t0)
    (out_dir / "escapement.json").write_text(
        json.dumps(doc, indent=2, default=float))
    return doc
