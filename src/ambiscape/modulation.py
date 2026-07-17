"""Environmental rhythm: multi-scale envelope modulation profile.

Soundscapes are rhythmic on very different time scales at once — strike
patterns (micro), traffic waves and surf (meso), duty cycles of machines and
human activity (macro). This module measures all three from cached envelopes,
no audio pass:

- **micro** (0.5–20 Hz) from the 20 ms broadband envelope (``env_hi``,
  extractor ≥ 0.2 caches; older caches fall back to the 8 Hz fast level,
  which limits micro to < 4 Hz);
- **meso** (0.01–0.5 Hz) from the 125 ms fast level;
- **macro** (below 0.01 Hz, floor set by session length) from the 1 s RMS.

``profile`` returns, per scale, a log-frequency modulation spectrum with the
dominant modulation frequency, its prominence, and the band modulation depth.
``modulation_spectrogram`` computes the windowed version — the "rhythm
spectrogram of the day" — and ``render`` writes the combined figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

EPS = 1e-20

SCALES = ("micro", "meso", "macro")
BANDS = {"micro": (0.5, 20.0), "meso": (0.01, 0.5), "macro": (None, 0.01)}


def modulation_spectrum(env: np.ndarray, dt: float, fmin: float, fmax: float,
                        n_bins: int = 48):
    """Welch modulation spectrum of a linear-power envelope, log-resampled.

    The envelope is normalized to zero-mean unit-mean (x = env/mean − 1) so
    spectra are comparable across levels; returns (freqs, power density).
    """
    x = env.astype(np.float64) / (env.mean() + EPS) - 1.0
    nper = int(min(len(x), max(64, round(8.0 / (fmin * dt)))))
    f, P = signal.welch(x, fs=1.0 / dt, nperseg=nper,
                        noverlap=nper // 2, detrend="linear")
    grid = np.geomspace(fmin, fmax, n_bins)
    idx = np.clip(np.searchsorted(f, grid), 1, len(f) - 1)
    return grid, P[idx]


def _scale_stats(f, P):
    i = int(np.argmax(P))
    prom_db = 10 * np.log10((P[i] + EPS) / (np.median(P) + EPS))
    # band modulation depth: sqrt of band-integrated modulation power of the
    # unit-mean envelope (spectral-domain, avoids low-frequency filtering)
    depth = float(np.sqrt(max(np.trapezoid(P, f), 0.0)))
    return {
        "peak_freq_hz": round(float(f[i]), 4),
        "peak_period_s": round(float(1.0 / f[i]), 2),
        "peak_prominence_db": round(float(prom_db), 1),
        "modulation_depth": round(depth, 3),
    }


def profile(F: dict) -> dict:
    """Three-scale modulation profile from cached features."""
    dur = float(len(F["t"]))
    out = {"scales": {}, "spectra": {}}
    sources = {
        "meso": (F["fast_db"], float(np.median(np.diff(F["t_fast"])))),
        "macro": (F["rms_w"] ** 2, 1.0),
    }
    if "env_hi" in F:
        sources["micro"] = (F["env_hi"], float(F["hi_dt"]))
    else:
        sources["micro"] = (F["fast_db"], sources["meso"][1])
        out["micro_limited"] = "no env_hi in cache; micro band tops out at 4 Hz"
    for scale in SCALES:
        env, dt = sources[scale]
        if scale in ("meso", "macro"):     # dB level -> linear power
            env = 10 ** (env.astype(np.float64) / 10) if scale == "meso" else env
        lo, hi = BANDS[scale]
        lo = max(lo or 4.0 / dur, 4.0 / dur)
        hi = min(hi, 0.45 / dt)
        if hi <= lo * 1.5:
            continue
        f, P = modulation_spectrum(env, dt, lo, hi)
        out["spectra"][scale] = {"freq_hz": [round(float(v), 5) for v in f],
                                 "power": [float(v) for v in P]}
        out["scales"][scale] = _scale_stats(f, P)
    return out


def modulation_spectrogram(env: np.ndarray, dt: float, win_s: float = 600.0,
                           step_s: float = 120.0, fmin: float = 0.02,
                           fmax: float = 20.0, n_bins: int = 64):
    """Windowed modulation spectra: the rhythm spectrogram of the session.

    Returns (t_centers, mod_freqs, S) with S in dB relative to each window's
    median (so rhythmic structure reads as ridges regardless of level).
    """
    nwin = int(win_s / dt)
    nstep = int(step_s / dt)
    fmax = min(fmax, 0.45 / dt)
    grid = np.geomspace(fmin, fmax, n_bins)
    ts, rows = [], []
    for i0 in range(0, len(env) - nwin + 1, nstep):
        f, P = modulation_spectrum(env[i0:i0 + nwin], dt, fmin, fmax, n_bins)
        rows.append(10 * np.log10((P + EPS) / (np.median(P) + EPS)))
        ts.append((i0 + nwin / 2) * dt)
    return np.array(ts), grid, np.array(rows)


def render(F: dict, prof: dict, out_path, title="", clock=None):
    """Combined figure: per-scale spectra + rhythm spectrogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(12.8, 7.2), dpi=130,
        gridspec_kw=dict(height_ratios=[1, 1.3]))
    colors = {"micro": "#2a78d6", "meso": "#d66a2a", "macro": "#3d9970"}
    for scale in SCALES:
        sp = prof["spectra"].get(scale)
        if not sp:
            continue
        f = np.array(sp["freq_hz"])
        P = np.array(sp["power"])
        ax0.plot(f, 10 * np.log10(P + EPS), color=colors[scale], lw=1.4,
                 label=f"{scale} (peak {prof['scales'][scale]['peak_period_s']} s)")
    ax0.set(xscale="log", xlabel="modulation frequency (Hz)",
            ylabel="power (dB)", title=f"{title} — envelope modulation "
            "spectra by scale")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.2, which="both")

    env, dt = (F["env_hi"], float(F["hi_dt"])) if "env_hi" in F else \
        (10 ** (F["fast_db"].astype(np.float64) / 10),
         float(np.median(np.diff(F["t_fast"]))))
    t0 = float(F["t_hi"][0] if "t_hi" in F else F["t_fast"][0])
    ts, mf, S = modulation_spectrogram(env, dt)
    if len(ts):
        pc = ax1.pcolormesh(t0 + ts, mf, S.T, cmap="magma", shading="auto",
                            vmin=0, vmax=max(6, np.percentile(S, 99)))
        ax1.set(yscale="log", ylabel="modulation frequency (Hz)",
                title="rhythm spectrogram (10 min windows, dB re window median)")
        if clock is not None:
            xt = ax1.get_xticks()
            ax1.set_xticks(xt)
            ax1.set_xticklabels([clock(x)[7:] for x in xt], fontsize=8)
            ax1.set_xlim(t0 + ts[0], t0 + ts[-1])
        fig.colorbar(pc, ax=ax1, pad=0.01)
    fig.tight_layout()
    fig.savefig(out_path)
    return out_path


def run_session(sess, out_dir) -> dict:
    """CLI driver: profile + figure + modulation.json."""
    import json
    from .features import load_features
    out_dir = Path(out_dir)
    F = load_features(sorted((out_dir / "features").glob("*.npz")))
    prof = profile(F)
    render(F, prof, out_dir / "modulation_profile.png", title=sess.name,
           clock=sess.clock)
    (out_dir / "modulation.json").write_text(json.dumps(prof, indent=2))
    return prof
