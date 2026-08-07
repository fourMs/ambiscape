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

All three scales are computed the same way: the source stream is converted
to a linear-power envelope, normalised to unit mean, and its Welch power
spectral density taken. One normalisation, so the per-scale curves live on a
single comparable dB axis; each scale is additionally computed half a decade
past its nominal band edges, so neighbouring scales overlap and their
agreement where they meet is visible rather than assumed.

``profile`` returns, per scale, a log-frequency modulation spectrum with the
dominant modulation frequency, its prominence, and the band modulation depth
(all three statistics taken within the nominal band, not the overlap).
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
# Each scale's spectrum is computed this factor past its nominal band edges
# (half a decade), so adjacent scales overlap and can be compared directly.
EXT = 10 ** 0.5


def modulation_spectrum(env: np.ndarray, dt: float, fmin: float, fmax: float,
                        n_bins: int = 48):
    """Welch modulation spectrum of a linear-power envelope, log-resampled.

    The envelope is normalised to zero-mean unit-mean (x = env/mean − 1) so
    spectra are comparable across levels; returns (freqs, power density).
    Welch bins are averaged into each log-grid cell (nearest bin where a
    cell is empty), which keeps the estimate stable even when the record
    only allows a single Welch segment.
    """
    x = env.astype(np.float64) / (env.mean() + EPS) - 1.0
    nper = int(min(len(x), max(64, round(8.0 / (fmin * dt)))))
    f, P = signal.welch(x, fs=1.0 / dt, nperseg=nper,
                        noverlap=nper // 2, detrend="linear")
    grid = np.geomspace(fmin, fmax, n_bins)
    edges = np.geomspace(fmin, fmax, n_bins + 1)
    lo = np.searchsorted(f, edges[:-1])
    hi = np.searchsorted(f, edges[1:])
    near = np.clip(np.searchsorted(f, grid), 1, len(f) - 1)
    out = np.array([P[a:b].mean() if b > a else P[n]
                    for a, b, n in zip(lo, hi, near)])
    return grid, out


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
    """Three-scale modulation profile from cached features.

    Every scale is a unit-mean linear-power envelope PSD (see
    ``modulation_spectrum``), so the three spectra share one normalisation
    and are directly comparable in level. Each is computed ``EXT`` past its
    nominal band edges (clipped to the record length and the stream's
    Nyquist); statistics are taken within the nominal band only.
    """
    dur = float(len(F["t"]))
    out = {"scales": {}, "spectra": {}}
    fast_pow = 10 ** (F["fast_db"].astype(np.float64) / 10)  # dB -> lin power
    meso_dt = float(np.median(np.diff(F["t_fast"])))
    sources = {
        "meso": (fast_pow, meso_dt),
        "macro": (F["rms_w"].astype(np.float64) ** 2, 1.0),
    }
    if "env_hi" in F:
        sources["micro"] = (F["env_hi"], float(F["hi_dt"]))
    else:
        sources["micro"] = (fast_pow, meso_dt)
        out["micro_limited"] = "no env_hi in cache; micro band tops out at 4 Hz"
    for scale in SCALES:
        env, dt = sources[scale]
        lo, hi = BANDS[scale]
        lo = max(lo or 4.0 / dur, 4.0 / dur)
        clo = max(lo / EXT, 4.0 / dur)
        chi = min(hi * EXT, 0.45 / dt)
        if chi <= clo * 1.5:
            continue
        f, P = modulation_spectrum(env, dt, clo, chi)
        out["spectra"][scale] = {"freq_hz": [round(float(v), 5) for v in f],
                                 "power": [float(v) for v in P]}
        band = (f >= lo) & (f <= min(hi, chi))
        if band.any():
            out["scales"][scale] = _scale_stats(f[band], P[band])
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
        db = 10 * np.log10(np.array(sp["power"]) + EPS)
        lo, hi = BANDS[scale]
        band = (f >= (lo or 0.0)) & (f <= hi)
        # full curve faint (the half-decade overlap into neighbouring
        # scales), nominal band solid on top
        ax0.plot(f, db, color=colors[scale], lw=1.0, alpha=0.35)
        st = prof["scales"].get(scale)
        lab = f"{scale} (peak {st['peak_period_s']} s)" if st else scale
        ax0.plot(f[band], db[band], color=colors[scale], lw=1.6, label=lab)
    for edge in (0.01, 0.5):
        ax0.axvline(edge, color="0.75", lw=0.8, ls=":", zorder=0)
    ax0.set(xscale="log", xlabel="modulation frequency (Hz)",
            ylabel="PSD (dB re 1/Hz)\nunit-mean power envelope",
            title=f"{title} — envelope modulation spectrum "
            "(macro | meso | micro, shared normalisation)")
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
