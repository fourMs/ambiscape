"""Event timbre templates: recurring event classes without machine learning.

Every transient event gets a spectral fingerprint — the strike-triggered
post/pre **rise spectrum** (what appeared) plus a per-band **decay slope**
(how it faded). Fingerprints are clustered by correlation distance into
template classes: "the same sound again" across a whole session, fully
transparent and corpus-comparable. Complements PANNs tagging (`[ml]`).

``run_session`` fingerprints the session's spectral events (see
:mod:`background`), clusters them, and writes ``timbre.json`` +
``timbre.png`` (class templates + counts + exemplar times).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

EPS = 1e-20
N_MEL = 48


def _melish_matrix(freqs, fmin=100.0, fmax=8000.0, n=N_MEL):
    """Triangular log-spaced band matrix (nbin -> n bands)."""
    edges = np.geomspace(fmin, fmax, n + 2)
    W = np.zeros((n, len(freqs)))
    for i in range(n):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs >= lo) & (freqs < mid)
        dn = (freqs >= mid) & (freqs < hi)
        W[i, up] = (freqs[up] - lo) / (mid - lo + EPS)
        W[i, dn] = (hi - freqs[dn]) / (hi - mid + EPS)
    return W, edges[1:-1]


def event_fingerprint(take, t_onset: float, nfft=8192, decay_s=1.0):
    """Rise spectrum (dB, mel-ish bands) + per-band decay slope (dB/s)."""
    fs = take.samplerate
    win = np.hanning(nfft)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    W, centers = _melish_matrix(freqs)
    i = int(t_onset * fs)
    n_dec = int(decay_s * fs / nfft)
    with sf.SoundFile(str(take.audio_path)) as f:
        if i < nfft or i + (n_dec + 1) * nfft > f.frames:
            return None, None, centers
        f.seek(i - nfft + int(0.02 * fs))
        x = take.mono_ref(f.read((n_dec + 2) * nfft, dtype="float64",
                                 always_2d=True))
    def spec(seg):
        return W @ (np.abs(np.fft.rfft(seg * win)) ** 2)
    pre = spec(x[:nfft])
    post = spec(x[nfft:2 * nfft])
    rise = 10 * np.log10((post + EPS) / (pre + EPS))
    tail = np.array([10 * np.log10(spec(x[(k + 1) * nfft:(k + 2) * nfft])
                                   + EPS) for k in range(n_dec + 1)])
    slope = np.polyfit(np.arange(n_dec + 1) * nfft / fs, tail, 1)[0]
    return rise, slope, centers


def cluster_events(fps: np.ndarray, th=0.35, min_size=2):
    """Average-linkage clustering of fingerprints by correlation distance.
    Returns labels (−1 = unclustered singleton)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    if len(fps) < 2:
        return np.zeros(len(fps), int)
    C = np.corrcoef(fps)
    d = np.clip(1 - C, 0, 2)
    np.fill_diagonal(d, 0)
    lab = fcluster(linkage(d[np.triu_indices_from(d, 1)], "average"),
                   th, criterion="distance")
    out = np.full(len(fps), -1)
    for l in np.unique(lab):
        m = lab == l
        if m.sum() >= min_size:
            out[m] = l
    return out


def run_session(sess, out_dir, max_events=150) -> dict:
    """Fingerprint + cluster the session's spectral events."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .background import band_background, foreground, spectral_events
    from .features import load_features

    out_dir = Path(out_dir)
    F = load_features(sorted((out_dir / "features").glob("*.npz")))
    take = sess.takes[0]
    bg = band_background(F["logspec"])
    rise_db, _frac = foreground(F["logspec"], bg)
    ev = spectral_events(rise_db, F["logf"])
    # one fingerprint per onset: merge blobs that start within 1 s
    seen, dedup = set(), []
    for e in ev:
        if e["t0_s"] not in seen:
            dedup.append(e)
            seen.update((e["t0_s"] - 1, e["t0_s"], e["t0_s"] + 1))
    ev = dedup
    if len(ev) > max_events:
        keep = np.argsort([-e["peak_rise_db"] for e in ev])[:max_events]
        ev = [ev[i] for i in sorted(keep)]
    fps, slopes, kept = [], [], []
    for e in ev:
        r, s, centers = event_fingerprint(take, float(e["t0_s"]))
        if r is not None:
            fps.append(r)
            slopes.append(s)
            kept.append(e)
    fps = np.array(fps)
    lab = cluster_events(fps) if len(fps) else np.array([], int)
    classes = []
    for l in sorted(set(lab.tolist()) - {-1}):
        m = lab == l
        classes.append({
            "n": int(m.sum()),
            "exemplar_t0_s": [kept[i]["t0_s"]
                              for i in np.flatnonzero(m)[:5]],
            "centroid_hz": int(np.exp((fps[m].mean(0) * np.log(centers)).sum()
                                      / (fps[m].mean(0).sum() + EPS))),
            # decay only over the bands the event actually excited
            "decay_median_db_s": round(float(np.median(
                [np.median(s[r > 6]) for r, s in
                 zip(fps[m], np.array(slopes)[m]) if (r > 6).any()] or
                [np.nan])), 1),
        })
    classes.sort(key=lambda c: -c["n"])
    doc = {"n_events_fingerprinted": len(fps),
           "n_classes": len(classes),
           "n_unclustered": int((lab == -1).sum()),
           "classes": classes}
    (out_dir / "timbre.json").write_text(json.dumps(doc, indent=2))

    if len(fps):
        order = np.argsort(lab)
        fig, ax = plt.subplots(figsize=(11.2, 5.2), dpi=130)
        pc = ax.pcolormesh(np.arange(len(fps)), centers, fps[order].T,
                           cmap="magma", shading="auto")
        for b in np.flatnonzero(np.diff(lab[order]) != 0):
            ax.axvline(b + 0.5, color="w", lw=0.8)
        ax.set(yscale="log", xlabel="event (grouped by class)",
               ylabel="Hz", title=f"{sess.name} — event rise-spectrum "
               f"fingerprints, {len(classes)} classes")
        fig.colorbar(pc, ax=ax, pad=0.01, label="rise (dB)")
        fig.tight_layout()
        fig.savefig(out_dir / "timbre.png")
        plt.close(fig)
    return doc
