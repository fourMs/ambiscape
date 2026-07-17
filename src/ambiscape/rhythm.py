"""Strike-level rhythm analysis of quasi-periodic pitched sources (bells,
machines, signals) in long ambisonic recordings.

The 1 Hz features in :mod:`features` are too coarse for strike rhythm, so this
module makes one extra streaming pass at ~20 ms resolution, restricted to the
narrowband partials of the sources of interest:

1. ``detect_partials`` — narrowband peaks from the cached per-minute mean PSD,
   contrasting source-active against quiet minutes;
2. ``partial_pass`` — streaming STFT pass storing, per frame, the power
   envelope and pseudo-intensity (for DOA) at each partial, plus a broadband
   spectral-flux onset function;
3. ``cluster_partials`` — group partials into sources by correlating their
   half-wave-rectified log-envelope derivatives (strike-synchronous);
4. ``pick_strikes`` — adaptive, strongest-first onset picking per source;
5. ``rayleigh_period`` / ``period_track`` — point-process periodicity
   (resultant length over a period grid, with harmonics for multi-strike
   cycles);
6. ``cycle_grid`` — repetition-with-variation statistics: per-cycle timing
   residuals and amplitudes for each position of the repeating pattern;
7. ``rise_spectrum`` — strike-triggered post/pre spectral rise, the tonal
   content of one rhythmic position (also exposes cross-talk: a stream whose
   rise spectrum shows another source's partials is leakage, not a strike).

All statistics are on the W channel except pseudo-intensity, which follows
the AmbiX ACN (W, Y, Z, X) convention used throughout ambiscape.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks

from .io import Take

EPS = 1e-14


# ---------------------------------------------------------------- partials

def detect_partials(F: dict, active: np.ndarray, quiet: np.ndarray,
                    band=(350.0, 4500.0), min_rise_db=6.0, max_n=30):
    """Narrowband partials of the active-state source(s).

    ``active``/``quiet`` are boolean masks over the minutes of ``F["minspec"]``
    (source clearly present / clearly absent). Returns (freqs, rise_db)
    sorted by frequency.
    """
    S_a = F["minspec"][active].mean(0)
    S_q = F["minspec"][quiet].mean(0)
    freqs = F["freqs"]
    m = (freqs >= band[0]) & (freqs <= band[1])
    ratio = 10 * np.log10((S_a[m] + EPS) / (S_q[m] + EPS))
    pk, props = find_peaks(ratio, height=min_rise_db, prominence=5.0,
                           distance=5)
    order = np.argsort(props["peak_heights"])[::-1][:max_n]
    keep = np.sort(pk[order])
    return freqs[m][keep], ratio[keep]


# ---------------------------------------------------------------- the pass

def partial_pass(take: Take, pfreq, nfft=4096, hop=960) -> dict:
    """Streaming STFT pass; per ~20 ms frame: 3-bin power envelope and
    pseudo-intensity at each partial, plus a 400-4500 Hz log-flux onset
    function. Returns dict of arrays keyed t/env/ix/iy/iz/odf/pfreq."""
    fs = take.samplerate
    hop = int(hop * fs / 48000)
    win = np.hanning(nfft).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    pbin = np.array([int(round(f * nfft / fs)) for f in np.atleast_1d(pfreq)])
    psel = np.stack([pbin - 1, pbin, pbin + 1], 1)
    fmask = (freqs >= 400) & (freqs <= 4500)

    env, ix, iy, iz, odf = [], [], [], [], []
    prevL = None
    carry = np.zeros((0, take.channels), np.float32)
    iW, iY, iZ, iX = take.wyzx
    with sf.SoundFile(str(take.path)) as f:
        while True:
            block = f.read(60 * fs, dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                break
            data = np.concatenate([carry, block]) if carry.shape[0] else block
            nwin = (data.shape[0] - nfft) // hop + 1
            if nwin <= 0:
                carry = data
                continue
            idx = np.arange(nfft)[None, :] + hop * np.arange(nwin)[:, None]
            W = np.fft.rfft(data[:, iW][idx] * win)
            Y = np.fft.rfft(data[:, iY][idx] * win)
            Z = np.fft.rfft(data[:, iZ][idx] * win)
            X = np.fft.rfft(data[:, iX][idx] * win)
            Pw = W.real ** 2 + W.imag ** 2
            env.append(Pw[:, psel].sum(2).astype(np.float32))
            ix.append((W.conj() * X).real[:, psel].sum(2).astype(np.float32))
            iy.append((W.conj() * Y).real[:, psel].sum(2).astype(np.float32))
            iz.append((W.conj() * Z).real[:, psel].sum(2).astype(np.float32))
            L = np.log1p(1e6 * Pw[:, fmask])
            Lp = np.concatenate([prevL[None] if prevL is not None else L[:1], L])
            odf.append(np.maximum(np.diff(Lp, axis=0), 0).sum(1)
                       .astype(np.float32))
            prevL = L[-1]
            carry = data[nwin * hop:].copy()
    odf = np.concatenate(odf)
    return dict(t=(np.arange(len(odf)) * hop + nfft // 2) / fs,
                env=np.concatenate(env), ix=np.concatenate(ix),
                iy=np.concatenate(iy), iz=np.concatenate(iz),
                odf=odf, pfreq=np.atleast_1d(pfreq).astype(np.float64))


# ---------------------------------------------------------------- sources

def cluster_partials(env: np.ndarray, mask=None, th=0.75, min_size=3):
    """Group partial columns into sources: correlate rectified log-envelope
    derivatives (with ±1-frame jitter tolerance), average-linkage cluster.
    Returns a list of column-index lists, largest first; singletons and
    groups below ``min_size`` are dropped (assign them afterwards with
    strike-triggered statistics if needed)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    e = env if mask is None else env[mask]
    dL = np.maximum(np.diff(np.log10(e + EPS), axis=0), 0)
    dL = maximum_filter1d(dL, 3, axis=0)
    C = np.corrcoef(dL.T)
    d = 1 - C
    np.fill_diagonal(d, 0)
    lab = fcluster(linkage(d[np.triu_indices_from(d, 1)], method="average"),
                   th, criterion="distance")
    groups = {}
    for i, l in enumerate(lab):
        groups.setdefault(l, []).append(i)
    out = [v for v in groups.values() if len(v) >= min_size]
    return sorted(out, key=len, reverse=True)


def source_odf(env: np.ndarray, cols) -> np.ndarray:
    """Mean rectified log-envelope derivative over one source's partials."""
    L = np.log10(env[:, cols] + EPS)
    return np.maximum(np.diff(L, axis=0, prepend=L[:1]), 0).mean(1)


def pick_strikes(odf, t, min_sep=0.5, k=1.5, t_max=None):
    """Adaptive strongest-first onset picking.

    Candidates are local maxima exceeding a running median by ``k`` MADs;
    they are accepted strongest-first subject to a ``min_sep`` guard (set it
    just below the shortest true inter-onset interval). Returns strike times.
    """
    dt = float(np.median(np.diff(t)))
    med = median_filter(odf, int(8.0 / dt) | 1)
    mad = median_filter(np.abs(odf - med), int(8.0 / dt) | 1) + 1e-9
    z = (odf - med) / mad
    ismax = odf == maximum_filter1d(odf, 2 * int(0.3 / dt) + 1)
    ok = ismax & (z > k)
    if t_max is not None:
        ok &= t < t_max
    cand = np.flatnonzero(ok)
    taken = np.zeros(len(t), bool)
    guard = int(min_sep / dt)
    keep = []
    for i in cand[np.argsort(z[cand])[::-1]]:
        if not taken[max(0, i - guard):i + guard].any():
            keep.append(i)
            taken[i] = True
    return np.sort(t[np.array(keep, int)]) if keep else np.array([])


# ------------------------------------------------------------- periodicity

def acf_structure(odf, dt, t_mask=None, max_lag_s=8.0, rel=0.25):
    """Cycle period and shortest intra-cycle gap from the ODF autocorrelation.

    Returns (cycle_period, min_gap): the lag of the strongest ACF peak, and
    the shortest peak lag whose value exceeds ``rel`` x the strongest —
    use ``0.8 * min_gap`` as the ``pick_strikes`` separation guard. Estimating
    the cycle from the ACF first keeps the later Rayleigh refinement off
    subharmonics."""
    x = odf if t_mask is None else odf[t_mask]
    y = x - x.mean()
    a = np.correlate(y, y, "full")[len(y) - 1:]
    a /= a[0] + EPS
    m = int(max_lag_s / dt)
    pk, props = find_peaks(a[:m], prominence=0.02)
    if not len(pk):
        return None, None
    vals = a[pk]
    near_max = pk[vals >= 0.9 * vals.max()]     # prefer the fundamental over
    cycle = float(near_max.min() * dt)          # its multiples
    strong = pk[vals >= rel * vals.max()]
    return cycle, float(strong.min() * dt)


def rayleigh_period(times, grid, harm=2):
    """Resultant length of the strike point process folded at each candidate
    period, summed over ``harm`` harmonics (multi-strike cycles)."""
    R = np.zeros(len(grid))
    for i, P in enumerate(grid):
        ph = 2 * np.pi * times / P
        R[i] = sum(np.abs(np.exp(1j * h * ph).mean())
                   for h in range(1, harm + 1))
    return R


def best_period(times, lo=0.5, hi=8.0, step=5e-4, harm=2) -> float:
    """Grid-search ``rayleigh_period`` with parabolic refinement."""
    grid = np.arange(lo, hi, step)
    R = rayleigh_period(times, grid, harm)
    i = int(np.argmax(R))
    if 0 < i < len(R) - 1:
        d = (R[i - 1] - R[i + 1]) / (2 * (R[i - 1] - 2 * R[i] + R[i + 1]))
        return float(grid[i] + d * step)
    return float(grid[i])


def period_track(times, P0, win=150.0, step=30.0, half_range=0.05, harm=2):
    """Sliding-window period estimates around ``P0``; returns (t, P)."""
    ts, Ps = [], []
    g = np.arange(P0 - half_range, P0 + half_range, 5e-4)
    for w0 in np.arange(times.min(), times.max() - win, step):
        sel = times[(times >= w0) & (times < w0 + win)]
        if len(sel) < 20:
            continue
        ts.append(w0 + win / 2)
        Ps.append(g[np.argmax(rayleigh_period(sel, g, harm))])
    return np.array(ts), np.array(Ps)


def phase_clusters(times, P, width=0.12):
    """Split one stream into phase clusters at period ``P`` (histogram modes).
    Returns (phase0, list of (center, mask)): phases are relative to the
    dominant cluster."""
    ph = (times / P) % 1.0
    h, e = np.histogram(ph, bins=100)
    main = e[np.argmax(h)] + 0.005
    ph0 = (ph - main) % 1.0
    out = []
    left = np.ones(len(times), bool)
    while left.any():
        h, e = np.histogram(ph0[left], bins=50, range=(0, 1))
        if h.max() < max(3, 0.02 * len(times)):
            break
        c = e[np.argmax(h)] + 0.01
        d = np.minimum(np.abs(ph0 - c), 1 - np.abs(ph0 - c))
        m = left & (d < width)
        out.append((float(c) % 1.0, m))
        left &= ~m
    return ph0, out


# ---------------------------------------------------- repetition/variation

def cycle_grid(streams: dict, P: float, t_max: float) -> dict:
    """Per-cycle timing residuals and hit rates for named event streams
    sharing one cycle period.

    ``streams`` maps name -> strike times. The cycle phase reference is the
    first stream. Returns per-stream position (s into cycle), residual array
    (NaN = missed cycle), and summary stats: timing sd, lag-1 autocorrelation,
    slow wander vs cycle-to-cycle sd, hit rate.
    """
    names = list(streams)
    ref = streams[names[0]]
    t0 = float(np.median((ref / P) % 1.0)) * P
    ncyc = int((t_max - t0) / P)
    out = {"P": P, "t0": t0, "ncyc": ncyc, "streams": {}}
    for name in names:
        tk = streams[name]
        pos = float(np.median(((tk - t0) / P) % 1.0)) * P
        res = np.full(ncyc, np.nan)
        for s in tk:
            c = int(round((s - t0 - pos) / P))
            if 0 <= c < ncyc:
                r = s - (t0 + c * P + pos)
                if abs(r) < 0.45 * P and (np.isnan(res[c]) or
                                          abs(r) < abs(res[c])):
                    res[c] = r
        v = res[~np.isnan(res)]
        g = ~np.isnan(res[:-1]) & ~np.isnan(res[1:])
        r1 = float(np.corrcoef(res[:-1][g], res[1:][g])[0, 1]) if g.sum() > 3 \
            else np.nan
        m = ~np.isnan(res)
        xi = np.interp(np.arange(ncyc), np.flatnonzero(m), res[m])
        slow = uniform_filter1d(xi, max(3, int(60.0 / P)))
        out["streams"][name] = dict(
            pos=pos, res=res, hit_rate=float(m.mean()),
            sd_ms=float(np.std(v) * 1e3), lag1=r1,
            slow_sd_ms=float(np.std(slow) * 1e3),
            fast_sd_ms=float(np.std(xi - slow) * 1e3))
    return out


# ------------------------------------------------------------------ tonal

def rise_spectrum(take: Take, times, nfft=8192, n_max=150, seed=1):
    """Strike-triggered mean post/pre log-spectral rise (dB) on W.

    The tonal fingerprint of one rhythmic position. If a stream's rise
    spectrum reproduces another source's partials, that stream is cross-talk
    rather than a distinct strike. Returns (freqs, rise_db).
    """
    fs = take.samplerate
    win = np.hanning(nfft)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    rng = np.random.default_rng(seed)
    sel = rng.choice(times, min(n_max, len(times)), replace=False)
    acc, n = np.zeros(len(freqs)), 0
    iW = take.wyzx[0]
    with sf.SoundFile(str(take.path)) as f:
        for s in sel:
            i = int(s * fs)
            if i < nfft or i + nfft > f.frames:
                continue
            f.seek(i - nfft + int(0.02 * fs))
            x = f.read(2 * nfft, dtype="float64", always_2d=True)[:, iW]
            if len(x) < 2 * nfft:
                continue
            pre = np.abs(np.fft.rfft(x[:nfft] * win)) ** 2
            post = np.abs(np.fft.rfft(x[nfft:] * win)) ** 2
            acc += 10 * np.log10((post + EPS) / (pre + EPS))
            n += 1
    return freqs, acc / max(n, 1)


def _activity_masks(F: dict):
    """Auto split minutes into source-active / quiet from 1-2 kHz octave
    power (top vs bottom quartile of per-minute medians)."""
    e = 10 * np.log10(F["oct_pow"][:, 5] + EPS)
    nmin = F["minspec"].shape[0]
    med = np.array([np.median(e[m * 60:(m + 1) * 60]) for m in range(nmin)])
    thr = float(med.mean())
    for _ in range(20):          # 2-means threshold on the per-minute medians
        new = (med[med > thr].mean() + med[med <= thr].mean()) / 2
        if abs(new - thr) < 1e-6:
            break
        thr = new
    return med > thr, med <= thr, med


def run_session(sess, out_dir, n_sources=2, t_stop=None, verbose=True):
    """Full rhythm pipeline for a single-take session; writes
    ``rhythm_overview.png`` and ``rhythm.json``, returns the summary dict."""
    import json
    from .features import load_features
    out_dir = Path(out_dir)
    fdir = out_dir / "features"
    F = load_features(sorted(fdir.glob("*.npz")))
    take = sess.takes[0]
    active, quiet, med = _activity_masks(F)
    if t_stop is None:
        sm = median_filter(10 * np.log10(F["oct_pow"][:, 5] + EPS), 31)
        thr = (np.median(sm[np.repeat(active, 60)[:len(sm)]])
               + np.median(sm[np.repeat(quiet, 60)[:len(sm)]])) / 2
        t_stop = float(np.flatnonzero(sm > thr).max())
    if verbose:
        print(f"  active section ends at {t_stop:.0f} s")
    pfreq, rise = detect_partials(F, active, quiet)
    if verbose:
        print(f"  {len(pfreq)} partials "
              f"({pfreq.min():.0f}-{pfreq.max():.0f} Hz)")
    P = partial_pass(take, pfreq)
    t = P["t"]
    groups = cluster_partials(P["env"], mask=t < t_stop)[:n_sources]
    summary = {"t_stop_s": round(t_stop, 1), "sources": []}
    streams = {}
    for gi, cols in enumerate(groups):
        name = chr(ord("A") + gi)
        odf = source_odf(P["env"], cols)
        dt = float(np.median(np.diff(t)))
        cycle, min_gap = acf_structure(odf, dt, t_mask=t < t_stop)
        if cycle is None:
            continue
        s0 = pick_strikes(odf, t, min_sep=0.8 * min_gap, t_max=t_stop)
        if len(s0) < 20:
            continue
        Pbest = best_period(s0, lo=0.9 * cycle, hi=1.1 * cycle)
        ph0, clusters = phase_clusters(s0, Pbest)
        src = {"name": name,
               "partials_hz": [round(float(pfreq[c]), 1) for c in cols],
               "period_s": round(Pbest, 4),
               "n_strikes": int(len(s0)),
               "phase_clusters": [round(c, 3) for c, _ in clusters]}
        az, el, R = strike_doa(P, s0, cols)
        src["azimuth_deg"], src["elevation_deg"], src["az_R"] = \
            round(az, 1), round(el, 1), round(R, 2)
        for ci, (c, m) in enumerate(clusters):
            streams[f"{name}{ci}"] = s0[m]
        summary["sources"].append(src)
    # flag phase clusters that coincide with another source's strikes
    # (cross-talk between partial groups, not independent strikes)
    for n, tk in streams.items():
        others = np.sort(np.concatenate(
            [v for m, v in streams.items() if m[0] != n[0]] or [np.array([])]))
        if not len(others) or not len(tk):
            continue
        i = np.clip(np.searchsorted(others, tk), 1, len(others) - 1)
        near = np.minimum(np.abs(tk - others[i - 1]), np.abs(tk - others[i]))
        frac = float((near < 0.06).mean())
        if frac > 0.5:
            summary.setdefault("crosstalk_suspects", {})[n] = round(frac, 2)
    if streams:
        P0 = summary["sources"][0]["period_s"]
        grid = cycle_grid(streams, P0, t_stop)
        summary["cycle"] = {
            n: {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in d.items() if k != "res"}
            for n, d in grid["streams"].items()}
        summary["cycle_period_s"] = P0
        _overview_figure(P, streams, grid, t_stop,
                         out_dir / "rhythm_overview.png", title=sess.name)
    (out_dir / "rhythm.json").write_text(json.dumps(summary, indent=2))
    return summary


def _overview_figure(P, streams, grid, t_stop, out_path, title=""):
    """Tempogram of the summed source ODFs, strike phase fold, per-position
    residual tracks, and IOI histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = P["t"]
    dt = float(np.median(np.diff(t)))
    Pc = grid["P"]
    colors = ["#2a78d6", "#d66a2a", "#3d9970", "#9467bd", "#8c564b"]
    fig, ax = plt.subplots(4, 1, figsize=(13, 12))

    odf = P["odf"]
    win, step = int(40 / dt), int(5 / dt)
    max_lag = int(min(4.5, 1.4 * Pc) / dt)
    starts = np.arange(0, int(t_stop / dt) - win, step)
    TG = np.zeros((len(starts), max_lag))
    for i, s0 in enumerate(starts):
        y = odf[s0:s0 + win] - odf[s0:s0 + win].mean()
        a = np.correlate(y, y, "full")[win - 1:win - 1 + max_lag]
        TG[i] = a / (a[0] + EPS)
    ax[0].pcolormesh((starts + win // 2) * dt, np.arange(max_lag) * dt, TG.T,
                     cmap="magma", vmin=0, vmax=0.35, shading="auto")
    ax[0].set(ylabel="lag (s)", title=f"{title} — onset tempogram")

    for i, (n, tk) in enumerate(streams.items()):
        c = colors[i % len(colors)]
        ax[1].plot(tk, (tk / Pc) % 1.0, ".", ms=2.5, color=c, alpha=0.6,
                   label=n)
        res = grid["streams"][n]["res"]
        ax[2].plot(grid["t0"] + np.arange(grid["ncyc"]) * Pc, res * 1e3, ".",
                   ms=2.5, color=c, alpha=0.6, label=n)
        ioi = np.diff(tk)
        ax[3].hist(ioi[ioi < 2 * Pc], bins=120, histtype="step", color=c,
                   label=n)
    ax[1].set(ylabel=f"phase @ {Pc:.3f}s", xlabel="time (s)", ylim=(0, 1))
    ax[2].set(ylabel="residual (ms)", xlabel="time (s)")
    ax[3].set(xlabel="inter-onset interval (s)", ylabel="count")
    for a in ax[1:]:
        a.legend(fontsize=8, loc="upper right")
        a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def strike_doa(P: dict, times, cols, dur=0.25):
    """Median per-strike azimuth/elevation (deg) from the pass arrays,
    energy-integrated over ``dur`` seconds after each strike."""
    t = P["t"]
    dt = float(np.median(np.diff(t)))
    az, el = [], []
    for s in times:
        sl = slice(int(s / dt), min(int(s / dt) + int(dur / dt), len(t)))
        Ix = P["ix"][sl][:, cols].sum()
        Iy = P["iy"][sl][:, cols].sum()
        Iz = P["iz"][sl][:, cols].sum()
        az.append(np.degrees(np.arctan2(Iy, Ix)))
        el.append(np.degrees(np.arctan2(Iz, np.hypot(Ix, Iy))))
    az = np.array(az)
    R = float(np.abs(np.exp(1j * np.radians(az)).mean()))
    return float(np.median(az)), float(np.median(el)), R
