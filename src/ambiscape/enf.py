"""Electric network frequency (ENF) traces from mains hum.

Buildings hum at the mains frequency and its harmonics (50 Hz nominal in
Europe; magnetostriction is strongest at 100 Hz), and the grid's *actual*
frequency wanders by tens of millihertz as load and generation balance.
A long indoor recording therefore carries a continuous, involuntary log of
the grid — usable as a session descriptor (how electrified is this room?),
as a source separator (a "50 Hz" line that does not follow the grid is a
rotor, not electricity), and forensically: matched against published
grid-frequency archives, an ENF trace timestamps a recording independently
of the recorder clock.

- ``hum_peak`` — sub-millihertz line frequency in one mono window
  (zero-padded FFT + parabolic interpolation) with its rise over the local
  spectral floor;
- ``enf_track`` — the trace: windows every ``step_s`` across a whole
  session, one or more harmonics, all scaled to the fundamental;
- ``enf_summary`` — mean/SD/max deviation, coverage, and cross-harmonic
  agreement — the latter is the authenticity check (independent acoustic
  lines reporting the same electrical frequency).

Needs raw audio (one streaming pass over the W channel); the cached
per-minute spectra are far too coarse (5.9 Hz bins) for millihertz work.
"""
from __future__ import annotations

import numpy as np

from .io import Session, read_span

EPS = 1e-30


def hum_peak(w: np.ndarray, fs: int, nominal: float = 50.0,
             search_hz: float = 0.2, nfft_mult: int = 4):
    """Frequency and floor-rise of the strongest line near ``nominal``.

    Zero-padded FFT of the Hann-windowed mono signal, parabolic
    interpolation of the log-power peak within ``nominal ± search_hz``.
    Returns ``(freq_hz, rise_db)``; rise is measured against the median
    power in a ±1.5 Hz-widened neighborhood, so a genuine line scores
    high even on a rumble shoulder.
    """
    n = len(w)
    W = np.fft.rfft(w * np.hanning(n), n * nfft_mult)
    f = np.fft.rfftfreq(n * nfft_mult, 1 / fs)
    P = W.real ** 2 + W.imag ** 2
    m = (f >= nominal - search_hz) & (f <= nominal + search_hz)
    j = int(np.flatnonzero(m)[0] + np.argmax(P[m]))
    a, b, c = (np.log(P[j - 1] + EPS), np.log(P[j] + EPS),
               np.log(P[j + 1] + EPS))
    d = 0.5 * (a - c) / (a - 2 * b + c + EPS)
    floor = np.median(P[(f >= nominal - search_hz - 1.5)
                        & (f <= nominal + search_hz + 1.5)])
    return float(f[j] + d * (f[1] - f[0])), \
        float(10 * np.log10(P[j] / (floor + EPS)))


def enf_track(sess: Session, step_s: float = 300.0, win_s: float = 60.0,
              nominal: float = 50.0, search_hz: float = 0.2,
              harmonics=(1, 2), channel: int = 0) -> dict:
    """Track the mains hum across a whole session.

    One window of ``win_s`` every ``step_s``, per take (windows start 1 s
    into each take and reads shorter than 90 % of the window are skipped —
    recorder 2 GB splits overlap by a fraction of a second, so a read at an
    exact take start can return a sliver of the previous file). Each
    harmonic ``k`` is searched at ``k*nominal ± k*search_hz`` and reported
    scaled to the fundamental.

    Returns ``{"t": absolute seconds, "f": {k: freq_hz/k}, "rise":
    {k: rise_db}}``.
    """
    ts, fk, rk = [], {k: [] for k in harmonics}, {k: [] for k in harmonics}
    for tk in sess.takes:
        t = tk.start + 1.0
        while t + win_s <= tk.end:
            x, fs = read_span(sess, t, win_s)
            if x.shape[0] >= 0.9 * win_s * fs:
                w = x[:, channel].astype(np.float64)
                for k in harmonics:
                    f, r = hum_peak(w, fs, nominal=k * nominal,
                                    search_hz=k * search_hz)
                    fk[k].append(f / k)
                    rk[k].append(r)
                ts.append(t)
            t += step_s
    return {"t": np.array(ts),
            "f": {k: np.array(v) for k, v in fk.items()},
            "rise": {k: np.array(v) for k, v in rk.items()}}


def enf_summary(track: dict, nominal: float = 50.0,
                min_rise_db: float = 6.0) -> dict:
    """Descriptors of an ENF trace.

    Statistics use only windows where the first harmonic rises
    ``min_rise_db`` above the floor; ``coverage`` is the fraction of
    windows that qualify. ``harmonic_agreement_mhz`` is the median absolute
    difference between the first two tracked harmonics (fundamental-scaled)
    where both are detected — millihertz-level agreement authenticates the
    line as electrical.
    """
    ks = sorted(track["f"])
    k0 = ks[0]
    good = track["rise"][k0] >= min_rise_db
    f = track["f"][k0][good]
    out = {
        "n_windows": int(len(track["t"])),
        "coverage": round(float(good.mean()), 2) if len(good) else 0.0,
        "mean_hz": round(float(f.mean()), 4) if len(f) else None,
        "sd_mhz": round(float(f.std() * 1000), 1) if len(f) else None,
        "max_dev_mhz": round(float(np.abs(f - nominal).max() * 1000), 1)
        if len(f) else None,
        "median_rise_db": round(float(np.median(track["rise"][k0])), 1)
        if len(good) else None,
    }
    if len(ks) > 1:
        k1 = ks[1]
        both = good & (track["rise"][k1] >= min_rise_db)
        if both.any():
            d = np.abs(track["f"][k0][both] - track["f"][k1][both])
            out["harmonic_agreement_mhz"] = round(float(np.median(d) * 1000),
                                                  2)
    return out
