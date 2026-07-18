"""Per-band running background and spectral foreground decomposition.

The broadband event detector in :mod:`analysis` misses band-limited events
riding on a loud bed in other bands (distant bells over traffic move their
octave a few dB while the broadband level barely changes). This module works
on the cached 1 Hz log-band spectrogram (``F["logspec"]``, 96 bands):

- ``band_background`` — running low-percentile background per band;
- ``foreground`` — dB exceedance and the per-second **foreground fraction**
  (share of total power sitting above the spectral background);
- ``spectral_events`` — connected spectro-temporal regions of exceedance
  (time x band blobs), each with onset, duration, band span, and peak rise;
- ``summarize_foreground`` — session descriptors appended to the analyze
  summary and README.

All functions are pure array transforms on cached features — no audio pass.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

EPS = 1e-20


def band_background(logspec: np.ndarray, win_s: float = 300.0,
                    pct: float = 10.0) -> np.ndarray:
    """Running ``pct``-percentile background per log band.

    ``logspec`` is the (nsec, nband) power array from the cached features;
    the window is in seconds (= rows). Returns the same shape.
    """
    n = max(3, int(round(win_s)) | 1)
    return ndimage.percentile_filter(logspec, pct, size=(n, 1),
                                     mode="nearest")


def foreground(logspec: np.ndarray, bg: np.ndarray):
    """dB rise above the spectral background and per-second foreground
    fraction (share of total power more than 3 dB above background)."""
    rise_db = 10 * np.log10((logspec + EPS) / (bg + EPS))
    fg_mask = rise_db > 3.0
    frac = (logspec * fg_mask).sum(1) / (logspec.sum(1) + EPS)
    return rise_db, frac


def spectral_events(rise_db: np.ndarray, logf: np.ndarray,
                    thresh_db: float = 6.0, min_dur_s: float = 2.0,
                    min_bands: int = 2) -> list[dict]:
    """Connected regions of band-wise exceedance as event dicts.

    A spectral event is a blob in the (time x band) plane where the rise
    exceeds ``thresh_db``, lasting >= ``min_dur_s`` and spanning >=
    ``min_bands`` bands. Returns onset/duration (s), band span (Hz), and
    peak rise (dB), sorted by onset.
    """
    lab, nlab = ndimage.label(rise_db > thresh_db)
    out = []
    for sl_t, sl_b in ndimage.find_objects(lab):
        dur = sl_t.stop - sl_t.start
        nb = sl_b.stop - sl_b.start
        if dur < min_dur_s or nb < min_bands:
            continue
        blob = rise_db[sl_t, sl_b]
        out.append({
            "t0_s": int(sl_t.start),
            "dur_s": int(dur),
            "f_lo_hz": round(float(logf[sl_b.start]), 1),
            "f_hi_hz": round(float(logf[min(sl_b.stop, len(logf) - 1)]), 1),
            "peak_rise_db": round(float(blob.max()), 1),
        })
    return sorted(out, key=lambda e: e["t0_s"])


def masking_index(F: dict, active: np.ndarray, quiet: np.ndarray) -> dict:
    """How much a dominant source hides the rest of the field — the "lo-fi"
    claim as a number.

    ``active``/``quiet`` are boolean second-masks (source on / off). Per log
    band, the floor elevation is the rise of the active-state median level
    above the quiet-state median: ambient sounds in that band must now
    exceed the elevated typical floor to be audible. Returns the median and
    maximum elevation over 250 Hz–8 kHz, the fraction of bands elevated by
    more than 6 dB, and the per-band curve.
    """
    ls = F["logspec"]
    logf = F["logf"]
    el = 10 * np.log10(np.median(ls[active], axis=0) + EPS) \
        - 10 * np.log10(np.median(ls[quiet], axis=0) + EPS)
    band = (logf[:-1] >= 250) & (logf[:-1] <= 8000)
    return {
        "floor_elevation_median_db": round(float(np.median(el[band])), 1),
        "floor_elevation_max_db": round(float(el[band].max()), 1),
        "bands_masked_gt6db_fraction": round(float((el[band] > 6).mean()), 2),
        "elevation_db_per_band": [round(float(v), 1) for v in el],
    }


def source_fingerprint(F: dict, active: np.ndarray, quiet: np.ndarray,
                       fmin: float = 25.0, fmax: float = 16000.0,
                       min_prom_db: float = 6.0, max_peaks: int = 20) -> dict:
    """Spectral fingerprint of a source: active-minus-quiet mean PSDs.

    ``active``/``quiet`` are boolean masks over the *minutes* of
    ``F["minspec"]`` (source clearly on / clearly off, e.g. from
    :func:`ambiscape.states.state_segments`). The rise curve is the dB
    difference of the two mean spectra — the source's own spectrum with the
    room ambience subtracted. Narrowband peaks of the rise are extracted and
    passed through the harmonic sieve, so a blade-pass or compressor comb
    reports its base frequency.

    Returns dict: ``freqs``/``rise_db`` (the full curve), ``rise_max_db``/
    ``rise_max_hz`` (the turbulence hump), ``peaks`` (list of
    ``{f_hz, rise_db}``), and ``comb`` (``{f0_hz, harmonicity}`` of the peak
    set, ``f0_hz`` None when there are no peaks).
    """
    from scipy.ndimage import median_filter as _medf
    from scipy.signal import find_peaks as _find_peaks
    from .tonality import harmonic_sieve

    freqs = np.asarray(F["freqs"], float)
    S_a = F["minspec"][np.asarray(active, bool)].mean(0)
    S_q = F["minspec"][np.asarray(quiet, bool)].mean(0)
    m = (freqs >= fmin) & (freqs <= fmax)
    rise = 10 * np.log10((S_a[m] + EPS) / (S_q[m] + EPS))
    fsel = freqs[m]

    # hump: broad maximum of the smoothed rise
    smooth = _medf(rise, size=51, mode="nearest")
    i_max = int(np.argmax(smooth))
    # peaks: narrowband lines above the smoothed curve
    line = rise - smooth
    pk, props = _find_peaks(line, height=min_prom_db, distance=5)
    order = np.argsort(props["peak_heights"])[::-1][:max_peaks]
    keep = np.sort(pk[order])
    peaks = [{"f_hz": round(float(fsel[i]), 1),
              "rise_db": round(float(rise[i]), 1)} for i in keep]
    f0, h = harmonic_sieve(fsel[keep], 10 ** (rise[keep] / 10)) \
        if len(keep) else (None, 0.0)
    return {
        "freqs": fsel, "rise_db": rise,
        "rise_max_db": round(float(smooth[i_max]), 1),
        "rise_max_hz": round(float(fsel[i_max]), 1),
        "peaks": peaks,
        "comb": {"f0_hz": round(f0, 1) if f0 else None, "harmonicity": h},
    }


def summarize_foreground(F: dict, win_s: float = 300.0) -> dict:
    """Foreground descriptors for the analyze summary."""
    bg = band_background(F["logspec"], win_s=win_s)
    rise_db, frac = foreground(F["logspec"], bg)
    ev = spectral_events(rise_db, F["logf"])
    dur_min = max(len(frac) / 60.0, 1e-9)
    return {
        "fg_fraction_median": round(float(np.median(frac)), 2),
        "fg_fraction_p90": round(float(np.percentile(frac, 90)), 2),
        "spectral_events_per_min": round(len(ev) / dur_min, 1),
        "spectral_event_median_dur_s": (
            round(float(np.median([e["dur_s"] for e in ev])), 1)
            if ev else None),
    }
