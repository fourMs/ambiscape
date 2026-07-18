"""Biophony measures: capturing nature and animal sounds by structure.

The ecoacoustic battery in :mod:`ambiscape.ecology` (NDSI, BI, ADI) reads
*energy in a band*; it cannot tell a dawn chorus from a ventilation hiss.
Biophony is distinguished by how it is **structured** — narrowband, tonal,
bursty in time, and (in an ambisonic recording) arriving from many elevated
bearings at once. This module measures that structure from the cached
features, no audio pass:

- ``narrowband_activity`` — persistent narrow spectral peaks in the bird
  band per minute (from the per-minute high-resolution PSD): birdsong is
  narrowband and tonal, wind and machines are broadband;
- ``band_temporal_entropy`` — Sueur temporal entropy of the bird-band
  envelope: structured vocalization concentrates energy in time (low Ht),
  a steady noise floor spreads it evenly (Ht → 1);
- ``band_activity`` — fraction of seconds and event rate where the bird
  band rises above its own running background (Towsey-style acoustic
  activity), restricted to the biophony band;
- ``spatial_dispersion`` — the ambisonic layer no other corpus tool has:
  the directional entropy and above-horizon energy fraction of the
  *bird-band foreground* — a chorus of many birds from many elevated
  directions is unmistakable, and it cross-checks a suspicious NDSI.

``summarize_biophony`` returns the descriptor set for the analyze summary.

Caveats: these are acoustic-structure proxies, not detections. A tonal
alarm, a whistling kettle, or a squealing fan belt can mimic biophonic
structure; confirm species with the BirdNET hook
(:func:`ambiscape.ml.birdnet_session`, ``[ml]`` extra) on the hi-fi
windows. The default band (2–11 kHz) targets temperate birdsong; widen it
(insects reach 8–16 kHz, many mammals sit below 2 kHz) per habitat.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label, percentile_filter

from .circstats import EPS

BIRD_BAND = (2000.0, 11000.0)


def _centers(F):
    if "fc" in F:
        return np.asarray(F["fc"], float)
    logf = np.asarray(F["logf"], float)
    return np.sqrt(logf[:-1] * logf[1:])


def _band_envelope(F, band):
    fc = _centers(F)
    m = (fc >= band[0]) & (fc <= band[1])
    return np.asarray(F["logspec"], float)[:, m].sum(1)


def narrowband_activity(F: dict, band=BIRD_BAND, min_prom_db: float = 6.0,
                        min_peaks: int = 2) -> dict:
    """Per-minute count of narrowband tonal peaks in ``band``.

    Uses :func:`ambiscape.tonality.tonal_peaks` on each row of the cached
    per-minute PSD. Returns median peaks/min, the per-minute counts, and
    the fraction of minutes with at least ``min_peaks`` (an "active"
    biophonic minute).

    On its own this is a *weak* biophony discriminator and must not be read
    as a bird count: (a) minute-averaging smears frequency-swept birdsong,
    so a busy dawn chorus shows only a modest per-minute peak count while
    its ``max`` and ``active_minute_fraction`` spike; (b) steady machine
    harmonics are narrowband too and score just as high. What separates
    birds from machines is the *combination* with the temporal
    (:func:`band_temporal_entropy`) and spatial
    (:func:`spatial_dispersion`) measures — biophony is narrowband **and**
    bursty **and** spread across elevated bearings; a machine tone is
    narrowband but steady, low, and directional.
    """
    from .tonality import tonal_peaks
    ms, freqs = np.asarray(F["minspec"], float), np.asarray(F["freqs"], float)
    counts = np.array([
        len(tonal_peaks(ms[i], freqs, fmin=band[0], fmax=band[1],
                        min_prom_db=min_prom_db)[0])
        for i in range(ms.shape[0])])
    return {
        "median_peaks_per_min": float(np.median(counts)) if len(counts) else 0.0,
        "max_peaks_per_min": int(counts.max()) if len(counts) else 0,
        "active_minute_fraction": round(float((counts >= min_peaks).mean()), 2)
        if len(counts) else 0.0,
        "per_min": counts,
    }


def band_temporal_entropy(F: dict, band=BIRD_BAND) -> float:
    """Sueur temporal entropy Ht of the bird-band envelope, in [0, 1].

    Low = energy concentrated in time (structured vocalization); near 1 =
    even over time (steady band, no biophonic events).
    """
    env = _band_envelope(F, band)
    p = env / (env.sum() + EPS)
    return round(float(-(p * np.log(p + EPS)).sum() / np.log(len(p))), 3) \
        if len(p) > 1 else 0.0


def band_activity(F: dict, band=BIRD_BAND, k_db: float = 3.0,
                  bg_win_s: float = 300.0, min_dur_s: int = 1) -> dict:
    """Acoustic activity of the bird band above its running background.

    The band envelope (dB) is compared to a running 10th-percentile
    background over ``bg_win_s``; seconds exceeding it by ``k_db`` are
    active. Returns the active-second fraction, event rate per minute
    (runs of >= ``min_dur_s`` active seconds), and median event duration.
    """
    env_db = 10 * np.log10(_band_envelope(F, band) + EPS)
    n = max(3, int(round(bg_win_s)) | 1)
    bg = percentile_filter(env_db, 10, size=n, mode="nearest")
    active = env_db > bg + k_db
    lab, nlab = label(active)
    durs = [int((lab == i).sum()) for i in range(1, nlab + 1)]
    durs = [d for d in durs if d >= min_dur_s]
    dur_min = max(len(env_db) / 60.0, 1e-9)
    return {
        "active_fraction": round(float(active.mean()), 3),
        "event_rate_per_min": round(len(durs) / dur_min, 1),
        "event_median_dur_s": round(float(np.median(durs)), 1) if durs else None,
    }


def _fg_weights(F, band, k_db=3.0, bg_win_s=300.0):
    """Per-second bird-band foreground energy (linear, above background)."""
    env = _band_envelope(F, band)
    env_db = 10 * np.log10(env + EPS)
    n = max(3, int(round(bg_win_s)) | 1)
    bg = percentile_filter(env_db, 10, size=n, mode="nearest")
    return np.where(env_db > bg + k_db, env, 0.0)


def spatial_dispersion(F: dict, band=BIRD_BAND, nbins: int = 36,
                       limit_deg: float = 10.0) -> dict:
    """Directional spread and elevation of the bird-band foreground.

    The azimuth histogram is weighted by the per-second bird-band
    foreground energy (band level above its running background), so only
    seconds carrying biophonic energy contribute. Returns the normalized
    directional entropy (0 = one bearing, 1 = all around) and the fraction
    of that foreground energy arriving from above ``limit_deg`` elevation
    (birds aloft) versus at/below the horizon.
    """
    w = _fg_weights(F, band)
    az, el = np.asarray(F["az"], float), np.asarray(F["el"], float)
    tot = w.sum() + EPS
    h, _ = np.histogram(az, bins=nbins, range=(-180, 180), weights=w)
    q = h / (h.sum() + EPS)
    ent = float(-(q * np.log(q + EPS)).sum() / np.log(nbins))
    return {
        "directional_entropy": round(ent, 3),
        "above_horizon_fraction": round(float(w[el > limit_deg].sum() / tot), 2),
        "below_horizon_fraction": round(float(w[el < -limit_deg].sum() / tot), 2),
    }


def summarize_biophony(F: dict, band=BIRD_BAND,
                       min_active_fraction: float = 0.02) -> dict:
    """Biophony descriptors for the analyze summary.

    The *spatial* biophony descriptors are only meaningful when the bird
    band actually carries foreground energy: in a quiet, birdless room a
    trickle of high-frequency energy that happens to arrive from above would
    otherwise read as ``above_horizon_fraction`` = 1.0, a false positive.
    When the band-active fraction is below ``min_active_fraction`` the
    directional and horizon descriptors are therefore reported as ``None``
    rather than as spurious numbers.
    """
    na = narrowband_activity(F, band)
    act = band_activity(F, band)
    active = act["active_fraction"] >= min_active_fraction
    sd = spatial_dispersion(F, band) if active else {}
    return {
        "bird_peaks_per_min": na["median_peaks_per_min"],
        "bird_active_minute_fraction": na["active_minute_fraction"],
        "bird_band_activity_pct": round(act["active_fraction"] * 100, 1),
        "bird_event_rate_per_min": act["event_rate_per_min"],
        "bird_temporal_entropy": band_temporal_entropy(F, band),
        "bird_directional_entropy": sd.get("directional_entropy"),
        "bird_above_horizon_fraction": sd.get("above_horizon_fraction"),
    }
