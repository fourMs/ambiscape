"""Ecoacoustic indices from the cached log-band spectrogram.

The standard soundscape-ecology battery, so sessions are reportable in the
idiom global acoustic-monitoring corpora expect. All computed from the
cached 1 Hz features — no audio pass:

- **ACI** (acoustic complexity, Pieretti et al. 2011): per-band temporal
  variation |ΔP|/ΣP summed over bands, averaged over 5-min chunks —
  sensitive to biophonic modulation, blind to steady drones;
- **ADI / AEI** (diversity / evenness, Villanueva-Rivera et al. 2011):
  Shannon entropy / Gini coefficient of the occupancy of 1 kHz bins
  (fraction of cells above a threshold re the band maximum);
- **NDSI** (Kasten et al. 2012): (biophony − anthropophony) /
  (biophony + anthropophony) with the conventional bands 2–8 kHz vs
  1–2 kHz, in [−1, 1];
- **BI** (bioacoustic index, Boelman et al. 2007): area of the mean
  2–8 kHz dB spectrum above its minimum;
- **acoustic entropy H** (Sueur et al. 2008): spectral entropy × temporal
  (envelope) entropy, in [0, 1].

Caveats for an ambisonic *indoor* corpus: these indices were designed for
outdoor terrestrial monitoring; report them for comparability, but read
NDSI/BI as "energy in the bird band", not as proof of birds — a 4 kHz
ventilation hiss scores as "biophony". Combine with the taxonomy layer
before interpreting.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-20


def _band_centers(logf):
    logf = np.asarray(logf, float)
    return np.sqrt(logf[:-1] * logf[1:])


def aci(F: dict, chunk_s: float = 300.0) -> float:
    """Acoustic complexity index, mean over ``chunk_s`` chunks."""
    S = np.asarray(F["logspec"], float)
    n = max(2, int(chunk_s))
    vals = []
    for i0 in range(0, S.shape[0] - n + 1, n):
        c = S[i0:i0 + n]
        vals.append(float((np.abs(np.diff(c, axis=0)).sum(0)
                           / (c.sum(0) + EPS)).sum()))
    return float(np.mean(vals)) if vals else 0.0


def _occupancy(F: dict, fmax: float = 10000.0, bin_hz: float = 1000.0,
               thresh_db: float = -50.0):
    """Fraction of spectrogram cells above ``thresh_db`` re the global
    maximum, per ``bin_hz`` frequency bin up to ``fmax``."""
    S = np.asarray(F["logspec"], float)
    fc = _band_centers(F["logf"])
    ref = S.max() + EPS
    lvl = 10 * np.log10(S / ref + EPS)
    edges = np.arange(0, fmax + bin_hz, bin_hz)
    occ = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fc >= lo) & (fc < hi)
        if m.any():
            occ.append(float((lvl[:, m] > thresh_db).mean()))
    return np.array(occ)


def adi_aei(F: dict, **kw):
    """Acoustic diversity (Shannon, normalized) and evenness (Gini)."""
    occ = _occupancy(F, **kw)
    p = occ / (occ.sum() + EPS)
    adi = float(-(p * np.log(p + EPS)).sum() / np.log(len(p) + EPS))
    x = np.sort(occ)
    n = len(x)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(x)
                 / (n * x.sum() + EPS))
    return adi, gini


def ndsi(F: dict, anthro=(1000.0, 2000.0), bio=(2000.0, 8000.0)) -> float:
    """Normalized difference soundscape index in [−1, 1]."""
    fc = _band_centers(F["logf"])
    S = np.asarray(F["logspec"], float).mean(0)
    a = S[(fc >= anthro[0]) & (fc < anthro[1])].sum()
    b = S[(fc >= bio[0]) & (fc < bio[1])].sum()
    return float((b - a) / (b + a + EPS))


def bioacoustic_index(F: dict, band=(2000.0, 8000.0)) -> float:
    """Boelman BI: area of the mean band dB spectrum above its minimum."""
    fc = _band_centers(F["logf"])
    m = (fc >= band[0]) & (fc <= band[1])
    s = 10 * np.log10(np.asarray(F["logspec"], float).mean(0)[m] + EPS)
    return float((s - s.min()).sum())


def acoustic_entropy(F: dict) -> float:
    """Sueur H = spectral entropy × temporal entropy, in [0, 1]."""
    S = np.asarray(F["logspec"], float)
    ps = S.mean(0)
    ps = ps / (ps.sum() + EPS)
    hf = float(-(ps * np.log(ps + EPS)).sum() / np.log(len(ps)))
    env = np.asarray(F["rms_w"], float)
    pe = env / (env.sum() + EPS)
    ht = float(-(pe * np.log(pe + EPS)).sum() / np.log(len(pe)))
    return hf * ht


def indices(F: dict) -> dict:
    """The full battery as one dict."""
    adi_, aei_ = adi_aei(F)
    return {
        "aci": round(aci(F), 1),
        "adi": round(adi_, 3),
        "aei": round(aei_, 3),
        "ndsi": round(ndsi(F), 3),
        "bi": round(bioacoustic_index(F), 1),
        "acoustic_entropy": round(acoustic_entropy(F), 3),
    }


def summarize_ecology(F: dict) -> dict:
    """Alias of :func:`indices` for the analyze-summary pipeline."""
    return indices(F)
