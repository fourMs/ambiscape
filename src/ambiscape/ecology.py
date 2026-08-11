"""Ecoacoustic indices from the cached log-band spectrogram.

The standard soundscape-ecology battery, so sessions are reportable in the
idiom global acoustic-monitoring corpora expect. All computed from the
cached 1 Hz features — no audio pass:

- **ACI** (acoustic complexity, Pieretti et al. 2011): per-band temporal
  variation |ΔP|/ΣP summed over bands, averaged over 5-min chunks —
  sensitive to biophonic modulation, blind to steady drones, and
  undefined (``None``) for recordings shorter than one chunk;
- **ADI / AEI** (diversity / evenness, Villanueva-Rivera et al. 2011):
  Shannon entropy / Gini coefficient of the occupancy of 1 kHz bins
  (fraction of cells above a threshold re the band maximum);
- **NDSI** (Kasten et al. 2012): (biophony − anthrophony) /
  (biophony + anthrophony) with the conventional bands 2–8 kHz vs
  1–2 kHz, in [−1, 1];
- **BI** (bioacoustic index, Boelman et al. 2007): area of the mean
  2–8 kHz dB spectrum above its minimum;
- **acoustic entropy H** (Sueur et al. 2008): spectral entropy × temporal
  (envelope) entropy, in [0, 1].

INDOORS, SOME OF THESE FAIL AND SOME DO NOT, and which is which was measured
rather than reasoned. Three dawn and dusk choruses against synthetic
ventilation noise (Jensenius 2026, *When ventilation outperforms the dawn
chorus*):

===================  ================  =============
index                ventilation       choruses
===================  ================  =============
ADI                  0.977             0.927–0.968
bird-band Ht         0.998             0.701–0.913
NDSI                 −0.139            0.707–0.997
acoustic entropy H   0.387             0.492–0.610
===================  ================  =============

ADI and bird-band temporal entropy rate a duct *above every chorus*, and Ht
gives the fan the highest value in the whole comparison, because a stationary
signal is perfectly uniform in time. NDSI, H, AEI and BI are not fooled by
this material.

The division is not about which band an index looks at. The indices that fail
read **occupancy and time** — how many cells are busy, how evenly spread
across the hours — and a stationary broadband source saturates both whatever
its spectral tilt. The ones that resist read **spectral shape**, and duct
noise falling steadily with frequency is neither bright nor flat. Expect the
same split for any steady mechanical source; expect NDSI to fail as well
wherever the machine's own energy sits inside the bio band, which is the 4 kHz
hiss case an earlier version of this note wrongly generalised from.

Scale is the other warning. Over 14 node-days of an inhabited home, ADI moves
less across an entire week — 0.031 on one node — than it does between two
microphones standing metres apart in the same room, 0.036. A descriptor whose
weekly variation is smaller than its disagreement between two positions in one
room is not measuring the week.

Report them for comparability with outdoor corpora, read the occupancy-and-
time pair as "is anything steady here", and go to :mod:`ambiscape.biophony`,
which measures structure rather than energy, before reading any of them as
life.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-20


def _band_centers(logf):
    logf = np.asarray(logf, float)
    return np.sqrt(logf[:-1] * logf[1:])


def aci(F: dict, chunk_s: float = 300.0) -> float | None:
    """Acoustic complexity index, mean over ``chunk_s`` chunks.

    ACI accumulates |ΔP| over a whole chunk, so its magnitude is a
    function of the chunk length: values are comparable only between
    recordings analysed with complete chunks of the same size. A
    recording shorter than one chunk therefore has no ACI, and ``None``
    is returned — a numeric zero would be indistinguishable from a
    measured minimum (clip corpora of 5–30 s are the common case).
    """
    S = np.asarray(F["logspec"], float)
    n = max(2, int(chunk_s))
    if S.shape[0] < n:
        return None
    vals = []
    for i0 in range(0, S.shape[0] - n + 1, n):
        c = S[i0:i0 + n]
        vals.append(float((np.abs(np.diff(c, axis=0)).sum(0)
                           / (c.sum(0) + EPS)).sum()))
    return float(np.mean(vals)) if vals else None


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
    """Acoustic diversity (Shannon, normalised) and evenness (Gini)."""
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
    aci_ = aci(F)                     # None below one full chunk (5 min)
    return {
        "aci": None if aci_ is None else round(aci_, 1),
        "adi": round(adi_, 3),
        "aei": round(aei_, 3),
        "ndsi": round(ndsi(F), 3),
        "bi": round(bioacoustic_index(F), 1),
        "acoustic_entropy": round(acoustic_entropy(F), 3),
    }


def summarize_ecology(F: dict) -> dict:
    """Alias of :func:`indices` for the analyze-summary pipeline."""
    return indices(F)
