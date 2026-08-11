"""Tonalness timeline, harmonic sieve, and inharmonicity.

Works entirely from the cached per-minute mean PSD (``minspec``):

- ``tonal_peaks`` — prominent narrowband components per minute (dB above a
  running spectral floor), the raw material for everything below;
- ``tonal_tracks`` — peaks linked across minutes into tracks (a hum, a bell
  partial, a beep), each with duration, median frequency, and cents drift:
  the tonalness timeline;
- ``group_tracks`` — those segments regrouped into lines, for a source that
  pauses or changes speed and so appears as several tracks;
- ``harmonic_sieve`` — best f0 explaining a minute's peak set as a harmonic
  series; the unexplained remainder is the inharmonic tonal content.
  Voices, engines, and music score high harmonicity; bells score low
  (their partial series 1 : 2 : 2.4 : 3 : 4 is not harmonic);
- ``pitch_class_profile`` — tonal peak energy folded onto the 12 pitch
  classes: what "key" the soundscape hums in.

``run_session`` writes ``tonality.json`` + ``tonality.png``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

EPS = 1e-20
NOTE = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def tonal_peaks(spec_row: np.ndarray, freqs: np.ndarray, fmin=40.0,
                fmax=8000.0, min_prom_db=8.0, max_n=40):
    """Prominent narrowband peaks in one mean spectrum.

    The floor is a wide median filter on the log spectrum; peaks must rise
    ``min_prom_db`` above it. Returns (freq, prominence_db, power) arrays.
    """
    m = (freqs >= fmin) & (freqs <= fmax)
    ls = 10 * np.log10(spec_row[m] + EPS)
    floor = median_filter(ls, size=101, mode="nearest")
    rise = ls - floor
    pk, props = find_peaks(rise, height=min_prom_db, distance=3)
    order = np.argsort(props["peak_heights"])[::-1][:max_n]
    keep = np.sort(pk[order])
    return freqs[m][keep], rise[keep], spec_row[m][keep]


def tonal_tracks(minspec: np.ndarray, freqs: np.ndarray, tol_cents=40.0,
                 min_minutes=2, **peak_kw):
    """Link per-minute peaks into tracks. Returns a list of dicts sorted by
    duration (longest first): median freq, span of minutes, mean prominence,
    and total drift in cents."""
    per_min = [tonal_peaks(minspec[i], freqs, **peak_kw)
               for i in range(minspec.shape[0])]
    open_tracks, done = [], []
    for mi, (fq, prom, _pw) in enumerate(per_min):
        used = np.zeros(len(fq), bool)
        for tr in list(open_tracks):
            cents = 1200 * np.abs(np.log2(fq / (tr["f"][-1] + EPS) + EPS))
            j = int(np.argmin(cents)) if len(cents) else -1
            if j >= 0 and cents[j] < tol_cents and not used[j]:
                tr["f"].append(float(fq[j]))
                tr["prom"].append(float(prom[j]))
                tr["m1"] = mi
                used[j] = True
            elif mi - tr["m1"] > 1:
                open_tracks.remove(tr)
                done.append(tr)
        for j in np.flatnonzero(~used):
            open_tracks.append(dict(f=[float(fq[j])], prom=[float(prom[j])],
                                    m0=mi, m1=mi))
    done += open_tracks
    out = []
    for tr in done:
        if tr["m1"] - tr["m0"] + 1 < min_minutes:
            continue
        f = np.array(tr["f"])
        out.append({
            "f_median_hz": round(float(np.median(f)), 1),
            "t0_min": tr["m0"], "t1_min": tr["m1"],
            "minutes": tr["m1"] - tr["m0"] + 1,
            "prominence_db": round(float(np.mean(tr["prom"])), 1),
            "drift_cents": round(float(1200 * np.log2(
                (f[-1] + EPS) / (f[0] + EPS))), 1),
        })
    return sorted(out, key=lambda t: -t["minutes"])


def group_tracks(tracks: list[dict], tol_cents: float = 60.0) -> list[dict]:
    """Group track segments that are the same line, interrupted.

    :func:`tonal_tracks` answers "how long was a line continuously present at
    this frequency". A source that pauses, or shifts frequency and comes back,
    is therefore several tracks --- correct as tracking and misleading as a
    description of the source. A dishwasher's circulation pump, which changes
    speed between programme phases, appears as five separate tracks.

    This regroups them: segments whose median frequencies lie within
    ``tol_cents`` become one line, regardless of the gaps between them.
    Returns one entry per line with ``f_median_hz``, ``minutes`` (the total
    across its segments), ``n_segments``, ``t0_min`` and ``t1_min`` spanning
    the first to the last, and ``prominence_db`` averaged over segments
    weighted by their length.

    Grouping by frequency alone is the assumption to be aware of: two
    unrelated sources that happen to share a frequency are merged, and one
    source that moves further than ``tol_cents`` between segments is not. Use
    it to count lines, not to attribute them.

    **Across a sensor network it separates a building from a room**, which is
    the second use it has earned. A machine bolted to a structure radiates a
    stable line and reaches rooms through the fabric, so matching the grouped
    lines between nodes says which steady sources belong to the building and
    which to one room. On the twelve SINS nodes, nine line groups below 1 kHz
    reach two or more rooms that are behind doors from one another --- 53, 98,
    107, 121, 342, 391, 473, 504 and 652 Hz --- and cannot have travelled
    between them through the air; the bedroom separately carries five lines
    between 45 and 84 Hz that nobody else hears, which are its own.

    **A line shared by every node may be the recorders rather than the
    building.** Identical hardware shares its artefacts. The most widely shared
    line in that corpus sits at 7,937 Hz in all five rooms across eleven nodes,
    and it is sensor self-noise: it falls in the band where every node of that
    deployment is already flagged floor-suspect. Split the candidates by band
    before reading a shared line as a shared source, and treat anything inside
    the known self-noise region as the instrument until shown otherwise.
    """
    if not tracks:
        return []
    order = sorted(tracks, key=lambda t: t["f_median_hz"])
    groups: list[list[dict]] = [[order[0]]]
    for t in order[1:]:
        ref = float(np.median([x["f_median_hz"] for x in groups[-1]]))
        if abs(1200 * np.log2(t["f_median_hz"] / ref)) <= tol_cents:
            groups[-1].append(t)
        else:
            groups.append([t])
    out = []
    for g in groups:
        mins = sum(x["minutes"] for x in g)
        out.append({
            "f_median_hz": round(float(np.median(
                [x["f_median_hz"] for x in g])), 1),
            "minutes": int(mins),
            "n_segments": len(g),
            "t0_min": min(x["t0_min"] for x in g),
            "t1_min": max(x["t1_min"] for x in g),
            "prominence_db": round(float(sum(
                x["prominence_db"] * x["minutes"] for x in g) / max(mins, 1)), 1),
        })
    return sorted(out, key=lambda t: -t["minutes"])


def harmonic_sieve(fq: np.ndarray, power: np.ndarray, f0_min=60.0,
                   f0_max=1200.0, tol_cents=35.0, max_harm=12):
    """Best f0 explaining the peak set as harmonics k*f0.

    Candidate f0s are every peak divided by k = 1..6; the score is the
    power-weighted fraction of peaks within ``tol_cents`` of a harmonic.
    Returns (f0, harmonicity in [0,1]) — harmonicity is the explained power
    fraction; 1 − harmonicity is the inharmonicity index.

    **The defaults are tuned for voices and music, and mislead on
    machinery.** Three of them, each demonstrated on a dishwasher's
    circulation pump whose peak structure puts its shaft near 46 Hz:

    ``f0_min`` at 60 Hz sits above many machine shaft rates. On that pump it
    excludes the fundamental and returns its second harmonic, 91.9 Hz. Lower
    it for anything mechanical.

    ``tol_cents`` is proportional, so 35 cents is ±5.6 Hz at 275 Hz and
    ±17 Hz at 825 Hz — wide enough that a candidate can collect high
    harmonics by coincidence. The default returns 68.6 Hz at harmonicity
    0.73; tightening to 8 cents returns 46.0 Hz at 0.45. The looser fit
    *scores higher while explaining fewer of the strong low peaks*, so a
    harmonicity figure means little without the tolerance beside it.

    ``max_harm`` at 12 is short for a machine comb; the same pump is
    tracked to k = 26.

    None of this is a defect in the sieve. It is that "which harmonic
    series is this" has more than one defensible answer, and the parameters
    choose between them rather than merely tuning precision. Report them.
    """
    if len(fq) == 0:
        return None, 0.0
    cands = np.concatenate([fq / k for k in range(1, 7)])
    cands = cands[(cands >= f0_min) & (cands <= f0_max)]
    best_f0, best = None, 0.0
    total = power.sum() + EPS
    for f0 in cands:
        k = np.clip(np.round(fq / f0), 1, max_harm)
        cents = 1200 * np.abs(np.log2(fq / (k * f0)))
        score = float(power[cents < tol_cents].sum() / total)
        if score > best:
            best_f0, best = float(f0), score
    return best_f0, round(best, 3)


def pitch_class_profile(minspec: np.ndarray, freqs: np.ndarray,
                        minutes=None, **peak_kw):
    """Tonal peak power folded onto 12 pitch classes (A4 = 440 Hz)."""
    pcp = np.zeros(12)
    rows = range(minspec.shape[0]) if minutes is None else minutes
    for i in rows:
        fq, _prom, pw = tonal_peaks(minspec[i], freqs, **peak_kw)
        if len(fq):
            pc = np.mod(np.round(12 * np.log2(fq / 440.0) + 69), 12).astype(int)
            np.add.at(pcp, pc, pw)
    return pcp / (pcp.sum() + EPS)


def run_session(sess, out_dir) -> dict:
    """Tonality timeline + per-minute tonalness/harmonicity + PCP + figure."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .features import load_features

    out_dir = Path(out_dir)
    F = load_features(sorted((out_dir / "features").glob("*.npz")))
    minspec, freqs = F["minspec"], F["freqs"]
    nmin = minspec.shape[0]

    tracks = tonal_tracks(minspec, freqs)
    tonalness, harmonicity = np.zeros(nmin), np.full(nmin, np.nan)
    for i in range(nmin):
        fq, _prom, pw = tonal_peaks(minspec[i], freqs)
        band = (freqs >= 40) & (freqs <= 8000)
        tonalness[i] = float(pw.sum() / (minspec[i][band].sum() + EPS))
        if len(fq) >= 3:
            _f0, h = harmonic_sieve(fq, pw)
            harmonicity[i] = h
    pcp = pitch_class_profile(minspec, freqs)

    doc = {
        "tracks": tracks[:40],
        "tonalness_median": round(float(np.median(tonalness)), 3),
        "harmonicity_median": round(float(np.nanmedian(harmonicity)), 3),
        "inharmonicity_median": round(1 - float(np.nanmedian(harmonicity)), 3),
        "pitch_class_profile": {NOTE[i]: round(float(pcp[i]), 3)
                                for i in range(12)},
        "top_pitch_classes": [NOTE[i] for i in np.argsort(pcp)[::-1][:3]],
    }
    (out_dir / "tonality.json").write_text(json.dumps(doc, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.6), dpi=130,
                           gridspec_kw=dict(width_ratios=[2.4, 1]))
    for tr in tracks:
        ax[0].plot([tr["t0_min"], tr["t1_min"] + 1],
                   [tr["f_median_hz"]] * 2, lw=max(0.8, tr["prominence_db"] / 6),
                   color="#2a78d6", alpha=0.7, solid_capstyle="butt")
    ax[0].set(yscale="log", xlabel="minute of session", ylabel="Hz",
              title=f"{sess.name} — tonal tracks (width = prominence)")
    ax[0].grid(alpha=0.2, which="both")
    ax[1].bar(range(12), pcp, color="#2a78d6")
    ax[1].set_xticks(range(12), NOTE, fontsize=8)
    ax[1].set(title="pitch-class profile", ylabel="tonal power share")
    ax[1].grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "tonality.png")
    plt.close(fig)
    return doc
