"""ISO 12913-3-style psychoacoustic indicators + level calibration.

Calibration
-----------
A session is calibrated by ``<folder>/calibration.json``::

    {"dbfs_to_dbspl": 94.0,
     "method": "SPL app next to mic, air pump running, LAeq 42 dB",
     "date": "2026-07-16"}

``dbfs_to_dbspl`` is the offset O such that a signal at −X dBFS corresponds
to (O − X) dB SPL. With it, dBFS descriptors become dB SPL and waveforms
convert to pascals for psychoacoustic metrics.

The same file may carry ``clock_offset_s`` — seconds added to every take's
start time when the recorder clock was found to be off (positive = clock was
slow; e.g. calibrated against a known external time reference). Applied in
:func:`ambiscape.io.open_session`, so all clock-labeled outputs agree. Both
keys are optional.

Indicators (via MoSQITo, optional dependency)
---------------------------------------------
ISO 532-1 time-varying loudness (N5, N50), DIN 45692 sharpness, and
Daniel & Weber roughness, computed per ear on a binaural render of the
B-format signal. If `ambiviz` (with its HRIR-based binauralizer) is
installed it is used; otherwise a documented fallback renders a
back-to-back cardioid pair at ±90° — a pseudo-binaural approximation
without pinna/ILD spectral cues. Uncalibrated sessions are computed with an
assumed offset and flagged: absolute sone/acum values are then indicative
only (their *ratios* between segments remain meaningful).

Beyond the MoSQITo set (pure numpy/scipy, always available)
-----------------------------------------------------------
MoSQITo (≤ 1.2.x) provides no fluctuation strength, so
:func:`fluctuation_strength` implements the Fastl & Zwicker
envelope-modulation *approximation* (~4 Hz weighting) — clearly not a
standardised metric — and :func:`fluctuation_index` is its cheap
broadband companion on the cached 20 ms envelope. :func:`tone_prominence`
/ :func:`prominent_tones` detect DIN 45681-style prominent tones
(spectral peak vs masking-band level, ΔL in dB) in the per-minute mean
spectra — the ventilation/appliance-hum detector.
:func:`summarize_psycho` folds both into the ``analyze`` summary.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

P_REF = 20e-6
ASSUMED_OFFSET = 94.0  # used (and flagged) when no calibration.json exists


def load_calibration(folder: str | Path) -> dict | None:
    p = Path(folder) / "calibration.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def take_offset(cal: dict | None, take_name: str) -> float | None:
    """The dbfs->dbspl offset for one take: per-take map, else global.

    Multi-device sessions (a Zoom and a phone running side by side) need
    different offsets per take; ``dbfs_to_dbspl_takes`` maps take filenames
    to offsets, falling back to the session-wide ``dbfs_to_dbspl``.
    """
    if not cal:
        return None
    per = cal.get("dbfs_to_dbspl_takes", {})
    if take_name in per:
        return float(per[take_name])
    if "dbfs_to_dbspl" in cal:
        return float(cal["dbfs_to_dbspl"])
    return None


def derive_offset(F: dict, laeq_spl: float, t0: float | None = None,
                  dur: float | None = None) -> dict:
    """Derive ``dbfs_to_dbspl`` from a field SPL-meter reading.

    ``laeq_spl`` is the LAeq in dB(A) read off a meter (or phone app) held
    at the microphone position over some span of the recording; ``t0`` /
    ``dur`` bound that span in seconds *from the start of the recording*
    (defaults: all of it). The recording's own LAeq over the same span
    comes from the cached A-weighted fast levels, and the offset is simply
    their difference: a signal at −X dBFS corresponds to (offset − X)
    dB SPL.
    """
    t = np.asarray(F["t_fast"], np.float64)
    t = t - t[0]                              # relative to recording start
    dba = np.asarray(F["fast_dba"], np.float64)
    m = np.ones(len(t), bool)
    if t0 is not None:
        m &= t >= t0
    if dur is not None:
        m &= t < (t0 or 0.0) + dur
    if not m.any():
        raise ValueError("span selects no samples")
    laeq_dbfs = 10 * np.log10(np.mean(10 ** (dba[m] / 10)) + 1e-30)
    return {"dbfs_to_dbspl": round(float(laeq_spl - laeq_dbfs), 1),
            "laeq_dbfs": round(float(laeq_dbfs), 1),
            "laeq_spl": float(laeq_spl),
            "span_s": [float(t[m][0]), float(t[m][-1])]}


def write_calibration(folder: str | Path, offset: float, method: str = "",
                      take: str | None = None) -> Path:
    """Write/merge an offset into ``<folder>/calibration.json``.

    Existing keys (``clock_offset_s`` etc.) are preserved. With ``take``,
    the offset lands in the per-take map ``dbfs_to_dbspl_takes``;
    otherwise it becomes the session-wide ``dbfs_to_dbspl``.
    """
    import datetime
    p = Path(folder) / "calibration.json"
    cal = json.loads(p.read_text()) if p.exists() else {}
    if take is not None:
        cal.setdefault("dbfs_to_dbspl_takes", {})[take] = float(offset)
    else:
        cal["dbfs_to_dbspl"] = float(offset)
    if method:
        cal["method"] = method
    cal["date"] = datetime.date.today().isoformat()
    p.write_text(json.dumps(cal, indent=2))
    return p


def to_pascal(x: np.ndarray, dbfs_to_dbspl: float) -> np.ndarray:
    return x.astype(np.float64) * P_REF * 10 ** (dbfs_to_dbspl / 20)


def apply_calibration(summary: dict, cal: dict) -> dict:
    """Add dB SPL versions of the level descriptors to a summary dict."""
    off = float(cal["dbfs_to_dbspl"])
    out = dict(summary)
    for key in ("leq_dbfs", "laeq_dbfs", "L10", "L50", "L90"):
        if key in summary and summary[key] is not None:
            out[key.replace("_dbfs", "") + "_db_spl"] = round(summary[key] + off, 1)
    out["calibration"] = {"dbfs_to_dbspl": off,
                          "method": cal.get("method", "")}
    return out


def binaural(x: np.ndarray, fs: int, order: str = "ambix",
             mode: str = "ambix") -> tuple[np.ndarray, str]:
    """Two-channel ear signals from a block, for the ISO psychoacoustic metrics.

    The treatment follows the take's ``mode`` so every input type is handled:

    * ``mono`` (or a 1-column block) -> the lone channel is duplicated to both
      ears.
    * ``stereo`` / ``binaural`` (or < 4 columns) -> the first two channels are
      already a left/right pair (binaural ear signals or a stereo mix) and
      pass through unchanged.
    * ``ambix`` first-order B-format -> the block is remapped to canonical
      AmbiX (W, Y, Z, X) via ``order``, so a ``fuma`` (W, X, Y, Z) take is
      decoded correctly, then binauralised with ambiviz's HRIR decoder,
      falling back to a +-90 deg cardioid pair (``0.5 * (W +- Y)``, no pinna cues).

    ``order`` is consulted only for ambix input. Returns an (n, 2) array and
    the method name.
    """
    if mode == "mono" or x.shape[1] == 1:
        m = x[:, 0]
        return np.stack([m, m], axis=1), "mono-duplicated"
    if mode in ("stereo", "binaural") or x.shape[1] < 4:
        return np.ascontiguousarray(x[:, :2]), "stereo-passthrough"
    wyzx = (0, 2, 3, 1) if order == "fuma" else (0, 1, 2, 3)
    xw = x[:, list(wyzx)]  # canonical AmbiX W, Y, Z, X
    try:
        from ambiviz.ambisonics.binauralizer import binauralize  # type: ignore
        y = binauralize(xw.T, fs)  # ambiviz convention: channels first, AmbiX
        return np.asarray(y).T[:, :2], "ambiviz-hrir"
    except Exception:
        w, ych = xw[:, 0], xw[:, 1]
        left = 0.5 * (w + ych)
        right = 0.5 * (w - ych)
        return np.stack([left, right], axis=1), "cardioid-pair-fallback"


def indicators(x_pa: np.ndarray, fs: int, rough_dur: float = 10.0) -> dict:
    """ISO 532-1 loudness (N5/N50), DIN 45692 sharpness, D&W roughness,
    and (approximate) fluctuation strength for one calibrated (pascal)
    channel.

    MoSQITo runs ~5x slower than realtime, and roughness is the costliest
    metric; it is therefore computed on a central `rough_dur`-second slice
    (roughness is a texture measure and stabilizes within seconds).
    Fluctuation strength is not in MoSQITo (≤ 1.2.x) and comes from the
    local :func:`fluctuation_strength` approximation instead.
    """
    from mosqito.sq_metrics import (loudness_zwtv,
                                    sharpness_din_from_loudness,
                                    roughness_dw)
    N, N_spec, _bark, _t = loudness_zwtv(x_pa, fs, field_type="diffuse")
    S = sharpness_din_from_loudness(N, N_spec)
    n_r = int(rough_dur * fs)
    mid = max(0, (len(x_pa) - n_r) // 2)
    R = np.atleast_1d(roughness_dw(x_pa[mid:mid + n_r], fs)[0])
    return {
        "N5_sone": round(float(np.percentile(N, 95)), 2),
        "N50_sone": round(float(np.percentile(N, 50)), 2),
        "sharpness_median_acum": round(float(np.median(S)), 2),
        "roughness_median_asper": round(float(np.median(R)), 3),
        "fluctuation_strength_vacil": round(
            fluctuation_strength(x_pa, fs), 3),
    }


def segment_indicators(sess, F: dict, folder: str | Path,
                       dur: float = 30.0, offset: float | None = None) -> dict:
    """Compute per-ear indicators on representative segments.

    Segments come from analysis.pick_segments (typical / quietest /
    most_active / transition); `dur` seconds from the start of each.
    """
    from .analysis import pick_segments
    from .io import read_span

    cal = load_calibration(folder)
    has_spl = bool(cal and "dbfs_to_dbspl" in cal)
    if offset is None:
        offset = float(cal["dbfs_to_dbspl"]) if has_spl else ASSUMED_OFFSET
    calibrated = has_spl or offset != ASSUMED_OFFSET

    out = {"calibrated": calibrated, "dbfs_to_dbspl": offset,
           "field_type": "diffuse", "segments": {}}
    if not calibrated:
        out["warning"] = (f"no calibration.json — assumed offset "
                          f"{ASSUMED_OFFSET} dB; absolute values indicative only")
    for pick in pick_segments(F, seg_s=dur):
        try:
            x, fs = read_span(sess, pick["t0"], dur)
        except ValueError:
            continue
        tk = next((t for t in sess.takes
                   if t.start <= pick["t0"] < t.end), None)
        ears, method = binaural(
            x, fs,
            order=(tk.order if tk else "ambix"),
            mode=(tk.mode if tk else "ambix"))
        seg = {"t0": sess.clock(pick["t0"]), "dur_s": dur,
               "binaural_method": method}
        # multi-device sessions: a per-take offset overrides the global one
        seg_offset = (take_offset(cal, tk.path.name)
                      if tk and calibrated else None) or offset
        for ch, name in ((0, "left"), (1, "right")):
            seg[name] = indicators(to_pascal(ears[:, ch], seg_offset), fs)
        seg["N5_sone_max_ear"] = max(seg["left"]["N5_sone"],
                                     seg["right"]["N5_sone"])
        out["segments"][pick["kind"]] = seg
    return out


# ------------------------------------- fluctuation strength (approximation)

# Zwicker critical-band (Bark) edges in Hz; adjacent pairs are ~1 Bark wide
BARK_EDGES = (20.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0,
              1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0,
              3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0)


def _fluctuation_weight(f_mod):
    """Fastl & Zwicker band-pass weighting of modulation frequency.

    ``2 / (f/4 + 4/f)`` — peaks at 1 for f = 4 Hz, falling toward slow
    level drifts on one side and roughness-rate modulation on the other.
    """
    f = np.maximum(np.asarray(f_mod, np.float64), 1e-6)
    return 2.0 / (f / 4.0 + 4.0 / f)


def _fluctuation_raw(x, fs, fmin_mod, fmax_mod, dl_cap):
    """Unnormalized Fastl-style sum: ΔL_band × w(f_mod) over Bark bands."""
    from scipy import signal as sg
    x = np.asarray(x, np.float64)
    frame = max(8, int(round(fs / 200.0)))          # ~5 ms envelope frames
    env_fs = fs / frame
    lp = sg.butter(4, min(32.0, 0.45 * env_fs), "low", fs=env_fs,
                   output="sos")
    envs, band_pow = [], []
    for lo, hi in zip(BARK_EDGES[:-1], BARK_EDGES[1:]):
        if hi >= 0.95 * fs / 2:
            break
        sos = sg.butter(4, (lo, hi), "bandpass", fs=fs, output="sos")
        y = sg.sosfilt(sos, x)
        n = (len(y) // frame) * frame
        e = (y[:n].reshape(-1, frame) ** 2).mean(1)
        if len(e) < 64:
            return 0.0
        e = np.maximum(sg.sosfiltfilt(lp, e), 0.0)  # keep < 32 Hz: fluctuation,
        envs.append(e)                              # not roughness, modulation
        band_pow.append(float(e.mean()))
    gate = (max(band_pow) if band_pow else 0.0) * 1e-4   # −40 dB energy gate
    raw = 0.0
    for e, p in zip(envs, band_pow):
        if p <= gate or p <= 0.0:
            continue          # filter leakage: dB depth is level-invariant,
        # so near-empty bands must not contribute
        L = 10 * np.log10(np.maximum(e, p * 1e-6))
        dl = float(np.clip(np.percentile(L, 95) - np.percentile(L, 5),
                           0.0, dl_cap))
        if dl <= 0.0:
            continue
        xm = e / p - 1.0
        nper = int(min(len(xm), max(64, round(8.0 * env_fs / fmin_mod))))
        f, P = sg.welch(xm, fs=env_fs, nperseg=nper, noverlap=nper // 2,
                        detrend="linear")
        m = (f >= fmin_mod) & (f <= fmax_mod)
        if not m.any() or not P[m].any():
            continue
        Pm = P[m]
        i = int(np.argmax(Pm))
        f_mod = float(f[m][i])
        # coherence of the modulation: share of modulation power at the
        # dominant frequency (±1 bin). ~1 for periodic AM, small for the
        # random envelope fluctuation of unmodulated noise.
        coh = float(Pm[max(0, i - 1):i + 2].sum() / Pm.sum())
        raw += dl * coh * float(_fluctuation_weight(f_mod))
    return raw


@lru_cache(maxsize=8)
def _fluctuation_reference(fs, fmin_mod, fmax_mod, dl_cap):
    """Raw model value for the 1-vacil reference: 1 kHz tone, 100 % AM at
    4 Hz (level-invariant model, so the amplitude is arbitrary)."""
    t = np.arange(int(4.0 * fs)) / fs
    ref = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t)) \
        * np.sin(2 * np.pi * 1000.0 * t)
    return _fluctuation_raw(ref, fs, fmin_mod, fmax_mod, dl_cap)


def fluctuation_strength(x: np.ndarray, fs: int, fmin_mod: float = 0.25,
                         fmax_mod: float = 32.0, dl_cap: float = 30.0) -> float:
    """Fluctuation strength, vacil — an *approximation*, not a standard.

    MoSQITo (≤ 1.2.x) offers no fluctuation strength, so this follows the
    Fastl & Zwicker envelope-modulation model in spirit: per Zwicker
    critical band, the envelope level depth ΔL (5th–95th percentile of the
    < 32 Hz band envelope, capped at ``dl_cap`` dB) is weighted by the
    band-pass modulation-frequency weighting ``2/(f/4 + 4/f)`` that peaks
    at 4 Hz, and summed over Bark bands. The sum is scaled so that the
    classic reference — a 1 kHz tone, 100 % amplitude-modulated at 4 Hz —
    reads 1 vacil.

    It is **not** an implementation of any standard (none exists for
    fluctuation strength): masking-based envelope depth, the level
    dependence, and interaction effects are all simplified, so treat
    absolute values as indicative and comparisons between recordings made
    with the same pipeline as the meaningful output. Needs ≥ ~2 s of
    signal; steady signals read ≈ 0.
    """
    ref = _fluctuation_reference(int(fs), float(fmin_mod), float(fmax_mod),
                                 float(dl_cap))
    if ref <= 0.0:
        return 0.0
    return float(_fluctuation_raw(x, fs, fmin_mod, fmax_mod, dl_cap) / ref)


def fluctuation_index(env: np.ndarray, dt: float, fmin: float = 0.25,
                      fmax: float = 20.0) -> float | None:
    """Broadband fluctuation index from a cached power envelope (unitless).

    The 4 Hz-weighted modulation depth of the unit-mean envelope:
    ``sqrt(∫ P(f) · w(f)² df)`` with the same Fastl-style weighting as
    :func:`fluctuation_strength`, computed from the cached 20 ms broadband
    envelope (``env_hi``) so ``analyze`` needs no audio pass. A relative
    index that tracks fluctuation strength (≈ 0 steady drone, high for
    ~4 Hz wobble), **not** vacil: no critical-band split, no absolute
    anchoring. Returns None when the envelope is too short.
    """
    from scipy import signal as sg
    env = np.asarray(env, np.float64)
    if len(env) < 64 or env.mean() <= 0.0:
        return None
    fmax = min(fmax, 0.45 / dt)
    if fmax <= fmin * 1.5:
        return None
    x = env / env.mean() - 1.0
    nper = int(min(len(x), max(64, round(8.0 / (fmin * dt)))))
    f, P = sg.welch(x, fs=1.0 / dt, nperseg=nper, noverlap=nper // 2,
                    detrend="linear")
    m = (f >= fmin) & (f <= fmax)
    if not m.any():
        return None
    w = _fluctuation_weight(f[m])
    return float(np.sqrt(max(np.trapezoid(P[m] * w ** 2, f[m]), 0.0)))


# --------------------------------------- tonal prominence (DIN 45681-style)

def critical_bandwidth(f_hz):
    """Zwicker & Terhardt (1980) critical bandwidth at ``f_hz``, in Hz."""
    f = np.asarray(f_hz, np.float64)
    return 25.0 + 75.0 * (1.0 + 1.4 * (f / 1000.0) ** 2) ** 0.69


def tone_prominence(spec_row: np.ndarray, freqs: np.ndarray,
                    fmin: float = 50.0, fmax: float = 10000.0,
                    min_dl_db: float = 6.0, max_n: int = 12) -> list[dict]:
    """DIN 45681-style prominent tones in one mean power spectrum.

    For each narrowband spectral peak, the decibel prominence
    ``ΔL = L_tone − L_noise`` compares the tone power (main-lobe bins,
    noise-corrected) against the masking-noise level in the surrounding
    critical band (median bin level × band width, i.e. the level the band
    would have without the tone). Tones with ``ΔL ≥ min_dl_db`` (default
    6 dB, the decisive audibility criterion of DIN 45681) are returned as
    ``{"f_hz", "dL_db"}``, strongest first.

    This follows the *method* of DIN 45681 (tone vs masking-band level)
    but is not a certified implementation: no frequency-dependent masking
    index, no uncertainty term. Pure numpy/scipy; expects a linear power
    spectrum on a uniform frequency grid (a ``minspec`` row).
    """
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks
    spec = np.asarray(spec_row, np.float64)
    freqs = np.asarray(freqs, np.float64)
    if len(freqs) < 32 or spec.max() <= 0.0:
        return []
    df = float(freqs[1] - freqs[0])
    eps = spec.max() * 1e-12
    ls = 10 * np.log10(spec + eps)
    floor = median_filter(ls, size=min(101, 2 * (len(ls) // 2) - 1),
                          mode="nearest")
    # cheap pre-filter: a band ΔL of 6 dB implies a much larger per-bin rise
    cand, _ = find_peaks(ls - floor, height=min_dl_db, distance=3)
    cand = cand[(freqs[cand] >= fmin) & (freqs[cand] <= fmax)]
    idx = np.arange(len(freqs))
    tones = []
    for i in cand:
        f0 = float(freqs[i])
        hw = max(float(critical_bandwidth(f0)) / 2.0, 6 * df)
        band = (freqs >= f0 - hw) & (freqs <= f0 + hw)
        tone = band & (np.abs(idx - i) <= 3)         # Hann main lobe + slack
        noise = band & (np.abs(idx - i) > 5)         # guard bins excluded
        if noise.sum() < 4:
            continue
        med = float(np.median(spec[noise]))          # masking noise per bin
        p_tone = float(spec[tone].sum()) - med * int(tone.sum())
        p_noise = med * int(band.sum())              # noise level of the band
        if p_tone <= 0.0 or p_noise <= 0.0:
            continue
        dl = 10 * np.log10(p_tone / p_noise)
        if dl >= min_dl_db:
            tones.append({"f_hz": round(f0, 1), "dL_db": round(float(dl), 1)})
    tones.sort(key=lambda t: -t["dL_db"])
    return tones[:max_n]


def prominent_tones(minspec: np.ndarray, freqs: np.ndarray,
                    min_fraction: float = 0.1, tol_cents: float = 50.0,
                    **tone_kw) -> list[dict]:
    """Time-aggregated prominent tones across the per-minute spectra.

    Runs :func:`tone_prominence` per minute and groups detections within
    ``tol_cents`` (or 2.5 bins at low frequency) into persistent tones.
    Tones present in at least ``min_fraction`` of the minutes are returned
    as ``{"f_hz", "dL_median_db", "dL_max_db", "present_fraction",
    "n_minutes"}``, strongest first — a ventilation hum shows up as one
    high-fraction line, a passing siren does not.
    """
    minspec = np.asarray(minspec, np.float64)
    freqs = np.asarray(freqs, np.float64)
    nrow = minspec.shape[0]
    if nrow == 0 or len(freqs) < 32:
        return []
    df = float(freqs[1] - freqs[0])
    groups: list[dict] = []
    for r in range(nrow):
        for t in tone_prominence(minspec[r], freqs, **tone_kw):
            for g in groups:
                fg = float(np.median(g["f"]))
                tol = max(fg * (2 ** (tol_cents / 1200.0) - 1.0), 2.5 * df)
                if abs(t["f_hz"] - fg) <= tol:
                    g["f"].append(t["f_hz"])
                    g["dl"].append(t["dL_db"])
                    g["rows"].add(r)
                    break
            else:
                groups.append({"f": [t["f_hz"]], "dl": [t["dL_db"]],
                               "rows": {r}})
    out = []
    for g in groups:
        frac = len(g["rows"]) / nrow
        if frac < min_fraction:
            continue
        out.append({"f_hz": round(float(np.median(g["f"])), 1),
                    "dL_median_db": round(float(np.median(g["dl"])), 1),
                    "dL_max_db": round(float(np.max(g["dl"])), 1),
                    "present_fraction": round(frac, 2),
                    "n_minutes": len(g["rows"])})
    return sorted(out, key=lambda t: -t["dL_median_db"])


def summarize_psycho(F: dict) -> dict:
    """Psychoacoustic summary keys from cached features (no audio pass).

    Adds to the ``analyze`` summary: the strongest persistent DIN
    45681-style tone (``tonal_prominence_db`` / ``_hz``, None when the
    scene has no prominent tone), the count of persistent tones, and the
    broadband :func:`fluctuation_index`. All are level-difference or
    normalized quantities, meaningful without SPL calibration. Degrades
    gracefully (None / 0) when the cache predates ``minspec``/``env_hi``.
    """
    out = {"tonal_prominence_db": None, "tonal_prominence_hz": None,
           "n_prominent_tones": 0, "fluctuation_index": None}
    if "minspec" in F and len(F["minspec"]):
        tones = prominent_tones(F["minspec"], F["freqs"])
        out["n_prominent_tones"] = len(tones)
        if tones:
            out["tonal_prominence_db"] = tones[0]["dL_median_db"]
            out["tonal_prominence_hz"] = tones[0]["f_hz"]
    if "env_hi" in F and "hi_dt" in F and len(F["env_hi"]):
        fi = fluctuation_index(F["env_hi"], float(F["hi_dt"]))
        out["fluctuation_index"] = round(fi, 3) if fi is not None else None
    return out


# ------------------------------------------------------- room noise criteria

NR_A = {31.5: 55.4, 63: 35.5, 125: 22.0, 250: 12.0, 500: 4.8,
        1000: 0.0, 2000: -3.5, 4000: -6.1, 8000: -8.0}
NR_B = {31.5: 0.681, 63: 0.790, 125: 0.870, 250: 0.930, 500: 0.974,
        1000: 1.000, 2000: 1.015, 4000: 1.025, 8000: 1.030}
# ANSI S12.2 tangent NC curves, octave levels 63 Hz .. 8 kHz per NC value
NC_TABLE = {
    15: (47, 36, 29, 22, 17, 14, 12, 11),
    20: (51, 40, 33, 26, 22, 19, 17, 16),
    25: (54, 44, 37, 31, 27, 24, 22, 21),
    30: (57, 48, 41, 35, 31, 29, 28, 27),
    35: (60, 52, 45, 40, 36, 34, 33, 32),
    40: (64, 56, 50, 45, 41, 39, 38, 37),
    45: (67, 60, 54, 49, 46, 44, 43, 42),
    50: (71, 64, 58, 54, 51, 49, 48, 47),
    55: (74, 67, 62, 58, 56, 54, 53, 52),
    60: (77, 71, 67, 63, 61, 59, 58, 57),
    65: (80, 75, 71, 68, 66, 64, 63, 62),
}
NC_FREQS = (63, 125, 250, 500, 1000, 2000, 4000, 8000)


def room_criteria(oct_spl_db: dict) -> dict:
    """NR, NC, and RC ratings of an octave-band SPL spectrum.

    ``oct_spl_db`` maps octave center frequency (Hz) to band SPL (dB).
    Ratings are only physically meaningful for *calibrated* levels
    (``dbfs_to_dbspl`` in ``calibration.json``); on uncalibrated dBFS they
    are relative numbers, comparable within one recorder+gain setup only.

    - **NR** (ISO/R 1996 Noise Rating): analytic curves ``L = a + b*NR``;
      the rating is the highest per-band NR value and
      ``NR_governing_hz`` names the band that sets it.
    - **NC** (ANSI S12.2 Noise Criterion): tangency against the tabulated
      curves, linearly interpolated per band (63 Hz–8 kHz).
    - **RC** (Blazier Room Criterion, simplified): arithmetic mean of the
      500/1000/2000 Hz levels; the reference line has a −5 dB/octave slope
      through (1 kHz, RC). ``RC_class`` is "R" (rumble) when any
      31.5–250 Hz band exceeds the line by > 5 dB, "H" (hiss) when any
      2–4 kHz band exceeds it by > 3 dB, "RH" for both, "N" (neutral)
      otherwise.
    """
    spec = {float(k): float(v) for k, v in oct_spl_db.items()}

    nr_per = {f: (spec[f] - NR_A[f]) / NR_B[f] for f in NR_A if f in spec}
    f_gov = max(nr_per, key=nr_per.get)
    nr = nr_per[f_gov]

    nc = None
    ncs = sorted(NC_TABLE)
    per_band = []
    for i, f in enumerate(NC_FREQS):
        if f not in spec:
            continue
        levels = np.array([NC_TABLE[n][i] for n in ncs], float)
        per_band.append(float(np.interp(spec[f], levels, ncs)))
    if per_band:
        nc = max(per_band)

    rc = None
    rc_class = None
    if all(f in spec for f in (500.0, 1000.0, 2000.0)):
        rc = (spec[500.0] + spec[1000.0] + spec[2000.0]) / 3
        ref = {f: rc + 5 * np.log2(1000.0 / f) for f in spec}
        rumble = any(spec[f] > ref[f] + 5 for f in (31.5, 63.0, 125.0, 250.0)
                     if f in spec)
        hiss = any(spec[f] > ref[f] + 3 for f in (2000.0, 4000.0)
                   if f in spec)
        rc_class = ("RH" if rumble and hiss else
                    "R" if rumble else "H" if hiss else "N")

    return {"NR": round(nr, 1), "NR_governing_hz": int(f_gov),
            "NC": round(nc, 1) if nc is not None else None,
            "RC": round(rc, 1) if rc is not None else None,
            "RC_class": rc_class}


def background_octaves_db(F: dict, pct: float = 50.0,
                          offset_db: float = 0.0) -> dict:
    """Per-octave percentile level (dB) from cached features, for
    :func:`room_criteria`. ``offset_db`` is the dBFS→dB SPL calibration
    offset (0 keeps uncalibrated dBFS)."""
    from .features import OCT_CENTERS
    lv = 10 * np.log10(np.asarray(F["oct_pow"], float) + 1e-20) + offset_db
    return {c: float(np.percentile(lv[:, i], pct))
            for i, c in enumerate(OCT_CENTERS) if c <= 8000}
