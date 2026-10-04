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

CHANGING ``nominal`` IS NOT A NEUTRAL ACT, and the trap it opens cost a
published claim. Railway supplies invite it: the Nordic countries, Germany,
Austria and Switzerland electrify at 16⅔ Hz, so ``nominal=16.667`` looks like
the obvious way to ask whether a recording was made on such a train. It is
not, because **16⅔ has 50 Hz as its third harmonic and 100 Hz as its sixth**.
A family built on 16⅔ therefore contains the European mains family inside it,
and any recording with mains in it will score well on "the Nordic railway
supply" — including a hotel foyer with no train within a kilometre, which is
the control that settled it on 2026-08-11. The Stavanger train's own 16.7 Hz
fundamental sat 3 dB *below* the noise around it, as did its second, third and
fourth harmonics; the score was carried entirely by 100 Hz, which is mains.

Two rules follow, and they generalise past railways to any hypothesised
family (a shaft rate, a chopper frequency, a fan's blade-pass):

1. **Check the rungs, not the mean.** A family whose fundamental and low
   harmonics are absent while one high harmonic is strong is not a family.
   :func:`ambiscape.tonality.family_prominence` returns the per-harmonic list
   for exactly this reason.
2. **Rank the hypothesis against the alternatives**, with
   :func:`ambiscape.tonality.family_percentile`, and run a control recording
   that cannot contain the source. Presence is not evidence; being
   *exceptional* is.
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
    power in a ±1.5 Hz-widened neighbourhood, so a genuine line scores
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

    COVERAGE IS NOT A PROXY FOR WHERE THE RECORDING WAS MADE, however
    reasonable that sounds — mains hum means mains nearby, so more hum
    should mean more indoors. Tested on 365 daily recordings sorted into
    seven kinds of place, the group medians ran from 0.590 down to 0.475, a
    total range of 0.115 against a within-group interquartile width of
    0.260: the differences between kinds of place were 2.3 times smaller
    than the differences within them, and Kruskal--Wallis returned H = 3.7
    at p = 0.72. Outdoor and semi-open sessions ranked sixth of seven, in
    among the indoor groups rather than below them. The measurement itself
    was in excellent health over the same year — the grid recovered on 364
    of 365 days at a median 49.9913 Hz — so this is a good measurement of
    the wrong thing, which is the kind that survives review.

    What coverage does track is the recording: gain, wind, the recorder's
    own noise floor, how much of the window something was leaning on the
    microphone. It is a quality figure wearing a location figure's clothes.
    It is also tied to ``win_s``, ``step_s`` and ``min_rise_db``, being a
    fraction of windows clearing a threshold, so two coverages compare only
    where all three match. And a single day's zero deserves a look at the
    file before it is believed: the most extreme reading of that year came
    from a WAV whose header declared 690 seconds over 379 MB of audio and
    returned no frames at all to libsndfile without raising anything — on an
    outdoor day, so the artefact was the one number that made the story come
    out the way it was expected to.
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


# --- Supply pickup: is the hum in the air or in the cable? -------------------

#: Prominence in W (dB over the ring) below which a mains line is "not there".
#: :func:`ambiscape.tonality.narrow_line_prominence` scores noise at about
#: 1-5 dB depending on how much was averaged, so 6 dB is the first value
#: clearly above that floor.
SUPPLY_MIN_PROM_DB = 6.0
#: Direction ratio of the line relative to its surround (dB) at or below which
#: the line counts as electrical pickup.
SUPPLY_PICKUP_DB = -10.0


def _welch_windows(path, n_windows, win_s, nperseg_s):
    """Average Welch PSD (n_freq x channels) over evenly spread windows, and
    the mono W windows themselves for :func:`hum_peak`."""
    import soundfile as sf
    from scipy import signal
    info = sf.info(str(path))
    fs, dur = info.samplerate, info.frames / info.samplerate
    lead = 60.0 if dur > win_s + 180 else 0.0      # skip setting the device down
    span = max(dur - 2 * lead - win_s, 0.0)
    starts = [lead + (span * i / (n_windows - 1) if n_windows > 1 else 0.0)
              for i in range(n_windows)] if span > 0 else [0.0]
    P, ws, f = None, [], None
    with sf.SoundFile(str(path)) as fh:
        for t0 in starts:
            fh.seek(int(t0 * fs))
            x = fh.read(int(win_s * fs), dtype="float64", always_2d=True)
            f, p = signal.welch(x, fs, nperseg=min(int(nperseg_s * fs), len(x)), axis=0)
            P = p if P is None else P + p
            ws.append(x[:, 0])
    return f, P / len(starts), ws, fs, info.channels, len(starts)


def supply_signature(path, nominal: float = 50.0, n_harmonics: int = 10,
                     n_windows: int = 6, win_s: float = 60.0, nperseg_s: float = 8.0,
                     wyzx=(0, 1, 2, 3)) -> dict:
    """Is the mains hum in a four-channel B-format file electrical pickup
    through the recorder's power supply, or sound in the air?

    Pickup reaches every capsule in phase, so after the A- to B-format matrix
    it lands in W and hardly at all in X, Y and Z. A sound in the air has a
    direction (or, diffuse, spreads over all three), so the first-order
    channels carry it too. For each harmonic ``k * nominal`` this measures the
    line's prominence in W (:func:`ambiscape.tonality.narrow_line_prominence`)
    and its *direction ratio*: the line's excess power over its ring in X+Y+Z
    against the same in W, in dB, minus the same ratio for the noise just
    beside the line (``line_minus_ring_db``). Subtracting the ring removes
    what the recorder's matrix and the room do to every frequency alike, so
    near 0 dB is acoustic and far below is pickup. The fundamental also gets
    its frequency and rise from :func:`hum_peak`, and the count of narrow lines
    between 1 and 20 kHz, where switching supplies whine, is returned.

    ``verdict`` is ``"pickup"`` when the fundamental stands at least
    ``SUPPLY_MIN_PROM_DB`` out of W and its direction ratio is at or below
    ``SUPPLY_PICKUP_DB``; ``"acoustic"`` when it stands out and sits above;
    ``"no line"`` when it does not stand out; ``"not ambix"`` for files that
    are not four-channel; ``"empty"`` for a file that holds no frames. The threshold rests on little: one overnight Zoom
    H3-VR session on a USB supply (the line 15 to 18.5 dB below its surround)
    against 45 files without pickup (-5.5 to +3.7 dB where a line stood out).

    ``wyzx`` gives the column of W, Y, Z and X (AmbiX: ``(0, 1, 2, 3)``;
    FuMa W, X, Y, Z: ``(0, 2, 3, 1)``); :attr:`ambiscape.io.Take.wyzx` has it.
    Six 60 s windows spread over the file, skipping the first and last minute
    where a long file allows.
    """
    import soundfile as sf
    from scipy import signal
    from scipy.ndimage import median_filter
    from .tonality import narrow_line_prominence
    info = sf.info(str(path))
    if info.channels != 4:
        return {"verdict": "not ambix"}
    if info.frames == 0:                        # a broken header reads as no audio
        return {"verdict": "empty"}
    f, P, ws, fs, _ch, nw = _welch_windows(path, n_windows, win_s, nperseg_s)
    P = P[:, list(wyzx)]                                 # -> W, Y, Z, X
    db = 10 * np.log10(P + EPS)
    lines = []
    for k in range(1, n_harmonics + 1):
        f0 = nominal * k
        if f0 + 6.0 >= f[-1]:
            break
        prom = narrow_line_prominence(f, db[:, 0], f0)
        pk = (f >= f0 - 0.35) & (f <= f0 + 0.35)
        ring = (np.abs(f - f0) > 1.5) & (np.abs(f - f0) <= 6.0)
        ex = [max(P[pk, c].max() - np.median(P[ring, c]), 0.0) for c in range(4)]
        rr = 10 * np.log10(np.median(P[ring, 1:].sum(1)) / np.median(P[ring, 0]))
        ratio = (10 * np.log10((sum(ex[1:]) + 1e-12 * ex[0]) / ex[0])) if ex[0] > 0 else None
        lines.append({"hz": f0, "prom_W_db": round(float(prom), 1),
                      "xyz_over_w_db": None if ratio is None else round(float(ratio), 1),
                      "ring_xyz_over_w_db": round(float(rr), 1),
                      "line_minus_ring_db": None if ratio is None else round(float(ratio - rr), 1)})
    band = (f >= 1000) & (f <= min(20000, f[-1]))
    n_hf = 0
    if band.sum() > 50:
        resid = db[band, 0] - median_filter(db[band, 0], 41, mode="nearest")
        n_hf = int(len(signal.find_peaks(resid, height=12.0, distance=8)[0]))
    hum = [hum_peak(w, fs, nominal) for w in ws]
    hz = float(np.median([h[0] for h in hum]))
    f1 = lines[0]
    if f1["prom_W_db"] < SUPPLY_MIN_PROM_DB or f1["line_minus_ring_db"] is None:
        verdict = "no line"
    elif f1["line_minus_ring_db"] <= SUPPLY_PICKUP_DB:
        verdict = "pickup"
    else:
        verdict = "acoustic"
    return {"verdict": verdict, "n_windows": nw, "win_s": win_s,
            "hum_hz_median": round(hz, 4), "hum_dev_mhz": round(abs(hz - nominal) * 1000, 1),
            "hum_rise_db_median": round(float(np.median([h[1] for h in hum])), 1),
            "fundamental": f1, "lines": lines, "n_hf_lines": n_hf}


def line_bearing(path, f0: float, n_windows: int = 6, win_s: float = 60.0,
                 nperseg_s: float = 8.0, wyzx=(0, 1, 2, 3)) -> dict:
    """Bearing of a steady narrow line, from the active intensity at ``f0``:
    the real parts of the cross-spectra of W with X, Y and Z, which point
    towards a plane wave's source. Summed over evenly spread windows for the
    bearing, kept per window to show whether the source moved. Azimuth is
    counter-clockwise from the recorder's front (X+), elevation up from the
    horizontal, both in the recorder's own frame."""
    import soundfile as sf
    from scipy import signal
    info = sf.info(str(path))
    if info.channels != 4:
        raise ValueError(f"{path}: line_bearing needs four-channel B-format")
    fs, dur = info.samplerate, info.frames / info.samplerate
    lead = 60.0 if dur > win_s + 180 else 0.0
    span = max(dur - 2 * lead - win_s, 0.0)
    starts = [lead + span * i / max(n_windows - 1, 1) for i in range(n_windows)]
    acc, per = np.zeros(3), []
    with sf.SoundFile(str(path)) as fh:
        for t0 in starts:
            fh.seek(int(t0 * fs))
            x = fh.read(int(win_s * fs), dtype="float64", always_2d=True)[:, list(wyzx)]
            W, Y, Z, X = x.T
            I = []
            for c in (X, Y, Z):
                f, C = signal.csd(W, c, fs, nperseg=min(int(nperseg_s * fs), len(W)))
                k = int(np.argmin(np.abs(f - f0)))
                k = max(k - 2, 0) + int(np.argmax(np.abs(C[max(k - 2, 0):k + 3])))
                I.append(C[k].real)
            I = np.array(I)
            acc += I
            per.append(round(float(np.degrees(np.arctan2(I[1], I[0])))))
    return {"f0_hz": f0, "az_deg": round(float(np.degrees(np.arctan2(acc[1], acc[0])))),
            "el_deg": round(float(np.degrees(np.arctan2(acc[2], np.hypot(acc[0], acc[1]))))),
            "per_window_az_deg": per}
