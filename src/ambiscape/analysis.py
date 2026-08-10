"""Session-level descriptors, event detection, and reverberation estimation.

Descriptor conventions follow the Intercontinental-database report
(2026-07-10): fast level = 125 ms RMS on W; events = fast level exceeding a
running background (10th percentile in a sliding 60 s window) by >= 8 dB for
>= 0.25 s; diffuseness/DOA from per-second pseudo-intensity vectors.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import percentile_filter, median_filter

EPS = 1e-20


def db(x, eps=1e-12):
    return 10 * np.log10(np.maximum(x, eps))


def running_background(fast_db: np.ndarray, fast_dt: float, win_s=60.0, pct=10):
    n = max(3, int(round(win_s / fast_dt)) | 1)
    return percentile_filter(fast_db, pct, size=n, mode="nearest")


def detect_events(fast_db, fast_dt, thresh_db=8.0, min_dur=0.25):
    """Return list of dicts (onset index, length, peak index, exceedance)."""
    bg = running_background(fast_db, fast_dt)
    above = fast_db > bg + thresh_db
    events = []
    i, n = 0, len(above)
    min_len = max(1, int(round(min_dur / fast_dt)))
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                k = i + int(np.argmax(fast_db[i:j + 1]))
                events.append(dict(i0=i, i1=j, ipk=k,
                                   exceed=float(fast_db[k] - bg[k])))
            i = j + 1
        else:
            i += 1
    return events, bg


def detect_cessations(fast_db, fast_dt, drop_db=6.0, min_before=60.0,
                      min_after=5.0):
    """Moments when a sustained level stops: figure-by-absence.

    A level-threshold detector finds sounds that start. It cannot find the
    event that happens when a sound *ends*, and indoors that is often the
    louder event of the two: a ventilation plant runs for nine hours and
    nobody attends to it, then it switches off and the birds, the clock and
    the water come back. Nothing rose; the ground fell, and the room
    changed character in two seconds.

    A cessation is recorded where a level held steady for at least
    ``min_before`` seconds, fell by at least ``drop_db``, and stayed down
    for at least ``min_after``. Returns a list of dicts with the index, the
    size of the drop, and the level either side.

    The asymmetry is the point. Attention is captured by change, not by
    level, and half the changes in a continuously occupied room are
    departures rather than arrivals.
    """
    x = np.asarray(fast_db, float)
    n = len(x)
    nb = max(1, int(round(min_before / fast_dt)))
    na = max(1, int(round(min_after / fast_dt)))
    if n < nb + na + 1:
        return []
    out, i = [], nb
    while i < n - na:
        before = x[i - nb:i]
        after = x[i:i + na]
        if not (np.isfinite(before).all() and np.isfinite(after).all()):
            i += 1
            continue
        lo, hi = float(np.median(before)), float(np.median(after))
        # steady before (a running machine holds a level), and clearly down
        if lo - hi >= drop_db and float(np.std(before)) <= drop_db / 2:
            out.append(dict(i=i, t_s=float(i * fast_dt),
                            drop_db=round(lo - hi, 1),
                            before_db=round(lo, 1), after_db=round(hi, 1)))
            i += nb            # one cessation per steady stretch
        else:
            i += 1
    return out


def trimmed_leq(level_db: np.ndarray, trim_pct: float = 5.0) -> float:
    """Energy mean in dB with the loudest ``trim_pct`` of frames discarded.

    An energy average is a mean of squared pressure, so it is decided by
    the loudest frames it contains: in a quiet room a handful of them can
    outweigh every other frame together. The trimmed level answers the
    companion question — what the average would be without that handful —
    and a large gap between the two says the plain average describes a few
    moments rather than the span. Reported next to ``laeq_dbfs`` in every
    session summary; see the descriptor guide's "Reading energy averages".
    """
    x = np.asarray(level_db, np.float64)
    if x.size == 0:
        return float("nan")
    keep = x <= np.percentile(x, 100.0 - trim_pct)
    if not keep.any():                       # every frame at one level
        keep = np.ones_like(x, bool)
    return float(db(np.mean(10 ** (x[keep] / 10))))


def intermittency_ratio(level_db: np.ndarray, dt: float,
                        k_db: float = 3.0) -> float:
    """Intermittency ratio IR (Wunderli et al. 2016), in percent.

    The share of total sound energy carried by "events": frames whose
    level exceeds the whole-period Leq by ``k_db`` (3 dB per the original
    definition, there on 1 s LAeq frames — here on the fast frames, which
    is equivalent for events longer than the frame). IR ≈ 0 for steady
    scenes (drones, dense traffic), high for scenes whose energy arrives
    in distinct events (rail, church bells, sparse traffic).
    """
    p = 10 ** (np.asarray(level_db, np.float64) / 10)
    leq = db(p.mean())
    mask = level_db >= leq + k_db
    return float(100.0 * p[mask].sum() / (p.sum() + EPS))


KNEE_MARGIN_DB = 3.0   # integrate until the decay is this close to the floor
MIN_FIT_SPAN_DB = 15.0  # narrowest decay T60 may be extrapolated from


def decay_metrics(x: np.ndarray, fs: int, bands=((250, 500), (500, 1000),
                  (1000, 2000), (2000, 4000), (4000, 8000)),
                  pre_roll: bool = True) -> dict:
    """T60, EDT, C50, C80 (dB) and D50 per octave band from an impulse.

    ``pre_roll=False`` says the samples before the peak are not a recording
    of the room but silence prepended by a caller so this estimator can run
    on a trimmed IR. The noise floor is then taken from the quietest part of
    the decay itself rather than from that silence. Read off the padding it
    comes out near 200 dB, which silently satisfies every dynamic-range
    guard below and lets T20/T30 be fitted over noise.

    Same truncated-Schroeder machinery as :func:`decay_time` (which is
    kept unchanged — its output feeds frozen corpus reports), plus the
    standard companions: EDT from the 0…−10 dB fit (perceived
    reverberance), clarity C50/C80 = 10·log10 of the early/late energy
    ratio at 50/80 ms, and definition D50 = early fraction at 50 ms.
    When the dynamic range allows (ISO 3382: floor at least 10 dB below
    the fit end) *and* the decay was observed that far before the signal
    ends, the fixed-range extrapolations T20 (−5…−25 dB) and T30
    (−5…−35 dB) are reported alongside the adaptive-range T60. The second
    condition matters for trimmed impulse responses, whose absent noise
    floor leaves the range guard unable to fire.
    T60 is additionally refused when the fitted range collapses: its lower
    limit is adaptive, so near the dynamic-range guard the fit can span only
    a few dB and still be extrapolated to 60. ``fit_db`` reports how wide the
    range actually was, and ``MIN_FIT_SPAN_DB`` is the narrowest accepted.
    Returns ``{band: {"T60", "T20", "T30", "EDT", "C50", "C80", "D50",
    "dr_db", "fit_db"}}`` (T20/T30 present only when supported by the
    range; T60 absent when ``fit_db`` is below the minimum).
    """
    from scipy import signal as sg
    pk_i = int(np.abs(x).argmax())
    env_bb = sg.convolve(x ** 2, np.ones(480) / 480, "same")
    tail = 10 * np.log10(env_bb[pk_i:pk_i + 3 * fs] + 1e-15)
    run_min = np.minimum.accumulate(tail)
    re = np.flatnonzero((tail - run_min > 8) & (np.arange(len(tail)) > fs // 10))
    cut = int(re[0]) if len(re) else 2 * fs
    out = {}
    for lo, hi in bands:
        sos = sg.butter(4, [lo, hi], "bandpass", fs=fs, output="sos")
        y = sg.sosfilt(sos, x)
        env = sg.convolve(y ** 2, np.ones(240) / 240, "same")
        pk = int(env[max(0, pk_i - 2400):pk_i + 2400].argmax()) \
            + max(0, pk_i - 2400)
        if pk < fs // 4:
            continue
        if pre_roll:
            noise = float(np.median(env[:pk - fs // 8]))
        else:
            # Nothing of the room was recorded before the peak, so the only
            # floor available is the quietest part of what the IR contains.
            after = env[pk:]
            noise = float(np.percentile(after, 10)) if len(after) else 0.0
        dr = 10 * np.log10(env[pk] / (noise + EPS))
        if dr < 20:
            continue
        # Stop integrating where the decay meets the noise. Backward
        # integration sums everything after a point, so an integral that runs
        # to the end of the file folds the whole tail's noise into every
        # earlier value and flattens the curve. Subtracting the noise first is
        # not enough: `maximum(..., 0)` rectifies the residual, so what is
        # left is positive-biased and still accumulates. Measured on a
        # synthetic 0.6 s decay with a 45 dB floor, integrating two seconds
        # gave T60 = 4.67 s; stopping at the knee gives 0.62. ISO 3382 asks
        # for this truncation (Lundeby); the fit-range guard below is a
        # different thing and cannot repair a curve that is already wrong.
        be = env[pk:pk + cut]
        below = np.flatnonzero(be <= noise * 10 ** (KNEE_MARGIN_DB / 10))
        knee = int(below[0]) if len(below) else len(be)
        knee = max(knee, int(0.05 * fs))          # never fit on a stub
        seg = np.maximum(y[pk:pk + knee] ** 2 - noise, 0)
        sch = np.cumsum(seg[::-1])[::-1]
        sch_db = 10 * np.log10(sch / (sch[0] + EPS) + 1e-15)
        tax = np.arange(len(sch_db)) / fs
        # An impulse response that has been trimmed (archive material, or
        # any IR cut before its decay finished) ends while still well above
        # the fixed fits' lower limit: the dynamic-range guard cannot fire,
        # because the truncated file has no noise floor to measure. Level
        # of the last 20 ms re the peak says how far the decay was actually
        # observed; below that, T20/T30 would extrapolate off the end.
        tail = env[pk:pk + knee][-max(1, int(0.02 * fs)):]
        obs_db = 10 * np.log10(float(tail.mean()) / (env[pk] + EPS) + EPS)
        res = {"dr_db": round(float(dr), 0)}
        for key, hi_db, lo_db, need_dr in (
                ("T60", -5.0, max(-35.0, -dr + 8), 0.0),
                ("T20", -5.0, -25.0, 35.0),
                ("T30", -5.0, -35.0, 45.0),
                ("EDT", 0.0, -10.0, 0.0)):
            if dr < need_dr:
                continue
            if key == "T60":
                # T60's lower limit is adaptive, so as the dynamic range
                # approaches the 20 dB guard above the fitted range collapses:
                # at dr = 20 it is 7 dB wide and is extrapolated 8.5x to reach
                # 60. That lever turns ordinary curvature into a large error
                # in the reported time, and it under-reports -- a 0.6 s decay
                # cut to 0.20 s measured 0.34 s. T20 and T30 are immune
                # because ISO 3382 fixes their ranges at 20 and 30 dB; the
                # adaptive estimate needs the same kind of floor. Refusing is
                # right: a T60 is a claim about 60 dB of decay, and 7 dB of
                # evidence does not support one.
                res["fit_db"] = round(float(hi_db - lo_db), 0)
                if hi_db - lo_db < MIN_FIT_SPAN_DB:
                    continue
            if key in ("T20", "T30") and obs_db > lo_db:
                continue                    # range not present in the file
            m = (sch_db <= hi_db) & (sch_db >= lo_db)
            if m.sum() < 150:
                continue
            A = np.vstack([tax[m], np.ones(int(m.sum()))]).T
            slope, _ = np.linalg.lstsq(A, sch_db[m], rcond=None)[0]
            if slope < 0:
                res[key] = round(-60.0 / slope, 2)
        for key, ms in (("C50", 50), ("C80", 80)):
            i = int(ms * fs / 1000)
            if i < len(sch) and sch[i] > 0:
                res[key] = round(float(10 * np.log10(
                    (sch[0] - sch[i]) / (sch[i] + EPS) + EPS)), 1)
        i50 = int(0.05 * fs)
        if i50 < len(sch):
            res["D50"] = round(float((sch[0] - sch[i50]) / (sch[0] + EPS)), 2)
        if "T60" in res:
            out[f"{lo}-{hi}"] = res
    return out


def quietest_channel(x, fs: int, pct: float = 5.0, frame_s: float = 1.0):
    """Which capsule of a multi-microphone node has the most room to measure in.

    A node's capsules share a housing, a preamp and a gain setting, so they
    should agree. When one does not — a blocked port, a damaged capsule — it
    reaches the same peaks as its siblings but on a raised floor, which costs
    dynamic range on every descriptor computed from it. Reading channel 0 by
    convention is then a coin toss.

    Measured in the SINS network: node 9's four capsules reached the same
    98th-percentile level within 0.7 dB, while channel 0's floor sat 5.5 dB
    higher than the other three. Analysing channel 0 halved that node's
    apparent dynamics and pushed the fraction of time it spent at its own
    floor from about 75 % to 96 %, which is most of what made it look broken.

    Run over that corpus it is decisive where it matters and indifferent
    where it does not: on node 9 it picks the same channel on every minute
    tried, away from the raised one; on a node whose capsules agree within
    2 dB it picks whichever is marginally lower, which is of no consequence.

    Returns ``(index, floors_db)`` — the channel with the lowest floor, and
    every channel's floor, so the caller can see how much the choice matters.
    A mono signal returns ``(0, [floor])``.
    """
    a = np.asarray(x, float)
    if a.ndim == 1:
        a = a[:, None]
    n = max(1, int(frame_s * fs))
    floors = []
    for c in range(a.shape[1]):
        y = a[:, c] - a[:, c].mean()
        lv = np.array([10 * np.log10((y[i:i + n] ** 2).mean() + EPS)
                       for i in range(0, max(len(y) - n, 1), n)])
        floors.append(float(np.percentile(lv, pct)) if len(lv) else float("nan"))
    return int(np.nanargmin(floors)), floors


FLOOR_SPREAD_THRESH_DB = 1.5

#: How close to its own floor a second must sit to count as "at the floor".
AT_FLOOR_WITHIN_DB = 3.0


FLOOR_MARGIN_DB = 6.0        # excess over the floor before a frame counts
FLOOR_MIN_COVERAGE = 0.10    # below this, report no level rather than a bad one
_MIN_STAT_BIAS_DB = 1.5      # the minimum of a noisy estimate sits below its mean


def track_noise_floor(level_db, dt: float, win_s: float = 120.0,
                      bias_db: float = _MIN_STAT_BIAS_DB) -> np.ndarray:
    """The recorder's own floor, followed over time rather than fixed once.

    A single floor figure per session cannot be right when the floor moves:
    a floor-dominated node in the SINS corpus swings 10.7 dB between night and
    midday as its electronics warm, which is larger than most of the
    differences such a figure would be used to interpret.

    Minimum statistics, after Martin (2001): the running minimum of power
    over a window long enough to contain a genuine gap in the source. The
    minimum of a fluctuating estimate sits below the mean of the noise it
    estimates, so it is lifted by ``bias_db`` rather than left to
    under-subtract.

    ``win_s`` is the one judgement. Too short and speech or music is
    mistaken for floor; too long and real drift is smoothed away. Two
    minutes suits domestic recordings, where gaps are frequent.

    **A source that never stops is a floor.** The method finds the quietest
    moment in each window, so anything continuous throughout — ventilation,
    a fridge, traffic hum — is absorbed into the estimate and subtracted
    away. That is the correct reading of "what is the recorder's own
    contribution" only when the steady sound *is* the recorder. Where a room
    has a genuine constant, this measures everything above it and reports
    the constant as floor. Say so when reporting, or widen ``win_s`` past
    the longest expected silence and accept the loss of drift tracking.
    """
    p = 10.0 ** (np.asarray(level_db, float) / 10.0)
    n = max(1, int(round(win_s / dt)))
    if n >= len(p):
        return np.full(len(p), 10 * np.log10(p.min()) + bias_db)
    pad = np.pad(p, (n // 2, n - 1 - n // 2), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, n)
    return 10 * np.log10(win.min(axis=1)[:len(p)]) + bias_db


def floor_corrected_level(level_db, dt: float,
                          margin_db: float = FLOOR_MARGIN_DB,
                          win_s: float = 120.0):
    """Level of the source alone, and where there is no source to measure.

    Returns ``(signal_db, floor_db, measurable)``. Noise adds as energy, so
    the floor is subtracted in power; subtracting decibels is a category
    error that happens to look plausible.

    Frames whose excess over the floor falls short of ``margin_db`` are
    returned as ``nan`` and marked unmeasurable. They are not zero and not
    the floor: they carry no information about the source, and giving them a
    value invents one. Clamping them instead is actively harmful — it turns
    a bias in level into a bias in *sampling*, because the frames that
    survive are the loud ones, and an average over survivors then reports a
    quiet span as loud. That error made a living room's midday read quieter
    than its night before this function existed.
    """
    lvl = np.asarray(level_db, float)
    p = 10.0 ** (lvl / 10.0)
    floor_db = track_noise_floor(lvl, dt, win_s=win_s)
    nfloor = 10.0 ** (floor_db / 10.0)
    excess = p - nfloor
    measurable = excess > nfloor * (10 ** (margin_db / 10.0) - 1.0)
    sig = np.full(len(p), np.nan)
    sig[measurable] = 10.0 * np.log10(excess[measurable])
    return sig, floor_db, measurable


def summarize_floor_corrected(signal_db, measurable,
                              min_coverage: float = FLOOR_MIN_COVERAGE) -> dict:
    """An energy mean over the measurable frames, with its coverage attached.

    ``coverage`` is not a footnote. A level computed over 4 % of a session
    and one computed over 96 % are different kinds of statement, and the
    level alone cannot tell them apart — in the SINS corpus a node reading a
    plausible 6.6 dB below the living room turned out to clear its own floor
    on 4 % of frames. Below ``min_coverage`` no level is returned at all,
    because there is nothing there to average.
    """
    measurable = np.asarray(measurable, bool)
    cov = float(measurable.mean()) if measurable.size else 0.0
    if cov < min_coverage:
        return {"level_db": None, "coverage": round(cov, 4),
                "n_measurable": int(measurable.sum()),
                "reason": "below the noise floor too often to measure"}
    v = np.asarray(signal_db, float)[measurable]
    v = v[np.isfinite(v)]
    return {"level_db": round(float(10 * np.log10(np.mean(10 ** (v / 10)))), 2),
            "coverage": round(cov, 4), "n_measurable": int(v.size),
            "reason": ""}


MACHINE_MIN_EXCESS_DB = 3.0    # steady sound this far over self-noise counts


def steady_sources(level_db, dt: float, short_s: float = 120.0,
                   long_s: float = 7200.0,
                   min_excess_db: float = MACHINE_MIN_EXCESS_DB) -> dict:
    """Separate a machine that cycles from the recorder that never stops.

    :func:`track_noise_floor` over a short window calls anything steady a
    floor, which is wrong for the sources this toolbox is usually pointed at.
    A fridge, a ventilation plant, a circulation pump: steady for minutes,
    and the object of study rather than the noise.

    What distinguishes them from the recorder's own contribution is that
    **they turn off**. Self-noise does not. So the floor is estimated twice —
    over minutes, which absorbs a running machine, and over hours, which does
    not, because the machine's off-phase falls inside the window. Where the
    short floor sits materially above the long one, the difference is
    machinery, and ``machine_duty`` says how much of the time it runs.

    Returns ``self_noise_db`` (the long-window floor, what never stops),
    ``steady_excess_db`` (how far the steady sound rises above it while
    running), ``machine_duty``, ``machine_detected``, and
    ``inseparable_steady_source``.

    That last flag is the honest case. A plant that runs continuously for
    longer than ``long_s`` cannot be told from self-noise by level alone — by
    this method or any other that only sees one number per frame — so it is
    flagged rather than quietly subtracted. Seeing it means either the source
    genuinely never stops, or ``long_s`` is shorter than its off-phase. A
    domestic fridge cycles over roughly three quarters of an hour, so two
    hours is a safe default; a building's ventilation may run all day, and no
    window will separate it.

    **How to tell that ``long_s`` is too short.** It does not fail loudly.
    ``machine_duty`` is understated first — a fridge running 60 % of the time
    reported at 17 % — and only then does ``self_noise_db`` climb toward the
    machine's own level. If the duty looks implausibly low for a machine you
    can hear in the recording, lengthen the window before believing the
    floor.
    """
    lvl = np.asarray(level_db, float)
    short = track_noise_floor(lvl, dt, win_s=short_s)
    long = track_noise_floor(lvl, dt, win_s=long_s)
    excess = short - long
    running = excess > min_excess_db
    duty = float(running.mean())
    detected = bool(0.02 < duty < 0.98)
    return {
        "self_noise_db": round(float(np.median(long)), 2),
        "steady_excess_db": round(float(np.median(excess[running]))
                                  if running.any() else 0.0, 2),
        "machine_duty": round(duty, 3),
        "machine_detected": detected,
        # Not "there is one" but "one cannot be ruled out". With no cycling
        # found, a perfectly constant source is indistinguishable from the
        # recorder by level alone, and is therefore inside self_noise_db.
        "steady_source_unresolved": not detected,
    }


# One ladder for the whole toolbox: the descriptor registry's bands, which
# now reach past a day. Two sets of timescale names in one library is a
# contradiction waiting to happen.
CYCLE_MIN_STRENGTH = 0.25


def cycle_band(period_s: float) -> str:
    """Which rung of the ladder a period sits on.

    Delegates to the descriptor registry so that a period and a descriptor
    window are named by the same scheme.
    """
    from .timescales import band_of
    return band_of(period_s)


def _full_acf(x):
    """Normalised autocorrelation at every lag, by FFT."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 8:
        return np.array([])
    x = x - x.mean()
    fft = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(fft * np.conj(fft), 2 * n)[:n]
    return acf / acf[0] if acf[0] > 0 else np.zeros(n)


def _prepare(level_db):
    x = np.asarray(level_db, float)
    ok = np.isfinite(x)
    if ok.sum() < 8:
        return None
    return np.where(ok, x, np.nanmean(x[ok]))


def _smooth(x, n):
    if n < 2:
        return x
    k = np.ones(int(n)) / float(int(n))
    return np.convolve(x, k, mode="same")


def _scaled_acf(x, dt, lo_lag, hi_lag):
    """Autocorrelation evaluated at the scale of the period being asked about.

    A cycle is only visible in a series that has been looked at from far
    enough away. A day-long rhythm inside a room's minute-to-minute
    fluctuation explains a small share of the total variance, so a raw
    autocorrelation scores it near 0.15 and any sane threshold discards it —
    even though the shape is textbook, negative at twelve hours and positive
    at twenty-four.

    So the series is smoothed to roughly an eighth of the period before its
    autocorrelation is read, one octave of periods at a time. This is the
    same principle the rest of this module keeps running into: the answer
    depends on the window, and the window has to be chosen to suit the
    question rather than left at whatever the recording happened to give.
    """
    out = np.zeros(hi_lag + 1)
    lag = max(1, lo_lag)
    while lag <= hi_lag:
        top = min(hi_lag, lag * 2)
        acf = _full_acf(_smooth(x, max(1, lag // 8)))
        if len(acf):
            hi = min(top, len(acf) - 1)
            if hi >= lag:
                out[lag:hi + 1] = acf[lag:hi + 1]
        lag = top + 1
    return out


def cycle_spectrum(level_db, dt: float, min_period_s: float = 60.0,
                   max_period_s: float = 6 * 3600.0, n_periods: int = 192):
    """How strongly a level series repeats, across a ladder of periods.

    The premise is that **what never changes cannot be used**. A recorder's
    own hiss is stationary; a room is not. A fridge turns over in tens of
    minutes, a ventilation plant in hours, a household in a day, a heating
    system in a season. So "signal or noise?" is really a question about
    periodicity, asked at every timescale at once, and the period that
    answers it also names the thing.

    Each period is judged on a version of the series smoothed to suit it —
    see :func:`_scaled_acf`. Returns ``(periods_s, strength)`` in 0–1, log
    spaced, each grid point carrying the strongest lag in its bin so a sharp
    peak survives being summarised. `nan` samples are tolerated.

    .. warning::
       **Validated at the ``cyclic`` band; provisional above it.** On real
       recordings this reliably recovers machinery — two rooms of a domestic
       network agreeing on a 62-minute cycle. At ``circadian`` and longer it
       is not yet trustworthy: over six days of real data it returned a
       48-hour harmonic rather than the 24-hour fundamental, and a spurious
       two-hour peak on a floor-dominated node. Six days is five repetitions of a
       daily cycle, which is thin, and the estimate is sensitive to how the
       series is smoothed. Treat any period beyond a few hours as a
       hypothesis to check by other means, and prefer a direct test — how a
       quantity varies by hour of day — for anything circadian.

    Working on the level series rather than the waveform is deliberate. What
    repeats at these scales is loudness, not pressure, and the level series
    survives coding, resampling and even a change of recorder.
    """
    x = _prepare(level_db)
    if x is None:
        return np.array([]), np.array([])
    lo = max(min_period_s, 2 * dt)
    hi = min(max_period_s, len(x) * dt / 2.5)
    if hi <= lo:
        return np.array([]), np.array([])
    acf = _scaled_acf(x, dt, int(lo / dt), int(hi / dt))
    edges = np.geomspace(lo, hi, n_periods + 1)
    periods, strength = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        i0, i1 = int(a / dt), max(int(a / dt) + 1, int(b / dt))
        seg = acf[i0:min(i1, len(acf))]
        if not len(seg):
            continue
        k = int(np.argmax(seg))
        periods.append((i0 + k) * dt)
        strength.append(float(seg[k]))
    return np.array(periods), np.clip(np.array(strength), 0.0, 1.0)


def dominant_cycles(level_db, dt: float, min_period_s: float = 60.0,
                    max_period_s: float = 6 * 3600.0, top: int = 3,
                    min_strength: float = CYCLE_MIN_STRENGTH,
                    min_prominence: float = 0.05) -> list[dict]:
    """The periods a level series actually repeats at, strongest first.

    Peaks of the autocorrelation, each reported with the band it belongs to.
    An empty list means the series is stationary at every scale asked about —
    which, for a recording of a room, is a statement about the recorder
    rather than about the room.

    Two things have to be got right or this measures the wrong quantity.

    **Prominence, not height.** Autocorrelation is high at short lag for
    anything that varies slowly, so a single 24-hour swing scores well at a
    ten-minute lag purely by being smooth. That is not a ten-minute cycle. A
    peak has to stand clear of the troughs around it, which is what
    distinguishes repeating from merely drifting.

    **Harmonics are not findings.** A cycle of period *T* also correlates at
    2*T* and 3*T*, so a peak near a low integer multiple of an accepted
    shorter period is dropped. The bound matters: a day is 32 fridge cycles
    long and is emphatically its own thing.
    """
    x = _prepare(level_db)
    if x is None:
        return []
    lo_lag = max(1, int(max(min_period_s, 2 * dt) / dt))
    hi_lag = min(len(x) - 1, int(min(max_period_s, len(x) * dt / 2.5) / dt))
    if hi_lag <= lo_lag + 2:
        return []
    acf = _scaled_acf(x, dt, lo_lag, hi_lag)

    peaks = []
    for i in range(lo_lag + 1, hi_lag):
        if not (acf[i] >= acf[i - 1] and acf[i] > acf[i + 1]):
            continue
        if acf[i] < min_strength:
            continue
        # prominence against the troughs on either side, out to one period
        w = max(2, i // 2)
        left = acf[max(lo_lag, i - w):i].min()
        right = acf[i + 1:min(hi_lag, i + w) + 1].min()
        if acf[i] - max(left, right) < min_prominence:
            continue
        peaks.append({"period_s": round(i * dt, 1),
                      "strength": round(float(acf[i]), 3),
                      "band": cycle_band(i * dt)})

    peaks.sort(key=lambda c: c["period_s"])          # fundamentals first
    kept: list[dict] = []
    for c in peaks:
        if any(1.85 <= (r := c["period_s"] / k["period_s"]) <= 8.0
               and abs(r - round(r)) < 0.15 for k in kept):
            continue
        kept.append(c)

    kept.sort(key=lambda c: -c["strength"])
    seen, out = set(), []
    for c in kept:
        if c["band"] in seen:
            continue
        seen.add(c["band"])
        out.append(c)
    return out[:top]


def cycle_residual(level_db, dt: float, min_period_s: float = 60.0,
                   max_period_s: float = 6 * 3600.0,
                   n_sigma: float = 5.0, min_gap_s: float = 60.0) -> dict:
    """What the room did *not* repeat — anomaly as the complement of rhythm.

    A rhythm and an anomaly are opposite readings of one series, and the
    difference matters practically. An outlier detector run on a kitchen
    flags the fridge thirty times a day, because every start is a step
    change; it is a perfectly good detector answering the wrong question.
    The fridge is not an anomaly, it is the room's normal behaviour, and what
    makes it normal is precisely that it repeats.

    So: find the strongest cycle, fold the series onto its phase to get what
    the room usually does at that point in the cycle, subtract it, and look
    at what survives. A spike has no period and cannot be folded away, so it
    stands out in the residual. A machine can, and does not.

    Returns the period used, the residual's spread, and any excursions past
    ``n_sigma`` robust deviations, each with its time and how far past the
    threshold it went.

    Three things this does not do, worth knowing before trusting it. It
    models one cycle, not several at once. It treats a *change* in the cycle
    — a fridge whose period drifts as it fails — as residual rather than as
    the more interesting finding it usually is. And with no cycle found it
    falls back to the plain series, where any slow drift will read as
    anomalous.
    """
    x = np.asarray(level_db, float)
    ok = np.isfinite(x)
    if ok.sum() < 8:
        return {"period_s": None, "residual_std_db": 0.0, "anomalies": []}
    x = np.where(ok, x, np.nanmean(x[ok]))

    cycles = dominant_cycles(x, dt, min_period_s, max_period_s, top=1)
    if cycles:
        period_s = cycles[0]["period_s"]
        lag = max(2, int(round(period_s / dt)))
        phase = np.arange(len(x)) % lag
        # the room's usual behaviour at each point of the cycle
        expected = np.zeros(lag)
        for k in range(lag):
            expected[k] = np.median(x[phase == k])
        resid = x - expected[phase]
    else:
        period_s = None
        resid = x - np.median(x)

    # robust spread: a spike must not inflate the threshold that finds it
    mad = float(np.median(np.abs(resid - np.median(resid))))
    sigma = 1.4826 * mad if mad > 0 else float(resid.std())
    thresh = n_sigma * sigma

    anomalies, last_t = [], -np.inf
    for i in np.flatnonzero(np.abs(resid) > thresh):
        t_s = float(i * dt)
        if t_s - last_t < min_gap_s:
            continue
        last_t = t_s
        anomalies.append({"t_s": round(t_s, 1),
                          "excess_db": round(float(abs(resid[i]) - thresh), 2),
                          "direction": "up" if resid[i] > 0 else "down"})
    return {"period_s": period_s,
            "residual_std_db": round(float(sigma), 3),
            "anomalies": anomalies}


def cycle_drift(level_db, dt: float, min_period_s: float = 600.0,
                max_period_s: float = 6 * 3600.0, n_windows: int = 6,
                min_drift_pct: float = 8.0) -> dict:
    """Is the rhythm itself changing? The case between a spike and a cycle.

    Three detectors already exist for a room and none of them sees this. An
    event detector sees a fridge start. An outlier detector sees the same
    start and calls it anomalous thirty times a day. A cycle finder sees the
    period and calls it normal. But a compressor that is beginning to fail
    does not produce anomalies and does not stop cycling — its **period
    drifts**, and that is what whoever owns the building would want to know.

    It is not an anomaly, because nothing is out of the ordinary from one
    moment to the next, and it is not the rhythm, because the rhythm is no
    longer what it was.

    The series is split into overlapping windows, the dominant period found
    in each, and a trend fitted across them. Returns the median period,
    whether it is drifting, the direction, and the drift as a percentage of
    the median.

    Needs a long recording: several windows, each holding several cycles, so
    perhaps twenty periods end to end. For a domestic fridge that is most of
    a day; for a ventilation plant, a week.
    """
    x = _prepare(level_db)
    if x is None:
        return {"period_s": None, "drifting": False, "direction": "",
                "drift_pct": 0.0, "periods_s": []}

    seg = len(x) // max(2, (n_windows + 1) // 2)
    if seg < int(4 * min_period_s / dt):
        seg = len(x)
    step = max(1, (len(x) - seg) // max(1, n_windows - 1)) if len(x) > seg else 1

    found = []
    for k in range(n_windows):
        i0 = k * step
        chunk = x[i0:i0 + seg]
        if len(chunk) < int(4 * min_period_s / dt):
            break
        c = dominant_cycles(chunk, dt, min_period_s, max_period_s, top=1)
        if c:
            found.append((i0 * dt + seg * dt / 2, c[0]["period_s"]))

    if len(found) < 3:
        return {"period_s": None if not found else round(
            float(np.median([p for _, p in found])), 1),
            "drifting": False, "direction": "", "drift_pct": 0.0,
            "periods_s": [round(p, 1) for _, p in found]}

    ts = np.array([t for t, _ in found], float)
    ps = np.array([p for _, p in found], float)
    med = float(np.median(ps))
    slope = float(np.polyfit(ts, ps, 1)[0])           # seconds of period per second
    change = slope * (ts[-1] - ts[0])
    pct = 100.0 * change / med if med > 0 else 0.0
    drifting = abs(pct) >= min_drift_pct
    return {"period_s": round(med, 1),
            "drifting": bool(drifting),
            "direction": ("lengthening" if pct > 0 else "shortening") if drifting else "",
            "drift_pct": round(float(pct), 1),
            "periods_s": [round(p, 1) for _, p in found]}


def cycle_profile(level_db, dt: float, min_period_s: float = 60.0,
                  max_period_s: float = 3 * 86400.0) -> dict:
    """What kind of thing is cycling here — a room, or the recorder?

    Periodicity alone does not separate them, and this is the correction the
    SINS corpus forced: a converter warms and cools with the building, so a
    node sitting at its own floor still shows a 24-hour cycle. A diurnal
    rhythm with
    *nothing faster underneath it* is the signature of an instrument
    breathing with the room temperature. A household leaves faster marks as
    well — a fridge, a kettle, a shower — and those are what
    ``has_sub_daily_cycle`` reports.
    """
    cycles = dominant_cycles(level_db, dt, min_period_s, max_period_s, top=6)
    sub = [c for c in cycles if c["band"] in ("meso", "macro", "cyclic")]
    diurnal = [c for c in cycles if c["band"] == "circadian"]
    return {
        "cycles": cycles,
        "has_sub_daily_cycle": bool(sub),
        "diurnal_only": bool(diurnal and not sub),
        "stationary": not cycles,
    }


ONSET_RISE = 0.25     # fraction of a series' own floor-to-peak range


def series_onset(series, rise: float = ONSET_RISE):
    """First index where a series passes `rise` of its own floor-to-peak range.

    Deliberately scale-free. It is used to compare series in different units —
    a motion measure in pixels against an acoustic energy — and any rule with
    an absolute threshold would compare the units instead.

    **The default is tuned for a difference, not for an absolute onset.**
    A quarter of a 40 dB floor-to-peak range is 30 dB below the peak, which on
    a recording of a sound-producing action is reached by the action's own
    small noises — an object picked up, a step, a hand on a surface — well
    before the sound the clip is *of*. Measured on the Sound Actions clips,
    where the lead-in sits a median 40 dB below the event peak but 5.5 dB
    above the clip floor and carries such transients, the default returns a
    median 1.78 s early against hand-checked onsets and agrees within a
    quarter second on 17 % of them; ``rise=0.75`` lands +0.01 s and agrees on
    77 %.

    So the choice is not noise versus signal but *which* sound is the onset.
    The default finds the first thing audible above the floor; a higher rise
    finds the event the clip was cut for.

    :func:`onset_lead` applies the same rule to both series, which is what
    keeps a lead comparable across modalities. How much of the bias survives
    the subtraction depends on the two series having similar shapes, and is
    not guaranteed. Pass a higher ``rise`` — 0.75 on this kind of material —
    whenever the answer wanted is an onset rather than a lead, and check it
    against a case whose answer is known.
    """
    s = np.asarray(series, float)
    s = s[np.isfinite(s)]
    if len(s) < 5:
        return None
    lo, hi = float(np.percentile(s, 5)), float(s.max())
    if hi <= lo:
        return None
    idx = np.flatnonzero(np.asarray(series, float) >= lo + rise * (hi - lo))
    return int(idx[0]) if len(idx) else None


def onset_lead(first, second, dt: float, rise: float = ONSET_RISE) -> dict:
    """How far one series begins before another — an action before its sound.

    A sound-producing action starts well before the sound it produces: an
    intention becomes neural and then muscular activity, then motion in the
    arm and the object, and only at the end an acoustic attack. A sound object
    therefore *embeds* an action, and the silence in front of the attack is
    not empty — it is where the action already is.

    Measured on 180 clips of the Sound Actions corpus, giving `first` a
    quantity-of-motion series from the video and `second` the audio energy on
    the same time grid: motion leads sound by a **median 0.72 s**, and does so
    in **84 %** of clips. The remaining sixth is worth keeping in view rather
    than treating as error — an object already moving when it is struck, or an
    action that happens out of frame, genuinely has no visible lead.

    Both series are given the same onset rule, applied to each one's own
    range, so the result does not depend on either modality's units. Returns
    ``lead_s`` (positive when `first` begins earlier), the two onset times,
    and which one led.

    This is the seam between the toolboxes rather than a video function: pass
    a motion series computed wherever motion is computed. It is what makes
    audio–video analysis more than two analyses side by side — the lead is a
    property of the *action*, and neither modality carries it alone.

``lead_s`` is the output this was validated on. ``first_onset_s`` and
    ``second_onset_s`` are returned for inspection and are *not* reliable
    onset times at the default ``rise``: on the Sound Actions clips they sit a
    median 1.78 s early, firing on the action's own handling noises rather
    than on the sound the clip is of. Raise ``rise`` before reading them as
    times — see :func:`series_onset`.
    """
    i, j = series_onset(first, rise), series_onset(second, rise)
    if i is None or j is None:
        return {"lead_s": None, "first_onset_s": None, "second_onset_s": None,
                "leads": "", "reason": "one series has no onset to find"}
    lead = (j - i) * dt
    return {"lead_s": round(float(lead), 4),
            "first_onset_s": round(i * dt, 4),
            "second_onset_s": round(j * dt, 4),
            "leads": "first" if lead > 0 else ("second" if lead < 0 else "neither"),
            "reason": ""}


def floor_occupancy(F: dict, within_db: float = AT_FLOOR_WITHIN_DB,
                    pct: float = 5.0) -> dict:
    """How much of a session sits at its own noise floor.

    :func:`floor_suspicion` asks whether a *band's* floor is self-noise. It
    is the right question and it has a limit: across a whole sensor network
    it can fire for every node, because every recorder's top octaves are its
    own hiss during quiet hours. It therefore cannot separate "this band is
    the instrument" from "this room was empty all week".

    This asks the second question. Each second's broadband level is compared
    against the session's own ``pct``-percentile floor, and the fraction
    within ``within_db`` of it is returned. A living room in use spends
    little time there; a bedroom occupied only to sleep spends most of the
    week there. That is a description of how a room is used, not a fault in
    the recorder — a distinction this corpus cost several hours to learn,
    when a microphone in a mostly-empty bedroom was diagnosed as dead.

    Being each session against its own floor, the measure does not move with
    recording gain, and so is comparable across uncalibrated instruments in
    a way an absolute level is not.

    Measured across the SINS network, one week each: the living-room and
    kitchen nodes sit at their own floor 28-56 % of the time (median 2.4-9.8
    dB above it), while the bedroom node sits there **96 %** of the time,
    a median of 0.4 dB above. One number, and it names the room.

    Returns ``{"at_floor_fraction", "floor_db", "median_above_floor_db"}``.
    """
    op = np.asarray(F["oct_pow"], float)
    if op.ndim != 2 or not len(op):
        return {"at_floor_fraction": None, "floor_db": None,
                "median_above_floor_db": None}
    lvl = 10 * np.log10(op.sum(axis=1) + EPS)
    floor = float(np.percentile(lvl, pct))
    above = lvl - floor
    return {"at_floor_fraction": round(float((above <= within_db).mean()), 3),
            "floor_db": round(floor, 1),
            "median_above_floor_db": round(float(np.median(above)), 1)}


def floor_suspicion(F: dict, chunk_s: float = 300.0, pct: float = 10.0,
                    spread_thresh_db: float = FLOOR_SPREAD_THRESH_DB,
                    min_chunks: int = 6, hf_min_hz: float = 2000.0) -> dict:
    """Flag high-frequency band floors that look like recorder self-noise.

    A genuine room background breathes: its low-percentile level moves with
    the day, the weather and the building. A microphone's self-noise floor
    does not — it is abnormally flat over time (and typically spectrally
    smooth). In the SINS sensor-network corpus the 4–8 kHz floor of a
    living room is flat to 0.8 dB across a full week (0.56 dB between six
    separate nights), while every band below 1 kHz varies by 2.4–5.3 dB
    over the same nights: the top of the spectrum is the instrument, and
    any L90-derived descriptor weighted towards it (LA90 in particular)
    measures the recorder rather than the room.

    The check works on the cached 1 s octave-band powers: the session is
    cut into ``chunk_s`` chunks, each chunk's ``pct``-percentile band level
    is that chunk's floor, and the temporal spread of the floor is taken as
    the median minus the 5th-percentile chunk floor — a low-tail statistic,
    so chunks whose floor is raised by activity (television, dishes) do not
    hide a pinned quiet-time floor. A band centred at or above
    ``hf_min_hz`` whose spread is below ``spread_thresh_db`` is suspect.
    The 1.5 dB default sits between the SINS self-noise band (≤ 0.8 dB
    over a week) and the quietest genuinely acoustic bands there
    (≥ 2.4 dB), with at least 0.7 dB of margin to each side. Bands with no
    content below the Nyquist frequency, and sessions shorter than
    ``min_chunks`` chunks (30 min at the defaults), are never flagged.

    This is an annotation, not a correction: no descriptor value changes.
    Returns ``floor_suspect`` (bool), the affected band range
    ``floor_suspect_lo_hz``/``floor_suspect_hi_hz`` (band edges, Hz), and
    ``floor_spread_db`` (the smallest spread among the flagged bands);
    the last three are None when nothing is flagged.
    """
    from .features import OCT_CENTERS
    out = {"floor_suspect": False, "floor_suspect_lo_hz": None,
           "floor_suspect_hi_hz": None, "floor_spread_db": None}
    op = F.get("oct_pow")
    if op is None or len(op) == 0:
        return out
    rows = max(1, int(round(chunk_s)))          # 1 s frames per chunk
    nchunk = len(op) // rows
    if nchunk < min_chunks:
        return out
    lvl = db(np.asarray(op[:nchunk * rows], np.float64))
    floors = np.percentile(lvl.reshape(nchunk, rows, lvl.shape[1]),
                           pct, axis=1)         # (nchunk, nband)
    spread = (np.percentile(floors, 50, axis=0)
              - np.percentile(floors, 5, axis=0))
    centers = np.asarray(OCT_CENTERS, float)[:lvl.shape[1]]
    nyq = float(F.get("fs", 48000)) / 2
    med = np.median(floors, axis=0)
    flagged = ((centers >= hf_min_hz) & (centers / np.sqrt(2) < nyq)
               & (spread < spread_thresh_db) & (med > -119.0))
    if flagged.any():
        idx = np.flatnonzero(flagged)
        out.update({
            "floor_suspect": True,
            "floor_suspect_lo_hz": int(round(centers[idx[0]] / np.sqrt(2))),
            "floor_suspect_hi_hz": int(round(min(centers[idx[-1]]
                                                 * np.sqrt(2), nyq))),
            "floor_spread_db": round(float(spread[idx].min()), 2),
        })
    return out


def circular_stats(az_deg, weights=None):
    """Energy-weighted circular mean (deg) and resultant length R."""
    from .circstats import mean_resultant
    mu, R = mean_resultant(np.radians(np.asarray(az_deg, float)), weights)
    return float(np.degrees(mu)), R


def summarize(F: dict) -> dict:
    """Session descriptor dict from concatenated features (see features.load_features)."""
    fast, fasta = F["fast_db"], F["fast_dba"]
    dt = float(np.median(np.diff(F["t_fast"]))) if len(F["t_fast"]) > 1 else 0.125
    leq = db(np.mean(10 ** (fast.astype(np.float64) / 10)))
    laeq = db(np.mean(10 ** (fasta.astype(np.float64) / 10)))
    l10, l50, l90 = (float(np.percentile(fast, q)) for q in (90, 50, 10))
    events, bg = detect_events(fast, dt)
    dur = float(len(F["t"]))  # 1 s per feature frame; robust across take gaps

    p = F["rms_w"].astype(np.float64) ** 2
    e_fg = p >= np.percentile(p, 75)
    e_bg = p <= np.percentile(p, 25)
    # direction is full 3-D (ambix), lateral-only (stereo), or absent (mono);
    # emit None for whatever this recording's channel layout cannot support
    az = np.asarray(F["az"], float)
    el = np.asarray(F["el"], float)
    psi = np.asarray(F["diffuse"], float)
    fin_az = np.isfinite(az)
    if fin_az.any():
        az_mean, R = circular_stats(az[fin_az], weights=p[fin_az])
        fg_az = e_fg & fin_az
        az_fg = (circular_stats(az[fg_az], weights=p[fg_az])[0]
                 if fg_az.any() else az_mean)
    else:
        az_mean = R = az_fg = None
    el_fg = (float(np.nanmedian(el[e_fg])) if np.isfinite(el[e_fg]).any()
             else None)
    has_psi = np.isfinite(psi).any()

    return {
        "duration_min": round(dur / 60, 1),
        "leq_dbfs": round(float(leq), 1),
        "laeq_dbfs": round(float(laeq), 1),
        "laeq_trim5_dbfs": round(trimmed_leq(fasta, 5.0), 1),
        "leq_minus_laeq_db": round(float(leq - laeq), 1),
        "L10": round(l10, 1), "L50": round(l50, 1), "L90": round(l90, 1),
        "dynamics_L10_L90": round(l10 - l90, 1),
        "events_per_min": round(len(events) / max(dur / 60, 1e-9), 1),
        "event_median_dur_s": round(float(np.median(
            [(e["i1"] - e["i0"] + 1) * dt for e in events])), 2) if events else None,
        "centroid_median_hz": int(np.median(F["centroid"])),
        "flatness_median": round(float(np.median(F["flatness"])), 3),
        "diffuseness_median": round(float(np.nanmedian(psi)), 2) if has_psi else None,
        "diffuseness_iqr": round(float(np.nanpercentile(psi, 75)
                                       - np.nanpercentile(psi, 25)), 2)
        if has_psi else None,
        "azimuth_mean_deg": round(az_mean, 0) if az_mean is not None else None,
        "azimuth_R": round(R, 2) if R is not None else None,
        "azimuth_fg_deg": round(az_fg, 0) if az_fg is not None else None,
        "elevation_fg_median_deg": round(el_fg, 0) if el_fg is not None else None,
        "n_events": len(events),
        "emergence_db": round(float(laeq - np.percentile(fasta, 10)), 1),
        "intermittency_ratio_pct": round(intermittency_ratio(fasta, dt), 1),
        **floor_suspicion(F),
    }


def decay_time(x: np.ndarray, fs: int, bands=((250, 500), (500, 1000),
               (1000, 2000), (2000, 4000), (4000, 8000))) -> dict:
    """T60 estimates from an impulse via truncated Schroeder integration.

    The decay is truncated at the first re-attack (envelope rising >= 8 dB
    above its running minimum) and at the noise floor; a linear fit of
    -5 dB .. max(-35 dB, floor + 8 dB) is extrapolated to 60 dB.
    Returns {band: (T60, dynamic_range_db)}.
    """
    from scipy import signal as sg
    pk_i = int(np.abs(x).argmax())
    env_bb = sg.convolve(x ** 2, np.ones(480) / 480, "same")
    tail = 10 * np.log10(env_bb[pk_i:pk_i + 3 * fs] + 1e-15)
    run_min = np.minimum.accumulate(tail)
    re = np.flatnonzero((tail - run_min > 8) & (np.arange(len(tail)) > fs // 10))
    cut = int(re[0]) if len(re) else 2 * fs
    out = {}
    for lo, hi in bands:
        sos = sg.butter(4, [lo, hi], "bandpass", fs=fs, output="sos")
        y = sg.sosfilt(sos, x)
        env = sg.convolve(y ** 2, np.ones(240) / 240, "same")
        pk = int(env[max(0, pk_i - 2400):pk_i + 2400].argmax()) + max(0, pk_i - 2400)
        if pk < fs // 4:
            continue
        noise = float(np.median(env[:pk - fs // 8]))
        dr = 10 * np.log10(env[pk] / (noise + EPS))
        if dr < 20:
            continue
        seg = np.maximum(y[pk:pk + cut] ** 2 - noise, 0)
        sch = np.cumsum(seg[::-1])[::-1]
        sch_db = 10 * np.log10(sch / (sch[0] + EPS) + 1e-15)
        tax = np.arange(len(sch_db)) / fs
        lo_db = max(-35.0, -dr + 8)
        m = (sch_db <= -5) & (sch_db >= lo_db)
        if m.sum() < 150:
            continue
        A = np.vstack([tax[m], np.ones(int(m.sum()))]).T
        slope, _ = np.linalg.lstsq(A, sch_db[m], rcond=None)[0]
        if slope < 0:
            out[f"{lo}-{hi}"] = (round(-60.0 / slope, 2), round(float(dr), 0))
    return out


def pick_segments(F: dict, n=4, seg_s=600.0) -> list[dict]:
    """Suggest representative windows: quietest, most active, median-typical,
    and (if present) the strongest state transition.

    Kinds can coincide: a session barely longer than one window has only
    one window to offer, and a stationary room has no most-active minute
    to distinguish from its quietest one. Coincident kinds are returned
    once, the window keeping the first kind's name and listing the others
    under ``also`` — so the degeneracy is visible rather than presented as
    several identical "representative" segments.
    """
    t, fast = F["t_fast"], F["fast_db"]
    dt = float(np.median(np.diff(t)))
    win = max(1, int(seg_s / dt))
    if len(fast) < win:
        return [dict(kind="whole", t0=float(t[0]), dur=float(t[-1] - t[0]))]
    k = np.ones(win) / win
    m_lvl = np.convolve(10 ** (fast.astype(np.float64) / 10), k, "valid")
    var = np.convolve((fast - fast.mean()) ** 2, k, "valid")
    cands = [("quietest", float(t[int(np.argmin(m_lvl))])),
             ("most_active", float(t[int(np.argmax(var))])),
             ("typical", float(t[int(np.argmin(np.abs(
                 db(m_lvl) - np.median(db(m_lvl)))))]))]
    smooth = median_filter(fast, size=max(3, int(30 / dt)) | 1)
    jump = np.abs(np.diff(smooth))
    if jump.max() > 6:
        cands.append(("transition",
                      float(max(t[0], t[int(np.argmax(jump))] - seg_s / 2))))
    picks: list[dict] = []
    for kind, t0 in cands:
        same = next((p for p in picks if abs(p["t0"] - t0) <= dt), None)
        if same is None:
            picks.append(dict(kind=kind, t0=t0, dur=seg_s))
        else:
            same.setdefault("also", []).append(kind)
    return picks[:n]
