"""Machine states: on/off segmentation, switch points, and duty cycles.

Domestic and mechanical sources (ventilation, fridges, pumps, HVAC) show up
in a soundscape as a *state* — a band-limited floor that is either present or
absent — rather than as events. This module segments a band-level timeline
into those states from the cached features, no audio pass:

- ``band_level`` — per-second dB level in a frequency band from the cached
  log-band spectrogram (the "machine band" of a source, e.g. 250–1000 Hz for
  a ventilation unit);
- ``state_segments`` — two-state (on/off) segmentation of that level with an
  automatic bimodal threshold, hysteresis, and a minimum duration, each
  segment carrying its median level and within-state stability (SD);
- ``switch_points`` — the transitions between segments (the 07:53:55
  switch-off moments);
- ``duty_cycle`` — cycle statistics of a cycling machine (a fridge's ~24 min
  period at ~50 % duty): period, duty fraction, cycle count;
- ``cycle_series`` — the same cycles as *series* rather than medians, because
  a machine's on-time and its period have different causes and can move
  independently;
- ``bimodal_separation`` — whether the timeline has two modes at all, which
  has to be asked before any of the above is believed.

Typical use: ``segs = state_segments(band_level(F, (250, 1000)))`` and mask
other analyses (fingerprints, masking, taxonomy states) by segment.

Ask ``bimodal_separation`` first whenever the machine may be faint. The
segmentation always returns something.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

EPS = 1e-20


def band_level(F: dict, band=(250.0, 1000.0)) -> np.ndarray:
    """Per-second dB level in ``band`` (Hz) from the cached ``logspec``."""
    logf = np.asarray(F["logf"], float)
    fc = np.sqrt(logf[:-1] * logf[1:])
    m = (fc >= band[0]) & (fc <= band[1])
    return 10 * np.log10(F["logspec"][:, m].sum(1) + EPS)


def bimodal_threshold(level_db: np.ndarray) -> float:
    """Otsu's threshold on the level histogram: the split that best
    separates the two modes of an on/off timeline."""
    lo, hi = np.percentile(level_db, (0.5, 99.5))
    hist, edges = np.histogram(level_db, bins=128, range=(lo, hi))
    p = hist.astype(float) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    var = (mu_t * w0 - mu) ** 2 / (w0 * (1 - w0) + EPS)
    k = int(np.argmax(var))
    # between-class variance is flat across an empty inter-mode gap; the
    # midpoint of the two class means splits the gap centrally
    mu0 = mu[k] / (w0[k] + EPS)
    mu1 = (mu_t - mu[k]) / (1 - w0[k] + EPS)
    return float((mu0 + mu1) / 2)


def transition_profile(level_db: np.ndarray, segments: list, dt_s: float = 1.0,
                       settle_tol_db: float = 1.0, max_settle_s: float = 120.0):
    """Characterise the boundaries between steady states, not the states.

    A machine starting or stopping is itself a sound action, and it has the
    morphology of one: abrupt, then settling. A refrigerator does not fade
    in. It strikes, clatters for a moment, and subsides into the steady hum
    that will be ignored for the next eleven minutes. Heard on its own that
    is an impulse followed by a sustain, which is to say a sound object in
    Schaeffer's sense, arriving involuntarily in a room rather than
    deliberately in front of a microphone.

    That matters for attention, because the transition is where a
    background briefly becomes a figure and then returns to being a
    background. The steady states either side are what a level summary
    describes; the crossings between them are what anybody in the room
    actually notices, and until now nothing here measured them.

    For each boundary in ``segments`` returns the direction, the size of
    the step, how abruptly it happened (the 10--90 % crossing time) and how
    long the level took to settle within ``settle_tol_db`` of its new
    median. ``None`` for a settling time means it had not settled within
    ``max_settle_s``, which is a finding rather than a gap: a transition
    that never settles is not a machine changing state.
    """
    x = np.asarray(level_db, float)
    out = []
    for a, b in zip(segments[:-1], segments[1:]):
        i = int(round(b["t0_s"] / dt_s))
        if i <= 0 or i >= len(x):
            continue
        lo, hi = float(a["median_db"]), float(b["median_db"])
        step = hi - lo
        if abs(step) < 1e-9:
            continue

        # abruptness: how long the level spends between 10% and 90% of the
        # step, searched in a window around the boundary
        w = int(round(min(max_settle_s, 30.0) / dt_s))
        seg = x[max(0, i - w):min(len(x), i + w)]
        lo_mark, hi_mark = lo + 0.1 * step, lo + 0.9 * step
        if step > 0:
            crossing = np.flatnonzero((seg >= lo_mark) & (seg <= hi_mark))
        else:
            crossing = np.flatnonzero((seg <= lo_mark) & (seg >= hi_mark))
        cross_s = float(len(crossing) * dt_s) if len(crossing) else 0.0

        # Settling: first index after the boundary from which the level
        # stays inside a band around the new state's median. The band is
        # the wider of the caller's tolerance and twice the new state's own
        # variability, because a tolerance tighter than the state's noise
        # would report that a steady state never settles -- which says
        # something about the tolerance, not about the room.
        tol = max(settle_tol_db, 2.0 * float(b.get("sd_db", 0.0) or 0.0))
        m = int(round(max_settle_s / dt_s))
        tail = x[i:min(len(x), i + m)]
        settle = None
        inside = np.abs(tail - hi) <= tol
        # "stays inside" as a fraction rather than as every sample: a
        # steady state that is merely noisy will throw the occasional
        # excursion past any band, and requiring perfection would report
        # that a settled room never settled.
        for k in range(len(inside)):
            if inside[k:].mean() >= 0.9:
                settle = float(k * dt_s)
                break

        out.append(dict(t_s=float(b["t0_s"]),
                        direction="onset" if step > 0 else "cessation",
                        step_db=round(step, 1),
                        crossing_s=round(cross_s, 1),
                        settle_s=None if settle is None else round(settle, 1),
                        settle_tol_db=round(tol, 1),
                        from_db=round(lo, 1), to_db=round(hi, 1)))
    return out


def state_segments(level_db: np.ndarray, thresh_db: float | None = None,
                   smooth_s: float = 11.0, hysteresis_db: float = 1.0,
                   min_dur_s: float = 30.0) -> list[dict]:
    """Two-state segmentation of a 1 Hz band-level timeline.

    The level is median-smoothed over ``smooth_s``; the threshold defaults
    to the bimodal (Otsu) split of the histogram — pass ``thresh_db`` when
    the timeline is not clearly bimodal. Hysteresis of ``hysteresis_db``
    around the threshold suppresses chatter, and segments shorter than
    ``min_dur_s`` are merged into their neighbors. Returns segments in time
    order as dicts: state ('on'/'off'), t0_s/dur_s (seconds into the
    timeline), median_db, and sd_db (within-state stability of the raw
    level — a running machine is *steady*, ambience is not).
    """
    x = np.asarray(level_db, float)
    k = max(3, int(round(smooth_s)) | 1)
    sm = median_filter(x, size=k, mode="nearest")
    th = bimodal_threshold(sm) if thresh_db is None else float(thresh_db)
    on = np.zeros(len(sm), bool)
    cur = sm[0] > th
    for i, v in enumerate(sm):
        if cur and v < th - hysteresis_db / 2:
            cur = False
        elif not cur and v > th + hysteresis_db / 2:
            cur = True
        on[i] = cur

    def bounds(mask):
        edges = np.flatnonzero(np.diff(mask.astype(int))) + 1
        return [0, *edges.tolist(), len(mask)]

    # merge runs shorter than min_dur_s into the surrounding state
    b = bounds(on)
    for i0, i1 in zip(b[:-1], b[1:]):
        if i1 - i0 < min_dur_s and i0 > 0 and i1 < len(on):
            on[i0:i1] = on[i0 - 1]
    b = bounds(on)
    segs = []
    for i0, i1 in zip(b[:-1], b[1:]):
        seg = x[i0:i1]
        segs.append({
            "state": "on" if on[i0] else "off",
            "t0_s": float(i0), "dur_s": float(i1 - i0),
            "median_db": round(float(np.median(seg)), 1),
            "sd_db": round(float(seg.std()), 2),
        })
    return segs


def switch_points(segments: list[dict]) -> list[dict]:
    """Transitions between consecutive segments: time and direction
    ('on' = machine starts, 'off' = machine stops)."""
    out = []
    for a, b in zip(segments[:-1], segments[1:]):
        out.append({"t_s": float(b["t0_s"]),
                    "direction": b["state"],
                    "step_db": round(b["median_db"] - a["median_db"], 1)})
    return out


def duty_cycle(segments: list[dict]) -> dict:
    """Cycle statistics of a cycling machine from its state segments:
    median period (consecutive on-starts), duty fraction (median on-time
    over period), and the number of complete cycles observed."""
    on_starts = np.array([s["t0_s"] for s in segments if s["state"] == "on"])
    on_durs = np.array([s["dur_s"] for s in segments if s["state"] == "on"])
    if len(on_starts) < 2:
        return {"period_s": None, "duty": None,
                "n_cycles": int(len(on_starts))}
    period = float(np.median(np.diff(on_starts)))
    return {"period_s": round(period, 1),
            "duty": round(float(np.median(on_durs)) / period, 3),
            "n_cycles": int(len(on_starts))}


def cycle_series(segments: list[dict]) -> dict:
    """On-time and period per cycle, and whether either is trending.

    A duty fraction hides the finding this exists for. Measured over one
    night, a domestic refrigerator held its on-time between 7.6 and 8.5
    minutes while its period lengthened from 30.5 to 38.0, because the two
    halves of a thermostat cycle have different causes: the compressor runs
    until the cabinet reaches its set point, which takes about as long each
    time and is a property of the appliance, and then waits until the cabinet
    drifts back, which takes longer as the room cools and is a property of the
    room. ``duty_cycle`` returns the ratio of the two and ``cycle_drift`` a
    median and a percentage; neither shows one holding while the other moves.

    Returns ``on_s`` (one per on-segment) and ``period_s`` (one per
    consecutive pair of on-starts, so one shorter), each with the Pearson
    correlation against cycle number and the change per cycle from a linear
    fit. A correlation near zero with a real spread is a machine whose
    interval is set by something that is not changing; a high correlation on
    the period with none on the on-time is the signature above.

    Read the trend against the spread, not on its own: a period that moves by
    a quarter of itself and an on-time that moves by a minute can both
    correlate at 0.9, and only one of them matters.
    """
    on = [s for s in segments if s["state"] == "on"]
    starts = [float(s["t0_s"]) for s in on]
    period_s = [b - a for a, b in zip(starts[:-1], starts[1:])]

    # A run still on when the series ends has an unknown length, and a short
    # tail dragged into the on-time trend will invert it. The period series
    # is unaffected: it is measured onset to onset, so the last onset still
    # closes the previous interval.
    truncated = bool(segments and segments[-1]["state"] == "on")
    on_s = [float(s["dur_s"]) for s in (on[:-1] if truncated else on)]

    def trend(v):
        if len(v) < 3:
            return {"rho": None, "per_cycle": None, "spread": None}
        i = np.arange(len(v), dtype=float)
        y = np.asarray(v, float)
        if y.std() == 0:
            return {"rho": 0.0, "per_cycle": 0.0, "spread": 0.0}
        slope = float(np.polyfit(i, y, 1)[0])
        return {"rho": round(float(np.corrcoef(i, y)[0, 1]), 3),
                "per_cycle": round(slope, 2),
                "spread": round(float(y.max() - y.min()), 2)}

    return {"n_cycles": len(on),
            "on_s": [round(v, 1) for v in on_s],
            "period_s": [round(v, 1) for v in period_s],
            "on_trend": trend(on_s),
            "period_trend": trend(period_s),
            "truncated_final_run": truncated}


def bimodal_separation(level_db: np.ndarray,
                       min_separation_db: float = 2.0,
                       min_fraction: float = 0.02) -> dict:
    """Does this timeline have two modes at all?

    ``bimodal_threshold`` is Otsu's method and will always return a number.
    Asked for a split of a level series with only one populated mode -- a
    machine too faint to clear the room, a recorder in a room the machine is
    not in -- it returns a value inside that single mode, ``state_segments``
    then divides noise, and ``duty_cycle`` reports a period for a machine
    that was never detected. Nothing in the chain says anything is wrong.

    That is not hypothetical. A refrigerator plainly cycling in a kitchen at
    a spread of 8.6 dB was sought in a second room of the same house, where
    it contributes 0.6 dB; the segmentation returned one segment covering the
    whole night and the duty calculation dutifully reported a cycle.

    Returns the two class means either side of the Otsu split, their
    separation in dB, the fraction of the series in the upper class, and
    ``bimodal``, which is False when the separation is under
    ``min_separation_db`` or either class holds less than ``min_fraction`` of
    the series. False does not prove the machine is absent -- only that a
    two-state split of this timeline is not evidence that it is present.
    """
    x = np.asarray(level_db, float)
    x = x[np.isfinite(x)]
    if x.size < 2 or x.max() - x.min() < EPS:
        return {"threshold_db": None, "separation_db": 0.0,
                "upper_fraction": 0.0, "lower_mean_db": None,
                "upper_mean_db": None, "bimodal": False}
    th = bimodal_threshold(x)
    upper, lower = x[x > th], x[x <= th]
    if not len(upper) or not len(lower):
        return {"threshold_db": round(float(th), 2), "separation_db": 0.0,
                "upper_fraction": float(len(upper)) / len(x),
                "lower_mean_db": None, "upper_mean_db": None,
                "bimodal": False}
    lo, hi = float(lower.mean()), float(upper.mean())
    frac = len(upper) / len(x)
    return {"threshold_db": round(float(th), 2),
            "separation_db": round(hi - lo, 2),
            "upper_fraction": round(frac, 4),
            "lower_mean_db": round(lo, 2),
            "upper_mean_db": round(hi, 2),
            "bimodal": bool(hi - lo >= min_separation_db
                            and min_fraction <= frac <= 1 - min_fraction)}
