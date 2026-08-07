"""Sound objects: event-level extraction and Schaeffer typing.

Schaeffer's *objet sonore* is a perceptual unit. It is what the ear can hold
whole in one act of attention — a door closing, a kettle's rattle, a two-second
tone. In the *Traité des objets musicaux* (1966) that horizon is on the order
of half a second to five seconds; below it there is nothing to hear as a shape,
above it attention stops holding the whole and starts following a texture.

A multi-minute steady level regime is therefore not a sound object at all. It
is Schafer's *keynote*: a ground that persists for minutes or hours, heard as
the condition of the place rather than as an event in it. Placing regimes on
the typo-morphology plane conflates two traditions and two timescales; this
module supplies what the plane actually asks for.

The unit of analysis here is the **detected event** — the fast level rising at
least 8 dB above its running background for at least 0.25 s (see
:func:`ambiscape.analysis.detect_events`). Events whose duration falls inside
the object window (:data:`OBJECT_MIN_S` to :data:`OBJECT_MAX_S`, 0.2–8 s by
default) are taken as candidate sound objects; the rest are counted and
reported, never silently dropped. Each surviving object is then characterised
on Schaeffer's two axes from its own signature in the cached features:

- **mass** — the spectral axis, tonic / tonic-complex / complex / noise, from
  the object's *excess* spectrum (what appeared over the running band
  background), via :func:`object_mass`;
- **facture** — the temporal axis, impulse / iteration / sustained (delimited)
  / sustained (unlimited), from the object's own amplitude envelope, via
  :func:`object_facture`.

Both rules are written out in the two functions' docstrings and their
thresholds are module constants, so any proposal can be traced to the number
that produced it. They remain machine-drafted proposals: no public domestic
corpus carries object-level ground truth against which they could be scored
(activity labels are minutes long, an order of magnitude coarser than an
object), so a typing here is a suggestion to confirm by listening, in the same
spirit as the rest of the draft stage.

Everything runs on the cached feature arrays — no audio pass — so a full
domestic day (tens of thousands of events) types in seconds.
"""
from __future__ import annotations

import numpy as np

from .analysis import detect_events

EPS = 1e-20

# --- the object window ----------------------------------------------------
OBJECT_MIN_S = 0.2      # below this there is no shape to hold in attention
OBJECT_MAX_S = 8.0      # above this attention follows a texture, not an object

# --- mass rule (spectral axis) --------------------------------------------
PEAK_PROMINENCE_DB = 6.0   # a band stands out of its octave neighbourhood by this
PEAK_SHARE_TONIC = 0.50    # share of the object's energy sitting in such peaks
PEAK_SHARE_COMPLEX = 0.20
SPREAD_OCT_NOISE = 1.2     # octaves of energy spread that read as broadband

# --- facture rule (temporal axis) -----------------------------------------
SUSTAIN_MAX_S = 5.0     # sustainment past this outlasts the attention horizon
IMPULSE_MAX_S = 1.0     # an impulse is over almost as soon as it starts
ATTACK_MAX_S = 0.08     # ... and gets there in one step
ITER_MIN_S = 0.4        # shorter than this, periodicity cannot be measured
ITER_LO_HZ, ITER_HI_HZ = 3.0, 20.0   # iteration rate band (grain, rattle, roll)
ITER_ACF = 0.35         # normalised envelope autocorrelation at the iteration lag


def _excess_spectrum(logspec: np.ndarray, bg: np.ndarray,
                     r0: int, r1: int) -> np.ndarray:
    """Linear excess power per band over the running background, >= 0.

    ``r0``/``r1`` are inclusive 1 Hz row indices of the object. What is left
    is the object's own spectrum — the bed it sounded over is subtracted, so a
    click over a ventilation drone is typed as a click and not as the drone.
    """
    seg = np.asarray(logspec[r0:r1 + 1], np.float64).mean(0)
    ref = np.asarray(bg[r0:r1 + 1], np.float64).mean(0)
    return np.maximum(seg - ref, 0.0)


def object_mass(excess: np.ndarray, bands_per_octave: float = 10.0) -> tuple:
    """Schaeffer mass from an object's excess spectrum, with its evidence.

    ``excess`` is linear power per log-frequency band, background already
    removed (:func:`_excess_spectrum`). Two quantities are read off it, both
    deliberately blind to the object's overall spectral tilt — brightness is
    not mass, and a bright hiss must not be mistaken for a high note:

    - **peak share** — the fraction of the object's energy sitting in *narrow
      spectral peaks*. A band counts as a peak when it stands
      ``PEAK_PROMINENCE_DB`` above the running median of its own octave. This
      is peakiness and harmonicity in one number: a single partial counts, and
      so does every member of a harmonic series, while a continuum of any
      shape counts for nothing;
    - **spread** — the energy-weighted standard deviation of log2 frequency,
      in octaves. A pitch has almost none; a knock has a fraction of an
      octave; hiss, rush and rustle spread across the spectrum whatever their
      tilt.

    The rules, applied in order:

    1. peak share >= 0.50 — most of what appeared is in peaks, so a pitch (or
       a harmonic series) is what one hears: **tonic**;
    2. peak share >= 0.20 — a pitch is audible over a continuum that carries
       most of the energy: **tonic-complex**;
    3. spread >= 1.2 octaves — no pitch, and the energy is spread wide enough
       to hear as a band of noise rather than as a body: **noise**;
    4. otherwise — energy in one or a few narrow regions, none of them a
       pitch: **complex**.

    The evidence also carries the peak's **prominence** in dB and a
    **flatness** reading (the Wiener entropy of the excess spectrum, floored
    40 dB below its peak: 0 for a single band, 1 for a perfectly even
    spectrum). Neither enters a rule; both are worth having beside the others
    when listening back.

    Returns ``(mass, evidence)``; ``(None, evidence)`` when nothing rose above
    the background, in which case there is no object spectrum to type.
    """
    from scipy.ndimage import median_filter
    e = np.asarray(excess, np.float64)
    tot = float(e.sum())
    ev = {"peak_share": 0.0, "spread_oct": 0.0, "prominence_db": 0.0,
          "flatness": 0.0}
    if tot <= 0 or not np.isfinite(tot):
        return None, ev
    peak = float(e.max())
    floored = np.maximum(e, peak * 1e-4)
    lvl = 10 * np.log10(floored)
    win = max(3, int(round(bands_per_octave)) | 1)
    res = lvl - median_filter(lvl, size=win, mode="nearest")
    is_peak = res >= PEAK_PROMINENCE_DB
    share = float(e[is_peak].sum() / tot)
    u = np.arange(len(e)) / bands_per_octave        # position in octaves
    w = e / tot
    mu = float((w * u).sum())
    spread = float(np.sqrt(max((w * (u - mu) ** 2).sum(), 0.0)))
    flat = float(np.exp(np.log(floored).mean()) / (floored.mean() + EPS))
    ev = {"peak_share": round(share, 3), "spread_oct": round(spread, 2),
          "prominence_db": round(float(res.max()), 1),
          "flatness": round(flat, 3)}
    if share >= PEAK_SHARE_TONIC:
        return "tonic", ev
    if share >= PEAK_SHARE_COMPLEX:
        return "tonic-complex", ev
    if spread >= SPREAD_OCT_NOISE:
        return "noise", ev
    return "complex", ev


def _iteration_strength(env: np.ndarray, dt: float) -> tuple:
    """Normalised envelope autocorrelation at its best iteration lag.

    Returns ``(strength, rate_hz)``. The envelope is mean-removed and its
    autocorrelation searched for a *local* maximum at lags between
    1/``ITER_HI_HZ`` and 1/``ITER_LO_HZ``; a single attack-and-decay has a
    monotonically falling autocorrelation and so scores nothing, while a
    rattle, roll or grain scores at its repetition period.
    """
    x = np.asarray(env, np.float64)
    if len(x) < 8:
        return 0.0, None
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return 0.0, None
    r = np.correlate(x, x, "full")[len(x) - 1:] / denom
    lo = max(1, int(round(1.0 / (ITER_HI_HZ * dt))))
    hi = min(len(r) - 2, int(round(1.0 / (ITER_LO_HZ * dt))))
    best, lag = 0.0, None
    for i in range(max(lo, 1), max(hi + 1, 1)):
        if i + 1 < len(r) and r[i] > r[i - 1] and r[i] >= r[i + 1] \
                and r[i] > best:
            best, lag = float(r[i]), i
    return best, (1.0 / (lag * dt) if lag else None)


def object_facture(env: np.ndarray, dt: float, dur: float) -> tuple:
    """Schaeffer facture from an object's amplitude envelope, with evidence.

    ``env`` is the object's own amplitude envelope (linear, one value every
    ``dt`` seconds; the 20 ms broadband envelope of the feature cache when it
    is available) and ``dur`` the object's duration in seconds. Two things are
    read off it: the **attack time**, the conventional 10-to-90 per cent rise
    of the envelope towards its peak, and the **iteration strength**, the
    normalised envelope autocorrelation at its best repetition lag between
    ``ITER_LO_HZ`` and ``ITER_HI_HZ`` (see :func:`_iteration_strength`).

    The rules, applied in order:

    1. duration >= 5 s — the sustainment outlasts what attention holds whole;
       the object has no audible end within its own present. This is
       Schaeffer's excentric case: **sustained (unlimited)**;
    2. iteration strength >= 0.35 over at least 0.4 s — energy is maintained by
       repetition, not continuously: **iteration**;
    3. attack <= 0.08 s and duration <= 1 s — all the energy arrives at once
       and nothing maintains it: **impulse**;
    4. otherwise — energy held continuously between a beginning and an end:
       **sustained (delimited)**.

    Returns ``(facture, evidence)``.
    """
    x = np.asarray(env, np.float64)
    ev = {"attack_s": None, "iter_strength": 0.0, "iter_rate_hz": None,
          "dur_s": round(float(dur), 2)}
    if dur >= SUSTAIN_MAX_S:
        return "unlimited", ev
    if len(x):
        top = float(x.max())
        hi = np.flatnonzero(x >= 0.9 * top)
        end = int(hi[0]) if len(hi) else int(np.argmax(x))
        below = np.flatnonzero(x[:end + 1] <= 0.1 * top)
        start = int(below[-1]) if len(below) else 0
        ev["attack_s"] = round((end - start) * dt, 3)
    strength, rate = _iteration_strength(x, dt)
    ev["iter_strength"] = round(strength, 3)
    ev["iter_rate_hz"] = round(rate, 1) if rate else None
    if dur >= ITER_MIN_S and strength >= ITER_ACF:
        return "iteration", ev
    if ev["attack_s"] is not None and ev["attack_s"] <= ATTACK_MAX_S \
            and dur <= IMPULSE_MAX_S:
        return "impulse", ev
    return "sustained", ev


def _row_index(t_axis: np.ndarray, t: float, n: int) -> int:
    """Index of the frame that contains time ``t`` (frames start at their own
    time, so the search is right-sided: an onset exactly on a frame boundary
    belongs to the frame it opens, not to the one before)."""
    return int(np.clip(np.searchsorted(t_axis, t, side="right") - 1, 0, n - 1))


def _band_background(logspec: np.ndarray, win_s: float = 300.0,
                     pct: float = 10.0) -> np.ndarray:
    """Running low-percentile background per band, on coarse blocks.

    The same quantity as :func:`ambiscape.background.band_background`, taken
    per ``win_s`` block of rows instead of per row: a sliding percentile over a
    whole day of 96 bands costs minutes, a blockwise one costs milliseconds,
    and the background it estimates is by construction slow.
    """
    x = np.asarray(logspec, np.float64)
    n, nb = x.shape
    step = max(1, int(round(win_s)))
    nblk = max(1, -(-n // step))
    bgb = np.empty((nblk, nb))
    for i in range(nblk):
        bgb[i] = np.percentile(x[i * step:(i + 1) * step], pct, axis=0)
    return np.repeat(bgb, step, axis=0)[:n]


def extract_objects(F: dict, min_dur: float = OBJECT_MIN_S,
                    max_dur: float = OBJECT_MAX_S,
                    thresh_db: float = 8.0) -> dict:
    """Sound objects of a session, typed on Schaeffer's two axes.

    Runs :func:`ambiscape.analysis.detect_events` on the cached fast level,
    keeps the events whose duration lies in ``[min_dur, max_dur]`` — the
    perceptual window in which something can be held whole in attention — and
    types each survivor with :func:`object_mass` and :func:`object_facture`.

    Returns ``{"objects": [...], "n_detected", "n_short", "n_long",
    "min_dur_s", "max_dur_s"}``, so the events that were *not* objects stay
    visible and countable. Each object is an annotation-shaped dict —
    ``mass``, ``facture``, ``kind`` (always ``"figure"``: what a sound object
    does in the place is a separate, human judgement), a one-element ``spans``
    on the session clock, its level and exceedance, ``_auto`` set, and the
    numbers behind both typings under ``_schaeffer`` — so it drops straight
    into the taxonomy figures.
    """
    tf = np.asarray(F["t_fast"], float)
    fast = np.asarray(F["fast_db"], float)
    dt = float(np.median(np.diff(tf))) if len(tf) > 1 else 0.125
    events, _bg = detect_events(fast, dt, thresh_db=thresh_db)
    out = {"objects": [], "n_detected": len(events), "n_short": 0,
           "n_long": 0, "min_dur_s": min_dur, "max_dur_s": max_dur}
    if not events:
        return out

    logspec = np.asarray(F["logspec"], np.float64) if "logspec" in F else None
    t_sec = np.asarray(F["t"], float) if "t" in F else None
    bg_spec = _band_background(logspec) if logspec is not None else None
    from .features import LOGF_RANGE
    nband = logspec.shape[1] if logspec is not None else 0
    bpo = nband / np.log2(LOGF_RANGE[1] / LOGF_RANGE[0]) if nband else 10.0

    has_hi = "env_hi" in F and "t_hi" in F
    if has_hi:
        t_hi = np.asarray(F["t_hi"], float)
        env_hi = np.sqrt(np.maximum(np.asarray(F["env_hi"], np.float64), 0.0))
        hi_dt = float(F.get("hi_dt", 0.02))

    for e in events:
        t0, t1 = float(tf[e["i0"]]), float(tf[e["i1"]]) + dt
        dur = t1 - t0
        if dur < min_dur:
            out["n_short"] += 1
            continue
        if dur > max_dur:
            out["n_long"] += 1
            continue
        mass, mass_ev = None, {}
        if logspec is not None and t_sec is not None:
            r0 = _row_index(t_sec, t0, len(logspec))
            r1 = max(r0, _row_index(t_sec, t1 - 1e-3, len(logspec)))
            mass, mass_ev = object_mass(
                _excess_spectrum(logspec, bg_spec, r0, r1), bpo)
        if has_hi:
            # a little pre-roll, so the attack is measured from the silence
            # before the object rather than from the detector's threshold
            pre = int(round(0.1 / hi_dt))
            j0 = max(0, _row_index(t_hi, t0, len(env_hi)) - pre)
            j1 = max(j0 + 1, _row_index(t_hi, t1 - 1e-3, len(env_hi)))
            env, edt = env_hi[j0:j1 + 1], hi_dt
        else:                       # pre-0.2 cache: the 8 Hz fast level only
            env = 10 ** (fast[e["i0"]:e["i1"] + 1] / 20)
            edt = dt
        facture, fac_ev = object_facture(env, edt, dur)
        out["objects"].append({
            "name": f"object at {_clock(t0)}",
            "kind": "figure", "mass": mass, "facture": facture,
            "spans": [[t0, t1]],
            "_auto": True, "_object": True,
            "_level_dbfs": round(float(fast[e["ipk"]]), 1),
            "_exceed_db": round(float(e["exceed"]), 1),
            "_schaeffer": {**mass_ev, **fac_ev},
        })
    return out


def _clock(t: float) -> str:
    day, s = int(t // 86400), int(t % 86400)
    hms = f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
    return f"{day} {hms}" if day else hms


def cell_counts(objects: list) -> dict:
    """``{(facture, mass): n}`` over typed objects — the map's full census."""
    out: dict[tuple, int] = {}
    for o in objects:
        key = (o.get("facture"), o.get("mass"))
        if None in key:
            continue
        out[key] = out.get(key, 0) + 1
    return out
