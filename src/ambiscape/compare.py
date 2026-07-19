"""Cross-session comparison: the same place on different days.

The catalog answers "how do my *places* differ"; this module answers "how
did *this place* differ between visits" — two or more analyzed sessions of
one room laid onto a common clock so that machines, weather, parties and
silences can be read against each other. Everything works from the cached
1 Hz features and ``summary.json``/``states.json`` of a prior ``analyze``
run; no audio is reopened.

- ``load_comparison`` --- features + summary + states for each session;
- ``laeq_timeline`` --- per-minute LAeq with NaN gaps between takes;
- ``clock_rows`` --- group sessions that share (or bridge) calendar days
  onto common clock-aligned rows;
- ``timeline_figure`` --- LAeq rows on a shared hour-of-day axis, detected
  states shaded;
- ``ltas_by_state`` / ``ltas_figure`` --- median band spectra split by each
  session's detected states, overlaid;
- ``line_prominence`` --- how strongly a known tonal line (a machine
  fingerprint) stands out of the per-minute minimum spectrum;
- ``band_level`` / ``band_timeline_figure`` --- one frequency band on the
  clock axis across sessions (dawn chorus, rain hiss, party bass);
- ``azimuth_rose_figure`` --- foreground energy by azimuth, side by side
  (mic frames differ between visits: compare shapes, not directions);
- ``floor_difference`` --- median minimum-spectrum difference between two
  time windows: the detector for near-floor sources (a quiet fan's shelf);
- ``duty_cycle`` --- period, duty and regularity of a cycling source (a
  fridge) from a band-level autocorrelation;
- ``run_compare`` --- orchestrate the above into figures + compare.json.

Times follow the feature axis: seconds since midnight of each session's
first calendar day (so hour 28.5 is 04:30 on day 2). The motivating corpus
is the Haarlem loft: the same room four days apart swapped a loud
ventilation drone for rain, a Saturday-night party, and its quietest
recorded floor.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import load_features
from .longitudinal import parse_date

EPS = 1e-12

# ---------------------------------------------------------------- loading


def load_comparison(folders: list[str | Path]) -> list[dict]:
    """Features + summary + states per analyzed session.

    Each entry: ``{"name", "folder", "date" (day-0 date or None),
    "F" (features), "summary", "states" (or None)}``. Sessions without a
    feature cache raise — run ``ambiscape analyze`` first.
    """
    out = []
    for folder in folders:
        folder = Path(folder)
        adir = folder / "analysis"
        paths = sorted((adir / "features").glob("*.npz"))
        if not paths:
            raise FileNotFoundError(
                f"no cached features in {adir} — run 'ambiscape analyze'")
        summary = {}
        sp = adir / "summary.json"
        if sp.exists():
            summary = json.loads(sp.read_text())
        states = None
        stp = adir / "states.json"
        if stp.exists():
            states = json.loads(stp.read_text()).get("states")
        date = parse_date(str(summary.get("date") or folder.name))
        out.append({"name": folder.name, "folder": folder, "date": date,
                    "F": load_features(paths), "summary": summary,
                    "states": states})
    return out


def in_intervals(t: np.ndarray, intervals) -> np.ndarray:
    """Boolean mask of ``t`` covered by ``[a, b)`` interval pairs."""
    m = np.zeros(len(t), bool)
    for a, b in intervals:
        m |= (t >= a) & (t < b)
    return m


def state_mask(sess: dict, state: str, min_s: float = 0.0) -> np.ndarray | None:
    """1 Hz mask for one named state; None if the session has no states."""
    if not sess["states"] or state not in sess["states"]:
        return None
    ivs = [iv for iv in sess["states"][state]["intervals_s"]
           if iv[1] - iv[0] >= min_s]
    return in_intervals(sess["F"]["t"], ivs)


# ---------------------------------------------------------------- timelines


def laeq_timeline(F: dict, bin_s: float = 60.0):
    """(bin centers s, LAeq per bin) from the fast A-weighted track.

    Bins with under 75 % coverage (the gaps between takes) are NaN so a
    plotted line breaks instead of bridging silence.
    """
    tf, x = F["t_fast"], F["fast_dba"]
    b0 = np.floor(tf[0] / bin_s)
    idx = (np.floor(tf / bin_s) - b0).astype(int)
    p = np.zeros(idx.max() + 1)
    n = np.zeros(idx.max() + 1)
    np.add.at(p, idx, 10 ** (x / 10))
    np.add.at(n, idx, 1)
    full = np.median(n[n > 0])
    la = 10 * np.log10(p / np.maximum(n, 1) + EPS)
    la[n < 0.75 * full] = np.nan
    t = (b0 + np.arange(len(p))) * bin_s + bin_s / 2
    return t, la


def clock_rows(sessions: list[dict]) -> list[list[tuple[int, float]]]:
    """Group sessions onto clock-aligned rows: ``[[(index, shift_h), ...]]``.

    Sessions whose calendar spans touch or overlap share a row; a session's
    shift is 24 h per day its day-0 lies after the row's reference day, so
    a night session (day 0) and the following day session (day 1) line up
    end to end. Sessions without a resolvable date each get their own row.
    """
    spans = []
    for i, s in enumerate(sessions):
        t = s["F"]["t"]
        if s["date"] is None:
            spans.append((None, None, i))
            continue
        o = s["date"].toordinal()
        spans.append((o + t[0] / 86400, o + t[-1] / 86400, i))
    rows, used = [], set()
    for a0, a1, i in sorted(spans, key=lambda x: (x[0] is None, x[0])):
        if i in used:
            continue
        if a0 is None:
            rows.append([(i, 0.0)])
            used.add(i)
            continue
        ref = sessions[i]["date"].toordinal()
        row, hi = [(i, 0.0)], a1
        used.add(i)
        for b0, b1, j in sorted(spans, key=lambda x: (x[0] is None, x[0])):
            if j in used or b0 is None:
                continue
            if b0 <= hi + 0.5:              # touches within 12 h: same row
                row.append((j, 24.0 * (sessions[j]["date"].toordinal() - ref)))
                hi = max(hi, b1)
                used.add(j)
        rows.append(row)
    return rows


def timeline_figure(sessions: list[dict], out_path: str | Path,
                    state: str = "machine_on", x0_hour: float | None = None,
                    colors=None):
    """Clock-aligned LAeq rows; intervals of ``state`` shaded per session."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = colors or _COLORS
    rows = clock_rows(sessions)
    lo = min(sessions[i]["F"]["t"][0] / 3600 + sh
             for row in rows for i, sh in row)
    hi = max(sessions[i]["F"]["t"][-1] / 3600 + sh
             for row in rows for i, sh in row)
    x0 = np.floor(lo) if x0_hour is None else x0_hour
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 3.2 * len(rows)),
                             sharey=True, squeeze=False)
    for ax, row in zip(axes[:, 0], rows):
        for i, shift in row:
            s = sessions[i]
            c = colors[i % len(colors)]
            t, la = laeq_timeline(s["F"])
            ax.plot(t / 3600 + shift - x0, la, color=c, lw=0.8,
                    label=s["name"])
            if s["states"] and state in s["states"]:
                for a, b in s["states"][state]["intervals_s"]:
                    if b - a > 300:
                        ax.axvspan(a / 3600 + shift - x0,
                                   b / 3600 + shift - x0,
                                   color=c, alpha=0.12, lw=0)
        ax.set_ylabel("LAeq 1-min (dBFS)")
        ax.set_xlim(0, np.ceil(hi) - x0)
        ticks = np.arange(0, np.ceil(hi) - x0 + 1, 2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int((x0 + h) % 24):02d}" for h in ticks])
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel(f"clock (h); shaded = {state}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------- spectra


def band_centers(F: dict) -> np.ndarray:
    """Geometric center frequencies of the log-spaced spectrum bands.

    ``logf`` holds the band *edges* (one more than the ``logspec`` width);
    the centers are the geometric means of consecutive edges.
    """
    logf = F["logf"]
    if len(logf) == F["logspec"].shape[1] + 1:
        return np.sqrt(logf[:-1] * logf[1:])
    return logf


def ltas_by_state(sess: dict, min_state_s: float = 0.0) -> dict:
    """Median band spectra (dB) for the whole session and each state."""
    ls = sess["F"]["logspec"]
    out = {"all": 10 * np.log10(np.median(ls, 0) + EPS)}
    for name in (sess["states"] or {}):
        m = state_mask(sess, name, min_state_s)
        if m is not None and m.any():
            out[name] = 10 * np.log10(np.median(ls[m], 0) + EPS)
    return out


def ltas_figure(sessions: list[dict], out_path: str | Path,
                min_state_s: float = 0.0, fmin: float = 90.0, colors=None):
    """Overlay per-state median spectra of every session on one axis.

    Sessions keep their color; the whole-session curve is drawn only when a
    session has no states, and states are distinguished by line style.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = colors or _COLORS
    styles = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, s in enumerate(sessions):
        fc = band_centers(s["F"])
        sel = fc >= fmin
        curves = ltas_by_state(s, min_state_s)
        names = [n for n in curves if n != "all"] or ["all"]
        for j, name in enumerate(names):
            label = s["name"] if name == "all" else f"{s['name']}: {name}"
            ax.plot(fc[sel], curves[name][sel], styles[j % len(styles)],
                    color=colors[i % len(colors)], lw=1.6, label=label)
    ax.set_xscale("log")
    ax.set_xlim(fmin, fc[-1])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Median band level (dBFS)")
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def line_prominence(sess: dict, freqs_hz, mask_min=None,
                    halfwidth_hz: float = 15.0,
                    bg_halfwidth_hz: float = 40.0) -> dict:
    """Prominence of known tonal lines in the per-minute minimum spectrum.

    For each target frequency: the peak within ±``halfwidth_hz`` against the
    median of the surrounding ±``bg_halfwidth_hz`` ring, on the median
    min-spectrum (optionally restricted to ``mask_min`` minutes). A machine
    fingerprint that survived the night keeps several dB of prominence; a
    machine that never ran leaves ≲ 1 dB. Returns
    ``{f0: {"peak_hz", "prominence_db"}}``.
    """
    freqs = sess["F"]["freqs"]
    ms = sess["F"]["minspec"]
    if mask_min is not None:
        ms = ms[mask_min]
    spec = 10 * np.log10(np.median(ms, 0) + EPS)
    out = {}
    for f0 in freqs_hz:
        w = (freqs > f0 - halfwidth_hz) & (freqs < f0 + halfwidth_hz)
        bg = ((freqs > f0 - bg_halfwidth_hz) & (freqs < f0 + bg_halfwidth_hz)
              & ~w)
        out[f0] = {
            "peak_hz": round(float(freqs[w][np.argmax(spec[w])]), 1),
            "prominence_db": round(float(spec[w].max() - np.median(spec[bg])),
                                   1)}
    return out


def floor_difference(sess_a: dict, hours_a, sess_b: dict, hours_b):
    """Median minimum-spectrum difference between two time windows (dB).

    ``hours_*`` are ``(h0, h1)`` on each session's own clock axis (h > 24 =
    day 2). This is the near-floor detector: a source too quiet for a level
    step — a ventilation fan on its low setting — still shows as a
    band-limited shelf of the A-minus-B difference. Returns
    ``(freqs, diff_db)``.
    """
    def med(sess, hours):
        mt = sess["F"]["min_t"]
        m = (mt >= hours[0] * 3600) & (mt < hours[1] * 3600)
        if not m.any():
            raise ValueError(f"no minutes in {hours} for {sess['name']}")
        return 10 * np.log10(np.median(sess["F"]["minspec"][m], 0) + EPS)
    return sess_a["F"]["freqs"], med(sess_a, hours_a) - med(sess_b, hours_b)


# ---------------------------------------------------------------- bands


def band_level(F: dict, f0: float, f1: float) -> np.ndarray:
    """1 Hz level (dB) of the summed log-spectrum bands inside [f0, f1]."""
    fc = band_centers(F)
    sel = (fc >= f0) & (fc < f1)
    return 10 * np.log10(F["logspec"][:, sel].sum(1) + EPS)


def band_timeline_figure(sessions: list[dict], out_path: str | Path,
                         f0: float, f1: float, hours=None,
                         smooth_s: int = 301, colors=None):
    """One frequency band on the clock axis across sessions.

    The band picks the phenomenon: 2–8 kHz for dawn chorus or rain hiss,
    100–300 Hz for party bass. ``hours=(h0, h1)`` restricts the clock
    window (h > 24 = day 2); ``smooth_s`` is a running-median width.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import median_filter
    colors = colors or _COLORS
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, s in enumerate(sessions):
        t = s["F"]["t"] / 3600
        e = band_level(s["F"], f0, f1)
        m = np.ones(len(t), bool) if hours is None else \
            (t >= hours[0]) & (t <= hours[1])
        if not m.any():
            continue
        ax.plot(t[m], median_filter(e[m], min(smooth_s, int(m.sum()))),
                color=colors[i % len(colors)], lw=1.4, label=s["name"])
    ax.set_xlabel("clock (h; > 24 = day 2)")
    ax.set_ylabel(f"{f0:.0f}-{f1:.0f} Hz level (dB, running median)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------- space


def azimuth_rose_figure(sessions: list[dict], out_path: str | Path,
                        fg_quantile: float = 0.75, nbins: int = 36,
                        colors=None):
    """Foreground energy by azimuth, one polar panel per session.

    Foreground = seconds in the top ``1 - fg_quantile`` of W energy. Mic
    frames usually differ between visits: compare the *shapes* (one machine
    lobe vs energy from everywhere), not absolute directions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = colors or _COLORS
    n = len(sessions)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4),
                             subplot_kw={"projection": "polar"},
                             squeeze=False)
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    for i, (ax, s) in enumerate(zip(axes[0], sessions)):
        w = s["F"]["rms_w"] ** 2
        sel = w >= np.quantile(w, fg_quantile)
        hist, _ = np.histogram(np.deg2rad(s["F"]["az"][sel]), bins=bins,
                               weights=w[sel])
        hist /= hist.max() + EPS
        ax.bar((bins[:-1] + bins[1:]) / 2, hist, width=np.diff(bins),
               color=colors[i % len(colors)], alpha=0.85, lw=0)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(s["name"], fontsize=9)
        ax.set_yticklabels([])
    fig.suptitle("Foreground energy by azimuth (mic frame)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


# ---------------------------------------------------------------- cycles


def duty_cycle(F: dict, t0: float, t1: float, f0: float = 63.0,
               f1: float = 500.0, period_range=(300.0, 5400.0),
               smooth_s: int = 61) -> dict:
    """Period, duty and regularity of a cycling source in one band.

    Threshold the smoothed band level midway between its 10th and 90th
    percentiles, autocorrelate the on/off square wave, and report the
    strongest period inside ``period_range``. An ``acf_peak`` above ~0.2
    marks a real cycler (a fridge); below ~0.1 there is no legible cycle —
    which is itself a finding when the same appliance was legible from
    another mic position.
    """
    from scipy.ndimage import median_filter
    t = F["t"]
    m = (t >= t0) & (t < t1)
    e = band_level(F, f0, f1)[m]
    if len(e) < 4 * period_range[0]:
        raise ValueError("window too short for the requested period range")
    sm = median_filter(e, smooth_s)
    q10, q90 = np.quantile(sm, 0.1), np.quantile(sm, 0.9)
    on = sm > (q10 + q90) / 2
    x = on.astype(float) - on.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0] + EPS
    lo, hi = int(period_range[0]), min(int(period_range[1]), len(ac) - 1)
    per = int(np.argmax(ac[lo:hi]) + lo)
    return {"period_min": round(per / 60, 1),
            "duty_pct": round(100 * float(on.mean()), 1),
            "acf_peak": round(float(ac[per]), 2),
            "swing_db": round(float(q90 - q10), 1)}


# ---------------------------------------------------------------- runner

_COLORS = ["#2a78d6", "#008300", "#eda100", "#e87ba4", "#1baf7a", "#eb6834"]

_TABLE_KEYS = ["duration_min", "leq_dbfs", "laeq_dbfs", "L10", "L50", "L90",
               "events_per_min", "intermittency_ratio_pct", "emergence_db",
               "centroid_median_hz", "flatness_median", "diffuseness_median",
               "azimuth_R", "directional_entropy", "aci", "ndsi",
               "bird_band_activity_pct"]


def run_compare(folders: list[str | Path], out_dir: str | Path,
                lines=None, band=None, hours=None,
                state: str = "machine_on") -> dict:
    """Compare analyzed sessions of one place; write figures + compare.json.

    Always: clock-aligned LAeq timelines, per-state LTAS overlay, azimuth
    roses, and a pooled + state-resolved descriptor table. Optional:
    ``lines`` (tonal-line prominence per session, e.g. a machine
    fingerprint) and ``band``/``hours`` (a band timeline, e.g. dawn
    chorus). Returns the compare.json document.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = load_comparison(folders)
    doc = {"sessions": [s["name"] for s in sessions],
           "pooled": {s["name"]: {k: s["summary"].get(k)
                                  for k in _TABLE_KEYS}
                      for s in sessions},
           "states": {s["name"]: {n: {k: v[k] for k in _TABLE_KEYS
                                      if k in v}
                                  for n, v in (s["states"] or {}).items()}
                      for s in sessions},
           "figures": {}}
    doc["figures"]["timelines"] = str(
        timeline_figure(sessions, out_dir / "compare_timelines.png",
                        state=state))
    doc["figures"]["ltas"] = str(
        ltas_figure(sessions, out_dir / "compare_ltas.png"))
    doc["figures"]["roses"] = str(
        azimuth_rose_figure(sessions, out_dir / "compare_roses.png"))
    if lines:
        doc["line_prominence"] = {}
        for s in sessions:
            m = None
            if s["states"] and state in s["states"]:
                m = in_intervals(s["F"]["min_t"],
                                 s["states"][state]["intervals_s"])
                m = m if m.any() else None
            doc["line_prominence"][s["name"]] = {
                str(f0): v for f0, v in
                line_prominence(s, lines, mask_min=m).items()}
    if band:
        doc["figures"]["band"] = str(
            band_timeline_figure(sessions, out_dir / "compare_band.png",
                                 band[0], band[1], hours=hours))
    (out_dir / "compare.json").write_text(json.dumps(doc, indent=1))
    return doc
