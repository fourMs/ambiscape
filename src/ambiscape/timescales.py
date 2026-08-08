"""Timescales: what each descriptor needs, and what rooms actually do.

A descriptor computed on too little audio is not a noisy descriptor, it is
a different quantity or none at all. The complexity index averages over
300 s chunks, so on a 30 s clip it has no chunk to average and returns
zero --- a confident-looking number that no measurement produced. Tonal
prominence aggregates per-minute spectra and has nothing to aggregate
below a minute. A median level exists at any length and moves by tens of
decibels until there is a minute of it.

This module is the one place those windows are written down. Everything
else reads from here: the guard in :func:`ambiscape.resolve.full_summary`,
the ``ambiscape timescales`` command and its figure, the validity section
of a deposit report, and the book's own figure. Writing the numbers twice
is how they drift.

Provenance is a field. Some windows were measured on this project's own
material and some are honest judgement; the difference matters to a reader
deciding how much weight to put on a bound, so ``source`` records which,
and the figure draws them differently.

The bands
---------
``micro``, ``meso`` and ``macro`` are joint spatial and temporal
categories, following the framework of *Sound Actions* (Jensenius 2022),
where mesomotion is defined at 1--100 cm and 0.5--5 s, the short-term
memory range, with macro and micro as everything longer/larger and
shorter/smaller. The pairing is deliberate: motion, and sound, cannot
sensibly be studied in space alone or in time alone.

Applied to rooms rather than bodies, the temporal extents carry over
unchanged, because they are anchored in perception --- the meso band is
Schaeffer's sound object, Godoy's chunking range and short-term memory at
once. The spatial extents move outward, and the spatial meso becomes the
*zone*: the kitchen zone, the dining-table zone, the sofa zone of one
open-plan room, experienced as different places and each with its own
soundscape.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Bands, as joint spatio-temporal categories (see the module docstring).
BANDS = {
    "micro": {
        "t_lo": 0.0, "t_hi": 0.5,
        "space_body": "< 1 cm", "space_room": "the ear, the head",
        "memory": "echoic / sensory",
    },
    "meso": {
        "t_lo": 0.5, "t_hi": 5.0,
        "space_body": "1-100 cm", "space_room": "the zone",
        "memory": "short-term / working",
    },
    "macro": {
        "t_lo": 5.0, "t_hi": float("inf"),
        "space_body": "> 1 m", "space_room": "the room, the building",
        "memory": "long-term",
    },
}


def band_of(t_s: float) -> str:
    """Which band a duration falls in."""
    for name, b in BANDS.items():
        if b["t_lo"] <= t_s < b["t_hi"]:
            return name
    return "macro"


@dataclass(frozen=True)
class Window:
    """The observation window one descriptor needs.

    ``kind`` is ``hard`` when the quantity does not exist below ``min_s``
    (no complete chunk, no minute to aggregate, no frame) and ``soft``
    when it exists but is not yet stable. ``source`` is ``measured`` when
    a number in this project established the bound and ``asserted`` when
    it is reasoning rather than measurement.
    """
    key: str
    min_s: float
    kind: str            # "hard" | "soft"
    why: str
    source: str          # "measured" | "asserted"
    ref: str = ""

    def __post_init__(self):
        if self.kind not in ("hard", "soft"):
            raise ValueError(f"{self.key}: kind must be hard or soft")
        if self.source not in ("measured", "asserted"):
            raise ValueError(f"{self.key}: source must be measured or asserted")


#: Descriptor windows. Keys are summary keys as emitted by
#: :func:`ambiscape.resolve.full_summary`.
WINDOWS: dict[str, Window] = {w.key: w for w in [
    Window("aci", 300.0, "hard",
           "averages over 300 s chunks; a shorter session has no complete "
           "chunk and the index is undefined",
           "measured", "dataset benchmark report, ESC-50 length study"),
    Window("tonal_prominence_db", 60.0, "hard",
           "aggregates per-minute spectra and requires a detection in a "
           "proportion of minutes",
           "measured", "dataset benchmark report"),
    Window("tonal_prominence_hz", 60.0, "hard",
           "companion to tonal_prominence_db", "measured",
           "dataset benchmark report"),
    Window("n_prominent_tones", 60.0, "hard",
           "companion to tonal_prominence_db", "measured",
           "dataset benchmark report"),
    Window("L50", 60.0, "soft",
           "the median level moved 38 dB between a 5 s and a 60 s session "
           "of identical composition",
           "measured", "dataset benchmark report"),
    Window("L90", 60.0, "soft",
           "a background percentile needs a quiet minute to find",
           "measured", "dataset benchmark report"),
    Window("L10", 60.0, "soft",
           "companion percentile", "measured", "dataset benchmark report"),
    Window("dynamics_L10_L90", 60.0, "soft",
           "a difference of two percentiles, no more stable than either",
           "measured", "dataset benchmark report"),
    Window("fg_fraction_median", 60.0, "soft",
           "foreground fraction was unstable below a minute",
           "measured", "dataset benchmark report"),
    Window("fg_fraction_p90", 60.0, "soft",
           "companion to fg_fraction_median", "measured",
           "dataset benchmark report"),
    Window("events_per_min", 60.0, "soft",
           "an event rate from a shorter window extrapolates from a "
           "handful of events, or from one",
           "measured", "dataset benchmark report"),
    Window("intermittency_ratio_pct", 60.0, "soft",
           "converges from above; needs a minute to settle",
           "measured", "dataset benchmark report"),
    Window("mech_periodicity_hz", 10800.0, "soft",
           "a periodicity estimate needs many cycles; it had not converged "
           "at six minutes, and domestic duty cycles run to tens of "
           "minutes, so hours are required rather than demonstrated",
           "asserted", "dataset benchmark report; SINS nocturnal search"),
    Window("mech_periodicity_strength", 10800.0, "soft",
           "companion to mech_periodicity_hz", "asserted",
           "dataset benchmark report"),

    # Meso-band descriptors: defined on a single sound object, so they need
    # an object rather than a minute. These are what makes the figure's
    # second row reach left of a minute at all -- before them the toolbox
    # could say nothing whatever about a six-second sound action.
    Window("attack_s", 0.2, "hard",
           "the 10-90% rise of one object's envelope; there is no attack "
           "without an object",
           "asserted", "objects.object_profile"),
    Window("decay_s", 0.2, "hard",
           "the fall from the peak back through a tenth of it",
           "asserted", "objects.object_profile"),
    Window("temporal_centroid", 0.2, "hard",
           "where the energy sits along one object, 0 at its start and 1 at "
           "its end",
           "asserted", "objects.object_profile"),
    Window("crest_db", 0.2, "soft",
           "peak over RMS; meaningful on an object, unstable on a fragment "
           "of one",
           "asserted", "objects.object_profile"),
    Window("iteration_hz", 0.2, "soft",
           "the envelope's best repetition rate needs a few cycles inside "
           "the object to be trusted",
           "asserted", "objects.object_profile"),
    Window("iteration_strength", 0.2, "soft",
           "companion to iteration_hz", "asserted",
           "objects.object_profile"),

    # Ecoacoustic indices. The length study ran to 360 s and these had not
    # settled by it, so the bound is where measurement stopped rather than
    # where the index becomes stable -- asserted, and deliberately not
    # dressed up as measured.
    Window("ndsi", 360.0, "soft",
           "did not converge across 5-360 s; the bound is where the length "
           "study stopped, not where the index settles",
           "asserted", "dataset benchmark report"),
    Window("bi", 360.0, "soft",
           "fell monotonically with length across 5-360 s",
           "asserted", "dataset benchmark report"),
    Window("aei", 360.0, "soft",
           "drifted across 5-360 s without settling",
           "asserted", "dataset benchmark report"),
    Window("fluctuation_index", 360.0, "soft",
           "moved by a factor of seven across 5-360 s",
           "asserted", "dataset benchmark report"),

    # Per-minute rates: computable below a minute only by extrapolation.
    Window("spectral_events_per_min", 60.0, "soft",
           "a per-minute rate extrapolated from less than a minute",
           "asserted", ""),
    Window("spectral_event_median_dur_s", 30.0, "soft",
           "stable from about 30 s in the length study",
           "measured", "dataset benchmark report"),
    Window("bird_peaks_per_min", 60.0, "soft",
           "a per-minute rate extrapolated from less than a minute",
           "asserted", ""),
    Window("bird_event_rate_per_min", 60.0, "soft",
           "a per-minute rate extrapolated from less than a minute",
           "asserted", ""),
    Window("bird_active_minute_fraction", 60.0, "hard",
           "a fraction of active minutes has no minute to count below one",
           "asserted", ""),
    Window("bird_band_activity_pct", 60.0, "soft",
           "rose monotonically with length across the length study",
           "measured", "dataset benchmark report"),
    Window("acoustic_entropy", 60.0, "soft",
           "rose from 0.085 to 0.763 between a 5 s and a 60 s session of "
           "identical composition",
           "measured", "dataset benchmark report"),
    Window("anthro_activity_fraction", 60.0, "soft",
           "rose monotonically with length across the length study",
           "measured", "dataset benchmark report"),
    Window("bird_temporal_entropy", 60.0, "soft",
           "a temporal distribution over too few frames; it also inverts on "
           "stationary noise, which no window fixes",
           "measured", "indoor ecoacoustic-indices article"),
]}

#: Summary keys deliberately without a window, with the reason. A key that
#: is neither here nor in WINDOWS fails the completeness test, which is
#: what stops the registry rotting as descriptors are added.
EXEMPT: dict[str, str] = {
    "duration_min": "the window itself",
    "leq_dbfs": "an energy mean is defined at any length (its fragility is "
                "a different problem, addressed by laeq_trim5_dbfs)",
    "laeq_dbfs": "as leq_dbfs",
    "laeq_trim5_dbfs": "as leq_dbfs",
    "leq_minus_laeq_db": "a difference of two energy means",
    "n_events": "a count, not a rate; honest at any length",
    "event_median_dur_s": "a median over detected events, not over time",
    "emergence_db": "defined at any length",
    "centroid_median_hz": "a spectral median over frames, stable quickly",
    "flatness_median": "as centroid_median_hz",
    "floor_suspect": "a flag from the noise floor, not a time average",
    "floor_suspect_lo_hz": "companion to floor_suspect",
    "floor_suspect_hi_hz": "companion to floor_suspect",
    "floor_spread_db": "companion to floor_suspect",

    # Band-energy fractions: ratios of spectral energy, defined at any
    # length. Their weaknesses are of a different kind -- the geophony
    # detector reads duct noise as wind however long it listens.
    "mechanical_index": "a band-energy ratio, defined at any length",
    "mech_lowfreq_fraction": "a band-energy ratio",
    "mech_rumble_db": "a band level",
    "anthrophony_index": "a band-energy ratio",
    "geophony_index": "a band-energy ratio",
    "geo_lowfreq_fraction": "a band-energy ratio",
    "geo_highband_fraction": "a band-energy ratio",
    "geo_wind_index": "a band-energy ratio",
    "geo_rain_index": "a band-energy ratio",
    "anthro_voiceband_fraction": "a band-energy ratio",
    "anthro_syllabic_mod": "envelope modulation depth in the syllabic band, "
                           "computable from a few seconds of envelope",
    "adi": "flat across the whole length study; its problem is that it "
           "does not discriminate, which no observation window repairs",

    # Directional descriptors: computed per frame and aggregated, so they
    # stabilise with frame count rather than with wall-clock duration.
    "azimuth_mean_deg": "a per-frame estimate aggregated over frames",
    "azimuth_fg_deg": "as azimuth_mean_deg",
    "azimuth_R": "circular concentration over frames",
    "elevation_fg_median_deg": "as azimuth_mean_deg",
    "diffuseness_median": "a per-frame estimate aggregated over frames",
    "diffuseness_iqr": "as diffuseness_median",
    "directional_entropy": "a distribution over per-frame directions",
    "below_horizon_fraction": "a fraction of per-frame directions",
    "above_horizon_fraction": "a fraction of per-frame directions",
    "fgbg_az_overlap": "an overlap of two per-frame direction sets",
    "bird_directional_entropy": "a distribution over per-frame directions",
    "bird_above_horizon_fraction": "a fraction of per-frame directions",
}


def check(summary: dict, duration_s: float) -> tuple[dict, list]:
    """Apply the registry to a summary computed over ``duration_s``.

    Returns ``(summary, low_confidence)``. Descriptors below a hard window
    are set to ``None`` --- never left at whatever a chunkless computation
    happened to return --- and descriptors below a soft window are kept
    and named in ``low_confidence`` with the window they missed.

    The summary is modified in place and also returned, so this reads the
    same whether a caller wants the value or the side effect.
    """
    low = []
    for key, w in WINDOWS.items():
        if key not in summary or duration_s >= w.min_s:
            continue
        if w.kind == "hard":
            summary[key] = None
        low.append({"key": key, "needs_s": w.min_s, "had_s": round(duration_s, 1),
                    "kind": w.kind, "why": w.why})
    return summary, low


def unregistered(summary: dict) -> list[str]:
    """Summary keys with neither a window nor an exemption."""
    return sorted(k for k in summary
                  if k not in WINDOWS and k not in EXEMPT
                  and k != "low_confidence")


def table() -> list[dict]:
    """The registry as rows, for printing, docs and report tables."""
    rows = []
    for w in sorted(WINDOWS.values(), key=lambda w: (w.min_s, w.key)):
        rows.append({"descriptor": w.key, "min_s": w.min_s, "kind": w.kind,
                     "band": band_of(w.min_s), "source": w.source,
                     "why": w.why, "ref": w.ref})
    return rows


# --------------------------------------------------------------- the figure

#: What rooms do, with this project's measured examples. Times are the
#: characteristic period or duration of the phenomenon, not a window.
PHENOMENA = [
    ("mains and rotating machinery", 1 / 121.0, 1 / 48.8,
     "a ~121 Hz line present in 72% of minutes of a Belgian living room",
     "measured"),
    ("sound objects", 0.2, 8.0,
     "21,593 in one domestic day; median detected event 0.5 s",
     "measured"),
    ("speech and conversation", 0.1, 10.0,
     "the syllabic band, and turns", "asserted"),
    ("activities", 5.0, 3600.0,
     "median labelled span 127.5 s in the SINS annotations", "measured"),
    ("appliance duty cycles", 300.0, 3600.0,
     "8.2 min on in every 35.9 for a kitchen refrigerator", "measured"),
    ("the diurnal envelope", 3600.0, 86400.0,
     "10.1 dB in the 25-100 Hz floor, minimum at 03:00-05:00", "measured"),
    ("the social week", 86400.0, 604800.0,
     "Saturday loudest and Tuesday quietest, on four independent nodes",
     "measured"),
    ("seasons", 604800.0, 3.15e7, "heating, windows, daylight", "asserted"),
]

#: What corpora and conventions supply.
CONVENTIONS = [
    ("clip corpora (ESC-50, UrbanSound8K, AudioSet, DCASE)", 4.0, 30.0,
     "the field's dominant unit", "measured"),
    ("soundwalk stops", 30.0, 180.0,
     "30 s in the SSID protocol; the three-minute figure is attributed to "
     "ISO/TS 12913-2 without a traceable clause", "measured"),
    ("activity annotations", 5.0, 600.0,
     "median 127.5 s in SINS; nothing shorter than 5 s", "measured"),
    ("365 Sound Actions: one action a day (foreground)", 4.0, 16.0,
     "365 audiovisual recordings of single sound-producing actions, one a "
     "day through 2022; measured p10-p90 of the deposited clips, median "
     "6.0 s, and 24% run past 10 s into the unannotated stratum",
     "measured"),
    ("StillStanding365: one standstill a day (background)", 480.0, 480.0,
     "365 daily standstill sessions through 2023, all 365 with audio. Ten "
     "minutes of standing, deposited as a median 481 s once the sync claps "
     "and the settling into stillness are trimmed, so the deposit is the "
     "steady middle. Deposited as 1 Hz non-identifying features rather "
     "than audio, the publication unit private interiors allow",
     "measured"),
    ("this book's own long-form recordings", 600.0, 604800.0,
     "Intercontinental sessions, minutes to a continuous week", "measured"),
]

#: The gap the figure exists to show.
GAPS = [
    ("no annotation at this granularity", 10.0, 120.0,
     "corpora are bimodal: events of about a second and activities of about "
     "two minutes. A long activity label spans this stratum without "
     "describing anything inside it, so structure here is unannotated "
     "rather than unrecorded"),
]


#: Descriptor families for the figure. Enumerating 25 descriptors buries
#: the argument; what a reader needs is the handful of distinct thresholds
#: and what sits at each. Members are checked against WINDOWS by the tests,
#: so a family cannot quietly disagree with the registry it summarises.
FAMILIES = [
    ("sound-object morphology (attack, decay, crest)", 0.2, "asserted"),
    ("spectral event duration", 30.0, "measured"),
    ("percentile levels, dynamics, event rates", 60.0, "measured"),
    ("tonal prominence (undefined below)", 60.0, "measured"),
    ("entropies and band activity", 60.0, "measured"),
    ("complexity index (undefined below)", 300.0, "measured"),
    ("ecoacoustic indices (ADI, NDSI, BI, AEI)", 360.0, "asserted"),
    ("machine periodicity", 10800.0, "asserted"),
]


def figure(out_path, dpi: int = 150):
    """Render the timescale figure from the registry.

    Three rows over one logarithmic time axis --- what rooms do, what
    descriptors need, what corpora supply --- against the micro/meso/macro
    bands. Asserted bounds are hatched so a reader can see at a glance
    which parts of the picture are measured.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lo, hi = 1e-3, 3.2e7
    fig, ax = plt.subplots(figsize=(11.0, 6.4), dpi=dpi)

    band_col = {"micro": "#eef2f7", "meso": "#dbe6f0", "macro": "#eef2f7"}
    for name, b in BANDS.items():
        a, z = max(b["t_lo"], lo), min(b["t_hi"], hi)
        if z <= a:
            continue
        ax.axvspan(a, z, color=band_col[name], zorder=0)
        # the meso band is narrow on a log axis spanning ms to years, which
        # is itself worth seeing: a great deal of perception happens in it
        mid = (a * z) ** 0.5
        anchor = mid if name != "meso" else 1.6
        ax.text(anchor, 1.105, name.upper(), ha="center",
                transform=ax.get_xaxis_transform(), fontsize=10.5,
                color="#33506e", weight="bold")
        ax.text(anchor, 1.062, b["space_room"], ha="center",
                transform=ax.get_xaxis_transform(), fontsize=8,
                color="#55708c")
        ax.text(anchor, 1.022, b["memory"], ha="center",
                transform=ax.get_xaxis_transform(), fontsize=8,
                color="#55708c", style="italic")

    rows = [("WHAT ROOMS DO", [(n, a, b, sc) for n, a, b, _w, sc in PHENOMENA],
             "#2b6cb0"),
            ("WHAT DESCRIPTORS NEED",
             [(n, t, hi, sc) for n, t, sc in FAMILIES], "#b7791f"),
            ("WHAT CORPORA SUPPLY",
             [(n, a, b, sc) for n, a, b, _w, sc in CONVENTIONS], "#2f855a")]

    y = 0.0
    yticks, ylabels = [], []
    for title, items, colour in rows:
        y -= 1.9
        ax.text(lo * 1.25, y + 0.3, title, fontsize=9.5, weight="bold",
                color=colour, va="center")
        y -= 0.5
        for label, t0, t1, source in items:
            y -= 1.0
            hatch = None if source == "measured" else "///"
            # A corpus of one fixed duration is a point, and a point has no
            # width on a log axis. Give it a visible mark rather than a
            # sliver that reads as missing data.
            left, right = (t0 / 1.35, t1 * 1.35) if t1 <= t0 else (t0, t1)
            ax.barh(y, width=right - left, left=left, height=0.6,
                    color=colour, alpha=.8, hatch=hatch,
                    edgecolor="white", linewidth=.6, zorder=3)
            yticks.append(y)
            ylabels.append(label)

    for label, t0, t1, _why in GAPS:
        ax.axvspan(t0, t1, color="#c53030", alpha=.09, zorder=1)
        ax.text((t0 * t1) ** 0.5, 0.30, label, ha="center", fontsize=8.5,
                color="#c53030", style="italic", rotation=90,
                va="bottom", transform=ax.get_xaxis_transform())

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(y - 1.0, 0.9)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("duration or period (seconds, logarithmic)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ticks = [1e-3, 1e-2, 1e-1, 1, 10, 60, 600, 3600, 86400, 604800, 3.15e7]
    names = ["1 ms", "10 ms", "0.1 s", "1 s", "10 s", "1 min", "10 min",
             "1 h", "1 day", "1 week", "1 year"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(names, fontsize=8)
    ax.grid(axis="x", alpha=.25, lw=.5, zorder=0)

    ax.legend(handles=[Patch(facecolor="#888", label="measured in this project"),
                       Patch(facecolor="#888", hatch="///",
                             label="asserted, not measured")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path
