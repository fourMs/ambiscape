"""What kind of evidence a descriptor is: signal, or something about a listener.

Every number this toolbox returns is a fact about a waveform. Some of them are
*also* meant as facts about hearing, and the distance between those two things
is where this project has made its worst mistakes. Four claims were withdrawn
in a single month, and every one of them was a perceptual quantity read off a
signal statistic: acoustic "zones" from a speech fraction, a "dead" channel
from a level, a building's rhythm from one day's periodicity, a reverberation
time from material containing no free decay. None was a coding error. Each was
a translation nobody had written down.

This module writes it down. It is the companion to :mod:`ambiscape.timescales`,
and deliberately the same shape: a registry, a check, a table of what is not yet
covered. Where the timescale registry answers *over how long is this valid*,
this one answers *what is it evidence about*.

The tiers
---------
``S`` — **signal only.** Defined by the mathematics of the waveform. No
listener anywhere. It may claim "this is what is in the recording" and nothing
about audibility, salience or annoyance.

``PM`` — **perceptually motivated.** Designed by analogy with hearing but not
validated against listeners in this domain. The analogy is a hypothesis. Most
of the event and source-category descriptors live here, because "an event is a
departure from the background" is a good guess about noticing that no one here
has tested.

``PC`` — **perceptually calibrated.** The transform embeds a measured property
of hearing, usually from listening experiments codified in a standard:
A-weighting is an equal-loudness contour, octave bands are roughly critical
bands. Note that calibrated is not the same as *right*: A-weighting is known to
misrepresent exactly the low-frequency steady sources this project is full of.

``PD`` — **perceptually defined.** The quantity only exists as a fact about a
listener, and the signal measure is a proxy that can simply be wrong.
Foreground and background are the standard case: a background is not a level,
it is a relation between a sound and someone not attending to it, and the level
is a stand-in. A ``PD`` number is evidence about the proxy until somebody asks
a listener.

The tier is not a quality ranking. ``S`` is not worse than ``PC``; a spectral
centroid is an excellent measurement of a spectral centroid. The tier says what
may be concluded, and the only real error is concluding one tier's worth of
thing from another tier's number.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The tiers, in order of how much of a listener is built into the number.
TIERS = {
    "S": "signal only — no listener in the number",
    "PM": "perceptually motivated — designed by analogy with hearing, "
          "not validated against listeners here",
    "PC": "perceptually calibrated — embeds a measured property of hearing",
    "PD": "perceptually defined — the quantity is a fact about a listener; "
          "this is a proxy for it",
}


@dataclass(frozen=True)
class Grounding:
    """One descriptor's evidence tier, with the reason and where it came from."""

    key: str
    tier: str
    why: str
    ref: str = ""

    def __post_init__(self):
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r} for {self.key!r}")


GROUNDINGS: dict[str, Grounding] = {g.key: g for g in [
    # ---------------------------------------------------------------- level
    Grounding("laeq_dbfs", "PC",
              "A-weighted: the curve is the 40-phon equal-loudness contour, "
              "from listening tests. Known to under-weight the low-frequency "
              "steady sources this project measures most",
              "IEC 61672"),
    Grounding("laeq_trim5_dbfs", "PC", "A-weighted, loudest 5 % discarded",
              "IEC 61672"),
    Grounding("leq_dbfs", "S", "unweighted energy mean; no listener"),
    Grounding("leq_minus_laeq_db", "PC",
              "the weighting itself, expressed as a difference"),
    Grounding("L10", "PC", "percentile of the A-weighted level"),
    Grounding("L50", "PC", "percentile of the A-weighted level"),
    Grounding("L90", "PC",
              "percentile of the A-weighted level. Widely *read* as 'the "
              "background', which is a PD claim this number cannot carry"),
    Grounding("dynamics_L10_L90", "PC",
              "spread of the A-weighted level. The figure/ground reading of "
              "it — that people widen a room's range rather than raise its "
              "floor — is measured and solid; calling the result "
              "figure and ground is the PD step"),
    Grounding("floor_spread_db", "S", "dispersion of the level floor"),
    Grounding("floor_suspect", "S", "instrument-facing: is this the recorder"),
    Grounding("floor_suspect_lo_hz", "S", "instrument-facing"),
    Grounding("floor_suspect_hi_hz", "S", "instrument-facing"),
    Grounding("duration_min", "S", "length of the recording"),

    # ------------------------------------------------------------- spectrum
    Grounding("centroid_median_hz", "S",
              "first moment of the spectrum. Routinely called 'brightness', "
              "which smuggles in a perceptual claim the number does not carry"),
    Grounding("flatness_median", "S", "spectral flatness; no listener"),
    Grounding("fluctuation_index", "PM",
              "modelled on Fastl & Zwicker's fluctuation strength in spirit "
              "only — the code says it is an approximation, not a standard, "
              "and it should not be reported as a psychoacoustic quantity"),

    # ------------------------------------------- events, and what a listener
    #                                              might have noticed
    Grounding("n_events", "PM",
              "an event is level above a running background by a fixed "
              "margin: a convention chosen to resemble noticing, validated "
              "against no listener"),
    Grounding("events_per_min", "PM", "rate of the same convention"),
    Grounding("event_median_dur_s", "PM", "duration of the same convention"),
    Grounding("emergence_db", "PM",
              "how far events stand above the floor; the assumption that "
              "standing above a floor is what gets noticed is untested here"),
    Grounding("intermittency_ratio_pct", "PM",
              "share of energy in events, on the same event definition",
              "Wunderli et al. 2016"),
    Grounding("spectral_events_per_min", "PM", "per-band event convention"),
    Grounding("spectral_event_median_dur_s", "PM", "per-band event convention"),

    # -------------------------------------------------- foreground/background
    Grounding("fg_fraction_median", "PD",
              "figure and ground are a relation between a sound and a "
              "listener, not a property of a signal. This is a level-based "
              "proxy for that relation and can be wrong about it"),
    Grounding("fg_fraction_p90", "PD", "as fg_fraction_median"),
    Grounding("fgbg_az_overlap", "PD",
              "whether figure and ground share a direction — inherits the "
              "figure/ground proxy and adds a spatial one"),

    # ------------------------------------------------------- the sound object
    Grounding("attack_s", "S",
              "10–90 % rise of the envelope. A number, deliberately, rather "
              "than a facture label — the typing is a separate, contestable "
              "step. Note it is unstable under repetition: across takes of "
              "one action it varies by about 85 %"),
    Grounding("decay_s", "S", "fall from the peak through 10 % of it"),
    Grounding("temporal_centroid", "S",
              "where the energy sits along the object. Stable under "
              "repetition (about 15 % across takes), which is why it is a "
              "better basis for typing than attack time"),
    Grounding("crest_db", "S", "peak over RMS"),
    Grounding("iteration_hz", "S", "best repetition rate of the envelope"),
    Grounding("iteration_strength", "S", "how strongly the envelope repeats"),

    # ------------------------------------------------------------- ecoacoustic
    Grounding("aci", "S",
              "accumulated spectral change; window-length dependent by "
              "construction. Interpreted as diversity, which is neither "
              "signal nor perception but ecology, and unvalidated indoors"),
    Grounding("adi", "S", "band occupancy entropy; as aci for interpretation"),
    Grounding("aei", "S", "band occupancy evenness; as aci"),
    Grounding("bi", "S", "area under the mean spectrum in a bird band"),
    Grounding("ndsi", "S", "ratio of two band energies"),
    Grounding("acoustic_entropy", "S", "entropy of the spectrum"),

    # ----------------------------------------------------------------- sources
    Grounding("anthrophony_index", "PM", "band-based source attribution"),
    Grounding("geophony_index", "PM", "band-based source attribution"),
    Grounding("mechanical_index", "PM", "band-based source attribution"),
    Grounding("anthro_activity_fraction", "PM", "voice-band activity"),
    Grounding("anthro_voiceband_fraction", "PM",
              "band chosen because speech lives there"),
    Grounding("anthro_syllabic_mod", "PM",
              "modulation range chosen because speech lives there"),
    Grounding("geo_rain_index", "PM", "spectral signature of rain"),
    Grounding("geo_wind_index", "PM", "spectral signature of wind"),
    Grounding("geo_lowfreq_fraction", "S", "band energy ratio"),
    Grounding("geo_highband_fraction", "S", "band energy ratio"),
    Grounding("mech_lowfreq_fraction", "S", "band energy ratio"),
    Grounding("mech_rumble_db", "S", "low-band level"),
    Grounding("mech_periodicity_hz", "S", "envelope periodicity"),
    Grounding("mech_periodicity_strength", "S", "envelope periodicity"),
    Grounding("bird_band_activity_pct", "S", "detector output in a band"),
    Grounding("bird_event_rate_per_min", "S", "detector output"),
    Grounding("bird_peaks_per_min", "S", "detector output"),
    Grounding("bird_active_minute_fraction", "S", "detector output"),
    Grounding("bird_temporal_entropy", "S", "detector output"),
    Grounding("bird_above_horizon_fraction", "S", "geometry of detections"),
    Grounding("bird_directional_entropy", "S", "geometry of detections"),

    # -------------------------------------------------------------- tonality
    Grounding("tonal_prominence_db", "PM",
              "prominence of a tonal peak; the threshold at which a tone "
              "becomes annoying is a perceptual question this does not answer"),
    Grounding("tonal_prominence_hz", "PM", "companion to tonal_prominence_db"),
    Grounding("n_prominent_tones", "PM", "companion to tonal_prominence_db"),

    # ----------------------------------------------------------------- space
    Grounding("diffuseness_median", "S",
              "pseudo-intensity coherence. Perceived spaciousness is a "
              "different quantity and is not measured here"),
    Grounding("diffuseness_iqr", "S", "spread of the same"),
    Grounding("directional_entropy", "S", "spread of arrival directions"),
    Grounding("azimuth_mean_deg", "S", "circular mean of arrival direction"),
    Grounding("azimuth_R", "S", "concentration of arrival direction"),
    Grounding("azimuth_fg_deg", "PD",
              "direction of the *foreground*, so it inherits the "
              "figure/ground proxy"),
    Grounding("elevation_fg_median_deg", "PD", "as azimuth_fg_deg"),
    Grounding("above_horizon_fraction", "S", "geometry"),
    Grounding("below_horizon_fraction", "S", "geometry"),
]}

#: Keys that are not descriptors and need no tier.
EXEMPT: set[str] = {"low_confidence", "grounding_cautions", "n_frames",
                    "fs", "path", "channels", "mode"}


def tier_of(key: str) -> str | None:
    """The evidence tier of a descriptor, or ``None`` if unregistered."""
    g = GROUNDINGS.get(key)
    return g.tier if g else None


def unregistered(summary: dict) -> list[str]:
    """Summary keys with neither a tier nor an exemption.

    Coverage is honest rather than assumed: a key nobody has classified shows
    up here instead of silently defaulting to ``S``, which would be the
    convenient answer and the wrong one.
    """
    return sorted(k for k in summary
                  if k not in GROUNDINGS and k not in EXEMPT)


def check(summary: dict) -> tuple[dict, list[str]]:
    """Annotate a summary with its perceptual cautions.

    Returns the summary with a ``grounding_cautions`` list added, and that
    list. A caution is raised for every ``PD`` quantity present, because those
    are the numbers most easily mistaken for the perceptual fact they stand in
    for. ``PM`` quantities are counted but not itemised: there are many of
    them, and warning on each would train the reader to ignore the warning.
    """
    cautions: list[str] = []
    for key in sorted(summary):
        g = GROUNDINGS.get(key)
        if g is not None and g.tier == "PD":
            cautions.append(f"{key}: {g.why}")
    n_pm = sum(1 for k in summary
               if (g := GROUNDINGS.get(k)) is not None and g.tier == "PM")
    if n_pm:
        cautions.append(
            f"{n_pm} further descriptors are perceptually motivated but "
            "unvalidated against listeners (tier PM); see ambiscape.grounding")
    out = dict(summary)
    out["grounding_cautions"] = cautions
    return out, cautions


def table() -> list[dict]:
    """The registry as rows, for printing, docs and report tables."""
    order = {t: i for i, t in enumerate(TIERS)}
    return [{"descriptor": g.key, "tier": g.tier, "meaning": TIERS[g.tier],
             "why": g.why, "ref": g.ref}
            for g in sorted(GROUNDINGS.values(),
                            key=lambda g: (order[g.tier], g.key))]


def counts() -> dict[str, int]:
    """How many registered descriptors sit in each tier."""
    out = {t: 0 for t in TIERS}
    for g in GROUNDINGS.values():
        out[g.tier] += 1
    return out
