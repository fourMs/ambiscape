"""Auto-generated per-session README skeleton."""
from __future__ import annotations

import json
from pathlib import Path

from .io import Session

TABLE_ROWS = [
    ("duration_min", "Duration (min)"),
    ("leq_dbfs", "Leq (dBFS)"),
    ("laeq_dbfs", "LAeq (dBFS)"),
    ("leq_minus_laeq_db", "Leq − LAeq (dB)"),
    ("L10", "L10 (dBFS)"), ("L50", "L50 (dBFS)"), ("L90", "L90 (dBFS)"),
    ("dynamics_L10_L90", "Dynamics L10−L90 (dB)"),
    ("events_per_min", "Events ≥ +8 dB (per min)"),
    ("event_median_dur_s", "Median event duration (s)"),
    ("intermittency_ratio_pct", "Intermittency ratio IR (%)"),
    ("emergence_db", "Emergence LAeq − LA90 (dB)"),
    ("fg_fraction_median", "Spectral foreground fraction, median"),
    ("fg_fraction_p90", "Spectral foreground fraction, P90"),
    ("spectral_events_per_min", "Spectral events (per min)"),
    ("spectral_event_median_dur_s", "Median spectral event duration (s)"),
    ("centroid_median_hz", "Spectral centroid, median (Hz)"),
    ("flatness_median", "Spectral flatness, median"),
    ("diffuseness_median", "Diffuseness ψ, median"),
    ("diffuseness_iqr", "Diffuseness ψ, IQR"),
    ("azimuth_mean_deg", "Dominant azimuth (mic frame, °)"),
    ("azimuth_R", "Azimuthal concentration R"),
    ("elevation_fg_median_deg", "Median foreground elevation (°)"),
    ("directional_entropy", "Directional entropy (azimuth)"),
    ("above_horizon_fraction", "Energy from above +10° elevation"),
    ("fgbg_az_overlap", "Foreground/background azimuth overlap"),
    ("ndsi", "NDSI (−1 anthropophony … +1 biophony band)"),
    ("adi", "Acoustic diversity ADI"),
    ("aci", "Acoustic complexity ACI (5-min mean)"),
    ("acoustic_entropy", "Acoustic entropy H"),
    ("bird_peaks_per_min", "Biophony narrowband peaks (per min)"),
    ("bird_band_activity_pct", "Biophony band activity (%)"),
    ("bird_temporal_entropy", "Biophony temporal entropy (low=structured)"),
    ("bird_directional_entropy", "Biophony directional entropy"),
    ("bird_above_horizon_fraction", "Biophony energy from above +10°"),
]


MARKER = "<!-- ambiscape:generated -->"

_FIGURES = [
    ("overview.png", "![overview](analysis/overview.png)"),
    ("ltas_percentiles.png", "![LTAS percentiles](analysis/ltas_percentiles.png)"),
    ("directogram.png", "![directogram](analysis/directogram.png)"),
]


def _recording_note(sess: Session) -> str:
    """Channel-layout-appropriate recording note for the README."""
    mode = getattr(sess.takes[0], "mode", "ambix") if sess.takes else "ambix"
    if mode == "stereo":
        return ("2-channel stereo. Levels are uncalibrated dBFS. Direction is "
                "a lateral left/right cue only: azimuth is the inter-channel "
                "energy balance (±90°, + = left, 0 = centre) and diffuseness "
                "is inter-channel decorrelation (0 = coherent point source, "
                "1 = decorrelated/enveloping) — no elevation or front/back.")
    if mode == "mono":
        return ("1-channel mono. Levels are uncalibrated dBFS; no directional "
                "information.")
    return ("Zoom H3-VR, AmbiX B-format (ACN: W-Y-Z-X, SN3D), upright. "
            "Levels are uncalibrated dBFS; directions are mic-relative.")


def _json_safe(o):
    """Recursively replace NaN/inf floats with None for valid JSON."""
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o

STATE_ROWS = [
    ("duration_min", "Duration (min)"),
    ("leq_dbfs", "Leq (dBFS)"),
    ("laeq_dbfs", "LAeq (dBFS)"),
    ("L90", "L90 (dBFS)"),
    ("dynamics_L10_L90", "Dynamics L10−L90 (dB)"),
    ("events_per_min", "Events (per min)"),
    ("intermittency_ratio_pct", "Intermittency ratio IR (%)"),
    ("diffuseness_median", "Diffuseness ψ, median"),
    ("directional_entropy", "Directional entropy"),
    ("ndsi", "NDSI"),
    ("bird_band_activity_pct", "Biophony band activity (%)"),
]


def state_table(states_doc: dict) -> str:
    """Compact Markdown table of state-resolved descriptors.

    ``states_doc`` maps state label -> summary dict (the ``states`` value of
    ``states.json``, or a plain ``{label: summary}``). Returns "" when
    empty."""
    states = states_doc.get("states", states_doc)
    if not states:
        return ""
    labels = list(states)
    head = "| Descriptor | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)
    lines = ["## State-resolved descriptors", "",
             "Auto-detected machine on/off states (see `analysis/states.json`; "
             "regenerate with `ambiscape resolve`). The pooled descriptor "
             "table above is a duration-weighted average of these.", "",
             head, sep]
    for key, label in STATE_ROWS:
        if any(key in states[s] for s in labels):
            cells = [str(states[s].get(key, "")) for s in labels]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_readme(sess: Session, summary: dict, out_dir: Path,
                 notes: str = "", extra: str = "",
                 states: dict | None = None) -> Path:
    """Write the session README.

    Everything above the ``MARKER`` line is hand-written and preserved
    across re-analysis; only the section below it is regenerated. When
    ``states`` (a ``{label: summary}`` dict) is given, a state-resolved
    table is appended.
    """
    path = sess.folder / "README.md"
    head = ""
    if path.exists():
        txt = path.read_text()
        if MARKER in txt:
            head = txt[:txt.index(MARKER)]
    lines = [head + MARKER, "", f"# {sess.name}", ""]
    if notes:
        lines += [notes, ""]
    lines += ["## Recording", "",
              "| File | Date | Start (recorder clock) | Duration |",
              "|---|---|---|---|"]
    for tk in sess.takes:
        lines.append(f"| `{tk.path.name}` | {tk.date} | {tk.clock} | "
                     f"{tk.duration/60:.1f} min |")
    lines += ["", _recording_note(sess),
              "", "## Descriptors", "", "| Descriptor | Value |", "|---|---|"]
    for key, label in TABLE_ROWS:
        v = summary.get(key)
        if v is not None:
            lines.append(f"| {label} | {v} |")
    fig_lines = [ln for fname, ln in _FIGURES if (out_dir / fname).exists()]
    if fig_lines:
        lines += ["", "## Figures", ""] + fig_lines + [""]
    if states:
        st = state_table(states)
        if st:
            lines += [st, ""]
    if extra:
        lines += [extra, ""]
    lines += ["---", "*Analyzed with [ambiscape](../ambiscape/) "
              "(streaming companion to "
              "[ambiviz](https://github.com/fisheggg/ambiviz)).*", ""]
    path.write_text("\n".join(lines))
    (out_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2))
    return path
