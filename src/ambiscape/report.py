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
]


def write_readme(sess: Session, summary: dict, out_dir: Path,
                 notes: str = "", extra: str = "") -> Path:
    lines = [f"# {sess.name}", ""]
    if notes:
        lines += [notes, ""]
    lines += ["## Recording", "",
              "| File | Date | Start (recorder clock) | Duration |",
              "|---|---|---|---|"]
    for tk in sess.takes:
        lines.append(f"| `{tk.path.name}` | {tk.date} | {tk.clock} | "
                     f"{tk.duration/60:.1f} min |")
    lines += ["",
              "Zoom H3-VR, AmbiX B-format (ACN: W-Y-Z-X, SN3D), upright. "
              "Levels are uncalibrated dBFS; directions are mic-relative.",
              "", "## Descriptors", "", "| Descriptor | Value |", "|---|---|"]
    for key, label in TABLE_ROWS:
        v = summary.get(key)
        if v is not None:
            lines.append(f"| {label} | {v} |")
    lines += ["", "## Figures", "",
              "![overview](analysis/overview.png)",
              "![LTAS percentiles](analysis/ltas_percentiles.png)",
              "![directogram](analysis/directogram.png)", ""]
    if extra:
        lines += [extra, ""]
    lines += ["---", "*Analyzed with [ambiscape](../ambiscape/) "
              "(streaming companion to "
              "[ambiviz](https://github.com/fisheggg/ambiviz)).*", ""]
    path = sess.folder / "README.md"
    path.write_text("\n".join(lines))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return path
