"""ISO 12913-2 Method A questionnaire → ISO/TS 12913-3 circumplex.

The rest of the toolbox measures a soundscape; this module asks people.
ISO 12913-2 (Method A) has respondents rate eight perceived affective
qualities — *pleasant, chaotic, vibrant, uneventful, calm, annoying,
eventful, monotonous* — each on a 5-point Likert scale (or a 100-point
slider in the common digital variant). ISO/TS 12913-3 projects the eight
ratings onto a two-dimensional circumplex::

    Pleasantness = (p − a) + cos45°·(ca − ch) + cos45°·(v − m)
    Eventfulness = (e − u) + cos45°·(ch − ca) + cos45°·(v − m)

normalised by ``ρ·(1 + √2)`` (ρ = the coded scale range: 4 for a 1–5
scale, 100 for a 0–100 slider), so both coordinates land in [−1, +1].
The quadrants carry the familiar labels: vibrant (+P, +E), chaotic
(−P, +E), monotonous (−P, −E), calm (+P, −E).

Input is a plain CSV, one row per respondent, headed by the eight scale
names (case-insensitive, any column order). The coded scale is
auto-detected: values within 1–5 read as 5-point, anything larger as a
0–100 slider. Extra columns (an id column, appropriateness or loudness
ratings, free text) ride along untouched and numeric extras are averaged
into the summary.

:func:`run_survey` writes ``survey.json`` + a circumplex ``survey.png``
into the session's analysis dir and folds ``srv_``-prefixed keys (mean
pleasantness/eventfulness, n, dispersion) into ``summary.json`` — the
same join as the ``vis_`` keys from :mod:`ambiscape.vision` — so
``ambiscape catalog`` can rank a corpus perceptually next to the
acoustic descriptors. When the session already has an acoustic summary,
the returned doc also carries a short perception-vs-measurement table
(LAeq vs pleasantness, events/min vs eventfulness).

The usual honesty note applies (see the acoustics guide): this supports
12913-2 *data handling and reporting*, it does not make a survey
protocol-conformant by itself.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

#: The eight ISO 12913-2 perceived-affective-quality scales, in the
#: conventional circumplex order (45° apart, pleasant at 0°).
SCALES = ("pleasant", "vibrant", "eventful", "chaotic",
          "annoying", "monotonous", "uneventful", "calm")
_ID_COLUMNS = ("respondent", "participant", "id", "subject")
_COS45 = np.cos(np.pi / 4)
_CHI2_95_2DF = 5.991464547  # chi2.ppf(0.95, 2): 95% coverage, 2 dof


def read_responses(path: str | Path) -> dict:
    """Parse a Method-A response CSV.

    Requires all eight :data:`SCALES` as columns (case-insensitive, any
    order). One of ``respondent``/``participant``/``id``/``subject``
    (if present) names each row; otherwise rows are numbered. All other
    columns are kept as extras (numeric ones parsed to float). Rows with
    a missing or non-numeric scale value are skipped and counted.

    Returns ``{"respondents": [{"id", "scales", "extras"}, ...],
    "scale": "5-point"|"100-point", "range": (lo, hi), "n_skipped": int,
    "extra_keys": [...]}``.
    """
    path = Path(path)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        by_lower = {c.strip().lower(): c for c in reader.fieldnames}
        missing = [s for s in SCALES if s not in by_lower]
        if missing:
            raise ValueError(
                f"{path.name} lacks ISO 12913-2 scale column(s): "
                f"{', '.join(missing)}")
        id_col = next((by_lower[c] for c in _ID_COLUMNS if c in by_lower),
                      None)
        scale_cols = {s: by_lower[s] for s in SCALES}
        extra_cols = [c for c in reader.fieldnames
                      if c not in scale_cols.values() and c != id_col]
        rows, n_skipped = [], 0
        for i, rec in enumerate(reader):
            try:
                scales = {s: float(rec[c]) for s, c in scale_cols.items()}
            except (TypeError, ValueError):
                n_skipped += 1
                continue
            extras = {}
            for c in extra_cols:
                v = (rec.get(c) or "").strip()
                if not v:
                    continue
                try:
                    extras[c] = float(v)
                except ValueError:
                    extras[c] = v
            rid = (rec[id_col].strip() if id_col and (rec.get(id_col) or "")
                   .strip() else f"r{i + 1:02d}")
            rows.append({"id": rid, "scales": scales, "extras": extras})
    if not rows:
        raise ValueError(f"{path.name} has no complete response rows")
    lo, hi = detect_scale(rows)
    return {"respondents": rows,
            "scale": "5-point" if hi == 5.0 else "100-point",
            "range": (lo, hi), "n_skipped": n_skipped,
            "extra_keys": extra_cols}


def detect_scale(respondents: list) -> tuple:
    """Coded scale range ``(lo, hi)`` from the pooled scale values.

    Everything within 1–5 reads as the printed 5-point Likert form;
    anything larger as the 0–100 digital slider. Out-of-range values
    (negative, or above 100) raise.
    """
    vals = np.array([v for r in respondents for v in r["scales"].values()])
    if vals.min() >= 1.0 and vals.max() <= 5.0:
        return (1.0, 5.0)
    if vals.min() >= 0.0 and vals.max() <= 100.0:
        return (0.0, 100.0)
    raise ValueError(f"scale values outside both 1–5 and 0–100: "
                     f"min {vals.min()}, max {vals.max()}")


def coordinates(scales: dict, lo: float = 1.0, hi: float = 5.0) -> tuple:
    """One respondent's ISO/TS 12913-3 (pleasantness, eventfulness).

    ``scales`` maps the eight scale names to ratings coded on
    ``lo``–``hi``. Both coordinates are normalised to [−1, +1] by
    ``(hi − lo)·(1 + √2)`` per the TS.
    """
    p, ch, v = scales["pleasant"], scales["chaotic"], scales["vibrant"]
    u, ca, a = scales["uneventful"], scales["calm"], scales["annoying"]
    e, m = scales["eventful"], scales["monotonous"]
    norm = (hi - lo) * (1 + np.sqrt(2))
    pl = ((p - a) + _COS45 * (ca - ch) + _COS45 * (v - m)) / norm
    ev = ((e - u) + _COS45 * (ch - ca) + _COS45 * (v - m)) / norm
    return float(pl), float(ev)


def quadrant(pleasantness: float, eventfulness: float) -> str:
    """The circumplex quadrant label of a point."""
    if eventfulness >= 0:
        return "vibrant" if pleasantness >= 0 else "chaotic"
    return "calm" if pleasantness >= 0 else "monotonous"


def ellipse_95(P: np.ndarray, E: np.ndarray) -> dict | None:
    """95% covariance ellipse of the respondent cloud (needs n ≥ 3).

    Returns full axis lengths ``width``/``height`` (major/minor) and the
    major axis' ``angle_deg`` counter-clockwise from the pleasantness
    axis; ``None`` when the cloud is too small or degenerate.
    """
    if len(P) < 3:
        return None
    cov = np.cov(np.stack([P, E]))
    if not np.all(np.isfinite(cov)):
        return None
    eigval, eigvec = np.linalg.eigh(cov)          # ascending
    if eigval[-1] <= 0:
        return None
    eigval = np.clip(eigval, 0, None)
    ang = float(np.degrees(np.arctan2(eigvec[1, -1], eigvec[0, -1])))
    return {"width": round(2 * float(np.sqrt(_CHI2_95_2DF * eigval[-1])), 3),
            "height": round(2 * float(np.sqrt(_CHI2_95_2DF * eigval[0])), 3),
            "angle_deg": round(ang, 1)}


def summarize(responses: dict) -> dict:
    """Per-respondent circumplex points + pooled statistics.

    ``responses`` is :func:`read_responses` output. Returns the
    ``survey.json`` document: points, mean, sd, dispersion (RMS distance
    of respondents from the mean point), quadrant of the mean, the 95%
    ellipse, and means of any numeric extra columns.
    """
    lo, hi = responses["range"]
    pts = []
    for r in responses["respondents"]:
        pl, ev = coordinates(r["scales"], lo, hi)
        pts.append({"id": r["id"], "pleasantness": round(pl, 3),
                    "eventfulness": round(ev, 3),
                    "scales": r["scales"], "extras": r["extras"]})
    P = np.array([p["pleasantness"] for p in pts])
    E = np.array([p["eventfulness"] for p in pts])
    disp = float(np.sqrt(np.mean((P - P.mean()) ** 2 + (E - E.mean()) ** 2)))
    doc = {
        "standard": "ISO 12913-2 Method A -> ISO/TS 12913-3 circumplex",
        "scale": responses["scale"], "n": len(pts),
        "n_skipped": responses["n_skipped"],
        "pleasantness_mean": round(float(P.mean()), 3),
        "eventfulness_mean": round(float(E.mean()), 3),
        "pleasantness_sd": round(float(P.std(ddof=1)), 3) if len(P) > 1
        else None,
        "eventfulness_sd": round(float(E.std(ddof=1)), 3) if len(E) > 1
        else None,
        "dispersion": round(disp, 3),
        "quadrant": quadrant(float(P.mean()), float(E.mean())),
        "ellipse_95": ellipse_95(P, E),
        "respondents": pts,
    }
    num_extras = {}
    for k in responses["extra_keys"]:
        vals = [p["extras"][k] for p in pts
                if isinstance(p["extras"].get(k), float)]
        if vals:
            num_extras[f"{k}_mean"] = round(float(np.mean(vals)), 3)
    if num_extras:
        doc["extras"] = num_extras
    return doc


def survey_summary_keys(doc: dict) -> dict:
    """The ``srv_`` rows folded into ``summary.json`` for the catalog."""
    out = {"srv_n": doc["n"],
           "srv_pleasantness_mean": doc["pleasantness_mean"],
           "srv_eventfulness_mean": doc["eventfulness_mean"],
           "srv_dispersion": doc["dispersion"],
           "srv_quadrant": doc["quadrant"]}
    if doc["pleasantness_sd"] is not None:
        out["srv_pleasantness_sd"] = doc["pleasantness_sd"]
        out["srv_eventfulness_sd"] = doc["eventfulness_sd"]
    return out


#: (summary key, label, paired perceptual key) rows for the
#: perception-vs-measurement table; the first summary key found wins
#: within each ("either/or") tuple.
_VS_ROWS = (
    (("laeq_db_spl", "laeq_dbfs"), "LAeq", "pleasantness_mean"),
    (("L90_db_spl", "L90"), "L90 background", "pleasantness_mean"),
    (("events_per_min",), "events/min", "eventfulness_mean"),
    (("ndsi",), "NDSI", "pleasantness_mean"),
)


def vs_measurement(doc: dict, summary: dict) -> list:
    """Perception-vs-measurement rows from an acoustic ``summary.json``.

    Pairs each available acoustic descriptor with the perceptual
    coordinate it is classically regressed against (LAeq and L90 vs
    pleasantness, event rate vs eventfulness). Returns a list of
    ``{"measured", "value", "perceived", "perceived_value"}`` rows —
    empty when the summary carries none of the keys.
    """
    rows = []
    for keys, label, pkey in _VS_ROWS:
        key = next((k for k in keys if summary.get(k) is not None), None)
        if key is None:
            continue
        unit = " (dB SPL)" if key.endswith("_db_spl") else \
            " (dBFS)" if key in ("laeq_dbfs", "L90") else ""
        rows.append({"measured": label + unit, "value": summary[key],
                     "perceived": pkey.replace("_mean", ""),
                     "perceived_value": doc[pkey]})
    return rows


def vs_table(rows: list) -> str:
    """The rows of :func:`vs_measurement` as a small Markdown table."""
    out = ["| measured | value | perceived | value |", "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['measured']} | {r['value']} | {r['perceived']} | "
                   f"{r['perceived_value']:+.3f} |")
    return "\n".join(out)


def render(doc: dict, out_path: str | Path, title: str = "") -> Path:
    """Circumplex plot: respondents, mean, 95% ellipse → a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from .figures import RC, BLUE, MAGENTA, GRID, MUT, SEC

    pts = doc["respondents"]
    P = [p["pleasantness"] for p in pts]
    E = [p["eventfulness"] for p in pts]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
        ax.grid(False)
        ax.set_aspect("equal")
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color=GRID, lw=1.0)
        for r in (0.25, 0.5, 0.75):
            ax.plot(r * np.cos(th), r * np.sin(th), color=GRID, lw=0.5)
        for ang in range(0, 360, 45):
            a = np.radians(ang)
            ax.plot([0, np.cos(a)], [0, np.sin(a)], color=GRID, lw=0.5)
        for name, ang in zip(SCALES, range(0, 360, 45)):
            a = np.radians(ang)
            ax.annotate(name, (1.1 * np.cos(a), 1.1 * np.sin(a)),
                        ha="center", va="center", color=SEC, fontsize=9)
        ax.scatter(P, E, s=26, color=BLUE, alpha=0.65, lw=0, zorder=3)
        ell = doc.get("ellipse_95")
        if ell:
            ax.add_patch(Ellipse(
                (doc["pleasantness_mean"], doc["eventfulness_mean"]),
                ell["width"], ell["height"], angle=ell["angle_deg"],
                fill=False, color=MAGENTA, lw=1.2, ls="--", zorder=4))
        ax.scatter([doc["pleasantness_mean"]], [doc["eventfulness_mean"]],
                   s=90, color=MAGENTA, marker="D", zorder=5,
                   label=f"mean (n={doc['n']})")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.tick_params(colors=MUT, labelsize=8)
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.set_xlabel("pleasantness")
        ax.set_ylabel("eventfulness")
        ax.set_title(f"{title} — ISO 12913-3 circumplex "
                     f"({doc['scale']}, {doc['quadrant']})",
                     loc="left", fontsize=10)
        ax.legend(loc="lower left", frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    return Path(out_path)


def run_survey(folder: str | Path, responses_csv: str | Path,
               out_dir: str | Path | None = None) -> dict:
    """CLI driver: response CSV → ``survey.json`` + ``survey.png`` +
    ``srv_`` keys in ``summary.json``.

    ``folder`` is the session folder (no audio is read); output goes to
    ``out_dir`` (default ``<folder>/analysis``). An existing acoustic
    ``summary.json`` gains the ``srv_`` keys and contributes the
    perception-vs-measurement rows (``doc["vs_measurement"]``); without
    one, a summary carrying only the ``srv_`` keys is written so the
    session still joins the catalog.
    """
    folder = Path(folder)
    out = Path(out_dir) if out_dir else folder / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    doc = summarize(read_responses(responses_csv))
    sp = out / "summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    rows = vs_measurement(doc, summary)
    if rows:
        doc["vs_measurement"] = rows
    summary.update(survey_summary_keys(doc))
    sp.write_text(json.dumps(summary, indent=2))
    (out / "survey.json").write_text(json.dumps(doc, indent=2))
    render(doc, out / "survey.png", title=folder.name)
    return doc
