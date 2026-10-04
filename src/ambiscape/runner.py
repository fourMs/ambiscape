"""One recording, one folder, one command: the per-session chain.

A session folder holds the takes and a ``session.json`` that says where and
how they were recorded. ``ambiscape run FOLDER`` then runs the whole chain
on it and leaves everything in the folder: the ambiscape passes
(``analyze``, ``draft``, ``anthrophony``, ``mechanical``, ``enf``,
``background --excerpt``), BirdNET where a location is given, the speech
gate, ISO psychoacoustics, frame-wise AudioSet tags, and a post stage that
writes ``energy.json``, ``tag_groups.json``, ``tag_timeline.png``,
``split.json`` (handled ends against the settled recording),
``supply.json`` (power-supply pickup, B-format only), ``tables.json`` and
listening clips. A ``REPORT.template.md`` in the folder, if present, is filled
from ``analysis/`` into ``REPORT.md`` (:func:`ambiscape.report.fill_template`).

The ambiscape passes and the machine-learning stages each run in their own
process (``python -m ambiscape ...``): tensorflow (BirdNET) imported before
torch hides the GPU from it, and a pass that fails should not take the rest
with it. Each step prints a ``### time step`` line, so a log of the run reads
as its record.

``session.json`` keys (``ambiscape init FOLDER`` writes a skeleton):

- ``title``, ``place``, ``device``, ``power``, ``orientation``,
  ``clock_note``: free text for the report and the README.
- ``notes``: passed to ``analyze --notes``.
- ``lat``, ``lon``: BirdNET's location filter; leave ``lat`` empty to skip
  BirdNET (an aircraft cabin, a corpus without a place).
- ``handling_s``: seconds at each end resolved as handling (default 60;
  ``resolve`` drops a state shorter than 30 s, so 15 is the least for a
  short clip).
- ``excerpt_s``: length of the ``background --excerpt`` minute (default 60).
- ``listen``: ``[[clock, seconds, label], ...]`` spans exported as stereo
  previews to ``listen/``.
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

CONFIG_NAME = "session.json"

SKELETON = {
    "title": "",
    "place": "",
    "device": "",
    "power": "",
    "orientation": "not recorded, so bearings are relative to the recorder",
    "clock_note": "",
    "notes": "",
    "lat": "",
    "lon": "",
    "handling_s": 60,
    "excerpt_s": 60,
    "listen": [],
}

STAGES = ("analyze", "birdnet", "speechgate", "iso", "tags", "post", "report")
ML_STAGES = ("birdnet", "speechgate", "tags")


def init_config(folder) -> Path:
    """Write a ``session.json`` skeleton into ``folder``; refuse to overwrite."""
    p = Path(folder) / CONFIG_NAME
    if p.exists():
        raise FileExistsError(f"{p} exists")
    p.write_text(json.dumps(SKELETON, indent=1, ensure_ascii=False) + "\n")
    return p


def load_config(folder) -> dict:
    """``session.json`` merged over the skeleton's defaults."""
    p = Path(folder) / CONFIG_NAME
    cfg = dict(SKELETON)
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg


def _log(*a):
    print(f"### {_dt.datetime.now():%Y-%m-%d %H:%M:%S}", *a, flush=True)


def _amb(*args) -> int:
    """Run an ambiscape subcommand in its own process."""
    _log("ambiscape", *args)
    r = subprocess.run([sys.executable, "-m", "ambiscape", *map(str, args)])
    if r.returncode not in (0, 2):              # speechgate returns 2 on FAIL
        print(f"!!! failed ({r.returncode}): ambiscape {' '.join(map(str, args))}", flush=True)
    return r.returncode


def _save(a: Path, name: str, obj) -> None:
    (a / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))


def session_tables(sess, split: dict | None, groups: dict | None) -> dict:
    """Markdown tables for a report: the takes, the per-state descriptors
    of ``split.json``, and the tag-group counts of ``tag_groups.json``."""
    from .ml import TAG_GROUPS
    from .report import markdown_table
    out = {"takes": markdown_table(
        [{"f": f"`{t.path.name}`", "start": sess.clock(t.start), "dur": round(t.duration / 60, 1),
          "ch": t.channels, "mode": t.mode} for t in sess.takes],
        ["f", "start", "dur", "ch", "mode"],
        ["Take", "Start (local)", "Duration (min)", "Channels", "Mode"])}
    if split:
        keys = [("duration_min", "Duration (min)"), ("laeq_dbfs", "LAeq (dBFS)"),
                ("laeq_trim5_dbfs", "LAeq, loudest 5 % trimmed (dBFS)"), ("L50", "L50 (dBFS)"),
                ("L90", "L90 (dBFS)"), ("dynamics_L10_L90", "L10−L90 (dB)"),
                ("events_per_min", "Events per min"), ("centroid_median_hz", "Centroid, median (Hz)"),
                ("diffuseness_median", "Diffuseness ψ, median"), ("azimuth_mean_deg", "Mean azimuth (°)"),
                ("azimuth_R", "Azimuthal concentration R"),
                ("elevation_fg_median_deg", "Foreground elevation, median (°)"), ("ndsi", "NDSI"),
                ("anthrophony_index", "Anthrophony index"), ("mechanical_index", "Mechanical index")]
        st = split["states"]
        labels = list(st)
        rows = [dict(k=lab, **{s: st[s].get(k) for s in labels}) for k, lab in keys
                if any(st[s].get(k) is not None for s in labels)]
        out["states"] = markdown_table(rows, ["k"] + labels, ["Descriptor"] + labels)
    if groups:
        ev = groups["events"]
        rows = [{"g": g, "n": n, "first": ev.get(g, {}).get("first"), "last": ev.get(g, {}).get("last")}
                for g, n in groups["counts"].items() if n and g in TAG_GROUPS]
        out["tags"] = markdown_table(rows, ["g", "n", "first", "last"], ["Group", "Windows", "First", "Last"])
    return out


def post(folder, cfg: dict | None = None) -> None:
    """The post stage: everything computed from the cached features and tags."""
    import numpy as np
    from . import analysis, enf, figures, io, ml, resolve
    from .features import load_features
    folder = Path(folder)
    cfg = cfg or load_config(folder)
    a = folder / "analysis"
    sess = io.open_session(folder)
    F = load_features(sorted((a / "features").glob("*.npz")))
    h = float(cfg.get("handling_s") or 60)

    _log("energy concentration")
    e = analysis.energy_concentration(F, handling_s=h)
    e["loudest_clock"] = [sess.clock(v)[-8:] for v in e["loudest_s"]]
    _save(a, "energy.json", e)

    groups = None
    if (a / "tags").is_dir() and all((a / "tags" / f"{t.path.stem}.npz").exists() for t in sess.takes):
        _log("tag groups and timeline")
        groups, arrays = ml.tag_groups(sess, a / "tags")
        _save(a, "tag_groups.json", groups)
        np.savez_compressed(a / "tag_groups.npz", **arrays)
        figures.tag_timeline(F, arrays, groups["counts"], a / "tag_timeline.png",
                             threshold=groups["threshold"],
                             title=f"{folder.name}: 50–1000 Hz level and AudioSet tag groups")

    _log("handling against settled")
    st = resolve.handling_states(F, handling_s=h)
    res = resolve.resolve(F, st, min_frames=30)
    split = {"handling_s": h, "intervals": {k: [[round(x, 1), round(y, 1)] for x, y in v] for k, v in st.items()},
             "states": res}
    _save(a, "split.json", split)

    if any(t.channels == 4 for t in sess.takes):
        _log("supply signature")
        takes = {t.path.name: enf.supply_signature(t.audio_path, wyzx=t.wyzx) for t in sess.takes}
        _save(a, "supply.json", {"min_prom_db": enf.SUPPLY_MIN_PROM_DB, "pickup_db": enf.SUPPLY_PICKUP_DB,
                                 "take0": next(iter(takes.values())), "takes": takes})

    _save(a, "tables.json", session_tables(sess, split, groups))
    if cfg.get("listen"):
        _log("listening clips")
        for old in (folder / "listen").glob("*.wav"):
            old.unlink()
        io.listening_clips(sess, cfg["listen"], folder / "listen")


def fill_report(folder) -> Path | None:
    """``REPORT.template.md`` filled from ``analysis/`` into ``REPORT.md``."""
    from .report import fill_template
    folder = Path(folder)
    tpl = folder / "REPORT.template.md"
    if not tpl.exists():
        return None
    _log("report")
    out = folder / "REPORT.md"
    out.write_text(fill_template(tpl.read_text(), folder / "analysis"))
    return out


def run(folder, stages=None, ml: bool = True) -> None:
    """Run ``stages`` (default all of :data:`STAGES`, in that order) on a
    session folder. ``ml=False`` skips the machine-learning stages."""
    folder = Path(folder)
    cfg = load_config(folder)
    stages = [s for s in STAGES if s in (stages or STAGES)]
    if not ml:
        stages = [s for s in stages if s not in ML_STAGES]
    a = folder / "analysis"
    if "analyze" in stages:
        _amb("analyze", folder, "--notes", cfg.get("notes") or "")
        _amb("draft", folder, *(() if ml else ("--max-tags", 0)))
        for cmd in ("anthrophony", "mechanical"):
            _amb(cmd, folder)
        _amb("enf", folder)
        _amb("background", folder, "--excerpt", cfg.get("excerpt_s") or 60)
    if "birdnet" in stages and str(cfg.get("lat", "")).strip():
        _amb("birdnet", folder, "--lat", cfg["lat"], "--lon", cfg["lon"])
    if "speechgate" in stages:
        _amb("speechgate", folder, "--json", a / "speechgate.json")
    if "iso" in stages:
        _amb("iso", folder)
    if "tags" in stages:
        _log("tags")
        subprocess.run([sys.executable, "-c",
                        "import sys; from ambiscape import io, ml; "
                        "ml.tag_session(io.open_session(sys.argv[1]), sys.argv[2])",
                        str(folder), str(a / "tags")])
    if "post" in stages:
        post(folder, cfg)
    if "report" in stages:
        fill_report(folder)
    _log("done", " ".join(stages))
