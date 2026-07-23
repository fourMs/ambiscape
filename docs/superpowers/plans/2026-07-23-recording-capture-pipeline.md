# Recording Capture & Documentation Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give ambiscape a student-friendly capture pipeline: a flat, lenient `session.txt` context form that flows into the README, `summary.json`, and the corpus catalog, plus richer self-explaining descriptor tables.

**Architecture:** A new `manifest` module parses a flat `key: value` form (custom lenient parser, no YAML dependency) with a free-text tail. New `ambiscape init`/`check` CLI commands scaffold and validate it. `analyze` ingests it into a README **Context** section and `ctx_*` keys in `summary.json`; the existing `catalog` union-of-keys logic then surfaces those columns automatically. Separately, `report.py` gains per-descriptor "How to read it" help.

**Tech Stack:** Python 3.12, argparse CLI, stdlib `difflib`/`re`, `soundfile` (tests), pytest. No new runtime dependencies.

## Global Constraints

- Python 3.12; install editable via `pip install --user --break-system-packages -e .`.
- **No new runtime dependency** — the form parser is hand-written (stdlib only).
- **Warn-only, never block:** a malformed/missing `session.txt` must never raise or fail an `analyze`/`catalog` run; parse errors become warnings.
- Repo has other active contributors: **`git fetch` + rebase on `origin/main` before pushing**; work stays on branch `capture-pipeline`.
- Follow existing test conventions: fixtures via `tests/conftest.py` `write_bwf`, or `soundfile.write` for plain WAVs (see `tests/test_pipeline_modes.py`).
- Commit message trailer on every commit:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Version bumps to `0.16.0` land in BOTH `pyproject.toml:7` and `src/ambiscape/__init__.py:23`.

---

## File Structure

- **Create** `src/ambiscape/manifest.py` — the whole form: field set, lenient parser, scaffold text, check report, README/summary projections.
- **Create** `tests/test_manifest.py` — parser, scaffold, check, projections.
- **Create** `tests/test_report.py` — README generator (help column, context, footer).
- **Modify** `src/ambiscape/report.py` — `DESCRIPTOR_HELP`, "How to read it" column, `context` param, footer link.
- **Modify** `src/ambiscape/cli.py` — `init`/`check` subparsers + dispatch; `analyze` manifest ingest.
- **Modify** `docs/guide/descriptors.md` — descriptor glossary.
- **Create** `docs/guide/capturing.md` + **Modify** `mkdocs.yml` — pipeline docs.
- **Modify** (non-git, in the database dir) `PROTOCOL.md` — rewrite around the pipeline.

---

## Task 1: README descriptor help + docs glossary

**Files:**
- Modify: `src/ambiscape/report.py`
- Test: `tests/test_report.py` (create)
- Modify: `docs/guide/descriptors.md`

**Interfaces:**
- Produces: `report.DESCRIPTOR_HELP: dict[str, str]` (help line per `TABLE_ROWS` key); `report.write_readme(...)` renders a 3-column Descriptors table and a `github.com/fourMs/ambiscape` footer link.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report.py`:

```python
"""The generated README: descriptor help column, footer link, coverage."""
from pathlib import Path
from types import SimpleNamespace

from ambiscape import report
from ambiscape.report import TABLE_ROWS, DESCRIPTOR_HELP


def _sess(tmp_path):
    tk = SimpleNamespace(path=Path("260706_124224.wav"), date="2026-07-06",
                         clock="12:42:24", duration=114.5, mode="stereo")
    return SimpleNamespace(folder=tmp_path, name="test-session", takes=[tk])


def _summary():
    return {"duration_min": 1.9, "leq_dbfs": -23.0, "diffuseness_median": 0.3}


def test_every_descriptor_has_help():
    missing = [k for k, _ in TABLE_ROWS if k not in DESCRIPTOR_HELP]
    assert missing == [], f"missing help for {missing}"


def test_readme_has_help_column_and_github_footer(tmp_path):
    report.write_readme(_sess(tmp_path), _summary(), tmp_path)
    md = (tmp_path / "README.md").read_text()
    assert "| Descriptor | Value | How to read it |" in md
    assert "github.com/fourMs/ambiscape" in md
    assert "../ambiscape/" not in md
    # a known help string is rendered next to its value
    assert DESCRIPTOR_HELP["diffuseness_median"] in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/alexanje/github/ambiscape && python -m pytest tests/test_report.py -v`
Expected: FAIL — `ImportError: cannot import name 'DESCRIPTOR_HELP'`.

- [ ] **Step 3: Add `DESCRIPTOR_HELP` and update the table + footer in `report.py`**

Add this dict directly below the `TABLE_ROWS` list (after line 43):

```python
DESCRIPTOR_HELP = {
    "duration_min": "Total recording length.",
    "leq_dbfs": "Energy-average level — the overall loudness.",
    "laeq_dbfs": "A-weighted Leq; approximates perceived loudness (de-emphasises deep bass).",
    "leq_minus_laeq_db": "Gap between the two — large = bass-heavy energy (traffic, machines, music).",
    "L10": "Level exceeded 10% of the time — the loud moments.",
    "L50": "Median level.",
    "L90": "Level exceeded 90% of the time — the quiet background floor.",
    "dynamics_L10_L90": "Spread of loud vs quiet — small = steady, large = dynamic.",
    "events_per_min": "How often the level jumps at least 8 dB above the background.",
    "event_median_dur_s": "Typical duration of those level events.",
    "intermittency_ratio_pct": "Share of energy in brief events vs the steady floor — high = spiky/impulsive.",
    "emergence_db": "How far foreground events rise above the floor — low = masked / 'camouflaged'.",
    "fg_fraction_median": "Typical share of the spectrum sitting above its own background.",
    "fg_fraction_p90": "Busiest-moment version of the same.",
    "spectral_events_per_min": "Rate of per-band spectral events (timbral changes).",
    "spectral_event_median_dur_s": "Typical duration of those spectral events.",
    "centroid_median_hz": "Spectral centre of gravity — low = dark/rumbly, high = bright/hissy.",
    "flatness_median": "Tonal vs noisy — near 0 = tonal (hums, pitches), near 1 = noise-like.",
    "diffuseness_median": "psi: 0 = one clear direction (point source), 1 = enveloping / diffuse field.",
    "diffuseness_iqr": "How much psi varies over time.",
    "azimuth_mean_deg": "Dominant bearing of arriving energy (mic frame: 0 deg = front, +90 = left).",
    "azimuth_R": "Directional concentration — 0 = from everywhere, 1 = all from one bearing.",
    "elevation_fg_median_deg": "Median height of the loudest moments (+ = above the mic).",
    "directional_entropy": "Spread of energy across bearings — low = focused, high = all around.",
    "above_horizon_fraction": "Share of energy arriving from above +10 deg elevation.",
    "fgbg_az_overlap": "How much foreground and background share directions — low = different places.",
    "ndsi": "Soundscape index: -1 = human/mechanical band dominates, +1 = biophony band.",
    "adi": "Acoustic Diversity — evenness of energy across frequency bands.",
    "aci": "Acoustic Complexity — amount of intensity fluctuation (bird/animal-activity proxy).",
    "acoustic_entropy": "Overall unpredictability of the spectrum (0-1).",
    "bird_peaks_per_min": "Narrowband tonal peaks in the bird band per minute (indoors: often whistles/music).",
    "bird_band_activity_pct": "Share of time the bird band is active (indoors, a false-positive proxy).",
    "bird_temporal_entropy": "Timing structure of bird-band activity — low = structured/rhythmic.",
    "bird_directional_entropy": "Directional spread of bird-band energy.",
    "bird_above_horizon_fraction": "Share of bird-band energy arriving from above +10 deg.",
}
```

In `write_readme`, replace the Descriptors block (currently lines 145-150):

```python
    lines += ["", _recording_note(sess),
              "", "## Descriptors", "",
              "Session-level summary values. See the "
              "[descriptors guide](https://fourms.github.io/ambiscape/guide/descriptors/) "
              "for full definitions.", "",
              "| Descriptor | Value | How to read it |", "|---|---|---|"]
    for key, label in TABLE_ROWS:
        v = summary.get(key)
        if v is not None:
            lines.append(f"| {label} | {v} | {DESCRIPTOR_HELP.get(key, '')} |")
```

Replace the footer (currently lines 160-162):

```python
    lines += ["---", "*Analyzed with "
              "[ambiscape](https://github.com/fourMs/ambiscape) "
              "(streaming companion to "
              "[ambiviz](https://github.com/fisheggg/ambiviz)).*", ""]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_report.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Add the glossary to `docs/guide/descriptors.md`**

Append this section to the end of the file:

```markdown
## Descriptor glossary

Every value in a session's `summary.json` and README, in one line. The
source/ecology indices have fuller treatments on their own pages.

| Key | Meaning |
|---|---|
| `leq_dbfs` / `laeq_dbfs` | Energy-mean level, unweighted and A-weighted (perceptual). |
| `L10` / `L50` / `L90` | Levels exceeded 10 / 50 / 90 % of the time (loud / median / floor). |
| `dynamics_L10_L90` | L10 − L90 spread; steadiness of the level. |
| `events_per_min`, `event_median_dur_s` | Rate and duration of ≥ 8 dB level events. |
| `intermittency_ratio_pct` | Energy fraction in brief events vs the steady floor. |
| `emergence_db` | How far foreground rises above the background floor (masking). |
| `fg_fraction_median` / `_p90` | Spectral foreground share, typical / busiest. |
| `centroid_median_hz`, `flatness_median` | Spectral brightness and tonal-vs-noise. |
| `diffuseness_median` / `_iqr` | ψ: point-source ↔ diffuse field, and its spread. See [Spatial analysis](spatial.md). |
| `azimuth_mean_deg`, `azimuth_R`, `directional_entropy` | Dominant bearing, its concentration, and directional spread. |
| `ndsi`, `adi`, `aci`, `acoustic_entropy` | Global soundscape indices. See [Ratings & global indices](indices.md). |
| `bird_*` | Biophony structural + spatial measures (indoors, often a music/whistle false positive). See [Biophony](biophony.md). |
| `ctx_place`, `ctx_environment`, `ctx_device_format` | Session context from `session.txt`. See [Capturing recordings](capturing.md). |
```

- [ ] **Step 6: Commit**

```bash
cd /home/alexanje/github/ambiscape
git add src/ambiscape/report.py tests/test_report.py docs/guide/descriptors.md
git commit -m "feat(report): per-descriptor help column + docs glossary; fix footer link

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Manifest parser (`manifest.py`)

**Files:**
- Create: `src/ambiscape/manifest.py`
- Test: `tests/test_manifest.py` (create)

**Interfaces:**
- Produces:
  - `manifest.FIELDS: list[tuple[str, str]]` (ordered key, comment)
  - `manifest.KNOWN: set[str]`
  - `manifest.parse(text: str) -> tuple[dict, list[str]]` — data has every known key (None if absent) plus `"interventions": list[dict]` and `"observations": str`; second element is warnings.
  - `manifest.load_manifest(folder) -> tuple[dict, list[str]]` — reads `session.txt`, falls back to `text.md`/`notes.md`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_manifest.py`:

```python
"""The session.txt capture form: lenient parse, fallbacks, projections."""
from ambiscape import manifest


FORM = """\
# a comment
city:            Bodø
environment:     indoor      # indoor / outdoor / mixed
expected_sources: voices, kitchen clatter, coffee machine
mic_height_m:
environmnet:     indoor
just a stray line
12:45:10  espresso grinder on

# --- observations (free text below) ---
People, kitchen noise. Ratio 3:1 busy.
"""


def test_parse_known_fields():
    data, warn = manifest.parse(FORM)
    assert data["city"] == "Bodø"
    assert data["environment"] == "indoor"          # inline comment stripped
    assert data["expected_sources"] == ["voices", "kitchen clatter", "coffee machine"]
    assert data["mic_height_m"] is None             # blank value


def test_typo_and_stray_line_warn_not_raise():
    data, warn = manifest.parse(FORM)
    joined = " ".join(warn)
    assert "environmnet" in joined and "environment" in joined   # did-you-mean
    assert any("stray" in w or "key: value" in w for w in warn)


def test_interventions_and_observations():
    data, warn = manifest.parse(FORM)
    assert data["interventions"] == [{"time": "12:45:10", "action": "espresso grinder on"}]
    assert data["observations"].startswith("People, kitchen noise")
    assert "3:1" in data["observations"]            # colons preserved in the tail


def test_empty_text_never_raises():
    data, warn = manifest.parse("")
    assert data["observations"] == ""
    assert data["interventions"] == []


def test_load_manifest_text_md_fallback(tmp_path):
    (tmp_path / "text.md").write_text("# note\nPeople, kitchen noise\n")
    data, warn = manifest.load_manifest(tmp_path)
    assert "People, kitchen noise" in data["observations"]
    assert any("text.md" in w for w in warn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ambiscape.manifest'`.

- [ ] **Step 3: Create `src/ambiscape/manifest.py`**

```python
"""The ``session.txt`` capture form.

A flat, forgiving ``key: value`` form (no indentation, no nesting) plus a
free-text ``observations`` tail and timestamped ``interventions`` lines.
Parsed by a hand-written tolerant reader — a typo is a warning, never an
error, and never blocks an analysis.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

# Ordered (key, scaffold-comment). This set is also the known-field set.
FIELDS: list[tuple[str, str]] = [
    ("city", ""),
    ("country", ""),
    ("venue", ""),
    ("room", ""),
    ("gps", "lat,lon"),
    ("environment", "indoor / outdoor / mixed"),
    ("device", ""),
    ("format", "(auto-checked against the file)"),
    ("mic_position", ""),
    ("mic_height_m", ""),
    ("x_plus_bearing_deg", ""),
    ("timezone", ""),
    ("clock_offset_s", ""),
    ("expected_sources", "comma-separated"),
    ("people_present", ""),
    ("weather", ""),
    ("speech_expected", "yes / no"),
    ("publishable", "yes / no"),
    ("spl_ref", 'e.g. "62 dBSPL @ HVAC" if metered'),
    ("intent", ""),
]
KNOWN = {k for k, _ in FIELDS}
LIST_FIELDS = {"expected_sources"}

_TIME = re.compile(r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s+(\S.*)$")
_OBS_MARK = re.compile(r"observations", re.I)


def _empty() -> dict:
    data = {k: None for k, _ in FIELDS}
    data["interventions"] = []
    data["observations"] = ""
    return data


def _closest(key: str) -> str | None:
    m = difflib.get_close_matches(key, KNOWN, n=1, cutoff=0.7)
    return m[0] if m else None


def parse(text: str) -> tuple[dict, list[str]]:
    """Parse form text. Returns (data, warnings); never raises."""
    data = _empty()
    warnings: list[str] = []
    obs_lines: list[str] = []
    in_obs = False
    for ln in text.splitlines():
        if in_obs:
            obs_lines.append(ln)
            continue
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if _OBS_MARK.search(stripped):
                in_obs = True
            continue
        m = _TIME.match(ln)
        if m:
            data["interventions"].append(
                {"time": m.group(1), "action": m.group(2).strip()})
            continue
        content = re.split(r"\s#", ln, maxsplit=1)[0]   # strip " # inline comment"
        if ":" not in content:
            warnings.append(f"ignored stray line (no 'key: value'): "
                            f"{stripped[:40]!r}")
            continue
        key, _, val = content.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if key not in KNOWN:
            sugg = _closest(key)
            hint = f' — did you mean "{sugg}"?' if sugg else ""
            warnings.append(f'unknown field "{key}"{hint} (ignored)')
            continue
        if val == "":
            data[key] = None
        elif key in LIST_FIELDS:
            data[key] = [p.strip() for p in val.split(",") if p.strip()]
        else:
            data[key] = val
    data["observations"] = "\n".join(obs_lines).strip()
    return data, warnings


def load_manifest(folder: str | Path) -> tuple[dict, list[str]]:
    """Read ``<folder>/session.txt``; fall back to legacy ``text.md``/``notes.md``."""
    folder = Path(folder)
    p = folder / "session.txt"
    if p.exists():
        return parse(p.read_text())
    for name in ("notes.md", "text.md"):
        q = folder / name
        if q.exists():
            data = _empty()
            data["observations"] = q.read_text().strip()
            return data, [f"no session.txt — read legacy {name} into observations"]
    return _empty(), ["no session.txt found"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ambiscape/manifest.py tests/test_manifest.py
git commit -m "feat(manifest): lenient session.txt form parser

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `ambiscape init` (scaffold)

**Files:**
- Modify: `src/ambiscape/manifest.py`
- Modify: `src/ambiscape/cli.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `manifest.FIELDS`.
- Produces: `manifest.scaffold_text(fmt: str | None = None, observations: str = "") -> str`; CLI `ambiscape init <folder> [--force]` writing `<folder>/session.txt`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_manifest.py`:

```python
import numpy as np
import soundfile as sf
from ambiscape.cli import main as cli_main


def test_scaffold_has_all_fields_and_prefilled_format():
    txt = manifest.scaffold_text(fmt="stereo", observations="hello")
    for k, _ in manifest.FIELDS:
        assert f"{k}:" in txt
    assert "format:" in txt and "stereo" in txt
    assert txt.rstrip().endswith("hello")
    # round-trips through the parser
    data, warn = manifest.parse(txt)
    assert data["format"] == "stereo"
    assert data["observations"] == "hello"


def test_init_writes_form_and_migrates_text_md(tmp_path):
    sf.write(str(tmp_path / "260706_124224.wav"),
             np.zeros((4800, 2), np.float32), 48000, subtype="PCM_16")
    (tmp_path / "text.md").write_text("People, kitchen noise\n")
    rc = cli_main(["init", str(tmp_path)])
    assert rc == 0
    form = (tmp_path / "session.txt").read_text()
    assert "format:" in form and "stereo" in form
    assert "People, kitchen noise" in form


def test_init_refuses_overwrite_without_force(tmp_path):
    (tmp_path / "session.txt").write_text("city: Oslo\n")
    rc = cli_main(["init", str(tmp_path)])
    assert rc == 1
    assert (tmp_path / "session.txt").read_text() == "city: Oslo\n"
    rc = cli_main(["init", str(tmp_path), "--force"])
    assert rc == 0
    assert "city:" in (tmp_path / "session.txt").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_manifest.py -k "scaffold or init" -v`
Expected: FAIL — `AttributeError: module 'ambiscape.manifest' has no attribute 'scaffold_text'`.

- [ ] **Step 3: Add `scaffold_text` to `manifest.py`**

Append to `src/ambiscape/manifest.py`:

```python
def scaffold_text(fmt: str | None = None, observations: str = "") -> str:
    """Render a blank, fully-labelled session.txt (format pre-filled if known)."""
    out = ["# Fill in what you know; leave the rest blank. "
           "Lines starting with # are ignored."]
    for k, comment in FIELDS:
        val = fmt if (k == "format" and fmt) else ""
        pad = " " * max(1, 20 - len(k) - 1)
        tail = f"   # {comment}" if comment else ""
        out.append(f"{k}:{pad}{val}{tail}")
    out += ["",
            "# --- interventions (one per line:  TIME  what happened) ---",
            "# 12:45:10  espresso grinder on",
            "",
            "# --- observations (free text below) ---",
            observations.strip()]
    return "\n".join(out) + "\n"
```

- [ ] **Step 4: Add the `init` subparser and dispatch in `cli.py`**

After the `draft` subparser block (ends at cli.py:41), add:

```python
    ini = sub.add_parser("init",
                         help="scaffold a session.txt capture form "
                              "(pre-filled from the audio)")
    ini.add_argument("folder")
    ini.add_argument("--force", action="store_true",
                     help="overwrite an existing session.txt")
    chk = sub.add_parser("check",
                         help="validate session.txt (warn-only)")
    chk.add_argument("folder")
```

Before the `if args.cmd == "deposit":` block (cli.py:249), add:

```python
    if args.cmd == "init":
        from . import manifest as mf
        folder = Path(args.folder)
        dest = folder / "session.txt"
        if dest.exists() and not args.force:
            print(f"{dest} exists — use --force to overwrite")
            return 1
        fmt = None
        try:
            from .io import open_session
            s = open_session(folder)
            fmt = getattr(s.takes[0], "mode", None) if s.takes else None
        except Exception:
            pass
        obs = ""
        for name in ("text.md", "notes.md"):
            q = folder / name
            if q.exists():
                obs = q.read_text().strip()
                break
        dest.write_text(mf.scaffold_text(fmt=fmt, observations=obs))
        print(f"wrote {dest}" + (f" (format: {fmt})" if fmt else ""))
        return 0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: PASS (8 tests).

- [ ] **Step 6: Commit**

```bash
git add src/ambiscape/manifest.py src/ambiscape/cli.py tests/test_manifest.py
git commit -m "feat(cli): ambiscape init scaffolds a pre-filled session.txt

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `ambiscape check` (validate, warn-only)

**Files:**
- Modify: `src/ambiscape/manifest.py`
- Modify: `src/ambiscape/cli.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `manifest.parse`, `manifest.KNOWN`.
- Produces: `manifest.check_lines(folder, fmt_actual: str | None = None) -> list[str]`; CLI `ambiscape check <folder>` printing them, always exit 0.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_manifest.py`:

```python
def test_check_lines_reports_typo_and_completeness(tmp_path):
    (tmp_path / "session.txt").write_text(
        "city: Bodø\nenvironmnet: indoor\n")
    lines = manifest.check_lines(tmp_path)
    blob = "\n".join(lines)
    assert "1 of" in blob                      # one real field filled
    assert 'did you mean "environment"' in blob


def test_check_lines_format_mismatch(tmp_path):
    (tmp_path / "session.txt").write_text("format: ambix\n")
    lines = manifest.check_lines(tmp_path, fmt_actual="stereo")
    assert any("ambix" in l and "stereo" in l for l in lines)


def test_check_nudges_on_empty_form(tmp_path):
    (tmp_path / "session.txt").write_text(manifest.scaffold_text())
    lines = manifest.check_lines(tmp_path)
    assert any("0 fields filled" in l for l in lines)


def test_check_cli_always_exit_zero(tmp_path, capsys):
    (tmp_path / "session.txt").write_text("environmnet: indoor\n")
    rc = cli_main(["check", str(tmp_path)])
    assert rc == 0
    assert "did you mean" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_manifest.py -k check -v`
Expected: FAIL — `AttributeError: ... has no attribute 'check_lines'`.

- [ ] **Step 3: Add `check_lines` to `manifest.py`**

Append to `src/ambiscape/manifest.py`:

```python
def check_lines(folder: str | Path, fmt_actual: str | None = None) -> list[str]:
    """Plain-language validation of a session.txt. Warn-only; empty-friendly."""
    folder = Path(folder)
    p = folder / "session.txt"
    if not p.exists():
        return [f'no session.txt in {folder} — run "ambiscape init {folder}"']
    data, warnings = parse(p.read_text())
    filled = sum(1 for k in KNOWN if data.get(k) not in (None, "", []))
    out = [f"session form: {filled} of {len(KNOWN)} fields filled"]
    out += [f"⚠ {w}" for w in warnings]
    fmt = data.get("format")
    if fmt_actual and fmt and fmt != fmt_actual:
        out.append(f'⚠ format says "{fmt}" but the file is {fmt_actual} '
                   f"— check device/format")
    if filled == 0:
        out.append("⚠ 0 fields filled — is that intentional?")
    return out
```

- [ ] **Step 4: Add the `check` dispatch in `cli.py`**

Immediately after the `init` dispatch block added in Task 3, add:

```python
    if args.cmd == "check":
        from . import manifest as mf
        fmt = None
        try:
            from .io import open_session
            s = open_session(args.folder)
            fmt = getattr(s.takes[0], "mode", None) if s.takes else None
        except Exception:
            pass
        for line in mf.check_lines(args.folder, fmt_actual=fmt):
            print(line)
        return 0
```

(The `check` subparser was already added alongside `init` in Task 3.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: PASS (12 tests).

- [ ] **Step 6: Commit**

```bash
git add src/ambiscape/manifest.py src/ambiscape/cli.py tests/test_manifest.py
git commit -m "feat(cli): ambiscape check validates session.txt, warn-only

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `analyze` ingest — README Context + summary.json + catalog columns

**Files:**
- Modify: `src/ambiscape/manifest.py`
- Modify: `src/ambiscape/report.py`
- Modify: `src/ambiscape/cli.py:748` (analyze dispatch)
- Test: `tests/test_manifest.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: `manifest.load_manifest`, `report.write_readme`.
- Produces: `manifest.context_summary_fields(data) -> dict` (`ctx_place`, `ctx_environment`, `ctx_device_format`); `manifest.context_readme(data) -> str`; `report.write_readme(..., context: str = "")` inserts the Context block.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_manifest.py`:

```python
def test_context_projections():
    data, _ = manifest.parse(
        "venue: café\ncity: Bodø\ncountry: NO\nenvironment: indoor\n"
        "format: stereo\nintent: café ground\n")
    fields = manifest.context_summary_fields(data)
    assert fields["ctx_place"] == "café, Bodø, NO"
    assert fields["ctx_environment"] == "indoor"
    assert fields["ctx_device_format"] == "stereo"
    md = manifest.context_readme(data)
    assert md.startswith("## Context")
    assert "café" in md and "café ground" in md


def test_context_readme_empty_is_blank():
    data, _ = manifest.parse("")
    assert manifest.context_readme(data) == ""
```

Append to `tests/test_report.py`:

```python
def test_readme_renders_context_block(tmp_path):
    ctx = "## Context\n\n| Field | Value |\n|---|---|\n| Place | Bodø |\n"
    report.write_readme(_sess(tmp_path), _summary(), tmp_path, context=ctx)
    md = (tmp_path / "README.md").read_text()
    assert "## Context" in md
    assert md.index("## Context") < md.index("## Descriptors")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_manifest.py -k context tests/test_report.py -v`
Expected: FAIL — `context_summary_fields` missing; `write_readme` has no `context` kwarg.

- [ ] **Step 3: Add the projections to `manifest.py`**

Append to `src/ambiscape/manifest.py`:

```python
def context_summary_fields(data: dict) -> dict:
    """Queryable context fields for summary.json / the catalog (None if absent)."""
    place = ", ".join(x for x in (data.get("venue"), data.get("city"),
                                  data.get("country")) if x)
    return {
        "ctx_place": place or None,
        "ctx_environment": data.get("environment"),
        "ctx_device_format": data.get("format"),
    }


def context_readme(data: dict) -> str:
    """Render a README '## Context' section from a manifest, or '' if empty."""
    rows: list[tuple[str, str]] = []
    place = ", ".join(x for x in (data.get("venue"), data.get("room"),
                                  data.get("city"), data.get("country")) if x)
    _add(rows, "Place", place)
    _add(rows, "Environment", data.get("environment"))
    _add(rows, "Device", data.get("device"))
    _add(rows, "Format", data.get("format"))
    src = data.get("expected_sources")
    _add(rows, "Expected sources", ", ".join(src) if src else None)
    _add(rows, "People present", data.get("people_present"))
    _add(rows, "Weather", data.get("weather"))
    _add(rows, "Mic", _mic_note(data))
    _add(rows, "Clock", _clock_note(data))
    _add(rows, "SPL reference", data.get("spl_ref"))
    _add(rows, "Intent", data.get("intent"))
    ivs = data.get("interventions") or []
    obs = data.get("observations") or ""
    if not rows and not ivs and not obs:
        return ""
    out = ["## Context", ""]
    if rows:
        out += ["| Field | Value |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in rows]
        out.append("")
    if ivs:
        out += ["**Interventions**", ""]
        out += [f"- `{i['time']}` — {i['action']}" for i in ivs]
        out.append("")
    if obs:
        out += ["**Observations**", "", obs, ""]
    return "\n".join(out).rstrip() + "\n"


def _add(rows: list, label: str, val) -> None:
    if val not in (None, "", []):
        rows.append((label, val))


def _mic_note(data: dict) -> str | None:
    bits = []
    if data.get("mic_position"):
        bits.append(str(data["mic_position"]))
    if data.get("mic_height_m"):
        bits.append(f"{data['mic_height_m']} m")
    if data.get("x_plus_bearing_deg"):
        bits.append(f"X+ {data['x_plus_bearing_deg']}°")
    return "; ".join(bits) or None


def _clock_note(data: dict) -> str | None:
    bits = []
    if data.get("timezone"):
        bits.append(str(data["timezone"]))
    if data.get("clock_offset_s"):
        bits.append(f"offset {data['clock_offset_s']} s")
    return "; ".join(bits) or None
```

- [ ] **Step 4: Add the `context` param to `report.write_readme`**

In `src/ambiscape/report.py`, change the signature (line 120-122):

```python
def write_readme(sess: Session, summary: dict, out_dir: Path,
                 notes: str = "", extra: str = "",
                 states: dict | None = None, context: str = "") -> Path:
```

Insert the context block right after the recording-note / before `## Descriptors`. Change the block that currently reads `lines += ["", _recording_note(sess), "", "## Descriptors", ...` so the recording note and Context come first:

```python
    lines += ["", _recording_note(sess), ""]
    if context:
        lines += [context, ""]
    lines += ["## Descriptors", "",
              "Session-level summary values. See the "
              "[descriptors guide](https://fourms.github.io/ambiscape/guide/descriptors/) "
              "for full definitions.", "",
              "| Descriptor | Value | How to read it |", "|---|---|---|"]
```

- [ ] **Step 5: Wire the manifest into the `analyze` dispatch in `cli.py`**

In the analyze block, replace the `report.write_readme(...)` call (cli.py:748-749) with:

```python
    from . import manifest as mf
    mdata, mwarn = mf.load_manifest(sess.folder)
    for w in mwarn:
        print(f"  ⚠ {w}")
    summary.update(mf.context_summary_fields(mdata))
    report.write_readme(sess, summary, out, notes=args.notes,
                        states=states_doc,
                        context=mf.context_readme(mdata))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_manifest.py tests/test_report.py -v`
Expected: PASS (all).

- [ ] **Step 7: Add a catalog column test**

Append to `tests/test_catalog.py`:

```python
def test_catalog_surfaces_context_columns(tmp_path):
    import json
    from ambiscape import catalog
    for name, place in [("s1", "café, Bodø, NO"), ("s2", None)]:
        d = tmp_path / name / "analysis"
        d.mkdir(parents=True)
        summ = {"leq_dbfs": -23.0, "ctx_place": place,
                "ctx_environment": "indoor" if place else None}
        (d / "summary.json").write_text(json.dumps(summ))
    col = catalog.collect(tmp_path)
    csv_path = catalog.to_csv(col, tmp_path / "catalog.csv")
    header = csv_path.read_text().splitlines()[0]
    assert "ctx_place" in header and "ctx_environment" in header
```

- [ ] **Step 8: Run the catalog test and a pipeline regression check**

Run: `python -m pytest tests/test_catalog.py -k context tests/test_pipeline.py -v`
Expected: PASS — context columns present; `analyze` pipeline still green with no manifest.

- [ ] **Step 9: Commit**

```bash
git add src/ambiscape/manifest.py src/ambiscape/report.py src/ambiscape/cli.py \
        tests/test_manifest.py tests/test_report.py tests/test_catalog.py
git commit -m "feat: analyze ingests session.txt into README Context + summary/catalog

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Docs + version bump

**Files:**
- Create: `docs/guide/capturing.md`
- Modify: `mkdocs.yml`
- Modify: `pyproject.toml:7`, `src/ambiscape/__init__.py:23`
- Modify (non-git, database dir): `PROTOCOL.md`

**Interfaces:** none (docs + metadata only).

- [ ] **Step 1: Create `docs/guide/capturing.md`**

```markdown
# Capturing recordings

A repeatable pipeline for documenting a new recording so `ambiscape`
extracts the most from it. Every step is safe to re-run.

```
1. copy WAV(s)          → YYYY-MM-DD_Place-descriptor/
2. ambiscape init <f>   → scaffolds session.txt, pre-filled from the audio
3. (you) fill session.txt
4. ambiscape check <f>  → friendly validation (optional)
5. ambiscape analyze <f>
6. ambiscape draft <f>  → annotations.draft.json
7. (you) curate → annotations.json
8. ambiscape taxonomy <f>
9. ambiscape catalog .
```

## The session.txt form

A flat `key: value` form — one field per line, no indentation, no quoting.
`ambiscape init` writes it fully labelled; you only replace the values.
Unknown or misspelled keys are warned and ignored; a blank value just means
"not provided". Nothing you type here can break an analysis.

Structured fields feed the README **Context** section, `summary.json`
(`ctx_place`, `ctx_environment`, `ctx_device_format`), and the corpus
catalog. A free-text `observations` tail and timestamped `interventions`
lines capture everything else. Run `ambiscape check <folder>` before
committing — it lists missing/typo fields and flags a `format` that
disagrees with the file, but never blocks.
```

- [ ] **Step 2: Add the page to `mkdocs.yml` nav**

Under the `Core:` section, after the `Sessions & conventions` line, add:

```yaml
      - Capturing recordings: guide/capturing.md
```

- [ ] **Step 3: Bump the version**

In `pyproject.toml` line 7: `version = "0.16.0"`.
In `src/ambiscape/__init__.py` line 23: `__version__ = "0.16.0"`.

- [ ] **Step 4: Verify the docs build and the CLI reports the new version**

Run: `python -m mkdocs build --strict 2>&1 | tail -5`
Expected: `INFO - Documentation built` with no warnings.

Run: `python -c "import ambiscape; print(ambiscape.__version__)"`
Expected: `0.16.0`.

- [ ] **Step 5: Commit**

```bash
git add docs/guide/capturing.md mkdocs.yml pyproject.toml src/ambiscape/__init__.py
git commit -m "docs: capture pipeline guide; bump to 0.16.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Rewrite the database `PROTOCOL.md` (non-git file edit)**

This file lives in the Intercontinental database directory (not a git repo),
so it is edited in place and NOT committed. Replace its
"Session notes (one `notes.md`...)" section with:

```markdown
## Session form (one `session.txt` per folder)

Run `ambiscape init "<folder>"` after copying the WAVs — it writes a
pre-filled `session.txt`. Fill in what you know (place, environment, device,
expected sources, interventions, observations), then `ambiscape check
"<folder>"` to catch typos. The old prose `notes.md` is superseded; `init`
migrates any existing `text.md`/`notes.md` into the form's observations tail.
```

Then update the "Folder & file conventions" step 3 to the full pipeline:
`init → fill → check → analyze → draft → curate → taxonomy → catalog`.

- [ ] **Step 7: Push the branch**

```bash
cd /home/alexanje/github/ambiscape
git fetch origin
git rebase origin/main
python -m pytest -q            # full suite green before pushing
git push
```

Expected: all tests pass; branch `capture-pipeline` updated on origin.

---

## Self-Review Notes

- **Spec coverage:** session.txt form + hybrid free tail (T2, T3) ✓; lenient parser, warn-only, text.md fallback (T2) ✓; `init` (T3); `check` with did-you-mean + format mismatch + completeness nudge (T4) ✓; analyze Context ingest + `summary.json.context` (T5) ✓; catalog columns (T5) ✓; README descriptor help + docs glossary + footer fix (T1) ✓; capturing.md + mkdocs + PROTOCOL.md rewrite + version bump (T6) ✓.
- **Deferred by design (YAGNI, per spec non-goals):** no GUI, no required fields, no pre-commit hook, no bulk back-catalogue migration.
- **Note:** `ctx_*` keys land in `summary.json` via `summary.update(...)`; catalog surfaces them through its existing union-of-keys logic — no catalog code change needed, only the regression test in T5.
