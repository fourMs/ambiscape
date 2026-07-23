# Recording capture & documentation pipeline — design

*2026-07-23. A systematic, student-friendly pipeline for documenting new
soundscape recordings so `ambiscape` extracts the maximum information from
each session. Supersedes the freeform-`text.md` habit and the un-ingested
prose `notes.md` described in the database `PROTOCOL.md`.*

## Problem

Today a recording enters the corpus as WAV file(s) plus a freeform text note
(e.g. Bodø's `text.md`: "People, kitchen noise"). Two gaps:

1. **Session context is never machine-read.** Place, indoor/outdoor, device,
   mic bearing, clock offset, expected sources, interventions, privacy — the
   things only the recordist knows and no algorithm can recover — live in
   prose that `ambiscape` ignores. It never reaches the README, `summary.json`,
   or the corpus catalog, so it can't be filtered or compared.
2. **No single documented flow.** The sound-object annotation path
   (`draft` → curate `annotations.json` → `taxonomy`) already exists but is
   not tied into a canonical, repeatable pipeline that a student can follow.

The manifest must be **robust to mistakes** and usable by students: a typo
must never crash an analysis or lose data.

## Goal

One documented pipeline, copy-in to committed analysis, with a structured but
forgiving session form that flows into every downstream artifact.

## The pipeline (per recording)

```
1. copy WAV(s)            → YYYY-MM-DD_Place-descriptor/
2. ambiscape init <f>     → scaffolds session.txt, pre-filled from the audio
3. (human) fill session.txt
4. ambiscape check <f>    → friendly validation (optional, warn-only)
5. ambiscape analyze <f>  → features, figures, README (+ Context), summary.json
6. ambiscape draft <f>    → annotations.draft.json (events + ML tags)
7. (human) curate         → annotations.json  (name the sound-objects)
8. ambiscape taxonomy <f> → Schaeffer map + Schafer timeline
9. ambiscape catalog .    → corpus table, now with context columns
```

Steps 2, 4, and the Context/catalog ingest are new. Steps 6–8 already exist
and are simply documented into the flow.

## `session.txt` — the hybrid form

A flat, one-key-per-line form. No indentation, no nesting, no braces — a form,
not a data structure. Structured fields on top; a free-text `observations`
tail at the bottom; `interventions` as timestamped free lines. `#` comments.

```
# Fill in what you know; leave the rest blank. Lines starting with # are ignored.
city:               Bodø
country:            NO
venue:              café
room:               main room
gps:
environment:        indoor          # indoor / outdoor / mixed
device:             Samsung S26
format:             stereo          # (auto-checked against the file)
mic_position:       on table
mic_height_m:       0.9
x_plus_bearing_deg:
timezone:           Europe/Oslo
clock_offset_s:     0
expected_sources:   voices, kitchen clatter, coffee machine
people_present:     Anne, Max
weather:
speech_expected:    yes
publishable:        no
spl_ref:                            # e.g. "62 dBSPL @ HVAC" if metered
intent:             café ground for the book

# --- interventions (one per line:  TIME  what happened) ---
# 12:45:10  espresso grinder on

# --- observations (free text below) ---
People, kitchen noise. Busy lunch service.
```

### Field set

| Field | Type | Flows to summary/catalog? |
|---|---|---|
| city, country, venue, room | text | yes (place) |
| gps | text (lat,lon) | summary only |
| environment | enum indoor/outdoor/mixed | yes |
| device | text | yes |
| format | enum, auto-verified vs file | yes (with mismatch warning) |
| mic_position, mic_height_m, x_plus_bearing_deg | text/number | README only |
| timezone, clock_offset_s | text/number | summary only |
| expected_sources | comma list | summary only |
| people_present | text | README only |
| weather | text | README only |
| speech_expected, publishable | yes/no | summary (gates deposit/speechgate advice) |
| spl_ref | text | README only |
| intent | text | README only |
| interventions | timestamped lines | README (+ hints for annotation) |
| observations | free text tail | README |

All fields optional. Unknown keys warned + ignored. Blank = not provided.

## ambiscape changes

### New: `io.load_manifest(folder) -> dict`
Lenient hand-written parser (no YAML dependency for the form):
- Split on the first `:` per line; trim; strip inline `# …` comments on
  structured lines (not in the observations tail).
- Case-insensitive, whitespace-tolerant keys. Comma-lists → list.
- Recognises the `interventions` and `observations` section markers; captures
  everything under `observations` verbatim (colons allowed there).
- Unknown key → append to a returned `warnings` list, keep going.
- Fallback: if no `session.txt`, read legacy `text.md`/`notes.md` into
  `observations` so old sessions still work.
- Never raises on malformed content; returns best-effort dict + warnings.

### New: `ambiscape init <folder>`
Scaffold `session.txt` pre-filled from audio/BWF metadata (date, duration,
detected channel `format`), every field present with its comment. Non-
destructive: refuses to overwrite an existing `session.txt` (`--force` to
replace). If a `text.md`/`notes.md` exists, migrate its body into the
`observations` tail.

### New: `ambiscape check <folder>`
Run `load_manifest`, print plain-language results: `✓ session form OK (N
fields)`, `⚠ line K: "<key>" not a known field — did you mean "<x>"?
(ignored)` (Levenshtein suggestion), and a `format`-vs-file mismatch warning. Also
prints a **completeness nudge**: `⚠ 3 of 18 fields filled — is that
intentional?` (a gentle prompt for students, never an error).
**Warn-only: exit 0 always.** No commit gating in this version.

### `analyze` ingest
Call `load_manifest`; pass a **Context** block to `report.write_readme`
(place, environment, device, mic, expected sources, interventions,
observations) rendered above the descriptor table; copy the queryable fields
into `summary.json` under a `context` key. Print manifest warnings. A missing
or broken form never fails the analysis.

### `catalog` columns
Read `context` from each `summary.json`; add `place`, `environment`,
`device_format` columns to `catalog.csv`/`.md` (blank where absent) so the
corpus is filterable.

### Folded-in: README descriptor help (from the prior mini-design)
`report.py` gains a `DESCRIPTOR_HELP` dict → a third **"How to read it"**
column in the Descriptors table + an intro link to
`https://fourms.github.io/ambiscape/guide/descriptors/`. Fix the dead
`../ambiscape/` footer link → the GitHub URL. `docs/guide/descriptors.md`
gains a **Descriptor glossary** table (every `summary.json` key → one-line
meaning → domain-guide link).

### Docs
Rewrite the database `PROTOCOL.md` around the 9-step pipeline (init/check/
form replace the prose `notes.md`). Add an ambiscape docs page
`guide/capturing.md` documenting the form + pipeline, linked in `mkdocs.yml`.

## Non-goals (YAGNI)

- No GUI/TUI form editor — a text form the student edits.
- No strict schema validation or required fields — warn-only.
- No pre-commit hook this version (chosen: warn-only).
- No new dependency (custom parser, not a YAML/TOML lib) for the form.
- No migration of the whole back-catalogue's context by hand; `init` is
  available if the user wants to backfill a session.

## Testing (test-first)

- `test_manifest.py`: lenient parse — valid form; misspelled key → warning +
  ignored; missing value → None; comma-list → list; observations tail with
  colons preserved; malformed lines never raise; `text.md` fallback.
- `test_init.py`: scaffolds all fields; pre-fills `format` from a fixture
  WAV; refuses to overwrite without `--force`; migrates `text.md` body.
- `test_check.py`: OK path; did-you-mean suggestion; format-vs-file mismatch;
  always exit 0.
- `test_report.py` (new): Context section renders from a manifest; Descriptors
  table has the "How to read it" column; every `TABLE_ROWS` key has a
  `DESCRIPTOR_HELP` entry; footer link is the GitHub URL.
- `catalog` test: context columns present, blank when manifest absent.

## Build order

1. `report.py` descriptor help + docs glossary (self-contained; the pending
   mini-design).
2. `io.load_manifest` + `test_manifest`.
3. `ambiscape init` + `test_init`.
4. `analyze` Context ingest + README Context section + `summary.json.context`.
5. `ambiscape check` + `test_check`.
6. `catalog` context columns.
7. Docs: `PROTOCOL.md` rewrite, `guide/capturing.md`, mkdocs nav; version
   bump + changelog. Pull --rebase before pushing (repo has other
   contributors).
