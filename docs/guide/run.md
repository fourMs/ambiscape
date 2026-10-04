# One folder per recording

A recording arrives as one or more files; `ambiscape run` turns the folder
that holds them into a finished analysis, without anything else around it.
Put the takes in a folder, describe them in a `session.json`, and run:

```bash
ambiscape init SESSION/          # writes a session.json skeleton
# fill in session.json: place, coordinates, notes, the spans to listen to
ambiscape run SESSION/
```

The folder then holds everything: the ambiscape outputs in `analysis/`, the
session `README.md` that `analyze` writes, listening clips in `listen/`, and,
if a `REPORT.template.md` is present, a `REPORT.md` filled from `analysis/`.
For a recording you keep making, such as a soundscape a day, the folder is the
unit that is copied, moved and deposited; nothing it needs lives elsewhere.

## The stages

`run` goes through seven stages in order. `--stage` picks some of them
(repeatable) and `--no-ml` skips the three that need `ambiscape[ml]`.

| Stage | What it runs | Writes |
|---|---|---|
| `analyze` | `analyze`, `draft`, `anthrophony`, `mechanical`, `enf`, `background --excerpt` | `summary.json`, figures, `README.md`, `annotations.draft.json`, the rest |
| `birdnet` | `birdnet` with `lat`/`lon`, skipped when `lat` is empty | `birdnet.json` |
| `speechgate` | `speechgate` on every take | `speechgate.json` |
| `iso` | `iso` | `iso_indicators.json` |
| `tags` | `ml.tag_session`: AudioSet posteriors every 2 s, one cache per take | `analysis/tags/*.npz` |
| `post` | energy concentration, tag groups and timeline, handled ends against the settled recording, supply pickup (B-format), tables, listening clips | `energy.json`, `tag_groups.json`, `tag_timeline.png`, `split.json`, `supply.json`, `tables.json`, `listen/` |
| `report` | `report.fill_template` on `REPORT.template.md` | `REPORT.md` |

Each pass of the first five stages runs in its own process. This matters
for the machine-learning stages: importing tensorflow (BirdNET) before torch
hides the GPU from torch, and in separate processes neither sees the other. A
failed pass prints a `!!! failed` line and the run continues, so one missing
extra does not cost the rest. The tag stage resumes: a take whose cache exists
is skipped.

A night takes about an hour, most of it in `birdnet`, `speechgate` and `iso`;
run long sessions as a memory-capped service rather than in a terminal you
might close, and send the output to a log, which then reads as the record of
the run.

## session.json

| Key | Meaning |
|---|---|
| `title`, `place`, `device`, `power`, `orientation`, `clock_note` | Free text for the report. |
| `notes` | Passed to `analyze --notes` and printed in the README. |
| `lat`, `lon` | BirdNET's location and season filter; leave `lat` empty to skip BirdNET. |
| `handling_s` | Seconds at each end resolved as handling (default 60). `resolve` drops a state shorter than 30 s, so 15 is the least that works for a short clip. |
| `excerpt_s` | Length of the characteristic excerpt (default 60). |
| `listen` | `[["HH:MM:SS", seconds, "label"], ...]`: spans exported as stereo previews, normalised, the applied gain in the filename. |

The clock in `listen`, and every clock in the outputs, is the session clock:
the recorder's own, moved by `calibration.json` where the recorder was off.
A recorder that kept its home time zone abroad needs a `clock_offset_s` there,
or every time in the report is an hour out.

## Handled ends

The first and last minutes of a recording are usually the recordist setting
the device down and picking it up. In a quiet room one set-down knock can
carry half of the acoustic energy of half an hour, so the session LAeq
describes the knock. `post` therefore resolves the ends as their own state
(`resolve.handling_states`) and writes both into `split.json`, and
`energy.json` says how few frames carry half the energy
(`analysis.energy_concentration`). Read the `settled` state for the place.
The same split is on the command line as `ambiscape resolve SESSION --by handling`.

## A report that cannot print a gap

`REPORT.template.md` is markdown with placeholders that name a JSON file in
`analysis/` and a path into it:

```markdown
The settled room has an LAeq of {{split.json:states.settled.laeq_dbfs}} dBFS
and a median centroid of {{summary.json:centroid_median_hz}} Hz.

{{tables.json:states}}
```

`{{file.json:key.path:fmt}}` adds a format spec (`.1f`, `+.0f`). A value
stored as text, such as a clock time or a table from `tables.json`, goes in as
it is; numbers print with a true minus sign. A missing key or a null value
stops the build with the placeholder's name, so a report never shows a blank
where a measurement should be, and a number in it is never typed by hand. The
prose around the placeholders is yours: `run` measures, the template says what
the measurement means.

Analyses that only one recording needs, such as a breathing-rate test for a
night with sleepers, stay in that folder as scripts that read and write
`analysis/`, and their outputs fill the same template.

## From Python

```python
from ambiscape import runner

runner.init_config("SESSION")
runner.run("SESSION", stages=["post", "report"])
runner.post("SESSION")                      # the post stage alone
```
