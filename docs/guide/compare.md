# Cross-session comparison

The catalogue answers *how do my places differ*; `compare` answers *how did this
place differ between visits*, by laying two or more analysed sessions of one
room onto a common clock, so machines, weather, parties and silences read
against each other. It works entirely from the cached 1 Hz features and the
`summary.json` / `states.json` of a prior `analyze` run; no audio is reopened.
For several recorders running *simultaneously* in different rooms of one
building, [`network`](network.md) reads the sessions as a coupling graph
instead.

```bash
ambiscape compare 2026-07-15-loft 2026-07-19-loft-night 2026-07-19-loft-day \
    --lines 146,258,650,820 --band 2000:8000 --hours 27:34
#   ... LAeq per session, tonal-line prominence per session ...
#   wrote .../comparisons/<names>/compare.json and 4 figure(s)
```

## What it produces

Always:

- **Clock-aligned LAeq timelines**—per-minute level on a shared hour-of-day
  axis. Sessions whose spans bridge (a night flowing into the next morning)
  share a row and line up end to end; separate visits get their own rows. The
  `--state` intervals are shaded.
- **Per-state LTAS overlay**—median band spectra split by each session's
  detected states, so a machine's broadband signature (or its absence) is
  visible at a glance.
- **Azimuth roses**—foreground energy by azimuth, one panel per session
  (mic frames usually differ between visits: compare *shapes*, not bearings).
- **Descriptor tables**—pooled and state-resolved, in `compare.json`.

Optional:

- `--lines A,B,C`—tonal-line prominence per session: how far each named
  frequency (a machine fingerprint) stands out of the per-minute minimum
  spectrum. A machine that ran keeps several dB; one that never ran, ≲ 1 dB.
- `--band F0:F1 [--hours H0:H1]`—a band timeline on the clock axis (2–8
  kHz for a dawn chorus or rain hiss, 100–300 Hz for party bass), optionally
  restricted to a clock window (hours > 24 = day 2).

## Programmatic helpers

`ambiscape.compare` also exposes the pieces directly:
`load_comparison`, `laeq_timeline`, `clock_rows`, `ltas_by_state`,
`line_prominence`, `band_level`, `floor_difference` (a near-floor source
detector, for a quiet fan's band shelf that never reaches a level step), and
`duty_cycle` (period, duty and regularity of a cycling source such as a
fridge, and its *absence* from a new mic position). See the API reference.

## Cross-node day figures (uncalibrated multi-room deployments)

For several *uncalibrated* recorders covering the same day of one building
(the SINS deployment style), `xnode_day_matrix`, `xnode_floor`,
`xnode_loudest` and `xnode_figure` build a day heatmap plus a loudest-room
strip from per-node full-day 1 Hz level arrays:

- **Heatmap** — clock-binned power means shown as *dB above each node's own
  day median*, since raw dB from uncalibrated instruments are not
  comparable. Bins without data (a node not yet recording, a gap between
  takes) render as neutral grey, never as a colour of their own.
- **Loudest-room strip** — a bin is awarded to a node only when *both*
  rules hold, and stays empty otherwise:
    1. *Margin rule*: the node's normalized level beats every other node's
       by more than `margin_db` (default 3 dB). Uncalibrated nodes differ
       by sensor gain, so a fractional-dB "win" says nothing about the
       sound — without this rule, a quiet night renders as a solid row for
       whichever node has the hotter gain.
    2. *Floor rule*: the node's absolute level clears its own noise floor
       (`xnode_floor`: a low percentile of the day's 1 Hz levels, raised by
       `adjust_db` when the session's analysis flagged `floor_suspect` —
       i.e. the recorded floor is instrument self-noise, so marginal
       excursions above it measure the recorder, not the room) by at least
       `floor_clear_db` (default 3 dB).

Both rules are restated in the figure's caption line. Historical note: fast
feature caches written by ambiscape ≤ 0.24.1 carried a few *unfilled*
125 ms frames past the last whole second of each take, stored as exactly
0 dBFS (full scale) — a false click at every take boundary that painted
thin bright columns onto such heatmaps. `extract_take` no longer writes
those frames, and `load_features` drops them from old caches on load, so
descendants of the fast streams (this module, `network`, LAeq timelines)
are clean without re-analysis.
