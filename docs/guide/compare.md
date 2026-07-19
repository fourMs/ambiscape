# Cross-session comparison

The catalog answers *how do my places differ*; `compare` answers *how did this
place differ between visits* — two or more analysed sessions of one room laid
onto a common clock, so machines, weather, parties and silences read against
each other. It works entirely from the cached 1 Hz features and the
`summary.json` / `states.json` of a prior `analyze` run; no audio is reopened.

```bash
ambiscape compare 2026-07-15-loft 2026-07-19-loft-night 2026-07-19-loft-day \
    --lines 146,258,650,820 --band 2000:8000 --hours 27:34
#   ... LAeq per session, tonal-line prominence per session ...
#   wrote .../comparisons/<names>/compare.json and 4 figure(s)
```

## What it produces

Always:

- **Clock-aligned LAeq timelines** — per-minute level on a shared hour-of-day
  axis. Sessions whose spans bridge (a night flowing into the next morning)
  share a row and line up end to end; separate visits get their own rows. The
  `--state` intervals are shaded.
- **Per-state LTAS overlay** — median band spectra split by each session's
  detected states, so a machine's broadband signature (or its absence) is
  visible at a glance.
- **Azimuth roses** — foreground energy by azimuth, one panel per session
  (mic frames usually differ between visits: compare *shapes*, not bearings).
- **Descriptor tables** — pooled and state-resolved, in `compare.json`.

Optional:

- `--lines A,B,C` — **tonal-line prominence** per session: how far each named
  frequency (a machine fingerprint) stands out of the per-minute minimum
  spectrum. A machine that ran keeps several dB; one that never ran, ≲ 1 dB.
- `--band F0:F1 [--hours H0:H1]` — a **band timeline** on the clock axis (2–8
  kHz for a dawn chorus or rain hiss, 100–300 Hz for party bass), optionally
  restricted to a clock window (hours > 24 = day 2).

## Programmatic helpers

`ambiscape.compare` also exposes the pieces directly:
`load_comparison`, `laeq_timeline`, `clock_rows`, `ltas_by_state`,
`line_prominence`, `band_level`, `floor_difference` (a near-floor source
detector — a quiet fan's band shelf that never reaches a level step), and
`duty_cycle` (period, duty and regularity of a cycling source such as a
fridge, and its *absence* from a new mic position). See the API reference.
