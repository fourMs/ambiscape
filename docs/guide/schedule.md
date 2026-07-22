# Schedule matching against civic time grids

Bells, chimes, and sirens keep the wall clock: hourly strikes, quarter-hour
chimes, fixed evening ringing. `ambiscape schedule` tests a session's event
stream against candidate **civic periods** (minute, 5 min, quarter hour,
half hour, hour, day) by folding the event times at each period and scoring
the alignment with circular statistics. A tight fold on the hour grid is
strong evidence of a clock-driven source.

```bash
ambiscape schedule <session-folder>   # needs a prior analyze run
```

Uses the cached broadband events (no audio pass). Writes `schedule.json` —
under `events`, the best-scoring periods, each with `period_s`, the `phase_s`
the events cluster on (seconds past the grid tick), the resultant length
`R`, circular SD, `rayleigh_p`, the event count `n`, and `n_cycles`.

## Reading a match

A meaningful match needs **both** a high `R` and enough events spread over
several grid cycles. `R` is trivially 1 when every event falls inside a
single cycle, so always read `n_cycles` — a period of 3600 s that only spans
one hour proves nothing. `rayleigh_p` guards against the same trap
statistically.

## In Python

```python
from ambiscape import schedule

# which civic grid does an event stream fit?
schedule.match_periods(times_abs)      # times in ABSOLUTE session seconds

# or scan every tick of a known grid for band-limited strikes
schedule.grid_scan(F, period_s=3600, phase_s=0, band=(300, 1500))
```

`match_periods` asks *which grid* fits; `grid_scan` is the complement — it
looks *at each tick* of a known grid (every hour + `phase_s`) for energy in
a band, catching a church clock in the bell band whether or not the
broadband detector heard it. Times must be **absolute** seconds on the
session clock (`take.start` + offset into the take), or the grid phases are
meaningless.

A consistent nonzero `offset_s` across ticks is recorder-clock error:
`schedule.clock_offset(observed_abs, true_clock_s)` turns one event of known
wall-clock time into the `clock_offset_s` correction for `calibration.json`.
