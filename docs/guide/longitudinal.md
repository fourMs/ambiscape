# Longitudinal analysis

How does a place sound across weeks, months, a year? Version 0.10 adds a
layer above the single session: trend and seasonal analysis of *dated*
session summaries across a corpus.

The design point matters for anyone planning a long study. The unit is the
**dated session summary**, not the audio. A year-long study is best run as
many short sessions---one a day, as in the StillStanding archive---each
analyzed to a small `summary.json`. A year is then 365 tiny rows, so the
longitudinal analysis is inherently out-of-core no matter how large the
underlying audio was. (A single *continuous* multi-month recording exceeds
what the in-memory feature pipeline holds; segment it into per-day sessions
first.)

```bash
ambiscape longitudinal CORPUS/ --key ndsi --key leq_dbfs
#   365 dated sessions, 2023-01-01..2023-12-31
#     ndsi: trend +0.02/yr, seasonal amp 0.31 (peak month 7)
#     leq_dbfs: trend -1.4/yr, seasonal amp 2.1 (peak month 1)
#   wrote CORPUS/analysis/longitudinal_ndsi.png ... and longitudinal.json
```

Each session needs a date. `ambiscape analyze` now writes one into
`summary.json` (from the recording's BWF timestamp); for older summaries the
date is parsed from the session folder name (`YYYY-MM-DD…` or `YYYYMMDD…`).

## In a notebook

```python
from ambiscape import longitudinal as lg

s = lg.collect_series("CORPUS/", keys=["bird_active_minute_fraction"])
dec = lg.decompose(s["dates"], s["series"]["bird_active_minute_fraction"])
lg.summarize_longitudinal(s["dates"], s["series"]["..."])
# {'trend_per_year': ..., 'seasonal_amplitude': ..., 'peak_month': 7, ...}
```

- **`decompose`** splits a descriptor into an additive **trend** (a
  day-windowed rolling median, default one year so the seasonal cycle
  averages out), a repeating **seasonal** component (the monthly
  climatology of the detrended series), and the **residual**.
- **`seasonal_climatology`** and **`trend_slope`** give the two components
  on their own; **`summarize_longitudinal`** reports trend-per-year,
  seasonal amplitude, and the peak/trough months.
- **`render`** draws the descriptor over time with its trend, beside the
  monthly climatology.

The motivating example is already in the StillStanding data: bird mentions
peak in July and fall to zero in winter---not because the birds leave, but
because the windows close. A longitudinal run on the biophony descriptors
recovers exactly that seasonal signature from the audio alone.
