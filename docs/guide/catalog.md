# Corpus catalog

Every `ambiscape analyze` writes `<session>/analysis/summary.json`. The
`catalog` command aggregates them across a whole corpus into one table —
the cross-session view, built in milliseconds from cached summaries alone
(no audio, no features).

```bash
ambiscape catalog CORPUS/ --sort diffuseness_median
#   10 sessions -> CORPUS/analysis/catalog.csv and catalog.md
#         0.90  stavanger-foyer
#         0.89  montpellier-airport
#         ...
```

`catalog.csv` is session-per-row for analysis (pandas, R); `catalog.md` is
the transposed descriptor-per-row table for a consolidated report. Sessions
with different descriptor sets — older caches, optional modules — are
handled by taking the union of keys and blanking where a session lacks one,
so a growing corpus never breaks the table.

## In a notebook

```python
from ambiscape import catalog

col = catalog.collect("CORPUS/")            # {session: summary}
catalog.to_csv(col, "catalog.csv")
catalog.rank(col, "ndsi")                   # sessions by any descriptor
catalog.outliers(col, "azimuth_R", z=1.5)   # what stands out, as z-scores
```

`rank` returns `(session, value)` pairs sorted on any numeric descriptor;
`outliers` flags sessions more than `z` standard deviations from the corpus
mean — the cheap "which session is unusual, and on what axis" query that
turns a growing database into a comparative instrument.

!!! note "Keep summaries current"
    The catalog reflects whatever is in each `summary.json`. After
    upgrading ambiscape, re-run `analyze` (or refresh the summaries) so
    every session carries the same descriptor set before cataloguing.
