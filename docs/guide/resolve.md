# State-resolved descriptors

A single descriptor row for a multi-state session averages soundscapes that
never coexisted. The Haarlem loft's pooled row is dominated by its 9-hour
air-pump night and barely reflects the hi-fi afternoon; its Leq, diffuseness,
and NDSI all sit between two states it represents neither of. `ambiscape.resolve`
splits the session by time and runs the **full** summary pipeline on each
state.

```python
from ambiscape import resolve

states = resolve.machine_states(F, band=(250, 1000))   # {on: iv, off: iv}
res = resolve.resolve(F, states)                       # {state: full_summary}
res["machine_off"]["ndsi"], res["machine_on"]["ndsi"]
```

## Automatic in `analyze`

`ambiscape analyze` runs `resolve.auto_states` after the pooled summary:
when a session is genuinely two-state (both states last long enough and the
machine band steps by ≥ 4 dB between them) it writes `analysis/states.json`
and appends a **state-resolved table** to the session README. Steady,
single-state sessions get no state rows — the gate prevents spurious
splitting. Disable with `analyze --no-resolve`.

For an explicit split (a specific band, or day/night), the standalone
command writes the same `states.json`:

```bash
ambiscape resolve SESSION/ --by machine --band 250,1000
#   machine_on:  533.7 min, Leq -47.5 dBFS, psi 0.63, events/min 0.3, NDSI -0.38
#   machine_off: 168.0 min, Leq -55.1 dBFS, psi 0.82, events/min 1.8, NDSI  0.22
ambiscape resolve SESSION/ --by diel --night 22,6      # day / night instead
```

## Discovering states

- **`machine_states`** — on/off from a machine band via
  [`states.state_segments`](states.md); the default 250–1000 Hz band suits
  ventilation, other bands suit other machines.
- **`diel_states`** — day / night from the wall clock (`night=(22, 6)`
  wraps midnight), for outdoor and long sessions where the diurnal cycle is
  the state variable.
- **Any intervals you supply** — `resolve(F, {"label": [(t0, t1), ...]})`
  takes absolute-second intervals, so states from the taxonomy annotations,
  a switch-off time, or a schedule work directly.

## Slicing directly

`slice_features(F, intervals)` returns a sub-`F` valid for every
`summarize_*` function and every per-window analysis — use it to run any
ambiscape measure (fingerprints, ENF, tonality) on one state alone.
`full_summary(F)` is the merged descriptor set (level, foreground,
ecoacoustic, spatial, biophony) for any `F`.

!!! note "Keep the pooled row too"
    State-resolved rows describe the session honestly; the pooled row keeps
    cross-session continuity in the [catalog](catalog.md). Report both — the
    pooled number for comparability, the states for interpretation.
