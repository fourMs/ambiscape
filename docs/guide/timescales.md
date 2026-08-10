# Timescales: what a descriptor needs

A descriptor computed on too little audio is not a noisy descriptor. It is
either a different quantity or none at all. The complexity index averages
over 300-second chunks, so a 30-second clip has no chunk to average and the
function returns `0.0` — a confident-looking number that no measurement
produced. Tonal prominence aggregates per-minute spectra and has nothing to
aggregate below a minute. A median level exists at any length and moves by
tens of decibels until there is a minute of it.

`ambiscape.timescales` is the one place those windows are written down.
Everything else reads from it: the guard inside `analyze`, the
`ambiscape timescales` command and its figure, and the validity section of
a deposit report. Writing the numbers twice is how they drift.

It has a companion. Where this registry answers *over how long is a descriptor
valid*, [`ambiscape.grounding`](grounding.md) answers *what is it evidence
about* — signal, or something about a listener. The two failure modes are
independent: a descriptor can be computed over a perfectly adequate window and
still be the wrong kind of evidence for the claim being made of it.

## The guard

Every descriptor that reaches `summary.json` passes through
`resolve.full_summary`, which checks it against the window it needs:

- below a **hard** window the value becomes `None`, because below it the
  quantity does not exist;
- below a **soft** window the value is kept and the key is listed in
  `low_confidence`, with the window it needed and the window it had.

```json
"aci": null,
"low_confidence": [
  {"key": "aci", "needs_s": 300.0, "had_s": 60.0, "kind": "hard",
   "why": "averages over 300 s chunks; a shorter session has no complete
           chunk and the index is undefined"}
]
```

This also applies to per-state summaries, where segments are short and the
problem is worst: a forty-second `machine_on` state cannot support an
index that needs five minutes, and now says so.

Pass `check_windows=False` to `full_summary` for the raw set — what the
computation produced rather than what it supports.

## Reading the registry

```bash
ambiscape timescales              # the table
ambiscape timescales --csv        # for a report
ambiscape timescales --figure timescales.png
```

Each window records whether it was **measured** or **asserted**. Some
bounds were established on this project's own material; others are honest
judgement — machine periodicity "needs hours" rests on its not having
converged by six minutes, which is weaker than a measurement. Ten of the
27 windows are asserted, and the figure hatches them so a reader can see
which parts of the picture are evidence.

## The bands

`micro`, `meso` and `macro` are joint spatial and temporal categories,
following *Sound Actions*, where mesomotion is defined at 1–100 cm and
0.5–5 s, the short-term memory range, with macro and micro as everything
longer/larger and shorter/smaller. The pairing is deliberate: sound, like
motion, is not sensibly studied in space alone or in time alone.

Applied to rooms rather than bodies the temporal extents carry over
unchanged, because they are anchored in perception — the meso band is
Schaeffer's sound object, Godøy's chunking range and short-term memory at
once. The spatial extents move outward, and the spatial meso becomes the
*zone*: the kitchen zone, the dining-table zone, the sofa zone of one
open-plan room.

| band | time | space (room) | memory |
|---|---|---|---|
| micro | < 0.5 s | the ear, the head | echoic / sensory |
| meso | 0.5–5 s | the zone | short-term / working |
| macro | > 5 s | the room, the building | long-term |

## The meso band

For a long time the answer to "what can this toolbox say about a
six-second sound action?" was: nothing. Every windowed descriptor was
invalid, because they are all session-scale, and only the band-energy
ratios and spectral medians survived.

`objects.object_profile` fills that band — attack, decay, temporal
centroid, crest, iteration rate and strength, each defined on a single
object and registered here at 0.2 s. It is why the figure's descriptor row
reaches left of a minute at all.

The bounds on those six are marked `asserted` rather than `measured`. They
follow from what the quantities are — there is no attack without an object
— rather than from a length study, and the figure hatches them accordingly.

## Adding a descriptor

A new summary key must be added either to `WINDOWS`, with a window, a kind
and a source, or to `EXEMPT`, with the reason it is safe at any length.
`tests/test_timescales.py` fails until one of the two is done, which is
deliberate: a new descriptor is neither trustworthy at all lengths nor
known to be fragile until somebody decides which.
