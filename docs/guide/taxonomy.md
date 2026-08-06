# The taxonomy workflow

## Two different men, two different questions

The annotation schema draws on two traditions whose founders have almost the same name. They are
easy to confuse and they answer different questions, so it is worth separating them before anything
else.

Pierre Schaeffer (1910–1995) asked what a sound *is*, considered on its own terms. His
*Traité des objets musicaux* (1966) describes the sound object through reduced listening, deliberately
setting aside where the sound came from, and classifies it by its internal make-up. In this package
that is the `mass` and `facture` pair, which together place an object on a simplified version of his
typo-morphological grid.

R. Murray Schafer (1933–2021) asked what a sound *does* in a place. *The Soundscape: Our Sonic
Environment and the Tuning of the World* (1977) classifies sounds by the role they play for the
people who live among them: the constant background, the deliberate foreground, the sound that marks
a community. In this package that is `kind` and `soundmark`.

A third, later tradition is also present and belongs to neither of them. Soundscape ecology, after
Bernie Krause and Bryan Pijanowski, divides sound by its physical origin into biophony, geophony and
anthrophony. That is the `source` field.

So the same object carries up to three independent labels, and they do not constrain one another. A
ventilation drone is `noise`/`unlimited` to Schaeffer, a `keynote` to Schafer, and `anthrophony` to
soundscape ecology. Nothing about the first determines the second or the third.

## The workflow

Instruments detect *when* things sound. Deciding what they are, in any of the three schemes, is a
listening judgement. The work is split accordingly:

```
ambiscape draft <folder>      # machine proposes
# ... you listen and edit ...
ambiscape taxonomy <folder>   # machine renders
```

## 1. Draft (`annotations.draft.json`)

From the cached features, `draft` pre-fills:

- steady-state keynote candidates: level regimes found by change-point detection with a fixed
  per-regime reference (median of the regime's first two minutes) and a two-minute confirmation
  window, so a slammed door does not split a regime but switching off the ventilation does;
- detected events, each with listening hints: clock time, exceedance, level, azimuth and elevation,
  diffuseness, and, with the `[ml]` extra, AudioSet tag suggestions from PANNs. Treat those as
  suggestions to confirm by ear, not as ground truth.

For a finer, object-level draft of the two Schaeffer axes, where every inter-onset sound object is
classified on a simplified TARTYP grid, see `music.tartyp_profile` in [Music](music.md). It needs the
`[music]` extra.

## 2. Annotate (`annotations.json`)

The fields are grouped below by which tradition they come from. Only `spans` and `events` are ours.

Schaeffer, what the sound is in itself:

| Field | Values |
|---|---|
| `mass` | `tonic` \| `tonic-complex` \| `complex` \| `noise` |
| `facture` | `impulse` \| `iteration` \| `sustained` \| `unlimited` |

Schafer, what the sound does in the place:

| Field | Values |
|---|---|
| `kind` | `keynote` \| `signal` \| `soundmark` \| `figure` |
| `soundmark` (optional) | `community` \| `dwelling` |

Soundscape ecology, where the sound physically comes from:

| Field | Values |
|---|---|
| `source` (optional) | `anthrophony` \| `biophony` \| `geophony` |

Timing, ours:

| Field | Values |
|---|---|
| `spans` / `events` | times as `"[D ]HH:MM:SS"` (D = days after day 0) |

There is also an optional `states` list for lo-fi spans, such as a masking drone. Hi-fi and lo-fi are
Schafer's terms as well: a hi-fi soundscape is one where individual sounds can be heard distinctly,
and a lo-fi one is where they are crowded out.

The full schema is documented in the [`taxonomy` module](../api.md), and worked examples live in the
Intercontinental database's Haarlem and Berlin session folders.

## 3. Render

`taxonomy` produces two figures, one per tradition.

The Schaeffer map places every object on the facture by mass plane, which is his question and his
alone. It is coloured by Schafer's `kind` only as a convenience, so that you can see whether the two
schemes happen to agree in your corpus. Keynotes tend to crowd the sustained and unlimited columns
and signals the impulse and iteration columns, but that is a finding about the material rather than
anything the classification enforces.

![Schaeffer map: annotated objects placed on the facture by mass plane, coloured by Schafer soundscape function.](../img/schaeffer_map.png)

The Schafer timeline gives one lane per object on the real session clock: keynote bars,
signal and soundmark event markers, lo-fi states shaded, and gap-aware panels for multi-take
sessions. This is where his question is answered, since a sound's function is a claim about how it
sits in time and in a place. It makes statements like "the community soundmark is audible only in
hi-fi windows" directly visible.

![Schafer timeline: one lane per object on the session clock, with keynote bars, signal and soundmark event markers, and shaded lo-fi states.](../img/schafer_timeline.png)

## A note on spelling

The field value is `anthrophony`, which is the spelling Pijanowski and most of the soundscape-ecology
literature use. Both forms appear in print, and the package also accepts
the older `anthropophony` spelling when reading an annotation file, so
existing annotations keep working; everything it writes uses `anthrophony`.
