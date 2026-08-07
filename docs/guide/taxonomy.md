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

- steady-state keynote *beds*: level regimes found by change-point detection with a fixed
  per-regime reference (median of the regime's first two minutes) and a two-minute confirmation
  window, so a slammed door does not split a regime but switching off the ventilation does. The
  regimes are then clustered by level similarity into beds (~6 dB bands), one draft object per bed
  carrying all of that band's spans, named by the bed's character and extent
  (`"quiet bed, -60 to -54 dBFS, 23 spans"`); the eight longest beds are kept and any remainder is
  pooled into one "other beds" object. A full domestic day therefore drafts as a handful of keynote
  candidates rather than one object per regime. With the `[ml]` extra, each bed also carries a
  `machine hint:` label with the dominant PANNs tag of its longest span, explicitly marked
  unverified;
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
anything the classification enforces. Points carry text only where it stays legible: on maps with
few objects every point is labelled, on crowded maps only points that sit alone in their grid cell
(the notable outliers) are. Cells with more objects than fit the regular layout are grouped into
~6 dB level bands, each band's points drawn together and given a concise identity label — level band
plus count, `"−46 to −40, n=17"` — so a crowded map never reduces to anonymous jitter.
Machine-drafted boilerplate is never printed per point; a single "machine-drafted labels; listen to
confirm" note under the title covers every drafted object.

![Schaeffer map: annotated objects placed on the facture by mass plane, coloured by Schafer soundscape function.](../img/schaeffer_map.png)

The Schafer timeline shows the session clock, and it has two layouts; which one you get depends on
whether human activity annotations are available (below).

**The acoustic-first layout** — the default, and the only one, when no activities are given — gives
one lane per hand-authored object: keynote bars, signal and soundmark event markers, lo-fi states
shaded, and gap-aware panels for multi-take sessions. This is where Schafer's question is answered,
since a sound's function is a claim about how it sits in time and in a place. It makes statements
like "the community soundmark is audible only in hi-fi windows" directly visible.

Machine-drafted steady-state regimes are the one exception to one-lane-per-object: when an
annotation file still contains many of them (an unedited draft, or an older per-regime one), they
are merged at render time into keynote-bed lanes by the same ~6 dB level banding the draft stage
uses — at most eight bed lanes plus "other beds", each labelled with its level range and span count
— so the figure height stays bounded however many regimes a long session proposes. Hand-authored
objects are never merged.

![Schafer timeline: one lane per object on the session clock, with keynote bars, signal and soundmark event markers, and shaded lo-fi states.](../img/schafer_timeline.png)

## Human activity ground truth (`--activities`)

Some datasets ship human annotations of what the people in the space were *doing*. The SINS
database (Dekkers et al. 2017) provides per-room activity logs as semicolon-separated CSVs with a
`Class;Start time;Stop time` header and absolute timestamps, and classes such as `absence`,
`sleeping`, `cooking`, `vacuumcleaner` or `watching_tv`. Both figures can overlay them:

```
ambiscape taxonomy <folder> --activities living_labels.csv
```

(or `taxonomy.render(folder, activities=...)` / the `activities` parameter of `schaeffer_map` and
`schafer_timeline` in the library, taking the list returned by `taxonomy.load_activities`). The
spans are aligned to the session clock via the session's day 0; if the file is missing the figures
render exactly as without the option.

### The activity-first timeline

With activities present the timeline inverts, by default, into the **activity-first layout**: the
human activities become the organising structure, not an overlay on the acoustic one. Each activity
class gets a lane of its own, ordered by total time (longest first), with classes under ~0.5% of the
session's labelled time pooled into an `other` lane at the bottom of the block. Each span's fill is
coloured by its measured acoustics — the median fast level within the span, in dB re the day's
median level, on the same palette as the cross-node day figures — so loud cooking and quiet cooking
read differently at a glance, and a vacuum-cleaner lane lights up against a dark absence lane. Lane
labels carry the acoustic summary computed from the session's cached features:
`"watching tv — 2.1 h, median −41 dBFS"`.

Hand-authored signals and soundmarks keep their own lanes and markers; the machine-drafted
keynote-bed structure no longer dominates the figure but is compacted into a single strip
(`keynote beds — machine draft, 62 spans`), its spans coloured by bed level on the same scale; the
events lane stays at the foot; lo-fi states remain shaded.

Which layout applies:

- no `--activities`: the acoustic-first lane timeline, exactly as before;
- `--activities <csv>`: the activity-first layout (`--layout auto`, the default, or
  `--layout activity`);
- `--activities <csv> --layout acoustic`: the acoustic-first lane timeline with the activity
  material as before — a compact ribbon of coloured class spans along the top (class legend shared
  with the map; `absence` and `other` deliberately muted so that colour means somebody was doing
  something), and each machine keynote bed's label gains its dominant concurrent activities by time
  share: `"quiet bed, -60 to -54 dBFS, 23 spans — during: absence 71%, sleeping 22%"`.

In the library this is `schafer_timeline(..., F=..., layout=...)`, where `F` is the features dict
from `features.load_features`; `render` loads the session's cached features itself when the
activity-first layout needs them (without a cache the figure still renders, with neutral span fills
and duration-only labels).

### The map with activities

On the map, points take the colour of their dominant concurrent activity instead of their Schafer
kind, with the class colours shared with the timeline ribbon's legend, and points that carry text
(sparse maps, or cell-singleton outliers) also say during which activity they occur
(`"— during cooking"`).

The two provenances are never conflated. The activities are **data** — human annotations shipped
with the dataset, attributed in the caption (`activities: human annotations, Dekkers et al. 2017`) —
while mass/facture, the level measurements and the bed structure remain machine output: the beds
keep their "listen to confirm" marking, and the acoustic summaries are measured, not judged.

## A note on spelling

The field value is `anthrophony`, which is the spelling Pijanowski and most of the soundscape-ecology
literature use. Both forms appear in print, and the package also accepts
the older `anthropophony` spelling when reading an annotation file, so
existing annotations keep working; everything it writes uses `anthrophony`.
