# Grounding: what a descriptor is evidence about

Every number this toolbox returns is a fact about a waveform. Some of them are
*also* meant as facts about hearing, and the distance between those two things
is where this project has made its worst mistakes.

Four claims were withdrawn in a single month. Every one of them was a
perceptual quantity read off a signal statistic:

| the claim | the number behind it | what was wrong |
|---|---|---|
| an open-plan room has two acoustic *zones* | speech fraction per microphone | a detection threshold, not a geography |
| one node is a *dead channel* | its level against its neighbours | a quiet room and a sensitive recorder look alike |
| the building has a 62-minute *rhythm* | periodicity on one day | it did not persist across the week |
| a loft's hum has a T60 of 17 s | Schroeder integration | the material contains no free decay to measure |

None was a coding error. Each was a *translation* nobody had written down.

`ambiscape.grounding` writes it down. It is the companion to
[`ambiscape.timescales`](timescales.md) and deliberately the same shape: a
registry, a check, and a list of what is not yet covered. Where the timescale
registry answers **over how long is this valid**, this one answers **what is it
evidence about**.

## The four tiers

| tier | meaning | what it may claim |
|---|---|---|
| `S` | **signal only** — defined by the mathematics of the waveform | "this is what is in the recording", and nothing about audibility, salience or annoyance |
| `PM` | **perceptually motivated** — designed by analogy with hearing, not validated against listeners in this domain | "this is a quantity a listener *might* track". The analogy is a hypothesis |
| `PC` | **perceptually calibrated** — the transform embeds a measured property of hearing | "this is weighted the way ears are". Still not a report of what anyone heard |
| `PD` | **perceptually defined** — the quantity only exists as a fact about a listener | nothing, without a listener. The number is evidence about the *proxy* until somebody asks one |

Current coverage is all **71** descriptors the toolbox emits:

| `S` | `PM` | `PC` | `PD` |
|---|---|---|---|
| 40 | 19 | 7 | 5 |

**The tier is not a quality ranking.** `S` is not worse than `PC`; a spectral
centroid is an excellent measurement of a spectral centroid. The tier says what
may be concluded, and the only real error is concluding one tier's worth of
thing from another tier's number.

Two consequences worth stating plainly, because both are easy to get backwards:

- *Calibrated is not the same as right.* A-weighting is a genuine
  equal-loudness contour from listening tests, and it is also known to
  under-weight exactly the low-frequency steady sources indoor work is full of.
  `PC` means a listener is built into the transform, not that the answer suits
  your material.
- *Signal-only is not the same as neutral.* `centroid_median_hz` is routinely
  called "brightness", which smuggles in a perceptual claim the number does not
  carry. Most `S` descriptors acquire a `PD` reading somewhere downstream, and
  that reading is where the risk lives — not in the arithmetic.

## The check

```python
from ambiscape import grounding

summary, cautions = grounding.check(summary)
for c in cautions:
    print(c)
```

On a summary carrying a foreground fraction, two events descriptors and a
couple of levels, that prints:

```text
fg_fraction_median: figure and ground are a relation between a sound and a
  listener, not a property of a signal. This is a level-based proxy for that
  relation and can be wrong about it
2 further descriptors are perceptually motivated but unvalidated against
  listeners (tier PM); see ambiscape.grounding
```

On a full summary, where all five perceptually-defined descriptors are
present, all five are named and the tally reads `19`.

`check` raises a caution for every `PD` quantity present and **counts** the
`PM` ones without itemising them. That asymmetry is deliberate: there are
nineteen `PM` descriptors, and a warning that fires on all of them every time
is a warning nobody reads.

The result is also written back into the summary under `grounding_cautions`,
so a deposited record carries its own caveats rather than relying on whoever
reads it having read this page.

## The five perceptually-defined descriptors

These are the ones to be careful with, and it is not a coincidence that they
are all the same thing:

- `fg_fraction_median`, `fg_fraction_p90` — how much of a session is
  "foreground"
- `fgbg_az_overlap` — whether figure and ground share a direction
- `azimuth_fg_deg`, `elevation_fg_median_deg` — where the foreground is

Foreground and background are not properties of a signal. A background is a
relation between a sound and someone who is not attending to it, and level is a
stand-in for that relation. The stand-in is a good one — across a year of
recordings, days with audible voices do not sit at a higher level but show a
wider range, which is a real and measured result — but it is still a stand-in.
Read these as evidence about the proxy.

## Coverage cannot rot

A test fails if a descriptor reaches the summary without a tier:

```python
def test_every_known_summary_key_has_a_tier():
    known = set(timescales.WINDOWS) | set(timescales.EXEMPT)
    missing = sorted(known - set(grounding.GROUNDINGS) - grounding.EXEMPT)
    assert not missing
```

The alternative — letting an unclassified descriptor default to `S` — is the
convenient answer and precisely how a perceptual claim gets attached to a
signal number without anyone deciding to. `grounding.unregistered(summary)`
reports the same thing at runtime for summaries from outside the toolbox.

## Inspecting the registry

```python
from ambiscape import grounding

grounding.tier_of("laeq_dbfs")      # 'PC'
grounding.tier_of("leq_dbfs")       # 'S'  -- unweighted, no listener
grounding.counts()                  # {'S': 40, 'PM': 19, 'PC': 7, 'PD': 5}

for row in grounding.table():
    print(row["tier"], row["descriptor"], row["why"])
```

Every entry carries a `why`, and a `ref` where the tier rests on a published
standard. If you disagree with a classification, the `why` is the thing to
argue with.

## What this does not cover

The registry covers the **session summary**. It does not yet reach:

- **object-level typing.** `object_facture` returns "impulsive", "sustained" or
  "iterative" — Schaeffer's categories, which are perceptual by definition and
  therefore `PD`. Three trained listeners agree unanimously on 158 of 365
  deliberately-recorded sound actions, so the ground truth is itself unstable.
  The label does not currently carry that.
- **impulse-response metrics.** T60, EDT, C50 and STI are all `PC` — the 50 ms
  and 80 ms splits and the 10 dB early-decay range come from listening
  experiments — but they are returned by `ir_metrics` rather than through the
  session summary.
- **the video side.** Quantity of motion and its relatives are `S`, and the
  same downstream risk applies.

These are gaps, listed so they are visible rather than assumed covered.
