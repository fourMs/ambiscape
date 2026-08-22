# Deposit export—publishing without publishing audio

Raw recordings from private spaces often cannot be shared; their acoustic
*envelope* can. `ambiscape deposit` writes per-take TSVs of 1 Hz features:

```
Time    level_dbfs    centroid_hz    low_frac    high_frac
0       -39.8         881            0.786       0.189
1       -44.9         135            0.966       0.007
```

These files are considered safe for open deposits (Zenodo and similar) where
the WAVs are withheld, since a 1 Hz loudness/spectral envelope is far below
speech timescales and carries no intelligible content.

The export follows the schema of the StillStanding365 deposit (365 daily
standstill sessions, fourMs/RITMO), which makes corpora that use it directly
poolable. Method deltas against that deposit's original extractor are
documented in the `deposit` module (W-channel at native rate vs an 8 kHz
four-channel downmix; power vs magnitude band fractions): trends and
dynamics are directly comparable, absolute fraction values differ slightly
by construction.

!!! warning "Directional data and channel conventions"
    When depositing *directional* products, record which B-format
    convention the source files used (see
    [Sessions & conventions](sessions.md)). A convention mismatch produces
    azimuth distributions collapsed onto one axis, which is an artefact
    that survives into downstream correlations and is invisible unless you
    know to look for it.

What still requires raw audio: fast-level descriptors (Leq/LAeq), events at
the 0.25 s criterion, diffuseness, elevation, spectra beyond the centroid,
psychoacoustic indicators. Plan deposits accordingly: features for
everyone, raw audio under controlled access where the ethics allow.

## Publishing an excerpt: the Freesound category

A feature deposit is one way material leaves a session; a curated audio
excerpt bound for a public repository is the other. Freesound has required a
category from its [Broad Sound
Taxonomy](https://freesound.org/help/broad-sound-taxonomy/) on every upload
since April 2025, and the category is also a search facet, so it decides
whether anyone finds the file. Recording it beside the WAV keeps the choice
reproducible across a pack of excerpts instead of retyping it into an upload
form:

```bash
ambiscape background my-session/ --excerpt 600 --bst ss-i \
    --tags "ambisonic,room-tone,ventilation" --licence "CC BY 4.0"
```

This writes `<excerpt>.wav.freesound.json` next to the excerpt, carrying the
category, its full name, the licence, the tags, and the excerpt's own
provenance (session, wall clock, duration, selection method).

A whole-room recording is a soundscape, so `--bst` accepts only the `ss-*`
categories unless you pass `--any-bst`:

| code | use it for |
|---|---|
| `ss-i` | interiors: rooms, offices, halls, anywhere closed |
| `ss-u` | outdoors with human intervention: streets, stations, transit halls |
| `ss-n` | natural habitats, and interiors an open window has handed over to one |
| `ss-s` | synthesised or computer-made ambiences |

Run the privacy gate before uploading and record its result in the same file:

```bash
ambiscape speechgate my-session/analysis/<excerpt>.wav
```

`deposit.freesound_sidecar(..., speech_fraction=...)` stores the fraction and
sets a `privacy_review` flag above 0.01. It is stored rather than enforced,
because what counts as an acceptable fraction is a judgement about the
recording and not a property of the format.

!!! note "This taxonomy classifies the file, not the sound"
    The Broad Sound Taxonomy cuts first by artefact type — music, instrument
    samples, speech, sound effects, soundscapes — which is a retrieval
    question, not an acoustic one. It is deliberately kept out of
    [`ambiscape.taxonomy`](taxonomy.md), where Schaeffer's, Schafer's and
    soundscape ecology's schemes each ask a question of the same *sound*. For
    a session recorded in a room the answer here is almost always `ss-i`, and
    a field with one value is no use as a descriptor.
