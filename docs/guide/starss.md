# DCASE STARSS validation

How good is the toolbox's own sense of direction? Every spatial descriptor in
ambiscape — pass-bys, directional entropy, azimuth roses — rests on the same
energy-based direction estimate: the pseudo-intensity azimuth
`atan2(Σ W·Y, Σ W·X)` over the ACN/SN3D first-order channels. The STARSS
datasets (Sony-TAu Realistic Spatial Soundscapes, the DCASE sound-event
localisation and detection task data) offer an independent yardstick: real
recorded scenes in first-order ambisonics with human-verified,
motion-capture-assisted direction labels every 100 ms. `doavalidate` runs
ambiscape's estimator over a folder of STARSS clips and reports how far its
azimuths sit from the labels.

```bash
ambiscape doavalidate STARSS23/foa_dev/dev-test-sony \
    --annotations STARSS23/metadata_dev/dev-test-sony
#   fold4_room23_mix001: 812 single-source frames, median |err| 14.2°
#   ...
#   overall: n=..., median |err| ...° (IQR ...°), bias ...°, within 20°: ...%
#   ; ... multi-source frame(s) excluded
# wrote .../analysis/doavalidate.json, .../analysis/doavalidate.png
```

## The STARSS format

Clips are 4-channel WAVs, 24 kHz, 16 bit — first-order ambisonics (ACN
channel order, SN3D normalisation) in the `foa` variant. Each clip has an
annotation CSV of the same stem (`fold4_room23_mix001.wav` ↔
`fold4_room23_mix001.csv`) whose headerless rows label one active source in
one 100 ms frame:

```
frame, class, source, azimuth, elevation[, distance]
```

- **frame** — integer index of 100 ms intervals from the clip start;
- **class** — index into the 13 STARSS sound-event classes (female/male
  speech, clapping, telephone, laughter, domestic sounds, footsteps, door,
  music, musical instrument, water tap, bell, knock);
- **source** — integer distinguishing simultaneous instances of a class;
- **azimuth** — degrees in [−180, 180], zero at the front, increasing
  counter-clockwise (+90 = left) — the same convention as ambiscape's
  pseudo-intensity azimuth, so labels and estimates compare directly;
- **elevation** — degrees in [−90, 90];
- **distance** — cm, present in STARSS23 but not STARSS22; both five- and
  six-column rows are read.

## Clips have no clock: `open_clips`

STARSS clips carry neither a BWF `bext` chunk nor a filename timestamp, so
[`open_session`](sessions.md) would fall back to file modification times —
download times, which say nothing about capture and typically pile every
clip onto one overlapping stretch of timeline. `open_clips(folder)` opens a
clip collection instead: takes are chained end-to-end in sorted filename
order from midnight of a nominal day 0 (1970-01-01). Positions on the
session timeline are deterministic and reproducible across machines;
clock-of-day readings, by construction, mean nothing for these takes.

## What is compared, and what is excluded

For each clip, the audio is band-passed to the corpus DOA band (80–3000 Hz)
and the pseudo-intensity azimuth is taken per 100 ms frame — the label grid.
On every frame labelled with *exactly one* active source, the signed
circular difference between estimated and labelled azimuth is recorded.

Multi-source frames are excluded by design, not for convenience: the energy
azimuth is a single broadband direction per frame, and when two sources are
active at once the pseudo-intensity vector points at an energy-weighted
mixture of them. Its deviation from either label then measures the mixture,
not the estimator, so scoring such frames would conflate a model limitation
(one direction per frame) with estimation error. The count of excluded
frames is reported alongside the statistics.

Elevation and distance labels are read but not scored; the validation
targets the horizontal energy direction that ambiscape's spatial
descriptors are built on.

## What it reports

`doavalidate.json` holds circular error statistics — median and IQR of the
absolute error, circular bias (mean signed error) with circular SD, the
fraction of frames within 20°, and a per-class breakdown (n, median, IQR)
— overall and per clip. `doavalidate.png` pairs an error rose (polar
histogram of the signed error; a spike at 0 = agreement) with a per-class
absolute-error chart.

Expect honest imperfection on real scenes: STARSS rooms are reverberant,
labels mark the source while the estimator integrates direct sound *and*
room reflections, and quiet labelled frames (a footstep's gap, a decaying
knock) are dominated by whatever else the room is doing. Median errors of
tens of degrees on real folds are a property of single-frame energy DOA in
rooms, and exactly what the per-class breakdown makes visible — steady
loud classes (music, speech) localise far better than sparse quiet ones.

## Python

```python
from ambiscape import open_clips, starss

sess = open_clips("foa_dev/dev-test-sony")   # synthetic clock, see above
rows = starss.read_annotations("metadata_dev/fold4_room23_mix001.csv")
doc = starss.validate_collection("foa_dev/dev-test-sony",
                                 "metadata_dev/dev-test-sony")
doc["overall"]["median_abs_deg"], doc["overall"]["per_class"]
```

Synthetic ground truth in the test suite closes the loop: a generated FOA
clip with a known panned source and a matching CSV recovers near-zero
error, and the same clip against labels rotated by 90° recovers the
rotation — the comparison is verified not to be vacuously small.
