# Quickstart

## Run something now, with no recordings of your own

Everything below points at `2026-07-15-Haarlem`, which is a session you do not have.
This writes one you do:

```python
import ambiscape

root = ambiscape.examples.demo_session("/tmp/demo-session")
```

```bash
ambiscape analyze /tmp/demo-session
```

Two takes an hour apart: background alone, then two swinging bells at azimuths of 30 and
60 degrees over the same background, four-channel AmbiX at 48 kHz. Every command on this
page works against it, and because the two bearings are known, the anglegram and the
spatial descriptors have a right answer you can check them against.

A soundscape is a property of a PLACE, and a synthetic session has no place in it. There
is no room here, no reverberation, no traffic and no birds, so the ecological and
source-domain indices will return numbers that mean nothing. Use it to learn what the
commands produce and to check an installation, not as material for a claim.

## The pipeline

A *session* is a folder of WAV files from one recording occasion. The whole
CLI pipeline, in the order you would actually run it:

```bash
# 1. What is this folder? (metadata only, instant)
ambiscape probe "2026-07-15-Haarlem"

# 2. Extract features, compute descriptors, render figures, write README.
#    Features are cached as npz — re-runs are fast.
ambiscape analyze "2026-07-15-Haarlem" --notes "Loft; mic on couch overnight"

# 3. Pre-fill taxonomy annotations from detected states and events
#    (adds AudioSet tag suggestions when the [ml] extra is installed)
ambiscape draft "2026-07-15-Haarlem"

# 4. LISTEN. Edit annotations.draft.json, save as annotations.json.

# 5. Render the Schaeffer map + Schafer timeline
ambiscape taxonomy "2026-07-15-Haarlem"

# 6. Optional: ISO 12913-3 psychoacoustic indicators ([iso] extra)
ambiscape iso "2026-07-15-Haarlem"

# 7. Before publishing any excerpt: the speech privacy gate ([ml] extra)
ambiscape speechgate path/to/segment.wav

# 8. Non-identifying 1 Hz TSV export for open deposits
ambiscape deposit "2026-07-15-Haarlem"
```

After `analyze`, the session folder contains a `README.md` (metadata,
descriptor table, figures) and an `analysis/` folder (cached features,
`summary.json`, PNGs).

This is the core pipeline, not the whole toolbox: the
[command overview](cli.md) lists every subcommand, including `survey`
(questionnaire responses onto the 12913-3 circumplex), `sweep`/`impulse`
(measured impulse responses, STI, auralization), and `entrain`
(sound–motion coupling with a body-worn accelerometer).

## The same from Python

```python
import ambiscape as asc

sess = asc.open_session("2026-07-15-Haarlem")
paths = asc.extract_session(sess, "features")     # streams, caches npz
F = asc.load_features(paths)                      # one absolute time axis

print(asc.summarize(F))                           # descriptor dict

# raw audio access anywhere on the session clock:
x, fs = asc.read_span(sess, t0=4.0, dur=6.0)
print(asc.decay_time(x[:, 0], fs))                # T60 from a clap at t≈4 s

asc.figures.overview(F, "overview.png", clock=sess.clock)
segs = asc.pick_segments(F)                       # representative windows
```

## Calibration (optional but recommended)

Drop a `calibration.json` next to the WAVs and `analyze` adds dB SPL
versions of the level descriptors:

```json
{"dbfs_to_dbspl": 94.0,
 "method": "SPL meter next to mic, HVAC running, LAeq 42 dB",
 "date": "2026-07-16"}
```

The same file can correct a wrong recorder clock: add
`"clock_offset_s": 665.0` (positive = clock was slow) and every
clock-labelled output uses corrected time. See
[Room acoustics & ISO](guide/acoustics.md) for details, and
[Strike-level rhythm](guide/rhythm.md) for the `rhythm` command.
