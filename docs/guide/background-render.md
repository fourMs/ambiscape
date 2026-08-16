# Background bed & characteristic excerpt

Schafer's foreground/background layers as audio operations. Two related
tools live behind `ambiscape background`:

- a background bed: the full recording with every emerging event, whether
  bells, voices or doors, capped down into the continuous background;
- a characteristic excerpt: the one minute (or any duration) of the
  *original, untouched* audio that best summarises what the place sounds
  like when nothing in particular is happening.

Use the bed when you need the background as a continuous signal (ambience
beds, keynote study, masking analysis); use the excerpt when you need a
compact, artefact-free *summary* of the space, a soundscape thumbnail.

The excerpt's filename is prefixed with its wall clock, as
`YYYYMMDD_HHMMSS`, which is the stamp `open_session` reads. A folder of
excerpts is therefore itself a session, on the clock it was cut from, and
can be re-opened and analysed like any other. `io.export_segment` takes
`stamp=False` where an exact filename is needed.

## Rendering the bed

```bash
ambiscape background SESSION/                 # keynote bed (+12 dB margin)
ambiscape background SESSION/ --margin-db 3   # strict continuous bed
```

Method: the short-time spectrogram is compared against a per-frequency
background surface, which is the median magnitude in 10 s blocks, followed
by a running *minimum* over ±60 s of blocks. The minimum-statistics step is
what makes it work on real material: a chime that rings for two minutes
straight cannot raise its own ceiling. Every time-frequency cell is then
capped at the surface plus the margin (phases untouched), the gain field
is lightly smoothed against musical noise, and re-capped so smoothing can
never lift a cell back above the surface. Output is a 24-bit WAV at the
take's rate and channel count, plus `background_render.json` with the
parameters and the fraction of cells meaningfully above the cap before
and after (the latter is 0.0000 by construction).

The margin is the layer boundary. At ~3 dB only the continuous bed
survives (city hum, ventilation, room tone). At ~12 dB (the default),
quiet iterative keynotes survive too, so that a tower clock's ticking keeps
~85 % of its envelope-periodicity prominence while chimes and speech
flatten. On the Belfort machine-room recording the carillon's emergence
drops from +36 dB to the cap; across a 26-session test corpus (cafés,
woods, escalators, an airplane cabin) residual exceedance is 0.00 %
everywhere. The best sanity check is that a steady-state soundscape like
the airplane passes through essentially unchanged, engine tones intact.

Long takes are processed in `chunk_s` pieces (default 20 min) with the
minimum-statistics context on each side and 50 ms crossfaded joins, so
memory stays bounded for any duration (a 98-minute café: 7.5 GB peak
instead of 22.6 GB one-shot).

## Cutting the characteristic excerpt

```bash
ambiscape analyze  SESSION/                # once, for the feature cache
ambiscape background SESSION/ --excerpt 60
```

Every 60 s window (5 s hop) is scored by two things: the mean per-band
distance between its median log-spectrum and the session's background
spectral profile (the 10th percentile per band over the whole take), and
the fraction of eventful seconds (broadband level more than 6 dB over the
session median), weighted at 20 dB. The winner is exported bit-exact from
the original take, with no processing whatsoever, together with
`background_excerpt.json` recording the wall-clock position and scores.

The picker's behaviour on real material is reassuringly boring: in a
uniformly quiet night recording it takes the first window (anywhere is
characteristic); in a 98-minute café it finds a calm stretch an hour in;
in the Belfort machine room it independently selects the same calm
post-chime minute a human picked for clock-tick analysis. That minute is
ticking, room tone and distant city, at 0.5 dB spectral distance from the
background profile.

## From Python

```python
import ambiscape as asc
from ambiscape import render, features

sess = asc.open_session("SESSION")
doc = render.run_session(sess, "SESSION/analysis", margin_db=12.0)

F = features.load_features(sorted(Path("SESSION/analysis/features").glob("*.npz")))
ex = render.characteristic_excerpt(sess, F, "SESSION/analysis", dur_s=60.0)
print(ex["clock"], ex["out_path"])
```

## Caveats

- **Capped is not recovered.** Where the foreground was loud, the bed
  contains that energy pushed down to background level. The true
  background masked beneath it is gone, and nothing can reconstruct it.
- Like `rhythm` and `carillon`, the command processes the first take
  of the session. Multi-device folders (a WAV and an M4A of the same
  scene) render whichever sorts first, so point at single-file folders for
  per-device beds.
- Channels are processed with independent gains; stereo and ambiX beds
  remain listenable, but exact spatial coherence under heavy capping is
  not guaranteed.
- On takes much shorter than a minute the background estimate collapses
  to a single block, so "background" is only weakly defined, and you
  should expect gentler foreground suppression there.
