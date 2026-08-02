# Prototype loop

`ambiscape loop` exports one seamlessly loopable segment per session: a
prototype of the place. The [characteristic
excerpt](background-render.md) seeks the *background*, the minute when
nothing in particular happens. The loop instead seeks the typical: the
window whose spectrum, level, *and event density* sit closest to the
take's own medians. A station loop keeps its normal rate of footsteps
and announcements; only the atypical (the one departure, the one quiet
lull) is avoided.

```bash
ambiscape loop SESSION/                    # 60 s window, 1 s crossfade
ambiscape loop SESSION/ --dur 90 --xfade 4 # longer, softer seam
ambiscape loop SESSION/ --take 1           # a specific take
```

Needs a prior `analyze` run (cached features). Output:
`analysis/loop_<take>_<dur>s.wav` (24-bit, native channel count, so 4-ch
AmbiX stays AmbiX) and `analysis/loop.json` with the scores and seam
residual.

## Selection

Windows of `--dur` seconds are scanned on a 5 s hop over the per-second
feature cache, scored by four distances (dB, lower is better):

1. **spectral typicality**: mean distance between the window's median
   log-spectrum and the take's median spectrum;
2. **level typicality**: |window L50 − take L50|;
3. **event-density typicality**: |window eventful-fraction − take
   eventful-fraction|, weighted ×20. We score for typicality rather than
   calm, so a window with the *representative* amount of activity beats
   both dead air and a burst of events;
4. **seam** (half weight): mismatch between the window's own first and
   last seconds, preferring windows that already end the way they begin.
   The end cut is then refined ±2 s for the best seam.

Sessions with time-overlapping takes (a Zoom and a phone running
simultaneously) are handled correctly: scoring uses only the chosen
take's feature rows (`load_features` carries per-row take identity), and
the audio is read from that same take.

## The seam

The final `--xfade` seconds are equal-power cross-faded into the head,
so the written file (length `dur − xfade`) loops without a click in any
player, so no DAW crossfade is needed. `loop.json` reports `seam_db`, the
RMS difference between the material entering and leaving the seam; on
steady soundscapes expect well under 1 dB.

## Library

```python
from ambiscape import features, render
import ambiscape as asc

sess = asc.open_session("2026-07-28-Antwerpen-train-station")
F = features.load_features(features.extract_session(sess, "analysis/features"))
doc = render.prototype_loop(sess, F, "analysis", dur_s=60.0, xfade_s=1.0)
```
