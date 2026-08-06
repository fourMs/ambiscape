# Resynthesis (teaching)

`ambiscape resynth` recreates a session's soundscape from layers of
basic synthesis models and renders the result as one self-contained
Web Audio page, with no samples and no external resources, one HTML file
you can open anywhere or hand to students.

```bash
ambiscape resynth SESSION/        # writes SESSION/resynthesis/index.html
```

Needs a prior `analyze` run; it also reads whatever module JSONs exist
(`tonality`, `modulation`, `spatial`, `summary`) and degrades gracefully
when one is missing.

## Why

The page is teaching material: a real place, decomposed into the four
canonical building blocks of a synthesis course, with every starting
value measured from the recording. Students toggle layers to hear
how a soundscape is assembled, then drag sliders to exaggerate or delete
each ingredient. Comparing two sessions' pages (say, a station and a
train carriage) shows how few parameters carry a place's identity.

## The four layers

| Layer | Synthesis model | Measured parameters |
|---|---|---|
| Bed | subtractive: white noise → per-octave peaking filters | 10th-percentile octave spectrum (the keynote as a filter curve) |
| Machine | additive: detuned oscillator pair per tonal line, shared AM LFO | tonal-track frequencies and prominences; micro-modulation rate and depth |
| Events | stochastic: Poisson-scheduled band-passed noise bursts; occasional pass-by swell | events/min, median event duration, spectral centroid, pass-by count/duration |
| Space | per-source panning + two cross-fed delays | diffuseness ψ (diffusion wet), azimuthal concentration R (panning spread) |

## Recipe

The mapping lives in `resynthesis/recipe.json` (also embedded in the
page), produced by `resynth.build_recipe(sess, F, analysis_dir)`. The recipe
stays small and human-readable, since it *is* the lesson: a place,
reduced to a couple of dozen numbers, remains recognisable.

```python
from ambiscape import features, resynth
import ambiscape as asc

sess = asc.open_session("2026-07-28-Antwerpen-train-station")
F = features.load_features(features.extract_session(sess, "analysis/features"))
doc = resynth.run_session(sess, F, "2026-07-28-Antwerpen-train-station/analysis")
```
