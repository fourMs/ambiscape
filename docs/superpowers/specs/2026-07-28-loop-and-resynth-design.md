# Design: `ambiscape loop` and `ambiscape resynth`

Date: 2026-07-28. Approved by user in session (train-recordings analysis).

## Motivation

Two user-facing needs from field practice:

1. **Loop**: for each session, export one seamlessly loopable WAV that is a
   *prototype* of the place: typical spectrum, typical level, typical event
   density. Use: installations, ambience beds, quick auditioning of a corpus.
2. **Resynth**: recreate a session's soundscape from layers of *basic
   synthesis models* (noise shaping, oscillator banks, stochastic events,
   simple diffusion), parameterised from the session's analysis, rendered as
   a self-contained Web Audio HTML page. Use: teaching material, where
   students deconstruct a real place into canonical synthesis blocks with
   sliders.

## 1. `ambiscape loop <session> [--dur 60] [--xfade 1.0] [--take N]`

New `prototype_loop(sess, F, out_dir, dur_s, xfade_s, take_idx)` in
`render.py` (sibling of `characteristic_excerpt`). Needs cached features.

**Window scoring** (dur_s window, 5 s hop, per-second features, restricted
to the chosen take; default take = longest):

- `d_spec`: mean |median log-spectrum(win) − median log-spectrum(session)| dB
- `d_level`: |L50(win) − L50(session)| dB
- `d_event`: |eventful_fraction(win) − eventful_fraction(session)| × 20 dB
  (eventful = fast level > session median + 6 dB), which scores *typicality*,
  not calm
- `d_seam`: mean |median log-spectrum(first 2 s) − (last 2 s)| dB
  + |level(first 2 s) − level(last 2 s)| dB
- score = d_spec + d_level + d_event + 0.5 · d_seam; lowest wins.

**Seam refinement:** end cut scanned ±2.5 s (0.25 s grid on per-second
features interpolated) for minimum seam mismatch.

**Export:** read winning span via `io.read_span` (native channel count),
equal-power crossfade of the final `xfade_s` into the head, output length
`dur − xfade`, peak-normalise to −1 dBFS if needed, write
`analysis/loop_<takestem>_<dur>s.wav` (PCM_24) + `analysis/loop.json`
(t0 session-time + clock, take, scores, seam residual dB, method note).

**Tests** (`tests/test_render.py`): synthetic session with a distinct
"typical" texture and an atypical loud stretch → picker lands in typical
region; loop WAV length = dur − xfade ± 1 s; channels preserved; seam
continuity: RMS in 50 ms window across the wrap (end→start concatenated)
within 3 dB of segment median RMS; loop.json exists with finite fields.

## 2. `ambiscape resynth <session>`

New module `resynth.py`: distill a recipe (plain dict → JSON) from
existing analysis outputs, then render `resynthesis/index.html` from a
string template (no new Python deps; page has zero external resources).

**Recipe inputs** (all optional except features/summary; layers degrade
gracefully when a JSON is missing):

- `summary.json` → L50, events/min, median event duration, centroid.
- features cache → background octave spectrum (10th pct of `oct_pow` per
  octave band, dB), typical spectrum (median), session duration.
- `tonality.json` → tracks (f_median_hz, prominence_db) → oscillator bank.
- `modulation.json` → micro-scale peak freq + depth → AM of machine layer.
- `spatial.json` → azimuth_R_median, directness → stereo width & diffusion
  wet level; passbys → optional "pass-by" event preset (panned sweep).
- `states.json` → machine_on/off Leq delta → machine layer level toggle.

**Web Audio layers** (each: gain slider + mute; parameters shown inline):

1. *Bed*: looped white-noise buffer → 10 peaking biquads at octave centres,
   gains from background octave spectrum (dB, normalised).
2. *Machine*: oscillators at tonal-track freqs (level ∝ prominence), slow
   sine AM (micro modulation peak freq/depth). Detuned pair per track.
3. *Events*: scheduler firing Poisson at events/min; each event = noise
   burst through a bandpass near the spectral centroid with attack/decay
   from median event duration; optional pass-by preset: 20 s filtered-noise
   swell panned across the field.
4. *Space*: per-layer StereoPanner + shared feedback-delay diffusion
   (wet ∝ diffuseness/1−directness).

Page: layer sections with comments teaching each synthesis model, master
gain, "all off/on". Recipe embedded as `<script type="application/json">`.

**CLI:** `ambiscape resynth <folder> [-o OUT]`; needs a prior analyze run;
uses whatever module JSONs exist.

**Tests** (`tests/test_resynth.py`): fixture session (conftest tone+noise)
after analyze-equivalent feature extraction + minimal JSONs → recipe has
finite octave gains, the known tonal line within 2%, events/min ≥ 0;
`index.html` written, contains the embedded recipe JSON (parseable) and
one `<section>` per layer.

## Out of scope

We leave out binaural/ambisonic resynthesis output, sample playback of
extracted events, DAW loop metadata (cue chunks), and per-state alternate
recipes.
