# Changelog

Written 2026-08-03, reconstructed from the tag history and the commits between tags. Entries before
that date are therefore summaries of what the commits say, not notes written at the time.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/); the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) loosely, in that the minor number has
been carrying feature work throughout a pre-1.0 life.

## [Unreleased]

### Added
- Sensor-noise-floor guard (`analysis.floor_suspicion`, run inside every `summarize`). Motivated by
  the SINS sensor-network corpus, where the 4–8 kHz background floor of a living room is flat to
  0.8 dB across a full week — microphone self-noise, so low-percentile descriptors there measure
  the instrument, not the room. Per octave band from 2 kHz up, the session is cut into 300 s
  chunks and the temporal spread of the chunkwise 10th-percentile floor is taken as the median
  minus the 5th-percentile chunk floor (a low-tail statistic, immune to activity-elevated chunks);
  a spread under 1.5 dB flags the band (SINS self-noise ≤ 0.8 dB over a week, quietest genuinely
  acoustic bands ≥ 2.4 dB — at least 0.7 dB of margin to each side). `summary.json` gains
  `floor_suspect`, `floor_suspect_lo_hz`/`floor_suspect_hi_hz` and `floor_spread_db`; the session
  README carries a warning that L90-derived descriptors in that range may reflect the instrument,
  not the room. Annotation only — no descriptor value changes. Sessions under six chunks (30 min)
  and bands with no content below the Nyquist frequency are never flagged.

## [0.23.1] — 2026-08-06

### Fixed
Five faults found by running the toolbox over public material (the MIT IR Survey, ESC-50 sessions
built at four durations, and xeno-canto recordings against synthetic ventilation):

- `ambiscape iso` raised a bare `ModuleNotFoundError` for `mosqito` on a stock install. The CLI now
  checks `iso.mosqito_available()` and exits with `MoSQITo not installed — pip install
  'ambiscape[iso]'`, and `iso.indicators` raises an `ImportError` naming the same extra, matching
  how the other optional-dependency subcommands behave.
- `ecology.aci` returned `0.0` for any recording shorter than one 300 s chunk — silently
  indistinguishable from a measured minimum, and the case for every clip corpus (ESC-50,
  UrbanSound8K, AudioSet, DCASE/TAU) as well as for archive recordings under five minutes. It now
  returns `None`, as does the `aci` key of `ecology.indices`.
- `iso.segment_indicators` labelled every segment with the requested duration even where less audio
  existed (`io.read_span` clamps at the end of a take). `dur_s` now reports the audio delivered,
  with the request kept as `dur_requested_s` when the two differ.
- `analysis.pick_segments` returned quietest, most-active and typical as separate entries when they
  resolved to the same window — a session only one segment long, or a stationary room with no
  distinct most-active minute. Coincident kinds are now returned once, listing the others under
  `also` (`also_kinds` in the `iso` output, and a note on the CLI), so the degeneracy is visible
  and the window is measured once rather than three times.
- `analysis.decay_metrics` reported T20/T30 on pre-trimmed impulse responses, whose absent noise
  floor leaves the dynamic-range guard unable to fire; 77 of the 270 MIT survey IRs returned a T30
  longer than the file itself. The fixed-range fits are now withheld unless the decay was observed
  as far as −25 / −35 dB before the signal ends. T60 (adaptive range) and EDT are unaffected.

## [0.23.0] — 2026-08-06

### Added
- **Sound–motion entrainment module** (`ambiscape entrain <session> --motion <csv>`): joins an
  analysed session with a body-worn accelerometer series on one common clock and computes the
  Guo–Riaz–Jensenius crossmodal measures — fast-level × quantity-of-motion correlation, audio-azimuth
  × sway-direction circular correlation, and per-band phase-locking values (0.1–4 Hz), all with
  circular time-shift surrogate significance. Writes `entrain.json` + `entrain.png` and folds
  `ent_` descriptors into `summary.json`. Adds `circ_corr` (Jammalamadaka–SenGupta) to
  `circstats`.
- **Perceptual survey module (`ambiscape survey`).** ISO 12913-2 Method-A questionnaire responses
  (CSV, one row per respondent; 5-point or 100-point scales auto-detected, extra columns such as
  loudness ratings kept) are projected onto the ISO/TS 12913-3 pleasantness–eventfulness
  circumplex: per-respondent points, mean, 95% ellipse, and a `survey.png` circumplex plot.
  `srv_`-prefixed keys join `summary.json` so `catalog` can rank sessions perceptually next to the
  acoustic descriptors, and sessions with an existing acoustic summary get a short
  perception-vs-measurement table (LAeq vs pleasantness, events/min vs eventfulness). Until now
  ISO 12913-2 was supported by protocol rather than software; this closes the data-handling half
  of that gap (collection remains a protocol matter).
- **Impulse-response module** (`ambiscape.impulse`), three CLI commands. `sweep` generates a
  Farina exponential sine sweep with matched inverse filter (peak −6 dBFS for playback headroom,
  raised-cosine fades, JSON parameter sidecar from which the inverse regenerates bit-identically).
  `impulse` deconvolves a recorded sweep to `ir.wav` (pre-ringing and harmonic-distortion images
  trimmed ahead of the direct sound) and reports octave-band T60/T20/T30, EDT, C50/C80 and D50
  through the existing truncated-Schroeder machinery, plus STI (IEC 60268-16 indirect method —
  noise-free assumption documented) and early IACC (0–80 ms) for binaural IRs. `auralize`
  convolves dry audio with a measured IR via uniformly partitioned FFT convolution, resampling
  the IR to the dry material's rate and peak-matching the output by default.
- `analysis.decay_metrics` also reports the ISO 3382 fixed-range T20 and T30 when the band's
  dynamic range supports them (`decay_time` remains frozen).
- **Fluctuation strength.** MoSQITo (≤ 1.2.x) does not provide it, so `iso.fluctuation_strength`
  implements the Fastl & Zwicker envelope-modulation approximation (~4 Hz weighting, anchored to
  the 1 kHz / 100 % AM at 4 Hz ≙ 1 vacil reference) — documented as an approximation, not a
  standard. Pure numpy/scipy; `ambiscape iso` now reports it per ear alongside loudness, sharpness
  and roughness, and the `analyze` summary gains the always-available broadband
  `fluctuation_index` from the cached 20 ms envelope.
- **Tonal prominence.** DIN 45681-style tone detection (`iso.tone_prominence` /
  `iso.prominent_tones`): spectral peaks vs masking-band level, decibel prominence ΔL per tone,
  aggregated over the per-minute spectra — the ventilation/appliance-hum detector. New
  `ambiscape tones` command writes `tones.json`; the `analyze` summary and README gain
  `tonal_prominence_db` / `tonal_prominence_hz` / `n_prominent_tones` (also per state), and the
  percentile-LTAS figure marks the top persistent tones.

## [0.22.0] — 2026-08-03

Five versions' worth of work that had been tagged only locally. The version had been bumped in
`pyproject.toml` and never released, so PyPI sat at 0.17.0.

### Changed
- **Schaeffer and Schafer are separated in the taxonomy documentation.** They are related and they
  are not the same thing, and running them together invited exactly the confusion the docs existed
  to prevent. Schaeffer's typo-morphology (mass, facture) and Schafer's soundscape functions
  (keynote, signal, soundmark) now stand apart, with Krause and Pijanowski's source classification
  a third axis.
- **`anthropophony` renamed to `anthrophony`.** The old documentation URL still resolves, via
  `mkdocs-redirects`, so existing links do not break.
- Sound objects with only some annotations are now tolerated rather than rejected, and the package
  boundary is stated explicitly.

### Fixed
- Six defects in the resynth page, two of them audible.

## [0.17.0] — 2026-07-25
### Added
- Escapement module and per-strike carillon transcription.

## [0.16.0] — 2026-07-25
### Added
- Music analysis: circular pulse clarity, fifths-circle statistics, TARTYP profile.
### Changed
- Reframed as a holistic soundscape toolbox for any recording type, rather than an ambisonics tool.

## [0.15.1] — 2026-07-23
### Fixed
- The rhythm figure no longer crashes; `iso` and `draft` honour `-o`.

## [0.15.0] — 2026-07-22
### Added
- Mechanical, anthropophony and geophony domain detectors, with CLI commands.
- Features wired into Schaeffer typo-morphology.

## [0.14.0] — 2026-07-22
### Added
- Six new documentation guide pages with reproducible generated illustrations.
- `vision.py` as a CLI command, for multimodal analysis.
- First-class binaural mode.
- Carillon bell-inventory module and informed-prior rhythm.
### Fixed
- `iso.binaural` channel-order bug; the module is now mode-aware across mono, stereo, binaural and
  ambisonics.

## [0.2.0] — 2026-07-18
### Added
- Rhythm module for strike-level analysis of quasi-periodic sources.
- Spatial, schedule and timbre modules; swing-FM and masking.
- Modulation profile, tonality, and circular-statistics modules.
- `clock_offset_s` calibration.
- Test suite and CI on a synthetic AmbiX fixture.

## [0.1.0] — 2026-07-17
First release.
