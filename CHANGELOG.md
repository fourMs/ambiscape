# Changelog

Written 2026-08-03, reconstructed from the tag history and the commits between tags. Entries before
that date are therefore summaries of what the commits say, not notes written at the time.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/); the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) loosely, in that the minor number has
been carrying feature work throughout a pre-1.0 life.

## [Unreleased]

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
