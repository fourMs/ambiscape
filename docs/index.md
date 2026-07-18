# ambiscape

Analysis toolkit for **long-duration first-order ambisonic soundscape
recordings** — hours to whole nights of AmbiX B-format audio from recorders
like the Zoom H3-VR, processed in a stream with constant memory.

## Why

Most audio analysis tooling assumes a file you can load whole. Long-form
field recording produces something else: tens of gigabytes per session,
split into 2 GB files by the recorder, in which the interesting structure
lives at time scales of minutes to hours — machine duty cycles, diurnal
traffic envelopes, day/night state changes. ambiscape treats the *session*
(a folder of WAVs on one absolute clock) as its unit of analysis and
produces:

- **descriptors** in the environmental-acoustics idiom (Leq, LAeq,
  L10/L50/L90, event statistics),
- **spatial timelines** only ambisonics can supply — direction of arrival,
  diffuseness, azimuthal concentration,
- **figures** (session overview, percentile spectra, directograms,
  taxonomy maps and timelines),
- **room acoustics** (T60 from claps or incidental impulses),
- **machine states** — on/off segmentation, duty cycles, source spectral
  fingerprints, and targeted civic-grid scans (church clocks, sirens),
- **mains hum / ENF** — millihertz tracking of the electric network
  frequency for grid forensics and hum characterization,
- **ratings & global indices** — NR/NC/RC room criteria, intermittency
  ratio and emergence, the ecoacoustic battery (ACI, ADI/AEI, NDSI, BI,
  H), EDT/C50/C80/D50, and spatial descriptors (directional entropy,
  horizon fractions, foreground/background direction overlap),
- **biophony** — structural measures of nature/animal sound (narrowband
  activity, temporal entropy, spatial dispersion) plus an optional BirdNET
  species layer gated to hi-fi windows,
- **strike-level rhythm** of quasi-periodic sources — bells, machines —
  with periodicity, phase, and repetition-vs-variation statistics,
- **ISO 12913-3 psychoacoustic indicators** and a calibration hook,
- **machine-listening assists** (AudioSet tagging, a speech privacy gate),
- **publication exports** (non-identifying 1 Hz features; curated segment
  selection),
- **corpus aggregation** — one cross-session table (CSV + Markdown) from
  every session's cached summary, with ranking and outlier queries.

## Relationship to ambiviz

ambiscape is the **streaming companion to
[ambiviz](https://github.com/fisheggg/ambiviz)**. ambiviz renders rich
spatial visuals (AEM spherical energy maps, anglegrams, directograms) from
audio it can load whole; ambiscape summarizes recordings too long for that,
and selects the short representative excerpts that ambiviz then visualizes
in detail. Plot names and conventions follow ambiviz where the two overlap.

## Where things are documented

- **This site** — user guide and API reference, versioned with the code.
- **[README](https://github.com/fourMs/ambiscape#readme)** — install and
  a one-page overview.
- **[Wiki](https://github.com/fourMs/ambiscape/wiki)** — research context,
  field-recording protocol, design decisions, recipes, and roadmap: the
  living material that evolves independently of releases.
