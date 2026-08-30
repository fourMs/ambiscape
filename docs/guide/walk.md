# Soundwalks (walk mode)

A walking recording breaks the toolbox's core assumption — one microphone,
one place, one soundscape whose statistics converge with time. On a
soundwalk the non-stationarity *is* the signal, so `walk` inverts the
pipeline: segment the recording into sub-soundscape **zones** first, then
describe each zone as a short stationary session, the way ISO/TS 12913-2
soundwalks treat their stop points.

## Usage

```bash
ambiscape analyze WALK/                     # cache features first
ambiscape walk WALK/                        # zones, cadence, regimes
ambiscape walk WALK/ --gps track.gpx        # + distance, speed, route map
```

Outputs land in `WALK/analysis/`: `walk.md` (the zone table), 
`walk_zones.tsv` (one row per zone), `route_profile.png` (spectrogram,
level, cadence and diffuseness strips with zone boundaries), and — with a
track — `route_map.png` (the route coloured by zone).

## What it computes

- **Zones** from multivariate change-point detection on the cached 1 Hz
  features (`ambiscape.segmentation`): Foote novelty over log-octave
  powers, log-centroid, log-flatness and diffuseness, on physical dB-like
  scales with an absolute contrast floor (`--threshold-db`, default 4).
  Transitions on a walk are gradients; read boundaries as centres of
  transition zones.
- **Per-zone descriptors**: Leq, L10−L90, band fractions, flatness,
  centroid, diffuseness, event rate — each zone treated as a short
  stationary session.
- **Step cadence** from the 50 Hz envelope: `steps_per_min` is a
  walking-speed proxy, `step_salience` says how much of the zone's
  foreground is the walker's own feet.
- **A figure–ground regime per zone**: `own-steps` (the gait owns the
  foreground), `external-events` (things keep emerging), `bed-only`
  (nothing does), `mixed`.
- **Self-noise gating**: footfalls (envelope impulses, dilated) and wind
  (per-second low-band × diffuseness gusts) are masked out of a gated
  level. Zones with clear gait report `leq_gated_dbfs` — what the place
  sounds like without the walker — plus `step_time_fraction` and
  `wind_time_fraction`. On the reference walk the walking group was
  2–4 dB of its own quiet zones.
- **GPS** (`--gps track.gpx`): GPX track points give each zone a
  distance, mean speed and midpoint position; track speed is an
  independent cross-check on the acoustic cadence. `--gps-offset` shifts
  session time when recorder and tracker do not share a clock
  (`calibration.json` `clock_offset_s` is applied to the session already).

## Session-level honesty

`analyze` itself now diagnoses non-stationarity: a session whose features
hold several regimes says so in its README, because session-level
descriptors average over different places. Ecoacoustic indices (ACI,
NDSI, ADI) assume a fixed sensor and are not meaningful for a walk;
between-zone comparisons within one walk and one rig are the defensible
use of everything here.

## Companions

`ambiscape draft` proposes keynote beds per spectral regime (so a walk
drafts one bed per sub-soundscape, not per level band) and `ambiscape
taxonomy` renders the Schafer timeline — the natural sequence view of a
walk — and the Schaeffer map, whose detected events inherit any named
annotation spans that cover them.

Design notes and literature grounding live in the wiki's Soundwalk-Mode
page; the reference case is a 12.8-minute walk across the Blindern
campus (MUS2640, 2026-08-24).
