# Ratings and global indices

Sessions are reportable in three established idioms beyond
the corpus's own descriptor set: HVAC/room criteria, environmental-noise
descriptors, and soundscape-ecology indices. There are also spatial
descriptors that only an ambisonic corpus can supply. The borrowed
idioms earn their place because a building-services engineer, a noise consultant
and a soundscape ecologist each want the same recording expressed in the
numbers their own field argues in. Everything except the room criteria is
appended automatically to the `analyze` summary and README table.

## Room noise criteria (ventilation & building services)

```python
from ambiscape import iso

spec = iso.background_octaves_db(F, pct=50, offset_db=cal["dbfs_to_dbspl"])
iso.room_criteria(spec)
# {'NR': 28.3, 'NR_governing_hz': 250, 'NC': 27.9, 'RC': 25.6,
#  'RC_class': 'R'}
```

*NR* (ISO/R 1996, analytic curves), NC (ANSI S12.2, tabulated), and
a simplified RC (Blazier) with rumble/hiss classification, which together
are the language HVAC noise is specified in worldwide. They mean something
in absolute terms only with SPL calibration
(`calibration.json: dbfs_to_dbspl`); uncalibrated
ratings compare rooms recorded with one recorder and gain setup only.
Compute them per machine state (`states.state_segments` masks) to rate
"vent on" vs "vent off" separately.

## Environmental-noise descriptors

- **Intermittency ratio IR** (`analysis.intermittency_ratio`, Wunderli
  et al. 2016): the share of energy carried by events, near 0 for steady
  drones and dense traffic, high when energy arrives as distinct events.
  It is the single best complement to events/min.
- **Emergence** LAeq − LA90: how far the energetic mean sits above the
  background, which is the classic "how eventful" number. Being built on
  an energy average it is set by the loudest frames of the span, so in a
  quiet room it should be read next to the trimmed level and the
  percentiles—see
  [Reading energy averages](descriptors.md#reading-energy-averages).

## Prominent tones (`ambiscape tones`)

DIN 45681-style tone detection on the cached per-minute mean spectra:
each narrowband spectral peak is compared against the masking-noise level
of its surrounding critical band, giving a decibel prominence
**ΔL = L_tone − L_noise** per tone (≥ 6 dB, the decisive audibility
criterion of DIN 45681, counts as prominent). Detections are aggregated
over time, so a ventilation or appliance hum shows up as one persistent
line with its presence fraction, while a passing siren does not:

```bash
ambiscape tones SESSION/          # needs a prior analyze run
#   152.3 Hz: ΔL 35.5 dB (max 35.5), present 100% (180 min)
```

The strongest persistent tone lands in the `analyze` summary and README
as `tonal_prominence_db` / `tonal_prominence_hz` (with
`n_prominent_tones`), per state in the state-resolved table — rate "vent
on" vs "vent off" tones separately — and the percentile-LTAS figure marks
the top tones. ΔL is a level *difference*, so it is meaningful without
SPL calibration. This follows the method of DIN 45681, not a certified
implementation (no frequency-dependent masking index or uncertainty
term); pure numpy/scipy, always available.

The summary also carries `fluctuation_index`, a broadband 4 Hz-weighted
modulation-depth index from the cached 20 ms envelope (≈ 0 for a steady
drone, high for a ~4 Hz wobble) — the cheap companion to the approximate
`iso.fluctuation_strength` (vacil) computed per ear by `ambiscape iso`
(see the room-acoustics page).

## Ecoacoustic indices (`ambiscape.ecology`)

ACI, ADI/AEI, NDSI, BI, and acoustic entropy H make up the standard
soundscape-ecology battery, computed from the cached 96-band spectrogram.
They are reported for cross-corpus comparability. Indoors, read NDSI/BI as
"energy in the 2–8 kHz band", not as proof of biophony, since a
ventilation hiss scores as birds. Combine with the taxonomy layer before
interpreting.

ACI is averaged over complete 5-minute chunks, and its magnitude depends on
the chunk length, so a recording shorter than one chunk has no comparable
value: `aci` is then `None`. Clip corpora of 5–30 s carry no ACI at all.

## Spatial descriptors (`ambiscape.spatial`)

- **Directional entropy**: how many directions the place sounds from
  (0 = one bearing, 1 = even around the horizon), the spatial analogue of
  a diversity index.
- **Horizon fractions**: energy from above / around / below ±10°
  elevation, which separates ceiling-mounted services and birds from
  footsteps and ground traffic.
- **Foreground/background azimuth overlap**: Bhattacharyya overlap of the
  loudest-25 % and quietest-25 % direction histograms, 1 when figure and
  ground share a direction (one-source rooms), 0 when they occupy
  different sectors.

## Room-acoustics companions

`analysis.decay_metrics` extends `decay_time` (unchanged, its numbers feed
frozen reports) with EDT, C50/C80, and D50 per octave from the
same truncated-Schroeder decay, which give reverberance and clarity from
any good impulse in the recording, a door slam or a dropped book.
