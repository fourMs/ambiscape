# Ratings and global indices

Version 0.5 makes sessions reportable in three established idioms beyond
the corpus's own descriptor set — HVAC/room criteria, environmental-noise
descriptors, and soundscape-ecology indices — plus spatial descriptors only
an ambisonic corpus can supply. Everything except the room criteria is
appended automatically to the `analyze` summary and README table.

## Room noise criteria (ventilation & building services)

```python
from ambiscape import iso

spec = iso.background_octaves_db(F, pct=50, offset_db=cal["dbfs_to_dbspl"])
iso.room_criteria(spec)
# {'NR': 28.3, 'NR_governing_hz': 250, 'NC': 27.9, 'RC': 25.6,
#  'RC_class': 'R'}
```

**NR** (ISO/R 1996, analytic curves), **NC** (ANSI S12.2, tabulated), and
a simplified **RC** (Blazier) with rumble/hiss classification — the
language HVAC noise is specified in worldwide. Meaningful absolutely only
with SPL calibration (`calibration.json: dbfs_to_dbspl`); uncalibrated
ratings compare rooms recorded with one recorder+gain setup only. Compute
them per machine state (`states.state_segments` masks) to rate "vent on"
vs "vent off" separately.

## Environmental-noise descriptors

- **Intermittency ratio IR** (`analysis.intermittency_ratio`, Wunderli
  et al. 2016): the share of energy carried by events — near 0 for steady
  drones and dense traffic, high when energy arrives as distinct events.
  The single best complement to events/min.
- **Emergence** LAeq − LA90: how far the energetic mean sits above the
  background — the classic "how eventful" number.

## Ecoacoustic indices (`ambiscape.ecology`)

ACI, ADI/AEI, NDSI, BI, and acoustic entropy H — the standard
soundscape-ecology battery, computed from the cached 96-band spectrogram.
Reported for cross-corpus comparability; indoors, read NDSI/BI as "energy
in the 2–8 kHz band", not as proof of biophony (a ventilation hiss scores
as birds). Combine with the taxonomy layer before interpreting.

## Spatial descriptors (`ambiscape.spatial`)

- **Directional entropy** — how many directions the place sounds from
  (0 = one bearing, 1 = even around the horizon); the spatial analogue of
  a diversity index.
- **Horizon fractions** — energy from above / around / below ±10°
  elevation: separates ceiling-mounted services and birds from footsteps
  and ground traffic.
- **Foreground/background azimuth overlap** — Bhattacharyya overlap of the
  loudest-25 % and quietest-25 % direction histograms: 1 when figure and
  ground share a direction (one-source rooms), 0 when they occupy
  different sectors.

## Room-acoustics companions

`analysis.decay_metrics` extends `decay_time` (unchanged, its numbers feed
frozen reports) with **EDT**, **C50/C80**, and **D50** per octave from the
same truncated-Schroeder decay — reverberance and clarity from any good
impulse in the recording.
