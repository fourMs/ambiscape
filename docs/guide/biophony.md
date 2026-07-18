# Biophony: nature and animal sounds

The ecoacoustic battery ([ratings & indices](indices.md)) reads *energy in
a band* — indoors a 4 kHz ventilation hiss scores as "biophony", and even
outdoors NDSI cannot tell a dawn chorus from cicadas from wind. Version 0.6
adds measures that capture biophony by its **structure**: narrowband,
tonal, bursty in time, and — the ambisonic advantage — arriving from many
elevated bearings at once.

Two layers: a cache-based structural set (`ambiscape.biophony`, no ML,
scales to a whole global corpus) and an optional species detector
(BirdNET via `ambiscape.ml`, `[ml]` extra) for ground truth on the good
windows.

## Structural measures (`ambiscape.biophony`)

```python
from ambiscape import biophony

biophony.summarize_biophony(F)
# {'bird_peaks_per_min': 5.0, 'bird_band_activity_pct': 21.3,
#  'bird_temporal_entropy': 0.71, 'bird_directional_entropy': 0.83,
#  'bird_above_horizon_fraction': 0.74, ...}
```

- **`narrowband_activity`** — persistent narrow spectral peaks in the bird
  band per minute (from the high-resolution per-minute PSD). Birdsong is
  narrowband and tonal; wind and machines are broadband.
- **`band_temporal_entropy`** — Sueur Ht of the bird-band envelope: low
  when energy is concentrated into vocalizations, near 1 for a flat band.
- **`band_activity`** — active-second fraction and event rate where the
  bird band rises above its own running background (Towsey-style).
- **`spatial_dispersion`** — the layer no other corpus tool has: the
  directional entropy and above-horizon fraction of the *bird-band
  foreground*. A chorus of many birds from many elevated bearings scores
  high on both; it cross-checks a suspicious NDSI.

The default band is 2–11 kHz (temperate birdsong). Widen it per habitat —
insects reach 8–16 kHz, many mammals and owls sit below 2 kHz.

!!! warning "Proxies, not detections"
    A tonal alarm, a kettle, or a squealing fan belt can mimic biophonic
    structure. These measures flag *where biophony is likely*; confirm
    species with BirdNET, and always read them beside the taxonomy layer.

## Species detection (`ambiscape.ml`, `[ml]` extra)

```python
from ambiscape import ml

doc = ml.birdnet_session(sess, F=F, hifi_max_diffuse=0.75,
                         lat=52.38, lon=4.64)   # Haarlem
# {'n_species': 3, 'species': [{'common_name': 'Eurasian Collared-Dove',
#   'species': 'Streptopelia decaocto', 'n': 6, 'max_conf': 0.82}, ...]}
```

Or `ambiscape birdnet <folder> --lat 52.38 --lon 4.64 --hifi-max-diffuse
0.75`. Passing the cached features `F` with `hifi_max_diffuse` runs BirdNET
**only on hi-fi windows** — where a masking drone has lifted and birds are
actually legible — instead of wasting inference on masked hours.
`lat`/`lon` enable BirdNET's location/season species filter, cutting false
positives. BirdNET analyzes the W channel resampled to 48 kHz; spatial
structure comes from the `biophony` measures, species identity from here.
