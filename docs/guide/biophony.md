# Biophony: nature and animal sounds

The ecoacoustic battery ([ratings & indices](indices.md)) reads *energy in
a band*. Indoors a 4 kHz ventilation hiss scores as "biophony", and even
outdoors NDSI cannot tell a dawn chorus from cicadas from wind. The
`biophony` module instead captures biophony by its structure: narrowband,
tonal, bursty in time, and, where the recording is ambisonic, arriving from
many elevated bearings at once.

The measures come in two layers: a cache-based structural set
(`ambiscape.biophony`, no ML, scales to a whole global corpus) and an
optional species detector (BirdNET via `ambiscape.ml`, `[ml]` extra) for
ground truth on the good windows.

## Structural measures (`ambiscape.biophony`)

```python
from ambiscape import biophony

biophony.summarize_biophony(F)
# {'bird_peaks_per_min': 5.0, 'bird_band_activity_pct': 21.3,
#  'bird_temporal_entropy': 0.71, 'bird_directional_entropy': 0.83,
#  'bird_above_horizon_fraction': 0.74, ...}
```

- **`narrowband_activity`**—persistent narrow spectral peaks in the bird
  band per minute (from the high-resolution per-minute PSD). Birdsong is
  narrowband and tonal; wind and machines are broadband.
- **`band_temporal_entropy`**—Sueur Ht of the bird-band envelope: low
  when energy is concentrated into vocalisations, near 1 for a flat band.
- **`band_activity`**—active-second fraction and event rate where the
  bird band rises above its own running background (Towsey-style).
- **`spatial_dispersion`**—the layer no other corpus tool has: the
  directional entropy and above-horizon fraction of the *bird-band
  foreground*. A chorus of many birds from many elevated bearings scores
  high on both; it cross-checks a suspicious NDSI.

The default band is 2–11 kHz (temperate birdsong). Widen it per habitat,
since insects reach 8–16 kHz and many mammals and owls sit below 2 kHz.

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
only on hi-fi windows, where a masking drone has lifted and birds are
actually legible, instead of wasting inference on masked hours.
`lat`/`lon` enable BirdNET's location/season species filter, cutting false
positives. BirdNET analyses the W channel resampled to 48 kHz; spatial
structure comes from the `biophony` measures, species identity from here.

!!! warning "A species list is a list of candidates"
    The classifier returns confident species where there were none, and it
    does so by more than one route.

    Run over a university corridor in January it returned eight Great
    Bittern detections, the best at 0.87, alongside Gray Heron, Tawny Owl
    and Red-throated Loon from other empty interiors. A bittern booms at
    roughly 150–200 Hz, which is where a ventilation plant lives, and that
    looked like the mechanism.

    It is one mechanism, not the mechanism. A second corpus of 37 sessions
    put a Long-eared Owl in 24 of them — 65 per cent, in daylight, in a
    city, across spring and summer — and the association with those
    sessions' mechanical index ran the *other* way: median 0.066 where the
    owl appeared against 0.292 where it did not, a factor of 4.5 at
    Mann–Whitney p = 0.00039. Those false detections lived in the quiet,
    where a low noise floor lets faint and ambiguous material through.

    Three habits follow:

    - **Run a control, not a higher threshold.** Same settings, same
      recorder, recordings that certainly contain no birds — a plant room,
      a corridor at night — and treat every species that comes back as
      confusable. Include quiet controls as well as loud ones. Report what
      the control removed, rather than quietly reporting the remainder.
    - **Read the rate, not the entry.** A regionally unremarkable species
      becomes a finding or an artefact depending on how often it appears.
      Long-eared Owls do live around that city; they are not in 24
      different places across two seasons in the middle of the day.
    - **Check the calendar against the range.** The same list held a Common
      Swift in October, and swifts have left the country by the end of
      August. One detection out of season disqualifies itself on a fact
      nothing acoustic can rescue.

    A low confidence threshold is defensible — a low threshold with an
    explicit validation step beats a high one that hides its own failures —
    and it is only defensible while the validation step actually happens.
