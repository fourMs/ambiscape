# Geophony

The non-biological, non-human ground of a soundscape: wind, rain, and water.
`geophony` reads the cached features (no audio pass) for two structures, and
its descriptors ride in the `analyze` summary.

## What it measures

- **Wind** — low-frequency (below ~200 Hz) energy that is *diffuse* and
  non-directional. Wind on the capsules is incoherent between channels, so
  ambisonic diffuseness rises; **`geo_wind_index`** is the low-band energy
  weighted by median diffuseness (it falls back to plain low-band energy for
  stereo/mono, which cannot resolve diffuseness).
- **Rain / water** — broadband high-frequency hiss that is spectrally flat and
  temporally steady (a shower, a fountain, a stream); **`geo_rain_index`** is
  the 2–16 kHz energy fraction times median spectral flatness.
- **`geo_lowfreq_fraction` / `geo_highband_fraction`** — the underlying band
  shares; **`geophony_index`** is the larger of wind and rain.

## Usage

```bash
ambiscape analyze <session>     # geo_* keys appear in the summary
ambiscape geophony <session>    # detail: writes geophony.json
```

```python
from ambiscape import features, geophony
F = features.load_features(sorted((out / "features").glob("*.npz")))
geophony.summarize_geophony(F)
```

## Caveats

Proxies, not detection. Wind and HVAC rumble share the low band (a train's
rumble reads partly as wind); rain, applause, and frying share the flat
high-band hiss. Diffuseness disambiguates wind only for ambisonic input.
Treat the indices as candidates to confirm by ear.
