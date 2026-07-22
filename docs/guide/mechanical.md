# Mechanical & transport

Engines, machinery, and rail/road traffic are broadband but
low-frequency-weighted, and temporally steady or quasi-periodic (engine
firing, bogies over rail joints, a compressor duty cycle). Before, the pieces
were scattered across `rhythm`, `spatial`, `states`, and `compare`;
`mechanical` gives the domain a first-class home. It reads the cached features
(no audio pass) and its descriptors ride in the `analyze` summary.

## What it measures

- **`mech_lowfreq_fraction`** — share of octave energy below 250 Hz. Traffic
  and machinery pile energy low; a train is typically >0.9.
- **`mech_rumble_db`** — mean level in the 31.5–125 Hz rumble band.
- **`mech_periodicity_hz` / `mech_periodicity_strength`** — the peak of the
  50 Hz broadband envelope's modulation spectrum in 0.3–12 Hz (engine / bogie
  / duty-cycle rhythm) and how strongly it stands out.
- **`mechanical_index`** — low-frequency weight scaled by periodicity
  prominence, in [0, 1].

## Usage

```bash
ambiscape analyze <session>      # mechanical_* keys appear in the summary
ambiscape mechanical <session>   # detail: writes mechanical.json
```

```python
from ambiscape import features, mechanical
F = features.load_features(sorted((out / "features").glob("*.npz")))
mechanical.summarize_mechanical(F)
```

## Caveats

These are acoustic-structure proxies, not detections. Indoor HVAC rumble and a
passing lorry look alike here, and a steady tonal machine may also light up
[Mains hum & ENF](enf.md). For directional confirmation use the pass-by view
in [Spatial analysis](spatial.md); for pitched machinery, the strike
periodicity in [Strike-level rhythm](rhythm.md).
