# Carillon bell inventory

`ambiscape carillon` answers a musical question a swinging-bell rhythm analysis
cannot: **which bells did a carillon play?** A carillon is a tuned, chromatic set
of cast bells played from a keyboard, so the object of interest is the *set of
distinct strike notes* — the instrument's range in use, its tuning, and the
pitch-class centre of the music.

```bash
ambiscape carillon <session-folder> --t0 60 --dur 3600   # needs the [music] extra
```

Writes `carillon.json` (detected bells with note names, frequencies and cents,
range, pitch-class profile, strike count) and `carillon.png` (the bell-salience
inventory with note labels, over a pitch-class profile).

## Why naive pitch tracking fails

A struck bell is *inharmonic*. Its principal partials sit at fixed ratios to the
strike note (the prime):

| partial | ratio | interval |
|---|---|---|
| hum | 1/2 | octave below |
| prime | 1 | the note |
| **tierce** | **6/5** | **minor third above** |
| quint | 3/2 | fifth above |
| nominal | 2 | octave above |

The loud tierce and nominal make a naive pitch tracker report notes a minor third
or an octave too high. `carillon` instead does **bell-template matching**: for
every candidate strike note it sums the energy at all five partial positions, so
a true strike note — which has energy at *all* of them — outscores any single
partial. Accumulated over the onset frames of a whole recital, the peaks of that
salience are the bells that were played.

The minor-third **tierce is the bell fingerprint**: an octave-down "hum ghost" or
a tierce/quint ghost of a real bell lacks its *own* tierce, so a genuine-tierce
gate removes them cleanly (validated on synthetic bells in the test suite).

## Caveats

- **Distant / soft / masked bells are missed.** Salience falls toward the noise
  floor in the treble; a bell struck rarely, or buried under the chord above it,
  may not clear the floor. The inventory is a *lower bound* on the instrument.
- **Tuning cents are relative to A4 = 440 equal temperament.** A consistent
  offset (e.g. a historic carillon tuned flat, or in meantone) shows as a
  systematic cents deviation across the detected bells.
- Runs on the W channel resampled to 22.05 kHz, in chunks, so an hour-long
  recital is fine.
