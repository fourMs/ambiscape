# Environmental rhythm (modulation profile)

A soundscape is rhythmic on very different time scales at once: bell strikes
and footsteps beat at a few hertz, traffic waves and surf breathe over tens
of seconds, and machines and human activity switch on a duty cycle of
minutes to hours. `ambiscape modspec` measures all three from the cached
envelopes, with no second pass over audio, as a modulation profile: a
log-frequency modulation spectrum per scale, plus a rhythm spectrogram of
the whole session.

![Envelope modulation spectra by scale (micro/meso/macro) over a rhythm spectrogram of the session.](../img/modulation_profile.png)

```bash
ambiscape modspec <session-folder>   # needs a prior analyze run
```

Writes `modulation.json` (per-scale dominant modulation frequency, period,
prominence and band modulation depth, plus the raw spectra) and
`modulation_profile.png` (the spectra by scale over a 10-minute-window
rhythm spectrogram, in dB relative to each window's median).

## The three scales

- **micro** (0.5–20 Hz) from the 20 ms broadband envelope: strike
  patterns, footstep cadence, flutter. This needs the high-rate `env_hi`
  cache (extractor ≥ 0.2); on older caches micro falls back to the fast
  level and tops out at 4 Hz (flagged as `micro_limited` in the JSON).
- **meso** (0.01–0.5 Hz) from the 125 ms fast level: traffic waves, surf,
  wind gusts, conversational turn-taking.
- **macro** (below 0.01 Hz, floor set by session length) from the 1 s RMS:
  ventilation and appliance duty cycles, diel activity.

Each scale comes from a different cached envelope, since no single
envelope rate resolves both a footstep and a fridge that runs for eight
minutes in every thirty-six.

## One spectrum, one normalisation

All three scales are computed identically: the source stream is converted
to a linear-power envelope, normalised to unit mean, and its Welch power
spectral density taken (Welch bins averaged into a log-frequency grid).
The three curves therefore share one axis — PSD in dB re 1/Hz of the
unit-mean power envelope — and are directly comparable in level, with no
per-band offsets. Each scale is additionally computed half a decade past
its nominal band edges, so adjacent scales overlap; in the figure the
overlap is drawn faint and the nominal band solid, and the curves joining
where they meet is the visible check that the shared normalisation holds
(on the SINS reference sessions, adjacent scales agree to within a couple
of dB in the overlaps). The reported statistics are always taken within
the nominal band only.

Each scale reports `peak_freq_hz`, `peak_period_s`, `peak_prominence_db`
(peak over the band median) and `modulation_depth` (band-integrated
modulation power of the unit-mean envelope). A sharp, prominent peak means
periodic structure; a flat spectrum means the level just wanders.

## In Python

```python
from ambiscape import modulation

prof = modulation.profile(F)          # F = load_features(...)
prof["scales"]["meso"]
# {'peak_freq_hz': 0.033, 'peak_period_s': 30.0,
#  'peak_prominence_db': 7.4, 'modulation_depth': 0.21}
```

`modulation.modulation_spectrogram(env, dt)` returns the windowed version
directly, as `(t_centers, mod_freqs, S)`, if you want to build the rhythm
spectrogram from a custom envelope.
