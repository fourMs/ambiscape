# Features and descriptors

## Streaming feature extraction

Files are read in 60-second blocks; nothing is ever held whole in memory.
Per take, cached as `.npz`:

| Rate | Features |
|---|---|
| 125 ms | fast RMS level on W, unweighted and A-weighted (IEC 61672 bilinear IIR) |
| 1 s | octave-band powers (31.5 Hz–16 kHz), spectral centroid and flatness (50 Hz–16 kHz), 96-band log-frequency spectrogram row, per-octave pseudo-intensity vectors, broadband DOA (azimuth/elevation, 80–3000 Hz), diffuseness ψ |
| 1 min | full-resolution (5.9 Hz bins) mean power spectrum — for narrowband hum tracking, comb/fingerprint analysis, room modes |

Spectra come from Welch-style averaged 8192-point Hann FFTs at 0.1 s hops.

## Spatial estimators

The **pseudo-intensity vector** is
`I(f) = Re{ W*(f) · [X, Y, Z](f) }`, integrated over 80–3000 Hz (below the
array's spatial-aliasing region, above wind/handling rumble). Azimuth and
elevation come from its direction; **diffuseness** is

ψ = 1 − 2‖⟨Re W*·v⟩‖ / ⟨|W|² + ‖v‖²⟩

which is 0 for a single plane wave and 1 for an ideally diffuse field.
Directional statistics over time use circular means and the resultant
length **R** (0 = no concentration, 1 = all energy from one bearing).
Foreground/background splits use energy quartiles (loudest vs quietest
25 % of seconds).

## Session descriptors (`summarize`)

Follows the conventions frozen for the Intercontinental-database reports so
rows stay comparable across studies:

- **Leq, LAeq** (energy means of the fast level), **L10/L50/L90**
  exceedance percentiles, **dynamics** L10−L90;
- **events**: fast level ≥ 8 dB above a running background (10th percentile
  in a sliding 60 s window) for ≥ 0.25 s — rate, count, median duration;
- spectral centroid and flatness medians;
- ψ median and IQR; energy-weighted mean azimuth and R; median foreground
  elevation.

!!! tip "Reading ψ and R together"
    High ψ + high R = diffuse but anisotropic (an airport hall that
    "leans" one way). Low ψ + high R = a point-source room (one running
    machine). Low R with any ψ = scattered sources. The two numbers do
    work neither does alone.

## Segment selection

`pick_segments` proposes representative windows — quietest, most active,
typical, and (when a >6 dB state change exists) the transition — for
listening, archiving, ISO indicators, or ambiviz rendering.
