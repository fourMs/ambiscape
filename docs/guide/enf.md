# Mains hum and the grid frequency (ENF)

Indoor recordings hum with the electricity supply: 50 Hz (nominal, Europe)
and harmonics, with magnetostriction strongest at 100 Hz. The grid's
*actual* frequency wanders by tens of millihertz as load and generation
balance — the electric network frequency, ENF — and a long recording
carries that wander as a continuous, involuntary log of the grid.
Version 0.4 adds a reader for it, developed on the Haarlem-loft case study
(9 h of hum tracked to ±50 mHz).

The cached per-minute spectra are far too coarse for this (5.9 Hz bins);
`ambiscape.enf` makes its own pass over the raw W channel.

## Tracking

```python
from ambiscape import enf

tr = enf.enf_track(sess, step_s=300, win_s=60, harmonics=(1, 2))
enf.enf_summary(tr)
# {'mean_hz': 49.9958, 'sd_mhz': 21.0, 'max_dev_mhz': 88.0,
#  'coverage': 1.0, 'harmonic_agreement_mhz': 0.9, ...}
```

`hum_peak` measures one window (zero-padded FFT + parabolic interpolation
→ millihertz precision on a 60 s window); `enf_track` walks a whole
session, skipping the sliver reads that recorders' overlapping 2 GB splits
produce at take boundaries.

## Reading the summary

- **`harmonic_agreement_mhz`** is the authenticity check: the 50 Hz and
  100 Hz hum are independent acoustic lines driven by the same electrical
  frequency, so millihertz agreement confirms you are looking at the grid
  and not at a machine. In Haarlem a second stable line at ~49.8 Hz — a
  rotor just under synchronous speed — masqueraded as mains in coarse
  spectra; its wander was 17× the grid's and it tracked nothing.
- A Continental-Europe trace should sit near 50.000 Hz with SD ~20 mHz,
  rarely leaving ±50 mHz (the ENTSO-E normal band). Systematic offsets
  mean either a mechanical line or a recorder sample-clock error.
- **Forensics:** matched against published grid-frequency archives, an ENF
  trace timestamps a recording to the second, independently of the
  recorder clock — a cross-check for `schedule.clock_offset`.

## As a corpus descriptor

`enf_summary`'s `median_rise_db` and `coverage` say how electrified a room
sounds: a machine-room measured +45 dB of 50 Hz line, a quiet hotel room
essentially none. Compare across sessions before interpreting any
low-frequency tonal finding — the hum is in nearly every indoor recording.
