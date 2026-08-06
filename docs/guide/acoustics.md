# Room acoustics, calibration, and ISO indicators

## Reverberation from claps (`decay_time`)

Any sufficiently loud impulse, whether a deliberate calibration clap, a
balloon pop or an incidental bang, yields octave-band T60 estimates via
truncated Schroeder integration. We added two safeguards after learning
the hard way that they were needed:

1. **noise-floor truncation**: the decay fit runs from −5 dB down to
   `max(−35 dB, floor + 8 dB)`, so a high ambient floor cannot masquerade
   as long reverberation;
2. **re-attack truncation**: integration stops at the first point where the
   envelope rises ≥ 8 dB above its running minimum, since movement right
   after a clap otherwise contaminates the tail.

Report T20/T30-style extrapolations only when they agree; treat impact
sources as ±20 % (they ring structurally, which reads as extra decay,
especially below 250 Hz). A balloon pop at ≥ 2 m with 5 s of stillness
around it approaches ISO 3382-2 survey grade.

For a deliberate measurement, a swept sine buys far more dynamic range
than any clap and adds STI, IACC and auralization — see
[Impulse response & auralization](impulse.md).

## Calibration

`calibration.json` in the session folder defines the offset `O` such that a
signal at −X dBFS corresponds to (O − X) dB SPL:

```json
{"dbfs_to_dbspl": 94.0, "method": "SPL meter vs running HVAC"}
```

With it, `analyze` adds `leq_db_spl`, `laeq_db_spl`, `L10/L50/L90_db_spl`
to the summary (ISO 1996-comparable), and the ISO indicators below run in
true pascals.

### Deriving the offset in the field (`ambiscape calibrate`)

You rarely know the offset, but any SPL meter (or a phone SPL app held
next to the microphone) gives you a reading you can anchor to later:

```bash
ambiscape calibrate SESSION/ --spl 62.5 --t0 0 --dur 60 \
    --method "Phone SPL app on the bench, LAeq over the first minute"
ambiscape calibrate SESSION/ --offset 94.0 \
    --method "1 kHz calibrator, 94 dB"          # skip derivation
```

The first form computes the recording's own LAeq (dBFS) over the stated
span from the cached features and stores `meter reading − recording LAeq`
as the offset; note the reading and the time window in your field notes
and run the command whenever. Existing keys in `calibration.json` (clock
corrections etc.) are preserved.

Multi-device sessions (a Zoom and a phone running side by side) need
different offsets per device: pass `--take <filename>` to store the
offset in a per-take map, which the ISO indicators resolve per segment:

```json
{"dbfs_to_dbspl": 83.2,
 "dbfs_to_dbspl_takes": {"260728_100056-station.m4a": 77.0}}
```

Field habit worth forming: at every session, note one SPL-app LAeq
reading and when you took it. It costs ten seconds and turns every
uncalibrated dBFS table into dB SPL, retroactively.

The same file may carry a clock correction:

```json
{"clock_offset_s": 665.0,
 "method": "ringing of the town bells ends 21:30:00 sharp"}
```

`clock_offset_s` seconds are added to every take's start time when the
session is opened (positive = the recorder clock was slow), so figures,
annotations, and reports all agree on corrected wall-clock time. Recorder
clocks drift; calibrate them against any event of known time, such as a
scheduled bell, a radio time signal, or a phone alarm captured on the
recording. Both keys are optional and independent.

## ISO 12913-3 indicators (`ambiscape iso`)

Computes, per ear, on each representative segment:

- **N5, N50**: ISO 532-1 time-varying loudness percentiles (sone),
- **sharpness**: DIN 45692 (acum),
- **roughness**: Daniel & Weber (asper),

via [MoSQITo](https://github.com/Eomys/MoSQITo) (validated against the
1 kHz / 60 dB ≙ 4 sone reference). Ear signals come from ambiviz's HRIR
binauralizer when installed, else a documented ±90° cardioid-pair fallback
(no pinna cues). Uncalibrated sessions run with an assumed offset and are
flagged: segment-to-segment ratios stay meaningful, absolute sones do
not.

!!! note "Cost"
    MoSQITo runs ~5× slower than realtime; defaults are 30 s per segment
    with roughness on a central 10 s slice (roughness is a texture measure
    and stabilises within seconds).

We do not claim full 12913-2 conformance, since that also implies a
calibrated Class-1 chain, which consumer recorders are not. The honest
claim is *"ISO 12913-informed collection and 12913-3-style indicators"*
with the protocol documented.
