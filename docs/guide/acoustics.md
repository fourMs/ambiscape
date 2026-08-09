# Room acoustics, calibration, and ISO indicators

## Reverberation from claps (`decay_time`)

Any sufficiently loud impulse, whether a deliberate calibration clap, a
balloon pop or an incidental bang, yields octave-band T60 estimates via
truncated Schroeder integration. Three safeguards protect the estimate,
each guarding against a failure mode common in field recordings:

1. **noise-floor truncation**: the decay fit runs from −5 dB down to
   `max(−35 dB, floor + 8 dB)`, so a high ambient floor cannot masquerade
   as long reverberation;
2. **re-attack truncation**: integration stops at the first point where the
   envelope rises ≥ 8 dB above its running minimum, since movement right
   after a clap otherwise contaminates the tail;
3. **unmeasured range**: T20 and T30 are withheld unless the decay was
   observed as far as −25 / −35 dB before the signal ends. A trimmed
   impulse response from an archive has no noise floor for safeguard 1 to
   find, so the fixed-range fits would otherwise extrapolate past the end
   of the file; T60 (adaptive range) and EDT still stand.

!!! warning "Camera audio can replace a recorder for T60 — and not for the rest"

    Measured on 7 days where a GoPro MAX and a Zoom H3-VR recorded the same
    room and the same clap, aligned by envelope cross-correlation and
    verified before comparison:

    | | camera − recorder (median) | recorder median |
    |---|---|---|
    | **T60** | **−0.01 s** | 0.41 s |
    | EDT | **+0.24 s** | 0.34 s |
    | C50 | **−5.5 dB** | 8.2 dB |

    Reverberation time survives the substitution almost exactly. The early
    field does not: the camera reports a slower early decay and a room 5.5 dB
    less clear than it is, consistent with its own gain and noise processing
    lifting the tail — which is where EDT and C50 live.

    This matters well beyond one corpus. Cameras are often the recorder with
    full coverage, and consumer cameras do not document their processing, so
    treat any camera-derived clarity or early-decay figure as unusable and
    any camera-derived T60 as sound. **Do not mix the two sources in one
    distribution without checking**, and do not assume the result transfers
    to another camera model: this was measured on one, and the processing is
    undocumented and may change with firmware.

    Matching matters too. The recorders here are unsynchronised — the
    measured offsets ran from −51 s to +14 s — so pairing by timestamp
    compares different events. Waveform correlation cannot verify a pair
    either, since the camera's coding changes the waveform of the same clap
    (peaks of 0.03–0.14 on every day tried). Align on the level envelope.

!!! warning "Trimmed IRs: regenerate anything measured before 0.28.1"

    `ir_metrics` prepends silence when the peak sits near the start of the
    buffer, so the estimator has samples before it. Until 0.28.1 the noise
    floor was then measured *on that silence*: a trimmed IR with a real
    45 dB floor reported 173 dB of dynamic range, which satisfies every
    guard above and lets T20/T30 be fitted over noise without complaint.

    Since 0.28.1 the floor is read from the quietest part of the decay
    instead, and `dr_db` tracks the real signal-to-noise ratio. Numbers
    change for trimmed material only — an IR with genuine pre-roll was
    never affected.

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

Computes, per ear, on each representative segment (`--dur` seconds each,
default 30; `--offset` overrides the calibration offset):

- **N5, N50**: ISO 532-1 time-varying loudness percentiles (sone),
- **sharpness**: DIN 45692 (acum),
- **roughness**: Daniel & Weber (asper),
- **fluctuation strength** (vacil, approximate — see below),

the first three via [MoSQITo](https://github.com/Eomys/MoSQITo), whose
loudness implementation follows the 1 kHz / 60 dB ≙ 4 sone reference;
fluctuation strength is computed locally (see below). Results are written
to `analysis/iso_indicators.json`, one block per segment with per-ear
values, the `binaural_method` used, and a `calibrated` flag.
Ear signals come from ambiviz's HRIR
binauralizer when installed, else a documented ±90° cardioid-pair fallback
(no pinna cues). Uncalibrated sessions run with an assumed offset and are
flagged: segment-to-segment ratios stay meaningful, absolute sones do
not.

!!! note "Cost"
    MoSQITo runs ~5× slower than realtime; defaults are 30 s per segment
    with roughness on a central 10 s slice (roughness is a texture measure
    and stabilises within seconds).

The toolbox does not claim full 12913-2 conformance, since that also
implies a calibrated Class-1 chain, which consumer recorders are not. The
honest claim is *"ISO 12913-informed collection and 12913-3-style
indicators"* with the protocol documented. The questionnaire half of
12913-2 — Method A responses projected onto the 12913-3 circumplex — is
covered by the [perceptual survey](survey.md) module. Two related rating
families live on the [ratings & global indices](indices.md) page:
NR/NC/RC room criteria (`iso.room_criteria`) and DIN 45681-style
prominent tones (`ambiscape tones`, `iso.prominent_tones`).

### Fluctuation strength (approximation)

MoSQITo (≤ 1.2.x) offers no fluctuation strength, and no standard exists
for it, so `iso.fluctuation_strength` implements the Fastl & Zwicker
envelope-modulation model in spirit: per Zwicker critical band, the
< 32 Hz envelope level depth ΔL is weighted by the band-pass
modulation-frequency weighting that peaks at **4 Hz** (the slow wobble
sitting between level drift and roughness) and by the coherence of the
dominant modulation, then summed over Bark bands and scaled so the classic
reference — a 1 kHz tone, 100 % AM at 4 Hz — reads **1 vacil**. Treat
absolute values as indicative and same-pipeline comparisons as the
meaningful output; masking-based depth and level dependence are
simplified. It is pure numpy/scipy (no `[iso]` extra needed on its own)
and also feeds the always-available `fluctuation_index` in the `analyze`
summary, computed from the cached 20 ms broadband envelope without an
audio pass.
