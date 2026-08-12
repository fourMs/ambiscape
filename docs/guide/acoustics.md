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

    Measured over a full year: a GoPro MAX and a Zoom H3-VR recording the
    same room and the same clap, 369 paired events, aligned by envelope
    cross-correlation and each window verified to contain a transient before
    it was measured. Reading the camera's *4-channel PCM ambisonic* track:

    | | camera − recorder | 95 % limits of agreement |
    |---|---|---|
    | **T60** | **+0.03 s** | −0.39 to +0.40 s |
    | EDT | **+0.31 s** | |
    | C50 | **−4.0 dB** | |

    Reverberation time survives the substitution in the median almost
    exactly. The early field does not: the camera reports a slower early
    decay and a room about 3.6 dB less clear than it is.

    **Read the limits, not just the median.** Those T60 limits span about
    0.8 s, against a median room of half a second. The camera is *unbiased and imprecise*: a distribution of T60
    computed from camera audio lands where the recorder's would, and any
    single measurement may be far out. Per event the camera falls within the
    ±20 % usually granted to a clap-derived estimate on 77 % of days. Use
    camera audio to populate a distribution; do not use it to certify a room.

    **Read the right track** — but for precision, not accuracy. A GoPro MAX
    `.360` carries both a processed stereo AAC mix and a 4-channel PCM
    ambisonic stream, and `ffmpeg` will hand you the stereo one by default.
    On bias the two layers are near-identical (C50 −3.6 dB against −3.8 dB);
    where they differ is spread, the stereo mix giving T60 limits of −0.59
    to +0.53 s against the ambisonic stream's −0.39 to +0.40. Note also that
    the small `.LRV` proxy carries *only* the stereo mix, so it cannot be
    used as a cheap stand-in for ambisonic work.

    *Corrected 2026-08-09.* This box previously said that "roughly half the
    apparent clarity deficit is the convenience track", from EDT +0.24 s and
    C50 −5.5 dB on the stereo track over 7 days. That was two artefacts, not
    a finding: the windows were cut from an alignment rather than from the
    transient, and the stereo layer was measured on its *left channel*
    while the ambisonic layers were measured on W. A directional channel
    collects a different share of the direct sound, which is exactly what
    C50 measures. Measured on the mid signal (L+R)/2, with windows centred
    on the clap, the difference between the layers' bias disappears.

    This matters well beyond one corpus. Cameras are often the recorder with
    full coverage, and consumer cameras do not document their processing, so
    treat any camera-derived clarity or early-decay figure as unusable and
    any camera-derived T60 as sound. *Do not mix the two sources in one
    distribution without checking*, and do not assume the result transfers
    to another camera model: this was measured on one, and the processing is
    undocumented and may change with firmware.

    Matching matters too. The recorders here are unsynchronised — the
    measured offsets ran from −51 s to +14 s — so pairing by timestamp
    compares different events. Waveform correlation cannot verify a pair
    either, since the camera's coding changes the waveform of the same clap
    (peaks of 0.03–0.14 on every day tried). Align on the level envelope.

    **Verify the alignment on the signal, not on the correlation peak.** The
    envelope correlation reports a peak height relative to its own median,
    and that number can be large and wrong: aligning the proxy `.LRV`
    instead of the `.360` agreed exactly with the full file on three days
    out of four and on the fourth returned an offset 60 s out — with a peak
    standing 50× above median, well past any plausible threshold. The cheap,
    decisive check is to cut the window at the predicted position and ask
    whether a transient is there at all: a real clap window shows 25–45 dB
    between the 20 ms after the onset and the 20 ms before it, and a
    mis-aligned one shows about zero. Gate on that, not on the correlation's
    self-report. (The proxy is also 60× faster to read, so it is tempting;
    treat it as a hint that must be confirmed against the full file.)

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

## Measuring the source, not the source plus the recorder

Every level a recorder reports is `P = S + N` — what the room did, plus what
the machine contributed. Nothing in the number says which is which, and
reading `P` as `S` is the single most productive source of error in this
project's history: it produced a "loudest room" rule that ranked floor depth,
a speech comparison that ranked microphone sensitivity, a dead-channel verdict
on a quiet bedroom, and a daily rhythm that turned out to be a converter
warming up.

```python
from ambiscape import analysis

sig_db, floor_db, measurable = analysis.floor_corrected_level(fast_dba, dt=0.125)
summary = analysis.summarize_floor_corrected(sig_db, measurable)
# {'level_db': -37.8, 'coverage': 0.32, 'n_measurable': 137_216, 'reason': ''}
```

Three parts, and the third is the one usually skipped:

1. **Track the floor over time** (`track_noise_floor`). One figure per session
   cannot be right when the floor moves — a floor-dominated node in the SINS corpus
   swings 10.7 dB between night and midday as its electronics warm, which is
   larger than most differences such a figure gets used to interpret. Minimum
   statistics after Martin (2001): the running minimum of power over a window
   long enough to contain a real gap, lifted by a bias compensation because
   the minimum of a fluctuating estimate sits below the mean of what it
   estimates.
2. **Subtract in power.** Noise adds as energy. Subtracting decibels is a
   category error that happens to look plausible.
3. **Censor and count; never clamp.** A frame that does not clear the floor
   carries no information about the source, so it comes back as `nan` and
   `measurable=False`. `coverage` travels with every level, and no level is
   returned below 10 %.

!!! danger "Clamping is worse than doing nothing"

    Setting sub-floor frames to zero and averaging what is left does not
    merely lose precision — it converts a bias in level into a bias in
    **sampling**. The frames that survive censoring are the loud ones, so a
    quiet span reports as loud. Measured on a real living-room node, dropping
    sub-floor hours and averaging the survivors made its *midday quieter than
    its night*, which is impossible for an occupied room. The mistake is
    invisible in the output; only the coverage figure exposes it.

!!! danger "A fridge is not noise — run `steady_sources` first"

    Minimum statistics looks for the quietest moment in each window, so
    anything steady for longer than the window is absorbed into the floor and
    subtracted away. For a ventilation plant, a fridge or a circulation pump
    that is precisely wrong: they are steady for minutes at a time and they
    are usually the thing being studied.

    What separates them from the recorder is that *they turn off*. Self-noise
    does not. `steady_sources` uses that:

    ```python
    analysis.steady_sources(fast_dba, dt=0.125, short_s=120.0, long_s=7200.0)
    # {'self_noise_db': -62.1, 'steady_excess_db': 6.3, 'machine_duty': 0.07,
    #  'machine_detected': True, 'steady_source_unresolved': False}
    ```

    The floor is tracked twice — over minutes, which absorbs a running machine,
    and over hours, which does not, because the machine's off-phase falls
    inside the window. The difference is the machinery, and `machine_duty` says
    how much of the time it runs. Set `long_s` longer than the machine's full
    cycle: a domestic fridge turns over in roughly three quarters of an hour,
    so two hours is a safe default.

    **When it cannot help.** A plant that runs continuously for longer than
    `long_s` is indistinguishable from self-noise by level alone, and
    `steady_source_unresolved` says so — not "there is a constant source" but
    "one cannot be ruled out, and it is inside `self_noise_db`". The failure of
    too short a `long_s` is quiet: `machine_duty` is understated first, a
    fridge running 60 % of the time reported at 17 %, and only then does the
    floor climb toward the machine. If the duty looks implausibly low for a
    machine you can hear, lengthen the window before believing the floor.

*What coverage is for.* A level computed over 4 % of a session and one
computed over 96 % are different kinds of statement, and the level alone
cannot tell them apart. In the corpus this came from, a node reading a
plausible 6.6 dB below the living room cleared its own floor on 5 % of frames;
the raw number invited a comparison the data could not support and said
nothing about it. Per hour, coverage also answers *when* a node can be
measured at all — a live living room runs 85–90 % through the evening, while a
floor-dominated node never exceeds 18 % at any hour of the day.

## The rhythm is the signal

The deepest version of the floor question is not "how loud is the noise?" but
"what here ever changes?" A recorder's own hiss is stationary. Everything else
in a room turns over eventually: a fridge in tens of minutes, a ventilation
plant in hours, a household in a day, a heating system across a season. So
whether something is signal or noise is a question about *periodicity across
timescales* — and the period that answers it also names the thing.

```python
analysis.dominant_cycles(level_db, dt=10.0, min_period_s=120, max_period_s=4*3600)
# [{'period_s': 3720.0, 'strength': 0.42, 'band': 'cyclic'}]
```

Periods are placed on a ladder that extends the descriptor registry upward,
where its `macro` band runs from 5 s to infinity and so cannot tell a fridge
cycle from a season:

| band | span | what lives there |
|---|---|---|
| `meso` | 0.5–5 s | the sound object |
| `macro` | 5 s – 10 min | an activity, a passage |
| `cyclic` | 10 min – 6 h | machinery, a meal, a rehearsal |
| `circadian` | 6 h – 3 days | the household's day |
| `seasonal` | 3 days – 1 year | heating, weather, term time |
| `archival` | > 1 year | the building's life |

On a domestic sensor network this separates the rooms from the instruments
without any calibration: on a given day the living room and the kitchen
independently show a 62-minute cycle, while the floor-dominated nodes are
stationary at every scale. *A node with no rhythm has nothing in it but its
own electronics.*

Check persistence before generalising, though. That 62-minute cycle held for
the day it was measured on and not for the week: across six 24-hour windows
only one carried it, and six-hour windows returned periods scattered from 30
to 60 minutes. Two nodes agreeing is strong evidence for *that day*. A machine
rhythm that is stable enough to characterise a building has to be shown to be
stable, and the way to show it is to look in several windows and see whether
the period holds.

!!! warning "Validated for machinery; provisional for anything daily"

    On real recordings `dominant_cycles` reliably finds machinery — two rooms
    of a domestic network agreeing on a 62-minute cycle. *At `circadian` and
    longer it is not yet trustworthy.* Over six days of real data it returned
    a 48-hour harmonic instead of the 24-hour fundamental, and a spurious
    two-hour peak on a floor-dominated node; six days is only five repetitions of a
    daily cycle, and the estimate moves with how the series is smoothed.

    For anything circadian, prefer a direct test — how the quantity varies by
    hour of day — and treat a long period from this function as a hypothesis
    rather than a finding.

!!! warning "The instrument has a rhythm too"

    Periodicity alone does not mean "this is the room". A converter warms and
    cools with the building, so a node that spends most of its week at its own
    floor still varies across the day — and the variation is the electronics,
    not the room. In the corpus behind this, such a node cleared its own floor
    in 8 % of night hours, 83 % of midday hours and 17 % of evening hours: the
    opposite phase to a household, which peaks in the evening. Five of the
    twelve nodes there behave this way, and all five are working microphones
    in quiet or isolated rooms. (That comes
    from the by-hour test, not from `dominant_cycles`; see the caution above.) What distinguishes a household is that it also leaves faster marks
    — a kettle, a shower, a fridge — which is what `cycle_profile` reports as
    `has_sub_daily_cycle`. A diurnal rhythm with nothing underneath it is an
    instrument breathing, not a home.

### Anomaly is the complement of rhythm

An outlier detector run on a kitchen flags the fridge thirty times a day. It
is a good detector answering the wrong question: what makes the fridge normal
is precisely that it repeats.

```python
analysis.cycle_residual(level_db, dt=10.0)
# {'period_s': 2700.0, 'residual_std_db': 0.41, 'anomalies': [...]}
```

The cycle is found, the series folded onto its phase to get what the room
usually does at that point in the cycle, and that subtracted. What survives is
what the room did *not* repeat. A spike has no period and cannot be folded
away; a machine can. So rhythm and anomaly are two readings of one series, and
finding the first is what makes the second meaningful.

Three limits worth knowing. It models one cycle, not several at once. A
*change* in the cycle — a fridge whose period drifts as it fails — appears as
residual rather than as the more interesting finding it usually is. And with
no cycle found it falls back to the plain series, where slow drift reads as
anomalous.

## An action begins before its sound

A sound-producing action does not start when the sound does. An intention
becomes a signal in the nervous system, then activity in muscle and fibre,
then motion in the arm and the object, and only at the end does the air move.
A sound object therefore *embeds* an action, and the silence before the attack
is the part of the event a microphone cannot reach.

```python
r = analysis.onset_lead(motion_series, audio_energy, dt=1/25)
# {'lead_s': 0.72, 'first_onset_s': 0.44, 'second_onset_s': 1.16, 'leads': 'first'}
```

`lead_s` is the quantity this is for. The two onset times are returned for
inspection and are not reliable *times* at the default `rise` — see the third
design point below.

Measured on 180 clips of a corpus recorded with both modalities, *motion
leads sound by a median 0.72 s, in 84 % of clips.* The remaining sixth is
real rather than error: an object already moving when it is struck, or an
action performed out of frame, has no visible beginning to find.

Two design points worth stating, because both are easy to get wrong.

*Same rule, both series.* Each onset is the first point passing a fixed
fraction of that series' *own* floor-to-peak range. A rule with an absolute
threshold would compare the units — pixels against acoustic energy — rather
than the events.

*The default rise finds a different onset than you may want.* Each onset is
the first point passing a quarter of that series' floor-to-peak range, and on
a recording of a person that is reached by the action's own handling noises —
an object picked up, a step — well before the sound the clip is of. On the
Sound Actions clips the default lands a median 1.78 s earlier than
hand-checked onsets and agrees within a quarter second on 17 % of them, where
`rise=0.75` lands +0.01 s and agrees on 77 %. It is not noise against signal
but which sound counts as the onset.

*That evidence is about audio, and motion wants the opposite.* The same rule
goes to the motion series in `onset_lead`, and there a high fraction is late:
checked against onsets marked by eye from video frames, `0.10` and `0.25` land
within a median 0.06 s of what a viewer calls the beginning, while `0.50` and
`0.75` are late on every clip checked, by a median 0.46 s and 0.66 s. Audio
carries a noise floor a low fraction fires on; motion has a still lead-in a
high fraction sits through. So raise `rise` for an audio onset time, leave it
low for a motion one, and where a lead uses both, say that it is a difference
between two differently-defined onsets rather than assuming the bias cancels.

*It takes series, not files.* Motion is computed wherever motion is computed;
this is the seam between toolboxes rather than a video function hidden in an
audio one. It is also what makes audio–video analysis more than two analyses
side by side: the lead belongs to the *action*, and neither modality carries
it alone.

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
modulation-frequency weighting that peaks at *4 Hz* (the slow wobble
sitting between level drift and roughness) and by the coherence of the
dominant modulation, then summed over Bark bands and scaled so the classic
reference — a 1 kHz tone, 100 % AM at 4 Hz — reads *1 vacil*. Treat
absolute values as indicative and same-pipeline comparisons as the
meaningful output; masking-based depth and level dependence are
simplified. It is pure numpy/scipy (no `[iso]` extra needed on its own)
and also feeds the always-available `fluctuation_index` in the `analyze`
summary, computed from the cached 20 ms broadband envelope without an
audio pass.
