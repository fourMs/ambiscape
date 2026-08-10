# Changelog

Written 2026-08-03, reconstructed from the tag history and the commits between tags. Entries before
that date are therefore summaries of what the commits say, not notes written at the time.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/); the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) loosely, in that the minor number has
been carrying feature work throughout a pre-1.0 life.

## [Unreleased]

## [0.35.0] — 2026-08-10

### Added
- **`states.cycle_series` — a cycle is two quantities, not one.** A thermostat's on-time is set by
  the appliance and its period by the room, so the two move independently; a duty fraction is their
  ratio and hides both. Returns `on_s` and `period_s` per cycle with the correlation against cycle
  number, the change per cycle and the spread, so a fixed stroke under a lengthening interval is
  visible rather than averaged away. A domestic refrigerator over one night: on-time 7.6 to 8.5
  minutes while the period went 30.5 to 38.0. A run still on when the series ends has an unknown
  length and is excluded from `on_s`, flagged by `truncated_final_run`; left in, a truncated stub
  inverts the on-time trend.

- **`states.bimodal_separation` — whether a two-state split means anything.** Otsu's method returns
  a threshold for any series, including one with a single populated mode; the split then divides
  noise and `duty_cycle` reports a period for a machine nothing detected, with nothing in the chain
  signalling a problem. Returns the two class means, their separation, the upper-class fraction and
  a `bimodal` flag. One refrigerator in two rooms of one house: 8.4 dB of separation and twelve
  cycles in the kitchen, 0.6 dB and a single night-long segment in the living room. A False flag
  says the split is not evidence the machine is present, not that it is absent.

- **`ambiscape.grounding` — what kind of evidence a descriptor is.** Every number this toolbox
  returns is a fact about a waveform; some are *also* meant as facts about hearing, and the
  distance between those two is where this project has made its worst mistakes. Four claims were
  withdrawn in a single month and every one was a perceptual quantity read off a signal statistic:
  acoustic "zones" from a speech fraction, a "dead" channel from a level, a building's rhythm from
  one day's periodicity, a reverberation time from material with no free decay in it. None was a
  coding error. Each was a translation nobody had written down.

  The registry writes it down, in the same shape as `ambiscape.timescales` — where that answers
  *over how long is this valid*, this answers *what is it evidence about*. Four tiers: `S` signal
  only, `PM` perceptually motivated but unvalidated, `PC` perceptually calibrated (the transform
  embeds listening-test data), `PD` perceptually defined (the quantity is a fact about a listener
  and the number is a proxy that can be wrong).

  All **71** descriptors the toolbox emits are classified — 40 `S`, 19 `PM`, 7 `PC`, 5 `PD` — and
  a test fails if a new descriptor arrives without a tier, so coverage cannot rot by default.
  `check()` raises a caution for every `PD` quantity in a summary and counts the `PM` ones without
  itemising them, on the principle that a warning which always fires is a warning nobody reads.

  The tier is not a quality ranking. A spectral centroid is an excellent measurement of a spectral
  centroid. The tier says what may be concluded, and the only real error is concluding one tier's
  worth of thing from another tier's number.

### Changed
- **`tonality.harmonic_sieve` documents which answer its defaults choose.** All three mislead on
  machinery, shown against a pump whose shaft sits near 46 Hz: `f0_min` at 60 Hz returns the second
  harmonic instead, `max_harm` at 12 is short for a comb running to k = 26, and `tol_cents` at 35
  selects a *different fundamental* — 68.6 Hz at harmonicity 0.73 against 46.0 at 0.45, the higher
  score explaining nine of twenty-seven peaks rather than fourteen and missing the second-strongest
  by two semitones. A cents window is proportional, so 35 cents is ±5.6 Hz at 275 Hz and ±17 Hz at
  825 Hz. Quote a harmonicity with the tolerance it was computed at.

- **`analysis.series_onset` and `onset_lead` say what the default rise is for.** A quarter of a
  40 dB range is 30 dB below the peak, which on a recording of a person is reached by the action's
  own handling noises before the sound arrives. Measured against hand-checked onsets, the default
  lands a median 1.78 s early and agrees within a quarter second on 17 % of clips; `rise=0.75`
  lands +0.01 s and agrees on 77 %. `lead_s` remains the quantity `onset_lead` is for; the absolute
  onset times it returns beside it are not reliable times at the default.


## [0.34.1] — 2026-08-10

### Fixed
- **`analysis.decay_metrics` refuses a T60 extrapolated from a collapsed fit range.** The `dr < 20`
  guard bounded the dynamic range but nothing bounded the *fitted* range. T60's lower limit is
  adaptive — `max(-35, -dr + 8)` — so as the dynamic range approached the guard the fit narrowed to
  a few dB and was still extrapolated to 60. At `dr = 20` that is 7 dB of evidence and an 8.5x
  lever.

  Measured on a synthetic 0.6 s decay cut to 0.20 s: **T60 reported 0.34 s, 43 % low**, with no
  warning. Note the direction — the truncation fault fixed in 0.33.0 *over*-estimated; this one
  *under*-estimates, so it could not have been caught by the same tests. T20 and T30 were never
  affected, because ISO 3382 fixes their ranges at 20 and 30 dB for precisely this reason; the
  adaptive estimate needed the same kind of floor.

  T60 is now reported only when the fit spans at least `MIN_FIT_SPAN_DB` (15 dB), and the width is
  returned as `fit_db` so the estimate can be judged rather than trusted. A T60 is a claim about
  60 dB of decay, and 7 dB of evidence does not support one.

  **No published number changes.** Across the 329 deposited StillStanding365 rooms carrying a T60
  at 1 kHz, the minimum dynamic range is 32 dB (a 19 dB fit span); the guard trips below 28 dB, so
  nothing in the deposit is affected. Real claps carry ample range — the case this protects is the
  short archive IR.

## [0.34.0] — 2026-08-10

### Added
- **`analysis.onset_lead` and `analysis.series_onset`** — how far a sound-producing action begins
  before its sound. An intention becomes neural and then muscular activity, then motion in the arm
  and the object, and only at the end an acoustic attack; a sound object therefore *embeds* an
  action, and the silence in front of the attack is where the action already is.

  Measured on 180 clips of the Sound Actions corpus, comparing a quantity-of-motion series from the
  video against audio energy on the same grid: **motion leads sound by a median 0.72 s, in 84 % of
  clips.** The remaining sixth is real rather than error — an object already moving when struck, or
  an action out of frame, has no visible lead.

  Both series get the same onset rule, applied to each one's own floor-to-peak range, so the result
  cannot depend on either modality's units. It takes series rather than files: motion is computed
  wherever motion is computed, and this is the seam between the toolboxes. It is also what makes
  audio–video analysis more than two analyses side by side, since the lead is a property of the
  action and neither modality carries it alone.

## [0.33.0] — 2026-08-10

### Fixed
- **Schroeder integration now stops where the decay meets the noise.** Backward integration sums
  everything after a point, so an integral running to the end of the file folded the whole tail's
  noise into every earlier value and flattened the decay curve. Subtracting the noise first did not
  help: `maximum(..., 0)` rectifies the residual, so what remains is positive-biased and still
  accumulates. On a synthetic 0.6 s decay with a 45 dB floor, T60 came out at **4.67 s**; with the
  integration truncated at the knee it is 0.63 s, and the estimate now holds from a 40 dB floor to
  none at all.

  Truncating the *fit range* — which the guide described and the code did — is a different thing
  and cannot repair a curve that is already wrong. ISO 3382 asks for the integration truncation
  (Lundeby); this implements it.

  **Regenerate any T60 measured before this.** The effect is largest exactly where a decay is long
  relative to its headroom, so trimmed archive impulse responses and quiet rooms move most. On 326
  deposited clap-derived rooms, 92 move by more than 0.05 s, twenty by more than half a second, and
  nineteen had values more than double the corrected ones; the median of that distribution falls
  from 0.540 s to 0.500 s and its 90th percentile from 1.57 s to 1.23 s. EDT was never affected —
  it is fitted over the first 10 dB, well clear of any floor.

## [0.32.0] — 2026-08-10

### Added
- **`analysis.cycle_spectrum`, `dominant_cycles`, `cycle_profile`, `cycle_residual`** — the rhythm
  is the signal. What separates a source from a recorder's own hiss is that the source *changes*:
  a fridge turns over in tens of minutes, a ventilation plant in hours, a household in a day, a
  heating system in a season. Self-noise is stationary. So "signal or noise?" becomes a question
  about periodicity asked at every timescale at once, and the period that answers it also names the
  thing found.

  `dominant_cycles` returns the periods a level series repeats at, each placed on a band —
  `meso`, `macro`, `cyclic`, `circadian`, `seasonal`, `archival`. The last four extend the ladder
  above the descriptor registry's `macro`, which runs 5 s to infinity and so cannot tell a fridge
  cycle from a season.

  On the SINS network, on one recording day, the living room and the kitchen independently show a
  62-minute cycle (r ≈ 0.5) while the floor-dominated nodes show none at all. **That cycle is not
  stable across the week**: of six 24-hour windows only one carries it, and six-hour windows give
  periods scattered between 30 and 60 minutes. Two nodes agreeing on a period is good evidence for
  that day; generalising it to the deployment is the same scale confusion this release exists to
  prevent.

- **`analysis.cycle_residual`** — anomaly as the complement of rhythm. An outlier detector run on a
  kitchen flags the fridge thirty times a day: a good detector answering the wrong question, since
  what makes the fridge normal is precisely that it repeats. The cycle is found, the series folded
  onto its phase and subtracted, and what survives is what the room did *not* repeat. A spike has
  no period and cannot be folded away; a machine can.

### Fixed
- Peak finding in the cycle spectrum uses the full FFT autocorrelation rather than sampling a log
  grid. A grid cannot see a peak narrower than its spacing, and these peaks are narrow: with a
  fridge cycling inside a day, a lag 1.4 % from 24 h is half a fridge cycle out, so the day's true
  peak of 0.83 reads as 0.25 a few hundred seconds either side and a 3 % grid steps over it.

## [0.31.0] — 2026-08-10

### Added
- **`analysis.track_noise_floor`, `analysis.floor_corrected_level`,
  `analysis.summarize_floor_corrected`** — measuring the source rather than the source plus the
  recorder. Every level a recorder reports is `P = S + N`; reading it as `S` produced four wrong
  findings in one sensor-network corpus (a loudest-room rule that ranked floor depth, a speech
  comparison that ranked sensitivity, a "dead channel" verdict on a quiet bedroom that was in fact working, and a diurnal
  rhythm that was a converter warming up).

  Three parts, and the third is the one usually missing. The floor is **tracked over time** by
  minimum statistics rather than fixed per session, because a floor can swing 10.7 dB between night
  and midday as electronics warm. The subtraction happens **in power**, since noise adds as energy
  and subtracting decibels is a category error that looks plausible. And frames with nothing above
  the floor are **censored and counted**, not clamped to zero: `summarize_floor_corrected` returns a
  `coverage` alongside every level and returns no level at all below 10 %.

  Clamping is not merely less precise, it is actively misleading — it converts a bias in level into
  a bias in *sampling*, because the frames that survive are the loud ones. Averaging survivors made
  a living room's midday read quieter than its night before this existed.

  On real data: a node whose raw Leq of −49.2 dB sits a plausible 6.6 dB below the living room
  clears its own floor on 5 % of frames, and now reports no level rather than that comparison.

- **`analysis.steady_sources`** — telling a fridge from the recorder. A short-window floor treats
  anything steady as noise, which is wrong for the sources this toolbox is usually pointed at: a
  fridge, a ventilation plant and a circulation pump are steady for minutes at a time and are the
  object of study. What separates them from the recorder is that **they turn off**, so the floor is
  tracked at two timescales — over minutes, which absorbs a running machine, and over hours, which
  does not, because the off-phase falls inside the window. The difference is the machinery, with a
  duty cycle.

  On the SINS corpus this separates the network cleanly: the living-room and kitchen nodes carry a
  cycling source 6.3 and 7.7 dB over their floor on the day measured, and the floor-dominated nodes carry none at all.

### Known limitation
- **A source that never stops cannot be separated from self-noise by level alone** — not by this
  method or any other that sees one number per frame. `steady_sources` therefore reports
  `steady_source_unresolved` whenever it finds no cycling: not a claim that a constant source
  exists, but a statement that one cannot be ruled out and would be inside `self_noise_db`.
  `floor_corrected_level` on its own will subtract such a source away, so run `steady_sources`
  first when a room has machinery in it.

  The failure of a too-short `long_s` is quiet: `machine_duty` is understated first — a fridge
  running 60 % of the time reported at 17 % — and only then does the floor climb toward the
  machine. If the duty looks implausibly low for a machine you can hear, lengthen the window before
  believing the floor.

## [0.30.1] — 2026-08-09

### Documentation
- **`compare.xnode_loudest` does not show which room is loudest, and now
  says so.** `xnode_gain_offsets` returns `floor_i − median(floors)`; the
  median term is identical for every node and cancels in the comparison, so
  what is ranked is `A_i − floor_i` — each node's level above its *own*
  noise floor. That is right only if the floors differ by gain. Where one
  room is genuinely quieter, it is credited with gain it does not have.

  On the SINS network the correlation between a node's floor depth and the
  bins it wins is **r = −0.81**: the two deepest-floored nodes take 37
  awards each while every other node takes 0–7. Synthetically, a node at
  −55 dBFS with a −80 dB floor beats one at −40 dBFS with a −60 dB floor.

  This is the pathology the function's own docstring warns about for the
  display normalisation `H`, reinstated with the floor in place of the day
  median. No behaviour is changed — deciding which room is loudest needs a
  common reference that uncalibrated nodes cannot supply — but the docstring
  now states the limitation, the figure's axis reads "highest above its own
  floor", and its caption opens by saying what the strip is not.

## [0.30.0] — 2026-08-09

### Added
- **`analysis.quietest_channel`** — which capsule of a multi-microphone node
  has the most room to measure in. A node's capsules share a housing, a
  preamp and a gain setting, so they should agree; when one does not, it
  reaches the same peaks on a raised floor and costs dynamic range on every
  descriptor computed from it, and reading channel 0 by convention becomes a
  coin toss. In the SINS network node 9's four capsules reached the same
  98th-percentile level within 0.7 dB while channel 0's floor sat 5.5 dB
  higher — analysing channel 0 halved that node's apparent dynamics and
  pushed its time-at-own-floor from about 75 % to 96 %.

  Run over that corpus it is decisive where it matters and indifferent where
  it is not: it picks the same channel on every minute of the affected node,
  and shrugs where capsules agree within 2 dB.

### Not changed
- An earlier draft of this release added a third rule to
  `compare.xnode_loudest`, disqualifying a node whose levels never depart
  from its own median. It was **withdrawn before release**: it passed its
  synthetic tests and changed nothing whatever on the corpus that motivated
  it (135 awarded bins before and after, identically distributed). The
  premise turned out to be wrong — the node in question does not behave as
  reported — so the rule solved a problem that had not been shown to exist.
  The loudest-room rule is unchanged and the underlying question is open.

## [0.29.0] — 2026-08-09

### Changed
- **`ml.speech_fraction` now normalises its input level before detection.**
  silero applies a fixed probability threshold to whatever level arrives, so
  the result was a function of recording gain as much as of speech. On one
  minute of a real recording, the *same conversation* measured 0.513 at
  unity gain, 0.289 at −12 dB, 0.110 at −24 dB and **0.000 at −30 dB**.
  Uncalibrated recorders were therefore not comparable with each other, and
  a gain difference between microphones read as a difference between rooms —
  which is exactly how a spurious two-zone result was produced from a
  seven-node array and believed for a day. After the change the same minute
  reads 0.537 at every one of those gains.

  `speech_fraction(..., normalize=False)` reproduces the old numbers.
  **Anything comparing speech fractions across recorders should be
  recomputed.** A single recorder compared against itself over time is
  much less affected, since its gain does not move.

  Digital silence is left untouched rather than normalised, so a silent file
  is not amplified into noise the detector can find.

### Added
- **`analysis.floor_occupancy`**: what fraction of a session sits within a
  few dB of its own noise floor. `floor_suspicion` asks whether a *band's*
  floor is self-noise, which is the right question but fires for every node
  in a sensor network, since every recorder's top octaves are its own hiss
  during quiet hours. This asks the second question and separates a room
  that is empty from a recorder that is deaf. Across the SINS network the
  living-room and kitchen nodes sit at their floor 28–56 % of the time while
  the bedroom node sits there 96 %, a median of 0.4 dB above it. Being each
  session against its own floor, it does not move with recording gain and is
  comparable across uncalibrated instruments.

## [0.28.1] — 2026-08-09

### Fixed
- **`impulse.ir_metrics` no longer reads a noise floor off its own padding.**
  It prepends half a second of digital silence when the peak sits near the
  start of the buffer, so the estimator has samples before it — but
  `decay_metrics` was then measuring the noise floor on that silence. A
  trimmed IR with a real 45 dB floor reported **173 dB** of dynamic range,
  which satisfies every guard downstream and lets T20/T30 be fitted over
  noise without complaint. `decay_metrics` now takes `pre_roll`, and when
  the pre-roll is fabricated it reads the floor from the quietest part of
  the decay instead.

  **This changes reported numbers for trimmed IRs** — archive material,
  anything cut before its decay finished. `dr_db` now tracks the real SNR
  (30 dB in gives 27, 40 gives 38, 60 gives 57), and bands that cannot
  support a fit are dropped as they always should have been. Results
  computed on trimmed IRs with 0.28.0 or earlier should be regenerated.
  IRs with genuine pre-roll are unaffected.
- **`compare.xnode_figure` panels now line up.** The colourbar was attached
  to the heatmap alone, so it stole width from that panel only and the
  loudest-room strip below no longer sat under the hours above it. It has
  its own gridspec column now, with a test asserting the two panels share
  their x extent. Hours tick every 4 rather than matplotlib's 5, the
  strip's marks are legible on faint row guides, and the caption states how
  many bins were awarded so a sparse strip reads as a result rather than a
  broken plot.

## [0.28.0] — 2026-08-08

### Added
- `impulse.iacc_e3`: octave-band early IACC and the IACC_E3 average (the
  mean of the 500, 1000 and 2000 Hz bands). `iacc_early` is broadband,
  while the concert-hall literature reports IACC_E3 — the two are not the
  same quantity, and low-frequency content moves the broadband value, so
  comparing a broadband number against published hall figures compares
  different things. `measure` now reports both, and the `impulse` command
  prints the per-band values beside the average.
- `impulse.iacc_signed` (inside the `iacc_e3` result): the signed
  correlation at the same lag as the reported IACC. ISO 3382-1 maximises
  the *modulus*, which discards the one case where the sign matters — ears
  receiving anti-phase sound read as strongly correlated. The CLI flags it
  only above `IACC_SIGN_FLOOR` (0.5), because the sign of a near-zero peak
  is noise: decorrelated ears would otherwise report anti-phase on every
  band.

### Note
- No behaviour changed in `iacc_early`. It already computed the maximum of
  the modulus, which *is* ISO 3382-1's definition; a note in the book's
  `toolbox-integration.md` claiming it deviated from a "signed maximum" was
  wrong and has been corrected there.

## [0.27.0] — 2026-08-08

### Added
- `objects.object_profile`: the morphology of one sound object as numbers
  rather than as a type — attack, decay, temporal centroid, crest, and the
  envelope's iteration rate and strength. `object_facture` was already
  measuring the attack and the iteration and discarding both once they had
  produced a label, which is a loss: two objects can share a facture and
  differ audibly, and a comparison needs the numbers behind the label. The
  temporal centroid is the useful addition, putting impulsive and sustained
  on a continuous axis (an impulse lands near 0.04, a held sound near 0.49)
  so two objects of the same facture can still be told apart. A truncated
  sound reports no decay rather than a decay of zero, because being cut off
  is not the same as ending.
- Six meso-band windows in `timescales`, registered at 0.2 s. Before them
  every windowed descriptor was invalid on a six-second recording, so the
  toolbox could say nothing whatever about a sound object — which is the
  length most corpus material arrives in. The figure's descriptor row now
  reaches left of a minute for the first time.
- `states.transition_profile`: the boundary between two states, described
  rather than merely located — direction, step, the 10–90 % crossing time
  and the settling time. A fridge and a slow fade can reach the same level
  while being nothing alike, and only the crossing tells them apart. The
  settling band adapts to the new state's own variability, because a
  tolerance tighter than the state's noise would report that a steady state
  never settles.
- `analysis.detect_cessations`: the departure half of event detection. A
  detector that looks for level rising above a background is a good
  definition of an arrival and no definition of an ending, so every machine
  that stopped was invisible — though indoors the ending is often the
  louder event, and attention is captured by change rather than by level.

## [0.26.0] — 2026-08-08

### Added
- `ambiscape.timescales`: the one place each descriptor's observation
  window is written down, with the reason and the provenance. A window is
  `hard` when the quantity does not exist below it (no complete chunk, no
  minute to aggregate) and `soft` when it exists but is not yet stable, and
  its `source` records whether the bound was measured on this project's
  material or asserted from reasoning. Ten of the 27 are asserted, and
  saying so is the point: an asserted bound should not harden into a
  measured one by repetition.
- `ambiscape timescales` prints the registry, `--csv` emits it for a
  report table, and `--figure` renders the three-row figure — what rooms
  do, what descriptors need, what corpora supply — over the micro/meso/
  macro bands, hatching the asserted bounds. The figure is generated from
  the registry, so the picture and the guard cannot disagree.
- `low_confidence` in `summary.json`: descriptors kept but computed below
  a soft window, each naming the window it needed and the window it had.

### Changed
- BREAKING: descriptors below a hard observation window are now `None`
  rather than whatever a chunkless computation returned. The complexity
  index returned `0.0` for any session under 300 s — indistinguishable
  from a measurement, and produced by every clip corpus in the field,
  since those are built from four to thirty seconds of audio. Anything
  reading `aci`, `tonal_prominence_db`, `tonal_prominence_hz`,
  `n_prominent_tones` or `bird_active_minute_fraction` from short sessions
  must now handle `None`. Long sessions are unaffected.
- `resolve.full_summary` applies the check and accepts
  `check_windows=False` for callers that want the raw computation. The
  per-state summaries of `resolve` are checked too, where segments are
  short and the problem is worst.

## [0.25.0] — 2026-08-08


### Changed
- The Schaeffer map is built at the sound-object level. Schaeffer's *objet sonore* is a perceptual
  unit of roughly 0.5–5 s, something attention holds whole; the map previously plotted multi-minute
  steady-state level regimes on the typo-morphology plane, which asks of an eight-hour ventilation
  bed the question Schaeffer asks of a closing door. Regimes are Schafer keynotes and now appear only
  on the Schafer timeline, which is unchanged. The map takes the session's detected events instead,
  keeps those inside a configurable duration window (0.2–8 s by default; events falling outside are
  counted in the caption, never silently dropped) and types each one on both axes from its own
  signature in the cached features. One point per object, jittered in its cell with the cell's full
  count printed, opacity by level, coloured by dominant concurrent activity where activity
  annotations are given; sessions of tens of thousands of objects are subsampled for the scatter,
  stratified so no occupied cell disappears and stated in the caption, while the counts stay
  complete. Hand-authored annotation entries that are themselves object-scale join the detected
  objects; the "machine-drafted, listen to confirm" note stands.
- `taxonomy.schaeffer_map` now takes a list of sound objects (an annotation dict is still accepted
  and passed through `taxonomy.map_objects`, which assembles the plotted objects and the census
  behind the caption). The regime-only helpers `_band_groups` and `_band_label` are gone with the
  level-band labelling they served.

### Added
- Trimmed energy level in every session summary: `analysis.trimmed_leq` and the summary key
  `laeq_trim5_dbfs`, the A-weighted energy mean with the loudest 5 % of frames discarded, now a
  standard row in the session README, in the state tables and in the `compare` descriptor table.
  An energy average is a mean of squared pressure, so in a quiet room a handful of loud frames
  decides it — contaminating 0.02 % of a week's frames moves that week's LAeq by several decibels
  while every percentile stays identical — and the trimmed level is the companion number that says
  whether LAeq describes the span or a few moments of it. Documented in the new descriptor-guide
  section "Reading energy averages" and in a wiki recipe for making the check.
- `ambiscape.objects`: event-level sound-object extraction and Schaeffer typing, on cached features
  with no audio pass (a full domestic day types in about two seconds). `object_mass` reads the
  object's excess spectrum over the running band background — peak share, the energy sitting in
  bands 6 dB above their own octave's running median, which counts a lone partial and a whole
  harmonic series alike and a continuum not at all; and spread, the energy-weighted standard
  deviation of log2 frequency — giving tonic / tonic-complex / complex / noise. `object_facture`
  reads the object's 20 ms amplitude envelope — the 10-to-90 per cent attack, and the normalised
  envelope autocorrelation at its best repetition lag between 3 and 20 Hz — giving impulse /
  iteration / sustained (delimited) / sustained (unlimited, for sustainment past the 5 s attention
  horizon). Every rule is written out in its docstring, every threshold is a module constant, and
  every object carries the numbers behind both readings under `_schaeffer`. These stay
  machine-drafted proposals: no public domestic corpus carries object-level ground truth to score
  them against, activity labels being minutes long.

### Fixed
- The documentation's Schafer timeline illustration was blank. Its demo annotations were written
  from time zero while the synthetic session's clock starts at 09:00, so every span and marker fell
  outside the figure's panels and nothing was drawn. The demo annotations now run on the session
  clock, and the bells are annotated as individual strokes (events) rather than as ninety-second
  spans carrying an `impulse` facture — which was the same scale confusion the map has now shed.
- False full-scale click at the end of every take in the fast feature streams: `extract_take`
  preallocated `fast_db`/`fast_dba`/`env_hi` for the take's full frame count but only filled
  whole-second spans, so up to 7 trailing 125 ms frames (and the matching 20 ms envelope
  frames) of each take survived as exactly 0 dBFS — full scale, ~60 dB above a quiet room.
  On multi-take days this painted a thin bright column at every take boundary in binned level
  heatmaps (the SINS cross-node figures showed them even at night, unaligned between nodes,
  because each node's takes start at different times). The extractor now trims the never-filled
  tail before saving, and `load_features` caps each take's fast/envelope streams at the filled
  whole-second span, so caches written by 0.24.1 and earlier are cleaned on load without
  re-analysis.
- Modulation profile: the nearly featureless meso line sitting ~15 dB above its neighbours in
  pre-fix figures was this same take-tail click. After the meso band's dB→linear conversion,
  the fake 0 dBFS frames (~60 dB above a quiet room; 92 frames on a SINS day) form a periodic
  impulse train at the per-take rate whose flat harmonic comb buried the entire 0.01–0.5 Hz
  band. The load-time cap removes it and meso recovers real structure (verified on Node02:
  spectrum range 0.9 → 12 dB). Macro (from the 1 s RMS, which never had the tail bug) and
  micro (`env_hi` is linear power, so its unfilled zeros were silence, ~0.2 dB effect) were
  essentially unaffected.
- Modulation profile figure no longer draws three disjoint per-scale segments at incomparable
  offsets. All three scales are now computed identically — unit-mean linear-power envelope
  PSD (the pre-0.2-cache micro fallback previously fed raw dB values) — on a single
  "PSD (dB re 1/Hz), unit-mean power envelope" axis, and each scale extends half a decade past
  its nominal band edges so adjacent scales overlap (drawn faint outside the nominal band,
  solid inside; dotted guides at the 0.01 and 0.5 Hz band edges). The curves joining in the
  overlaps is the visible check that the shared normalisation holds; reported statistics stay
  within the nominal band. The log-frequency grid now averages the Welch bins falling in each
  cell instead of picking single bins, which stabilises the single-segment macro estimate
  (macro/meso overlap agreement on Node02: ~4 → ~1.6 dB median).

### Added
- Cross-node day comparison helpers in `compare`: `xnode_day_matrix` (clock-binned heatmap
  rows as dB above each node's own day median — uncalibrated nodes are never compared on raw
  dB), `xnode_floor` (a node's day noise floor from a low percentile of the 1 Hz levels,
  raised by `adjust_db` when the analysis flagged `floor_suspect`), `xnode_loudest` (loudest
  node per bin only where the inter-node margin exceeds `margin_db`, default 3 dB, **and** the
  winner's absolute level clears its floor by `floor_clear_db`, default 3 dB — near-ties and
  near-floor bins stay unmarked, so sensor gain can no longer decide a whole night), and
  `xnode_figure` (heatmap with neutral-grey no-data bins + loudest strip, both rules stated in
  the caption line). Guide: cross-node section in `docs/guide/compare.md`.
- Human activity ground truth on the taxonomy figures: `ambiscape taxonomy <folder>
  --activities <csv>` (library: `taxonomy.load_activities` and the `activities` parameter of
  `render`, `schaeffer_map`, `schafer_timeline`) reads SINS-style per-room activity logs
  (`Class;Start time;Stop time`, semicolon-separated, absolute timestamps; Dekkers et al. 2017)
  and aligns the spans to the session clock via the session's day 0. The Schafer timeline gains
  a compact activity ribbon along the top (coloured spans per class with a legend; `absence`
  and `other` muted) and each machine-drafted keynote bed's label its dominant concurrent
  activities by time share (`"quiet bed, -60 to -54 dBFS, 23 spans — during: absence 71%,
  sleeping 22%"`). On the Schaeffer map, points that carry text also say during which activity
  they occur. The two provenances stay separate in the captions: activities are dataset data
  ("activities: human annotations, Dekkers et al. 2017"), while mass/facture remain
  machine-drafted "listen to confirm" proposals. A missing or absent CSV leaves both figures
  exactly as before.
- Activity-first Schafer timeline: whenever `--activities` is given the timeline now inverts
  by default, making the human activities the organising structure instead of an overlay on
  the acoustic level-beds. One lane per activity class (ordered by total duration within the
  session, classes under ~0.5% of the labelled time pooled into `other`), each span's fill
  coloured by its measured fast level in dB re the day median (magma, the same palette and
  scale as the cross-node day figures, with a colourbar) so loud cooking and quiet cooking
  read differently at a glance; lane labels carry the acoustic summary from the session's
  feature cache (`"watching tv — 2.1 h, median −41 dBFS"`, typographic minus). Hand-authored
  signals/soundmarks keep their lanes and markers, the machine keynote-bed structure compacts
  to a single strip coloured by bed level, and the events lane stays at the foot. The
  acoustic-first lane timeline is unchanged without activities and stays available with them
  via `--layout acoustic` (library: `schafer_timeline(..., F=..., layout=...)`;
  `render(..., layout=...)` loads the cached features itself). Caption provenance unchanged:
  activities are dataset annotations (Dekkers et al. 2017), levels are measured, machine
  drafts keep their "listen to confirm" note. Guide: both layouts and when each applies in
  `docs/guide/taxonomy.md`.
- Schaeffer map de-anonymised: crowded cells no longer collapse to bare jittered points with
  a count. Their points are grouped into ~6 dB level bands, each group labelled concisely
  with its band and count (`"−46 to −40, n=17"`); with `--activities`, points are coloured by
  their dominant concurrent activity (class colours shared with the timeline, legend
  included) instead of Schafer kind. The single machine-drafted caption note is kept.
- `array` module and `ambiscape array <recording.wav> --geometry <json>` command: spatial
  analysis from spaced-microphone recordings (SINS-style linear MEMS nodes) — the third spatial
  paradigm next to `spatial` (a soundfield sampled at one point) and `network` (one mic per
  room). Per frame, pairwise GCC-PHAT time differences of arrival searched over physically
  possible lags with parabolic refinement (`tdoa`); for linear arrays, a prominence-weighted
  least-squares bearing in 0–180° from the array axis with a GCC-prominence confidence stream
  (`bearing`) — front–back ambiguous and endfire-blind by construction, and documented as such.
  Per window, inter-channel magnitude-squared coherence versus frequency against the analytic
  diffuse-field curve for each spacing (`sinc(2fd/c)²`), yielding `gamma_array`, an
  energy-weighted direct/diffuse proxy deliberately named apart from the ambisonic diffuseness
  psi (`coherence_profile`). Across two or more nodes on a floor plan, `triangulate` intersects
  bearing streams by least squares over all front/back sign combinations, rejects fixes behind
  a node, reports the RMS ray residual as coarse uncertainty, and flags fixes `ambiguous` when
  mirror-symmetric geometry genuinely cannot resolve the fold; `triangulate_figure` draws the
  floor-plan scatter. The CLI writes `array.json`, `array_bearing.png` (time × bearing, colour
  = confidence) and `array_coherence.png`; triangulation is a library call with a worked
  synthetic example in the docs. Verified on synthetic ground truth: fractional-delay plane
  waves recover pair TDOAs within half a sample and bearings within 3°; decorrelated noise
  yields high `gamma_array` and low bearing confidence; a two-node fixture recovers a known
  position and a parallel-axes fixture is flagged ambiguous.
- DCASE STARSS support (`starss` module, `ambiscape doavalidate <folder> --annotations <dir>`):
  the toolbox now ingests STARSS22/23-format data — first-order ambisonic WAV clips (24 kHz,
  ACN/SN3D) with per-clip annotation CSVs of headerless 100 ms rows
  (`frame, class, source, azimuth, elevation[, distance]`; the distance column is STARSS23-only
  and both shapes are read) — and validates its own energy-based direction estimate against the
  labelled DOAs. Per clip, the pseudo-intensity azimuth is taken on the 100 ms label grid
  (band-passed to the corpus 80–3000 Hz DOA band) and compared with the labelled azimuth on
  single-source frames only; multi-source frames are excluded because a single broadband energy
  direction points at the energy-weighted mixture, not at either source. Reports circular error
  statistics (median and IQR of |error|, circular bias and SD, fraction within 20°, per-class
  and per-clip breakdowns) to `doavalidate.json`, with an error rose + per-class chart in
  `doavalidate.png`. Verified on synthetic ground truth: a generated FOA clip with a known
  panned source and matching CSV recovers near-zero error; the same clip against labels rotated
  by 90° recovers the rotation.
- `open_clips(folder)`: opens a folder of dataset clips (STARSS folds, contributed excerpt
  collections) as takes chained end-to-end on a synthetic clock from midnight of a nominal
  day 0 (1970-01-01). Such clips carry no BWF `bext` chunk and no filename timestamp, so
  `open_session`'s fallback would use file modification times — download times, which pile
  every clip onto one meaningless overlapping timeline. Session-time positions are deterministic
  and reproducible; clock-of-day readings are documented as meaningless for these takes.

### Changed
- `draft` clusters steady-state regimes into keynote *beds* (~6 dB level bands): one object per
  bed carrying all of that band's spans, named `"quiet bed, -60 to -54 dBFS, 23 spans"`, the eight
  longest beds kept and the remainder pooled into "other beds". A full domestic day drafts as a
  handful of keynote candidates instead of 60+ per-regime objects. Beds carry `_level_dbfs` and
  `_auto`; where PANNs tags are available the dominant tag of the bed's longest span becomes a
  `machine hint: … (PANNs, unverified)` label. The per-object
  "AUTO — mass/facture proposed…" boilerplate label is gone.
- `taxonomy` timeline (`schafer_timeline`): machine-drafted keynote regimes are merged at render
  time into the same level-banded bed lanes (`merge_keynote_beds`), capped at eight beds plus
  "other beds", so figure height is bounded at any regime count — a real SINS domestic day now
  renders as a handful of bed lanes instead of a ~4000 px staircase of 60+ mostly empty lanes.
  Hand-authored objects always keep one lane each. Figures with machine-drafted content carry a
  single "machine-drafted labels; listen to confirm" note under the title.

### Fixed
- `taxonomy` map (`schaeffer_map`): per-point machine boilerplate is never printed (it overplotted
  into illegible smears on drafted sessions); point labels appear only on maps with few objects or
  for points alone in their grid cell. Cells with more than five objects — whose extra points were
  previously silently dropped by the fixed offset list — now draw every object with deterministic
  jitter, count-scaled marker size, and an `n=` count in the cell corner.

### Changed
- `compare.xnode_loudest` decides the loudest node on **level**, not on the display
  normalization. The margin rule previously ranked nodes on `H`, each node's level minus that
  node's own day median, which is the right quantity for the heatmap (where every row is read
  against its own baseline) and the wrong one for deciding who is loudest: the largest `H` belongs
  to the node whose day departs furthest from its own baseline, which is the peakiest node and can
  be the quietest one in the building. Found on a four-node domestic day where one node took 110 of
  113 awarded bins while having the lowest noise floor of the four and the lowest daily median on
  six of seven days; it swept by construction, having the largest excursion above its own median
  every day. The rule now ranks on absolute level, optionally corrected by `gain_offsets_db`.
  BREAKING: `xnode_loudest` no longer takes `H` — the signature is
  `xnode_loudest(names, A, floors_db=None, margin_db=3.0, floor_clear_db=3.0,
  gain_offsets_db=None)`. Callers passing `H` positionally must drop it. `xnode_figure` is
  unchanged and still takes `H`, which remains correct for the heatmap.

### Added
- `compare.xnode_gain_offsets`: per-node gain offsets estimated from the nodes' own noise floors.
  Nodes of one building hear the same diffuse field when the place is empty, so their measured
  floors should agree and the spread between them reads as sensor gain. Subtracting the offsets
  makes absolute levels comparable across uncalibrated recorders, which is what deciding which room
  is loudest requires; the hours when nobody is home are what make the occupied hours comparable.
  Documented with its own limit: the estimate conflates gain with position, since a node in a
  genuinely quieter corner also shows a lower floor, and separating the two needs a source every
  node hears or the assumption that an empty building is diffuse.

### Fixed
- The package version has one source. `__version__` in `src/ambiscape/__init__.py` is now the only
  place the number is written, and setuptools reads it from there via a dynamic version; the static
  `version` in `pyproject.toml` is gone. The 0.24.1 entry below claims this was fixed and it was
  not: that release bumped `pyproject.toml` alone, so the tagged 0.24.1 reports `__version__ ==
  "0.24.0"`, the same drift as the two releases before it. `tests/test_version.py` now fails if a
  static version reappears in `pyproject.toml`, if the build stops reading the module attribute, or
  if the number setuptools would package differs from the one the module reports.

## [0.24.1] — 2026-08-07

### Fixed
- The `__version__` attribute had been stale at 0.22.0 since that release;
  it now reports the installed version correctly.

## [0.24.0] — 2026-08-07

### Added
- `network` module and `ambiscape network <folder>` command: multi-recorder acoustic-network
  analysis of one building — several recorders in different rooms on a common clock (the SINS
  deployment style), the inverse paradigm of ambisonics. Works entirely from the cached 8 Hz fast
  A-weighted level streams: all nodes are placed on one uniform clock grid, and every pair's
  detrended dB envelopes are cross-correlated in non-overlapping windows (default 120 s) with a
  ±4 s lag search, giving per-window coupling (adjacency) and antisymmetric lag matrices
  (positive lag = the first room leads). Numpy-only graph measures per window and per hour of
  day: node strength (the acoustic-hub reading), edge density, and transitivity as a simple
  clustering indicator. Outputs `network.json` (median coupling/lag matrices, strength, hub,
  density, hourly breakdown, strongest pair), `network.png` (house graphs at the quietest, median
  and busiest hours — node size = strength, edge width = coupling, arrows = lag direction — over
  a density-of-the-day timeline), and `net_` keys folded into the building's `summary.json`
  (`net_density_median`, `net_hub_node`, …) so it joins the corpus catalogue as one row.
  Verified on synthetic trios: two known edges with 0.5 s lags recover coupling, direction and
  hub; an independence null recovers no edges.

## [0.23.3] — 2026-08-07

### Fixed
- `ambiscape draft` raised `StopIteration` on any session proposing more than 16 steady-level
  regimes: the keynote labels came from a fixed 16-letter iterator. Found on a single ordinary
  home day from the SINS corpus, which produced 72 regimes. Regime labels are now generated
  without bound in spreadsheet-column style (A…Z, AA…AZ, BA…), so every regime gets a unique
  name and the draft → taxonomy path handles arbitrarily long sessions.

## [0.23.2] — 2026-08-07

### Added
- Sensor-noise-floor guard (`analysis.floor_suspicion`, run inside every `summarize`). Motivated by
  the SINS sensor-network corpus, where the 4–8 kHz background floor of a living room is flat to
  0.8 dB across a full week — microphone self-noise, so low-percentile descriptors there measure
  the instrument, not the room. Per octave band from 2 kHz up, the session is cut into 300 s
  chunks and the temporal spread of the chunkwise 10th-percentile floor is taken as the median
  minus the 5th-percentile chunk floor (a low-tail statistic, immune to activity-elevated chunks);
  a spread under 1.5 dB flags the band (SINS self-noise ≤ 0.8 dB over a week, quietest genuinely
  acoustic bands ≥ 2.4 dB — at least 0.7 dB of margin to each side). `summary.json` gains
  `floor_suspect`, `floor_suspect_lo_hz`/`floor_suspect_hi_hz` and `floor_spread_db`; the session
  README carries a warning that L90-derived descriptors in that range may reflect the instrument,
  not the room. Annotation only — no descriptor value changes. Sessions under six chunks (30 min)
  and bands with no content below the Nyquist frequency are never flagged.

## [0.23.1] — 2026-08-06

### Fixed
Five faults found by running the toolbox over public material (the MIT IR Survey, ESC-50 sessions
built at four durations, and xeno-canto recordings against synthetic ventilation):

- `ambiscape iso` raised a bare `ModuleNotFoundError` for `mosqito` on a stock install. The CLI now
  checks `iso.mosqito_available()` and exits with `MoSQITo not installed — pip install
  'ambiscape[iso]'`, and `iso.indicators` raises an `ImportError` naming the same extra, matching
  how the other optional-dependency subcommands behave.
- `ecology.aci` returned `0.0` for any recording shorter than one 300 s chunk — silently
  indistinguishable from a measured minimum, and the case for every clip corpus (ESC-50,
  UrbanSound8K, AudioSet, DCASE/TAU) as well as for archive recordings under five minutes. It now
  returns `None`, as does the `aci` key of `ecology.indices`.
- `iso.segment_indicators` labelled every segment with the requested duration even where less audio
  existed (`io.read_span` clamps at the end of a take). `dur_s` now reports the audio delivered,
  with the request kept as `dur_requested_s` when the two differ.
- `analysis.pick_segments` returned quietest, most-active and typical as separate entries when they
  resolved to the same window — a session only one segment long, or a stationary room with no
  distinct most-active minute. Coincident kinds are now returned once, listing the others under
  `also` (`also_kinds` in the `iso` output, and a note on the CLI), so the degeneracy is visible
  and the window is measured once rather than three times.
- `analysis.decay_metrics` reported T20/T30 on pre-trimmed impulse responses, whose absent noise
  floor leaves the dynamic-range guard unable to fire; 77 of the 270 MIT survey IRs returned a T30
  longer than the file itself. The fixed-range fits are now withheld unless the decay was observed
  as far as −25 / −35 dB before the signal ends. T60 (adaptive range) and EDT are unaffected.

## [0.23.0] — 2026-08-06

### Added
- **Sound–motion entrainment module** (`ambiscape entrain <session> --motion <csv>`): joins an
  analysed session with a body-worn accelerometer series on one common clock and computes the
  Guo–Riaz–Jensenius crossmodal measures — fast-level × quantity-of-motion correlation, audio-azimuth
  × sway-direction circular correlation, and per-band phase-locking values (0.1–4 Hz), all with
  circular time-shift surrogate significance. Writes `entrain.json` + `entrain.png` and folds
  `ent_` descriptors into `summary.json`. Adds `circ_corr` (Jammalamadaka–SenGupta) to
  `circstats`.
- **Perceptual survey module (`ambiscape survey`).** ISO 12913-2 Method-A questionnaire responses
  (CSV, one row per respondent; 5-point or 100-point scales auto-detected, extra columns such as
  loudness ratings kept) are projected onto the ISO/TS 12913-3 pleasantness–eventfulness
  circumplex: per-respondent points, mean, 95% ellipse, and a `survey.png` circumplex plot.
  `srv_`-prefixed keys join `summary.json` so `catalog` can rank sessions perceptually next to the
  acoustic descriptors, and sessions with an existing acoustic summary get a short
  perception-vs-measurement table (LAeq vs pleasantness, events/min vs eventfulness). Until now
  ISO 12913-2 was supported by protocol rather than software; this closes the data-handling half
  of that gap (collection remains a protocol matter).
- **Impulse-response module** (`ambiscape.impulse`), three CLI commands. `sweep` generates a
  Farina exponential sine sweep with matched inverse filter (peak −6 dBFS for playback headroom,
  raised-cosine fades, JSON parameter sidecar from which the inverse regenerates bit-identically).
  `impulse` deconvolves a recorded sweep to `ir.wav` (pre-ringing and harmonic-distortion images
  trimmed ahead of the direct sound) and reports octave-band T60/T20/T30, EDT, C50/C80 and D50
  through the existing truncated-Schroeder machinery, plus STI (IEC 60268-16 indirect method —
  noise-free assumption documented) and early IACC (0–80 ms) for binaural IRs. `auralize`
  convolves dry audio with a measured IR via uniformly partitioned FFT convolution, resampling
  the IR to the dry material's rate and peak-matching the output by default.
- `analysis.decay_metrics` also reports the ISO 3382 fixed-range T20 and T30 when the band's
  dynamic range supports them (`decay_time` remains frozen).
- **Fluctuation strength.** MoSQITo (≤ 1.2.x) does not provide it, so `iso.fluctuation_strength`
  implements the Fastl & Zwicker envelope-modulation approximation (~4 Hz weighting, anchored to
  the 1 kHz / 100 % AM at 4 Hz ≙ 1 vacil reference) — documented as an approximation, not a
  standard. Pure numpy/scipy; `ambiscape iso` now reports it per ear alongside loudness, sharpness
  and roughness, and the `analyze` summary gains the always-available broadband
  `fluctuation_index` from the cached 20 ms envelope.
- **Tonal prominence.** DIN 45681-style tone detection (`iso.tone_prominence` /
  `iso.prominent_tones`): spectral peaks vs masking-band level, decibel prominence ΔL per tone,
  aggregated over the per-minute spectra — the ventilation/appliance-hum detector. New
  `ambiscape tones` command writes `tones.json`; the `analyze` summary and README gain
  `tonal_prominence_db` / `tonal_prominence_hz` / `n_prominent_tones` (also per state), and the
  percentile-LTAS figure marks the top persistent tones.

## [0.22.0] — 2026-08-03

Five versions' worth of work that had been tagged only locally. The version had been bumped in
`pyproject.toml` and never released, so PyPI sat at 0.17.0.

### Changed
- **Schaeffer and Schafer are separated in the taxonomy documentation.** They are related and they
  are not the same thing, and running them together invited exactly the confusion the docs existed
  to prevent. Schaeffer's typo-morphology (mass, facture) and Schafer's soundscape functions
  (keynote, signal, soundmark) now stand apart, with Krause and Pijanowski's source classification
  a third axis.
- **`anthropophony` renamed to `anthrophony`.** The old documentation URL still resolves, via
  `mkdocs-redirects`, so existing links do not break.
- Sound objects with only some annotations are now tolerated rather than rejected, and the package
  boundary is stated explicitly.

### Fixed
- Six defects in the resynth page, two of them audible.

## [0.17.0] — 2026-07-25
### Added
- Escapement module and per-strike carillon transcription.

## [0.16.0] — 2026-07-25
### Added
- Music analysis: circular pulse clarity, fifths-circle statistics, TARTYP profile.
### Changed
- Reframed as a holistic soundscape toolbox for any recording type, rather than an ambisonics tool.

## [0.15.1] — 2026-07-23
### Fixed
- The rhythm figure no longer crashes; `iso` and `draft` honour `-o`.

## [0.15.0] — 2026-07-22
### Added
- Mechanical, anthropophony and geophony domain detectors, with CLI commands.
- Features wired into Schaeffer typo-morphology.

## [0.14.0] — 2026-07-22
### Added
- Six new documentation guide pages with reproducible generated illustrations.
- `vision.py` as a CLI command, for multimodal analysis.
- First-class binaural mode.
- Carillon bell-inventory module and informed-prior rhythm.
### Fixed
- `iso.binaural` channel-order bug; the module is now mode-aware across mono, stereo, binaural and
  ambisonics.

## [0.2.0] — 2026-07-18
### Added
- Rhythm module for strike-level analysis of quasi-periodic sources.
- Spatial, schedule and timbre modules; swing-FM and masking.
- Modulation profile, tonality, and circular-statistics modules.
- `clock_offset_s` calibration.
- Test suite and CI on a synthetic AmbiX fixture.

## [0.1.0] — 2026-07-17
First release.
