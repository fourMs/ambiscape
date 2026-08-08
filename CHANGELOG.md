# Changelog

Written 2026-08-03, reconstructed from the tag history and the commits between tags. Entries before
that date are therefore summaries of what the commits say, not notes written at the time.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/); the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) loosely, in that the minor number has
been carrying feature work throughout a pre-1.0 life.

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
