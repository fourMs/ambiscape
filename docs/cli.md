# Command overview

Every command works on a *session*—a folder of WAV files on one absolute clock—unless noted otherwise. Many commands read the feature cache that `analyze` writes, so run `analyze` once per session first; those are marked *cache*. Commands that need an optional extra are marked with the extra's name.

## Core

| Command | What it does |
|---|---|
| `ambiscape probe <session>` | Show session metadata: takes, clock, duration, channels. |
| `ambiscape analyze <session>` | Extract features, compute descriptors, render figures, write the session `README.md`. |
| `ambiscape draft <session>` | Pre-fill `annotations.draft.json` from detected states and events (*cache*; tag hints with `[ml]`). |
| `ambiscape taxonomy <session>` | Render the Schaeffer map and Schafer timeline from `annotations.json`. |
| `ambiscape resolve <session>` | Per-state descriptors: split by machine on/off (`--by machine`) or day/night (`--by diel`) (*cache*). |
| `ambiscape scenes <folder>` | Analyse each WAV in a folder as an independent one-off scene. |

See [Sessions & conventions](guide/sessions.md), [Features & descriptors](guide/descriptors.md), [State-resolved descriptors](guide/resolve.md), and [Taxonomy](guide/taxonomy.md).

## Space & acoustics

| Command | What it does |
|---|---|
| `ambiscape spatial <session>` | Direct/diffuse split, pass-by events, azimuth organisation timeline (*cache*). |
| `ambiscape iso <session>` | ISO 12913-3 psychoacoustic indicators on representative segments (*cache*, `[iso]`). |
| `ambiscape survey <session> --responses <csv>` | ISO 12913-2 Method-A questionnaire responses projected onto the 12913-3 circumplex. |
| `ambiscape calibrate <session> --spl <dB>` | Derive and store the dBFS to dB SPL offset from a field SPL reading. |
| `ambiscape sweep` | Generate an exponential sine sweep plus matched inverse filter for impulse-response measurement. |
| `ambiscape impulse <recording.wav>` | Deconvolve a recorded sweep to an impulse response; octave-band T60/T20/T30, EDT, C50/C80, D50, STI, IACC. |
| `ambiscape auralize <dry.wav> --ir <ir.wav>` | Convolve dry audio with a measured impulse response. |

See [Spatial analysis](guide/spatial.md), [Room acoustics & ISO](guide/acoustics.md), [Perceptual survey](guide/survey.md), and [Impulse response & auralization](guide/impulse.md).

## Time & rhythm

| Command | What it does |
|---|---|
| `ambiscape rhythm <session>` | Strike-level rhythm of quasi-periodic pitched sources: periods, phase clusters, variation (*cache*). |
| `ambiscape modspec <session>` | Multi-scale envelope modulation profile: micro, meso, and macro (*cache*). |
| `ambiscape schedule <session>` | Match event streams against civic time grids—church clocks, sirens (*cache*). |
| `ambiscape escapement <session>` | Tick-level regularity of clockwork: beat period, jitter versus wander, Allan deviation. |

See [Strike-level rhythm](guide/rhythm.md), [Modulation profile](guide/modulation.md), and [Civic-grid schedule matching](guide/schedule.md).

## Pitch, timbre & music

| Command | What it does |
|---|---|
| `ambiscape tonality <session>` | Tonal tracks, harmonicity, pitch-class profile (*cache*). |
| `ambiscape tones <session>` | DIN 45681-style prominent tones: spectral peaks against the masking-band level (*cache*). |
| `ambiscape timbre <session>` | Event timbre templates: rise/decay fingerprints, clustered without ML (*cache*). |
| `ambiscape background <session>` | Render a background-only bed, or with `--excerpt SEC` export the most characteristic original minute. |
| `ambiscape loop <session>` | Export a seamlessly loopable prototype segment, chosen for typicality (*cache*). |
| `ambiscape music <session>` | Librosa tempogram and chromagram (`[music]`). |
| `ambiscape carillon <session>` | Which bells a carillon played: strike-note inventory (`[music]`). |

See [Tonality](guide/tonality.md), [Event timbre](guide/timbre.md), [Background bed & excerpt](guide/background-render.md), [Prototype loop](guide/loop.md), [Music](guide/music.md), and [Carillon](guide/carillon.md).

## Ecology & sources

| Command | What it does |
|---|---|
| `ambiscape mechanical <session>` | Engines, machinery, traffic: low-frequency fraction, rumble, envelope periodicity (*cache*). |
| `ambiscape anthrophony <session>` | Human speech and activity: voice band, syllabic modulation, activity (*cache*). |
| `ambiscape geophony <session>` | Wind, rain, water: diffuse low-band and flat high-band indices (*cache*). |
| `ambiscape birdnet <session>` | BirdNET bird-species detections, optionally gated to hi-fi windows (`[ml]`). |
| `ambiscape enf <session>` | Track the mains hum (50/60 Hz) at millihertz resolution: electrification descriptor and forensic grid trace. |

Biophony's structural measures are library calls—see [Biophony](guide/biophony.md)—alongside [Ratings & global indices](guide/indices.md), [Mains hum & ENF](guide/enf.md), and [Machine listening](guide/ml.md).

## Interpretation & multimodal

| Command | What it does |
|---|---|
| `ambiscape resynth <session>` | Recreate the soundscape from basic synthesis models as a self-contained Web Audio page (*cache*). |
| `ambiscape vision <video-or-folder>` | Per-frame visual features: brightness, colour, light direction, motion. |
| `ambiscape entrain <session> --motion <csv>` | Sound–motion entrainment against a body-worn accelerometer series (*cache*). |

See [Resynthesis](guide/resynth.md), [Visual features](guide/vision.md), and [Sound–motion entrainment](guide/entrain.md).

## Corpus & workflow

| Command | What it does |
|---|---|
| `ambiscape catalog <corpus>` | Aggregate every session's `summary.json` into one CSV and Markdown table. |
| `ambiscape compare <session> <session> …` | Cross-session comparison of the same place: clock-aligned timelines, per-state LTAS, azimuth roses. |
| `ambiscape longitudinal <corpus>` | Trend and seasonal analysis of dated session summaries. |
| `ambiscape capture <root>` | Always-on feature-extraction daemon; audio discarded per block (`[capture]`). |
| `ambiscape speechgate <wav-or-folder>` | Speech privacy check before publishing (`[ml]`). |
| `ambiscape deposit <session>` | Non-identifying 1 Hz TSV export for open deposits (*cache*). |

See [Corpus catalog](guide/catalog.md), [Cross-session comparison](guide/compare.md), [Longitudinal analysis](guide/longitudinal.md), [Always-on capture](guide/capture.md), and [Deposit export](guide/deposit.md).
