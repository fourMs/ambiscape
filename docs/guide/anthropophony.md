# Anthropophony

Beyond the privacy VAD gate ([Machine listening](ml.md)), a soundscape wants a
*descriptor* of human presence. Speech and conversation carry two signatures
the cached features already hold: energy in the voice band (~250–2000 Hz) and,
above all, amplitude modulation at the 3–8 Hz **syllabic rate** — the strongest
model-free speech cue. `anthropophony` reads the features (no audio pass, no
ML) and its descriptors ride in the `analyze` summary.

## What it measures

- **`anthro_voiceband_fraction`** — share of octave energy in 250–2000 Hz.
- **`anthro_syllabic_mod`** — share of the 50 Hz-envelope modulation energy in
  3–8 Hz (conversation / announcement cadence).
- **`anthro_activity_fraction`** — seconds where the voice band rises above its
  own running background.
- **`anthropophony_index`** — the three combined, in [0, 1].

## Usage

```bash
ambiscape analyze <session>        # anthro_* keys appear in the summary
ambiscape anthropophony <session>  # detail: writes anthropophony.json
```

```python
from ambiscape import anthropophony, features
F = features.load_features(sorted((out / "features").glob("*.npz")))
anthropophony.summarize_anthropophony(F)
```

## Caveats

Proxies, not detection. Music, radio, and TV also fill the voice band and
modulate syllabically; confirm actual speech with `ambiscape.ml.speech_fraction`
(silero-VAD, `[ml]` extra) before treating this as human talk. The privacy
stance is the deposit module's: publish features, not audio.
