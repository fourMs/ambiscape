# Machine listening (`[ml]` extra)

Two assistants, both deliberately positioned as *helpers around* the human
ear rather than replacements for it. Both operate on the W (omni) channel,
downmixed and resampled — the models are trained on 16/32 kHz mono internet
audio, and quiet domestic material is out of their training distribution.

## AudioSet tagging (PANNs)

`ambiscape draft` runs PANNs CNN14 (527 AudioSet classes) on 10-second
windows around detected events and steady states, and writes the top
classes into the draft's listening hints:

```json
{"t": "07:56:16", "az": 150.0, "el": 5.0,
 "tags": [{"label": "Cupboard open or close", "p": 0.49},
          {"label": "Door", "p": 0.49}]}
```

AudioSet's taxonomy includes remarkably apt classes for indoor soundscape
work (*Air conditioning, Refrigerator, Church bell, Pigeon/dove, Water tap,
Footsteps, Speech*). The intended reading: the tagger says **what**, the
intensity vector says **from where** — together, a labeled spatial event.
Confirm by ear; scores on low-SNR ambience are suggestions.

## Speech privacy gate (silero-vad)

```bash
ambiscape speechgate segment.wav              # one file
ambiscape speechgate segments/ --threshold 0.01
```

Reports the fraction of speech per file and a PASS/FAIL verdict (default:
fail above 1 % speech). Run it on **every excerpt before publishing**
(Freesound, Zenodo, supplementary material) — recordings made in homes
routinely catch a few words the recordist forgot. Exit code 2 on any
failure, so it slots into scripts.

The gate detects *voice activity*, not intelligibility — a conservative
proxy. For a stricter check on borderline files, listen to the flagged
regions (`first_speech_at_s` is reported).
