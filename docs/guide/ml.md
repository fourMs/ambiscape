# Machine listening (`[ml]` extra)

Two assistants, both of which are positioned as *helpers around* the human
ear rather than replacements for it. Both operate on the W (omni) channel,
downmixed and resampled, since the models are trained on 16/32 kHz mono
internet audio, and quiet domestic material is out of their training
distribution.

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
Footsteps, Speech*). The intended reading is that the tagger says what and
the intensity vector says from where, which together give a labelled
spatial event. Confirm by ear; scores on low-SNR ambience are suggestions.

## Speech privacy gate (silero-vad)

```bash
ambiscape speechgate segment.wav              # one file
ambiscape speechgate segments/ --threshold 0.01
```

Reports the fraction of speech per file and a PASS/FAIL verdict (default:
fail above 1 % speech). Run it on every excerpt before publishing
(Freesound, Zenodo, supplementary material), since recordings made in homes
routinely catch a few words the recordist forgot. Exit code 2 on any
failure, so it slots into scripts.

!!! warning "Level normalisation, and comparing recorders (0.29.0)"

    silero applies a fixed probability threshold to whatever level reaches
    it, so before 0.29.0 the result was a function of recording gain as much
    as of speech. The same minute of a real recording measured 0.513 speech
    at unity gain, 0.289 at −12 dB, 0.110 at −24 dB and *0.000 at −30 dB*.

    `speech_fraction` now scales its input to a fixed RMS first, and that
    minute reads 0.537 at all four gains. Anything that compared speech
    fractions *between* recorders before 0.29.0 was comparing their gains
    too, and should be recomputed; a single recorder against itself over
    time is far less affected. `normalize=False` reproduces the old numbers.

The gate detects *voice activity*, not intelligibility, which makes it a
conservative proxy. For a stricter check on borderline files, listen to the flagged
regions (`first_speech_at_s` is reported).
