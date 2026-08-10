# Working with the other fourMs packages

ambiscape handles sound. When a question involves bodies, video or music corpora, another package
owns it, and the packages are meant to meet at file boundaries rather than by importing one another.

## Who owns what

| Package | Owns | Weight |
|---|---|---|
| `ambiscape` | spatial audio in, soundscape features out | numpy, scipy, soundfile, matplotlib |
| [`micromotion`](https://fourms.github.io/micromotion/) | motion time series: mocap, IMU, force plate | numpy, scipy, pandas |
| `musicalgestures` (MGT) | video in, visual features out | ~282 MB — opencv, librosa, numba, scikit-image, ffmpeg |
| `musiscape` | music corpora, fingerprints, similarity | + librosa, ambiscape |

The dependency arrow points from the heavy packages to the light ones. ambiscape does not import
MGT or micromotion; MGT's `_soundscape.py` consumes ambiscape's *output* and adapts it at the seam.
It stays that way because someone analysing a field recording should not have to install a
computer-vision stack, and someone analysing accelerometer data should not have to install an audio
one.

## Crossing to micromotion

The common case is a recording where a body and its surroundings were captured together, and the
question is whether they relate. ambiscape produces the per-second sound side; micromotion produces
the per-second motion side; the join happens in your own analysis on a shared clock.

Two things that have cost real time on exactly this join:

*Clocks are not shared just because the recordings are simultaneous.* Devices started separately
drift, and phone apps in particular can suspend and lose time from the timeline entirely rather than
leaving a gap. See micromotion's [formats guide](https://fourms.github.io/micromotion/formats/) for
what Physics Toolbox does to its own timestamps. Align on a physical event that appears in both
signals, such as a tap that registers in the accelerometer *and* the microphone, rather than on file
timestamps.

*A cross-modal null needs a positive control.* If motion does not track sound, the honest question
is whether the pipeline could have detected it. Correlating two *environmental* channels, say
ambiscape's loudness against a visual-change trace, gives that control cheaply: if they track each
other and the body tracks neither, the null is about the body.

## Crossing to MGT

MGT reads the video that ambiscape's session folder usually sits beside, whether that is 360
recordings, action cameras or room video. For GoPro `.360` and similar, MGT's `flatten_gopro360`
handles the projection.

A GoPro MAX carries four audio channels which can be read as pseudo-ambisonics giving horizontal
direction, so a session with no dedicated ambisonic recorder may still have a usable sound field.
That is worth checking before writing a day off as audio-less.

## When the overlap is real

Some functionality genuinely exists in more than one package. Prefer the package whose *domain* the
question belongs to, and say in your analysis which implementation produced a number, since two
implementations of "the same" measure rarely agree exactly, and a figure without its provenance
cannot be reproduced or compared.
