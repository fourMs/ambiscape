# ambiscape

[![CI](https://github.com/fourMs/ambiscape/actions/workflows/ci.yml/badge.svg)](https://github.com/fourMs/ambiscape/actions/workflows/ci.yml)
[![docs](https://github.com/fourMs/ambiscape/actions/workflows/docs.yml/badge.svg)](https://fourms.github.io/ambiscape/)
[![PyPI version](https://img.shields.io/pypi/v/ambiscape)](https://pypi.org/project/ambiscape/)
[![Python](https://img.shields.io/pypi/pyversions/ambiscape.svg)](https://pypi.org/project/ambiscape/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21965234.svg)](https://doi.org/10.5281/zenodo.21965234)

ambiscape is a Python toolbox for analysing soundscapes—the sonic ambiences of rooms and other places. It reads mono, stereo, binaural, or first-order ambisonic recordings of any length and describes a place's sound as a whole: level, spectrum, space, rhythm, sources, and more. Several recorders spread through a building can be read together as one acoustic network.

## Install

```bash
pip install ambiscape
```

Optional extras add psychoacoustic indicators, machine listening, music analysis, live capture, and spatial visuals. See the [install guide](https://fourms.github.io/ambiscape/install/).

## Quickstart

Point `analyze` at a *session*—a folder of WAV files from one recording occasion:

```bash
ambiscape analyze my-session/
```

This streams the audio in constant memory, however long it is. It extracts features, computes descriptors (Leq, LAeq, percentile levels, event statistics, diffuseness, and more), renders overview figures, and writes a `README.md` summarising the session. Start by reading that README and looking at `analysis/overview.png`:

![Session overview figure: level timeline, spectrogram, anglegram, and diffuseness lane on one clock.](docs/img/overview.png) The [quickstart guide](https://fourms.github.io/ambiscape/quickstart/) continues from there, on the command line and in Python.

## Commands

`analyze` is one of nearly forty subcommands. The others cover taxonomy annotation, rhythm and tonality, room acoustics and impulse responses, ecological and source-domain indices, perceptual surveys, multi-recorder building networks (`network`), corpus aggregation, and privacy-aware publishing. The [command overview](https://fourms.github.io/ambiscape/cli/) lists them all; `ambiscape --help` prints the same list.

## Documentation

- **[User guide & API reference](https://fourms.github.io/ambiscape/)**—the session model, feature and descriptor definitions, and a page per analysis module.
- **[Wiki](https://github.com/fourMs/ambiscape/wiki)**—field-recording protocol, recipes, worked case studies, design rationale, and research context.

ambiscape analyses sound; its sister toolbox [MGT-python](https://github.com/fourMs/MGT-python) analyses video. The two meet at file boundaries—see [Working with other packages](https://fourms.github.io/ambiscape/interop/).

## The four toolboxes

Four packages from the fourMs lab, each released separately on PyPI. Which one you want is
decided by what you have in hand rather than by what you want to know:

| you have | use | it gives you |
|---|---|---|
| a recording of a place — mono, stereo, binaural or ambisonic | ambiscape (this one) | the sonic ambience of that place: level, spectrum, space, rhythm, sources |
| a motion time series from a body — optical markers, an accelerometer, a respiration belt, a force plate | [micromotion](https://github.com/fourMs/micromotion) | quantity of motion, posture, balance, and the band conventions the others follow |
| a video file, with or without its sound | [musicalgestures](https://github.com/fourMs/MGT-python) | motiongrams, videograms, motion analysis from ordinary video |
| a folder of music, or a concert recording | [musiscape](https://github.com/fourMs/musiscape) | many tracks and albums compared at a glance |

Where a measure appears in more than one package it has a single owner and a single
implementation, so the answer does not depend on which package you called. micromotion owns
filtering, lag estimation and circular statistics; ambiscape owns the soundscape descriptors;
musicalgestures owns everything that starts from pixels. Music analysis moved out of ambiscape
into musiscape on 2026-08-12, so a release of either from before then may still carry the
other's functions.

ambiscape installs and runs without any of the others. It keeps its own copy of six short
circular-statistics primitives rather than taking a dependency for them, which is a deliberate
exception to the single-owner rule: `tests/test_circstats_agreement.py` checks them against
micromotion's and skips when it is not installed. That test exists because the two Rayleigh
implementations were once found to disagree on about a fifth of random cases — ambiscape used
Zar's series expansion and micromotion uses Wilkie's approximation, both published, neither
wrong, and nothing anywhere saying they were meant to match.

## Licence

MIT—see [LICENSE](LICENSE).

## Credits

ambiscape is developed as part of the [AMBIENT project](https://www.uio.no/ritmo/english/projects/ambient/index.html) at [fourMs / RITMO](https://www.uio.no/ritmo/english/), University of Oslo, supported by the Research Council of Norway. It is the streaming companion to [ambiviz](https://github.com/fisheggg/ambiviz), which renders rich spatial visuals from short ambisonic files.

## Citing

Cite the concept DOI, which always resolves to the newest version:

> Jensenius, A. R., & Guo, J. (2026). *ambiscape: analysis of soundscapes from mono, stereo, binaural and ambisonic recordings* [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.21965234

Where the exact behaviour matters, add the version you ran. Every release has its own DOI, listed on
the [Zenodo record](https://doi.org/10.5281/zenodo.21965234).

`CITATION.cff` in this repository carries the same information in machine-readable form.
