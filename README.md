# Ambiscape

[![CI](https://github.com/fourMs/ambiscape/actions/workflows/ci.yml/badge.svg)](https://github.com/fourMs/ambiscape/actions/workflows/ci.yml)
[![docs](https://github.com/fourMs/ambiscape/actions/workflows/docs.yml/badge.svg)](https://fourms.github.io/ambiscape/)
[![PyPI version](https://img.shields.io/pypi/v/ambiscape)](https://pypi.org/project/ambiscape/)
[![Python](https://img.shields.io/pypi/pyversions/ambiscape.svg)](https://pypi.org/project/ambiscape/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21948997.svg)](https://doi.org/10.5281/zenodo.21948997)

Ambiscape is a Python toolbox for analysing soundscapes—the sonic ambiences of rooms and other places. It reads mono, stereo, binaural, or first-order ambisonic recordings of any length and describes a place's sound as a whole: level, spectrum, space, rhythm, sources, and more. Several recorders spread through a building can be read together as one acoustic network.

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

Ambiscape analyses sound; its sister toolbox [MGT-python](https://github.com/fourMs/MGT-python) analyses video. The two meet at file boundaries—see [Working with other packages](https://fourms.github.io/ambiscape/interop/).

## Licence

MIT—see [LICENSE](LICENSE).

## Credits

Ambiscape is developed as part of the [AMBIENT project](https://www.uio.no/ritmo/english/projects/ambient/index.html) at [fourMs / RITMO](https://www.uio.no/ritmo/english/), University of Oslo, supported by the Research Council of Norway. It is the streaming companion to [ambiviz](https://github.com/fisheggg/ambiviz), which renders rich spatial visuals from short ambisonic files.

## Citing

Cite the CONCEPT DOI, which always resolves to the newest version:

> Jensenius, A. R., & Guo, J. (2026). *ambiscape: analysis of soundscapes from mono, stereo, binaural and ambisonic recordings* (Version 0.42.0) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.21948997

Where the exact behaviour matters, cite the version you ran instead. Version 0.42.0 is https://doi.org/10.5281/zenodo.21948998.

`CITATION.cff` in this repository carries the same information in machine-readable form.
