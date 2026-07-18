# ambiscape

[![PyPI version](https://img.shields.io/pypi/v/ambiscape)](https://pypi.org/project/ambiscape/)
[![Documentation](https://img.shields.io/badge/docs-fourms.github.io%2Fambiscape-blue)](https://fourms.github.io/ambiscape/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Analysis toolkit for **long-duration first-order ambisonic soundscape
recordings** (Zoom H3-VR and other AmbiX/SN3D B-format sources) — hours to
whole nights of audio, processed in a stream with constant memory.

Built as the **streaming companion to
[ambiviz](https://github.com/fisheggg/ambiviz)**: ambiviz renders rich
spatial visuals from files it can load whole; ambiscape summarizes
recordings too long for that, and selects the short representative excerpts
that ambiviz then visualizes in detail.

## Install

```bash
pip install ambiscape            # core
pip install "ambiscape[iso]"     # + ISO 532-1 loudness/sharpness/roughness
pip install "ambiscape[ml]"      # + AudioSet tagging, speech privacy gate
pip install "ambiscape[viz]"     # + ambiviz (HRIR binaural, AEM visuals)
```

## Quickstart

```bash
ambiscape probe      <session-folder>   # metadata
ambiscape analyze    <session-folder>   # features, descriptors, figures, README
ambiscape draft      <session-folder>   # pre-fill taxonomy annotations
ambiscape taxonomy   <session-folder>   # Schaeffer map + Schafer timeline
ambiscape rhythm     <session-folder>   # strike-level rhythm of periodic sources
ambiscape modspec    <session-folder>   # micro/meso/macro modulation profile
ambiscape tonality   <session-folder>   # tonal tracks, harmonicity, pitch classes
ambiscape spatial    <session-folder>   # direct/diffuse split, pass-bys, azimuth R(t)
ambiscape schedule   <session-folder>   # match events against civic time grids
ambiscape timbre     <session-folder>   # event timbre templates (no-ML clustering)
ambiscape music      <session-folder>   # librosa tempogram + chromagram [music]
ambiscape iso        <session-folder>   # ISO 12913-3 indicators
ambiscape speechgate <wav-or-folder>    # privacy check before publishing
ambiscape deposit    <session-folder>   # non-identifying 1 Hz TSV export
```

A *session* is a folder of WAVs on one absolute clock (BWF timestamps,
parsed natively). `analyze` produces a per-session `README.md` with a
descriptor table (Leq, LAeq, L10/L50/L90, events, diffuseness ψ, azimuthal
concentration R, …) and overview figures (level + spectrogram + anglegram +
ψ timeline, percentile spectra, directogram).

## In notebooks

Everything the CLI does is a library call, and version 0.3 adds a
notebook-oriented case-study toolbox — machine on/off states, source
fingerprints, civic-grid scans, bit-exact segment export:

```python
import ambiscape as asc
from ambiscape import background, features, schedule, states

sess = asc.open_session("2026-07-15-Haarlem-loft")
F = features.load_features(
    features.extract_session(sess, "analysis/features"))

segs = states.state_segments(states.band_level(F, (250, 1000)))  # vent on/off
fp = background.source_fingerprint(F, night_minutes, morning_minutes)
bells = schedule.grid_scan(F, 900.0, band=(350, 800))            # church clock
asc.export_segment(sess, t0, 600.0, "seg6_vent_switchoff.wav")

from ambiscape import enf                       # v0.4: grid-frequency traces
enf.enf_summary(enf.enf_track(sess))            # mains ENF wander, mHz-level
```

See the [machine-states guide](https://fourms.github.io/ambiscape/guide/states/)
and the executable session report it was built for.

## Documentation

- **[User guide & API reference](https://fourms.github.io/ambiscape/)** —
  the session model and conventions, feature/descriptor definitions, room
  acoustics and ISO indicators, the taxonomy workflow, machine listening,
  deposit export.
- **[Wiki](https://github.com/fourMs/ambiscape/wiki)** — research context,
  field-recording protocol, design decisions, recipes, roadmap.

## License

MIT — see [LICENSE](LICENSE). Developed in the
[AMBIENT project](https://www.uio.no/ritmo/english/projects/ambient/index.html)
at [fourMs / RITMO](https://www.uio.no/ritmo/english/), University of Oslo.
