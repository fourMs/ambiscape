# ambiscape

Analysis toolkit for long-duration first-order ambisonic soundscape recordings (Zoom H3-VR and other AmbiX/SN3D B-format sources).

Built as the streaming companion to [ambiviz](https://github.com/fisheggg/ambiviz): ambiviz renders rich spatial visuals (AEM spherical energy maps, anglegrams, directograms) from files it can load whole; ambiscape handles the other end of the problem: recordings of hours (tens of GB) that must be processed in a stream and produces session-level summaries, timelines, and short representative excerpts that ambiviz then visualises in detail. Plot names and conventions follow ambiviz where the two overlap.

## What it does

- **Session model** — a folder of WAVs = one session; BWF `bext` timestamps (parsed natively, no ffmpeg needed) put all takes on an absolute clock,
  including 2 GB recorder splits.
- **Streaming features** (constant memory, per take, cached as `.npz`) 125 ms fast level (unweighted + A-weighted), per-second octave-band powers, spectral centroid/flatness, 96-band log spectrogram, per-octave pseudo-intensity vectors, broadband DOA (azimuth/elevation), diffuseness ψ, and per-minute full-resolution spectra for narrowband hum tracking.
- **Descriptors** — Leq, LAeq, L10/L50/L90, dynamics, event statistics (+8 dB over a running 60 s 10th-percentile background, ≥ 0.25 s), circular direction statistics (mean azimuth, resultant length R), foreground/background energy-quartile splits.
- **Figures** — 4-panel session overview (level + background, log spectrogram, anglegram, ψ(t)), percentile LTAS, foreground/background directogram.
- **Room acoustics** — noise-aware truncated-Schroeder T60 from claps or incidental impulses (`analysis.decay_time`).
- **Segment selection** — quietest / most active/typical / transition windows for archiving, listening, or ambiviz rendering (`analysis.pick_segments`).
- **Reports** — auto-generated per-session `README.md` with metadata, descriptor table, and figures.
- **Taxonomy figures** — from a hand-authored `annotations.json` in the session folder (the interpretive layer instruments can't supply), render a **Schaeffer typo-morphology map** (objects on the facture × mass plane, colored by Schafer function) and **Schafer timeline** (keynote lanes, signal/soundmark events, lo-fi states shaded, gap-aware panels). Schema documented in `taxonomy.py`.
- **Calibration** — drop a `calibration.json` in the session folder (`{"dbfs_to_dbspl": 94.0, "method": "SPL app next to mic, pump running"}`) and `analyse` adds dB SPL versions of Leq/LAeq/L10/L50/L90 to the summary, making them ISO 1996-comparable.
- **ISO 12913-3 indicators** — `ambiscape iso <folder>` computes ISO 532-1 time-varying loudness (N5, N50), DIN 45692 sharpness and Daniel & Weber roughness (via [MoSQITo](https://github.com/Eomys/MoSQITo), validated here against the 1 kHz/60 dB ≙ 4 sone reference) per ear on a binaural render of each representative segment — ambiviz's HRIR binauralizer when installed, otherwise a documented ±90° cardioid-pair fallback. Uncalibrated sessions are computed with an assumed offset and flagged (ratios between segments stay meaningful; absolute sones don't). MoSQITo runs ~5× slower than real-time, hence 30 s segments and a 10 s roughness slice by default.
- **Draft annotations** — `ambiscape draft <folder>` pre-fills `annotations.draft.json` from the cached features: steady level regimes (fixed-reference change-point detection become keynote candidates with spans, and detected events are listed with listening hints (clock time, level, azimuth/elevation, diffuseness). You supply the ears: name the objects, fill mass/facture/kind, save as `annotations.json`.

## Install

```bash
pip install -e .            # from this folder
pip install -e ".[viz]"     # + ambiviz for AEM/segment visuals
```

## Use

```bash
ambiscape probe    "folder"
ambiscape analyze  "folder" --notes "Living room, mic on table"
ambiscape draft    "folder"   # pre-fill annotations.draft.json
ambiscape taxonomy "folder"   # needs <folder>/annotations.json
ambiscape iso      "folder"   # ISO 12913-3 indicators (MoSQITo)
```

```python
import ambiscape as asc

sess = asc.open_session("2026-07-15-Haarlem")
paths = asc.extract_session(sess, "features")   # streams, caches npz
F = asc.load_features(paths)                    # one absolute time axis
print(asc.summarize(F))

x, fs = asc.read_span(sess, t0=4.0, dur=6.0)    # raw audio anywhere
print(asc.decay_time(x[:, 0], fs))              # T60 from a clap at t≈4 s

asc.figures.overview(F, "overview.png", clock=sess.clock)
```

## Zoom H3-VR notes

The toolbox was built using Ambisonics recordings captured with a Zoom H3-VR. The H3-VR records B-format as either AmbiX (W,Y,Z,X) or FuMa (W,X,Y,Z); `directional_energy.py` defaults to `fuma` while ambiscape auto-detects the convention from the recorder's `zTRK` tags in the BWF `bext` chunk (`io.channel_order`). Processing AmbiX files as FuMa swaps X↔Y — a reflection of all azimuths about the ±45° diagonal, which a best-circular-shift correlation does *not* absorb. Worth verifying which mode the 2023 raw files used before comparing directional results across the two corpora.

## Conventions

- AmbiX ACN channel order (W, Y, Z, X) as written by the H3-VR, SN3D.
- Azimuth: 0° = front (X+), +90° = left (Y+), ±180° = rear; elevation + up. All directions are mic-relative.
- Levels are dBFS (uncalibrated); within-session structure is exact; between-session absolute comparisons are indicative.
- Diffuseness ψ = 1 − 2‖⟨Re W*·v⟩‖ / ⟨|W|² + ‖v‖²⟩ (0 = plane wave, 1 = diffuse).
