# ambiscape

Analysis toolkit for **long-duration first-order ambisonic soundscape
recordings** (Zoom H3-VR and other AmbiX/SN3D B-format sources).

Built as the **streaming companion to
[ambiviz](https://github.com/fisheggg/ambiviz)**: ambiviz renders rich spatial
visuals (AEM spherical energy maps, anglegrams, directograms) from files it
can load whole; ambiscape handles the *other* end of the problem — recordings
of hours to whole nights (tens of GB) that must be processed in a stream —
and produces session-level summaries, timelines, and the short representative
excerpts that ambiviz then visualizes in detail. Plot names and conventions
follow ambiviz where the two overlap.

## What it does

- **Session model** — a folder of WAVs = one session; BWF `bext` timestamps
  (parsed natively, no ffmpeg needed) put all takes on an absolute clock,
  including 2 GB recorder splits.
- **Streaming features** (constant memory, per take, cached as `.npz`):
  125 ms fast level (unweighted + A-weighted), per-second octave-band powers,
  spectral centroid/flatness, 96-band log spectrogram, per-octave
  pseudo-intensity vectors, broadband DOA (azimuth/elevation), diffuseness ψ,
  and per-minute full-resolution spectra for narrowband hum tracking.
- **Descriptors** — Leq, LAeq, L10/L50/L90, dynamics, event statistics
  (+8 dB over a running 60 s 10th-percentile background, ≥ 0.25 s),
  circular direction statistics (mean azimuth, resultant length R),
  foreground/background energy-quartile splits.
- **Figures** — 4-panel session overview (level + background, log
  spectrogram, anglegram, ψ(t)), percentile LTAS, foreground/background
  directogram.
- **Room acoustics** — noise-aware truncated-Schroeder T60 from claps or
  incidental impulses (`analysis.decay_time`).
- **Segment selection** — quietest / most active / typical / transition
  windows for archiving, listening, or ambiviz rendering
  (`analysis.pick_segments`).
- **Reports** — auto-generated per-session `README.md` with metadata,
  descriptor table, and figures.
- **Taxonomy figures** — from a hand-authored `annotations.json` in the
  session folder (the interpretive layer instruments can't supply), renders a
  **Schaeffer typo-morphology map** (objects on the facture × mass plane,
  colored by Schafer function) and a **Schafer timeline** (keynote lanes,
  signal/soundmark events, lo-fi states shaded, gap-aware panels). Schema
  documented in `taxonomy.py`; worked examples in the Haarlem and Berlin
  session folders.
- **Calibration** — drop a `calibration.json` in the session folder
  (`{"dbfs_to_dbspl": 94.0, "method": "SPL app next to mic, pump running"}`)
  and `analyze` adds dB SPL versions of Leq/LAeq/L10/L50/L90 to the summary,
  making them ISO 1996-comparable.
- **ISO 12913-3 indicators** — `ambiscape iso <folder>` computes ISO 532-1
  time-varying loudness (N5, N50), DIN 45692 sharpness and Daniel & Weber
  roughness (via [MoSQITo](https://github.com/Eomys/MoSQITo), validated here
  against the 1 kHz/60 dB ≙ 4 sone reference) per ear on a binaural render of
  each representative segment — ambiviz's HRIR binauralizer when installed,
  otherwise a documented ±90° cardioid-pair fallback. Uncalibrated sessions
  are computed with an assumed offset and flagged (ratios between segments
  stay meaningful; absolute sones don't). MoSQITo runs ~5× slower than
  realtime, hence 30 s segments and a 10 s roughness slice by default.
- **Draft annotations** — `ambiscape draft <folder>` pre-fills
  `annotations.draft.json` from the cached features: steady level regimes
  (fixed-reference change-point detection; verified to recover the Haarlem
  air-pump switch-off and Berlin drone-off to within the smoothing window)
  become keynote candidates with spans, and detected events are listed with
  listening hints (clock time, level, azimuth/elevation, diffuseness). You
  supply the ears: name the objects, fill mass/facture/kind, save as
  `annotations.json`.

## Install

```bash
pip install -e .            # from this folder
pip install -e ".[viz]"     # + ambiviz for AEM/segment visuals
```

## Use

```bash
ambiscape probe    "2026-07-15-Haarlem"
ambiscape analyze  "2026-07-15-Haarlem" --notes "Loft, mic on couch overnight"
ambiscape draft    "2026-07-15-Haarlem"   # pre-fill annotations.draft.json
ambiscape taxonomy "2026-07-15-Haarlem"   # needs <folder>/annotations.json
ambiscape iso      "2026-07-15-Haarlem"   # ISO 12913-3 indicators (MoSQITo)
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

## Relation to the StillStanding365 pipeline

The StillStanding365 Zenodo deposit extracts non-identifying 1 Hz audio
features (`level_dbfs`, `centroid_hz`, `low_frac`, `high_frac`) so the
acoustic environment can be published where raw audio cannot. ambiscape
adopts that philosophy and format: **`ambiscape deposit <folder>`** writes
the same TSV schema from the cached features (method deltas documented in
`deposit.py`: W-channel at native rate vs 8 kHz downmix; power vs magnitude
band fractions — trends comparable, absolute fractions differ slightly).

Directional analysis differs by estimator, not by goal: StillStanding's
`directional_energy.py` accumulates per-frame *argmax* directions of a
horizontal virtual-speaker decode (the Guo et al. 2025 / ambiviz directogram
approach); ambiscape integrates the *pseudo-intensity vector*. Both yield
azimuthal energy distributions; intensity additionally gives elevation and
diffuseness, argmax-decode better resolves multiple discrete reflections.

**Channel-convention caveat.** The H3-VR records B-format as either AmbiX
(W,Y,Z,X) or FuMa (W,X,Y,Z); `directional_energy.py` defaults to `fuma`
while ambiscape auto-detects the convention from the recorder's `zTRK` tags
in the BWF `bext` chunk (`io.channel_order`). Processing AmbiX files as FuMa
swaps X↔Y — a reflection of all azimuths about the ±45° diagonal, which a
best-circular-shift correlation does *not* absorb. Worth verifying which
mode the 2023 raw files used before comparing directional results across
the two corpora.

## Conventions

- AmbiX ACN channel order **W, Y, Z, X** (as written by the H3-VR), SN3D.
- Azimuth: 0° = front (X+), +90° = left (Y+), ±180° = rear; elevation + up.
  All directions are **mic-relative**.
- Levels are dBFS (uncalibrated); within-session structure is exact,
  between-session absolute comparisons are indicative.
- Diffuseness ψ = 1 − 2‖⟨Re W*·v⟩‖ / ⟨|W|² + ‖v‖²⟩ (0 = plane wave,
  1 = diffuse).

## Roadmap

- [ ] wind/handling detector for outdoor takes
- [ ] cyclic-machine detector (fridge/HVAC duty cycles) as a first-class module
- [x] taxonomy figures (Schaeffer map, Schafer timeline) from annotations.json
- [ ] tonal-event finder (bells, beeps) with schedule matching
- [ ] AEM export adapter for direct ambiviz hand-off
- [ ] tests + CI, PyPI release; move to its own git repository
