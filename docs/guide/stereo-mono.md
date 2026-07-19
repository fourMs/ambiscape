# Stereo & mono inputs

ambiscape was built for four-channel AmbiX B-format, but from v0.13 it also
analyses **stereo** and **mono** recordings, and ingests compressed audio (a
phone's AAC `.m4a`, say) alongside WAV/FLAC. The channel count of a file
picks a processing *mode*, and the mode decides how much direction can be
reported. Everything else — levels, spectra, events, ecology indices,
reverberation, the whole descriptor table — is computed the same way from a
single mono reference (the W channel for ambix, the L/R mean for stereo, the
lone channel for mono).

| Mode | Channels | Azimuth | Elevation | "Diffuseness" |
|---|---|---|---|---|
| ambix | ≥ 4 | full 360° (pseudo-intensity) | yes | 1 − intensity/energy |
| stereo | 2 | **lateral only**, ±90° balance | — | 1 − inter-channel coherence |
| mono | 1 | — | — | — |

## What stereo direction means

A two-channel recording carries no front/back or up/down information, but the
*difference* between the channels still says something about lateral position
and spatial width. ambiscape derives two cues per second, over the
80 Hz–3 kHz band:

- **Azimuth** = the inter-channel energy balance mapped to ±90°,
  `90·(P_L − P_R)/(P_L + P_R)`. `0` is centre, positive is left, negative is
  right. It is a *lateral cue*, not a calibrated bearing: a hard-panned source
  reads toward ±90° but the mapping depends on the recording's stereo base.
- **Diffuseness** = one minus the magnitude coherence between the channels,
  `1 − |Σ L·conj(R)| / √(Σ|L|²·Σ|R|²)`. A coherent, centred point source
  reads near 0; a decorrelated, enveloping field (reverberation, a wide
  ambient bed) reads near 1.

The azimuth-based descriptors (`azimuth_mean_deg`, `azimuth_R`,
`directional_entropy`, `fgbg_az_overlap`) are reported for stereo as lateral
quantities; the elevation-based ones (`elevation_fg_median_deg`,
`above_horizon_fraction`, `below_horizon_fraction`) are `null`. For mono every
directional descriptor is `null` and the directogram is skipped.

```bash
ambiscape analyze path/to/cafe          # a folder with one stereo .m4a
#   2-channel stereo. azimuth is a lateral L/R balance; no elevation.
#   ... full descriptor table, overview + LTAS + directogram ...
```

## Ingest: timestamps and transcoding

Recordings need not be BWF WAVs. On ingest ambiscape:

- **Transcodes** containers libsndfile cannot open (`.m4a`, `.aac`, `.opus`,
  …) to a cached WAV under `<folder>/.ambiscape_decoded/` with ffmpeg (native
  rate and channels, 16-bit PCM), reused while newer than the source. ffmpeg
  must be on `PATH`.
- **Dates** a take from its BWF timestamp if present, else a
  `YYYYMMDD_HHMMSS` / `YYMMDD_HHMMSS` stamp in the filename (phones and field
  recorders write these), else the file's modification time.

## Caveats

Lossy phone audio and a narrow stereo base make the directional read
*indicative, not calibrated* — good for "is this centred or lateral, point-like
or enveloping, and does the foreground come from where the background sits,"
not for degree-accurate bearings. And remember the biophony/bird-band
descriptors key off spectral energy in the 2–8 kHz band: indoors they respond
to music and machines, not birds. For true 3-D direction, record ambisonics.
