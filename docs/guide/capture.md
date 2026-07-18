# Always-on capture

For a study that runs for months---a year in one room---recording raw
ambisonic audio is impractical (terabytes) and a privacy hazard (speech).
The `capture` daemon (v0.11, optional `[capture]` extra) extracts *features*
continuously on a small always-on computer and keeps only those.

```bash
pip install "ambiscape[capture]"
ambiscape capture --list-devices          # find your 4-channel interface
ambiscape capture /data/room --device 2 --block-seconds 3600
```

It captures each block (hourly by default) to a short-lived WAV, runs it
through the same tested extractor as `analyze`, saves a feature `.npz`, and
then **deletes the audio**. Audio exists only one block at a time---robust
(a crash leaves the block on disk to reprocess) and privacy-preserving. At
each midnight the day is rolled up into a catalog- and longitudinal-ready
`summary.json` (plus a non-identifying deposit unless `--no-deposit`):

```
/data/room/2024-03-21/features/100000.npz   # one per block
/data/room/2024-03-21/analysis/summary.json # rolled up at day end
```

A year is then 365 daily scenes; point `ambiscape longitudinal /data/room`
at the root to see trend and season. `--keep-audio` retains the block WAVs
(for ground truth); `Ctrl-C`/`SIGTERM` stops gracefully and rolls up the day
in progress.

## Soundfield (A-format) microphones

A soundfield mic that outputs A-format (four raw capsules) needs converting
to B-format (AmbiX W/Y/Z/X) before extraction. In the library:

```python
from ambiscape.capture import CaptureDaemon, aformat_to_bformat
CaptureDaemon("/data/room", aformat_matrix=my_mic_matrix).run()
```

The 4×4 matrix comes from the microphone maker's calibration. A mic that
already outputs B-format needs no matrix; pass `--order fuma` if it is FuMa
rather than AmbiX.

## Running it for real

The daemon is a plain long-running process; on a Raspberry Pi run it under
`systemd` for auto-start, restart-on-failure, and clean shutdown, log to an
SSD (not the SD card), and keep the clock true with NTP. A complete
hardware-and-setup guide for a year-long room logger (with an optional light
sensor) lives in the companion **ambient-pi** repository.
