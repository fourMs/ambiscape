# Machine states and source fingerprints

Domestic and mechanical sources — ventilation units, fridges, pumps, HVAC —
structure a soundscape as *states* rather than events: a band-limited floor
that is either present or absent, sometimes for hours. Version 0.3 adds a
notebook-oriented toolbox for working with them, developed on the
2026 Haarlem-loft case study (an air pump droning for nine hours, a fridge
cycling every ~24 minutes, a church clock at the noise floor).

All of it runs from the cached features of a prior `ambiscape analyze` —
no audio pass except segment export.

## On/off segmentation (`ambiscape.states`)

```python
from ambiscape import states

lvl  = states.band_level(F, (250, 1000))       # the source's "machine band"
segs = states.state_segments(lvl, min_dur_s=120)
```

`state_segments` median-smooths the band level, splits it at an automatic
bimodal (Otsu) threshold with hysteresis, and merges segments shorter than
`min_dur_s`. Each segment reports its median level and within-state SD — a
running machine is *steady* (the Haarlem pump held ±0.2 dB for 9 h), ambience
is not. `switch_points` lists the transitions (the moment someone presses the
off button); `duty_cycle` summarizes a cycling machine as period, duty
fraction, and cycle count (a fridge: ~24 min at ~50 %).

Pass an explicit `thresh_db` when the timeline is not clearly bimodal.
Segment times (`t0_s`) index the 1 Hz feature rows — map them through
`F["t"]` for absolute clock time in multi-take sessions.

## Source fingerprints (`background.source_fingerprint`)

With minute masks for "source clearly on" and "clearly off" (e.g. derived
from the state segments), the fingerprint is the dB difference of the two
mean PSDs — the source's own spectrum with the room ambience subtracted:

```python
fp = background.source_fingerprint(F, active_minutes, quiet_minutes)
fp["rise_max_db"], fp["rise_max_hz"]   # the broadband turbulence hump
fp["peaks"]                            # narrowband lines riding on it
fp["comb"]                             # {f0_hz, harmonicity} of the lines
```

A blade-pass or compressor comb reports its base frequency via the harmonic
sieve — 130 Hz for the Haarlem pump (~1950 rpm × 4 blades). Combine with
`background.masking_index` to quantify how much the source hides the rest of
the field.

## Civic grid scans (`schedule.grid_scan`)

The complement of `schedule.match_periods`: instead of asking which grid an
event stream fits, look *at every tick* of a known grid for band-limited
energy — a church clock in the bell band, whether or not the broadband
detector heard it:

```python
scans = schedule.grid_scan(F, 900.0, band=(350, 800), win_s=120)
```

Each quarter-hour tick reports `detected`, the peak `rise_db` above the
running band background, and the `offset_s` of that peak from the tick. A
consistent nonzero offset across ticks is recorder-clock error — feed it to
`schedule.clock_offset` and store the result as `clock_offset_s` in
`calibration.json`.

## Segment export (`io.export_segment`, `io.stereo_preview`)

```python
from ambiscape.io import export_segment, stereo_preview

export_segment(sess, t0, 600.0, "seg6_vent_switchoff.wav")   # bit-exact AmbiX
st = stereo_preview(x)                                       # ±90° cardioids
```

`export_segment` copies samples in the source's own PCM subtype (no float
round trip), so a report's representative segments stay citable against the
raw takes. `stereo_preview` decodes an AmbiX block to side-facing cardioids
for listenable previews; write the result with `soundfile`.
