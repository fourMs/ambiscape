# Visual features

The AMBIENT project treats a room as an audio-*visual* subject, and a room's
*look* has a diurnal rhythm just as its sound does. `ambiscape.vision`
(v0.12) extracts a compact descriptor from a single video frame, so a camera
can log visual *behaviour* rather than store imagery — the same
"features, not recordings" stance as the [capture daemon](capture.md). It is
numpy-only (no camera or OpenCV dependency): frame *grabbing* lives in the
capture rig, the feature *definitions* live here so they are versioned and
tested with ambiscape.

```python
from ambiscape import vision

f = vision.frame_features(rgb)          # rgb: (H, W, 3), uint8 0..255 or float
# {'brightness': 0.41, 'warm_cool_ratio': 1.8, 'colourfulness': 0.22,
#  'spatial_entropy': 0.88, 'bright_centroid_x': 0.63, ...}
m = vision.frame_delta(prev_rgb, rgb)   # motion proxy (0..1)
vision.summarize_vision(frames, motion=motions)   # a day -> vis_* summary
```

Per frame:

- **brightness / brightness_sd** — overall light level and its unevenness
  (Rec.709 luma);
- **r_frac / g_frac / b_frac, warm_cool_ratio** — colour balance and a
  warm/cool proxy (daylight vs incandescent vs the blue of a screen);
- **saturation, colourfulness** — how colourful the scene is
  (Hasler–Süsstrunk), near zero for a grey room;
- **spatial_entropy** — entropy of a coarse brightness grid: 1 when the room
  is evenly lit, low when the light is concentrated — the visual analogue of
  acoustic diffuseness;
- **bright_centroid_x / _y** — the luma-weighted centre of light in the
  frame: *where* the light comes from, a visual direction-of-arrival.

`frame_delta` is a motion proxy; `summarize_vision` rolls a day of per-frame
features into a `vis_`-prefixed summary that joins the audio `summary.json`,
so a day's row can carry sound, light, and vision together.

The companion **ambient-pi** repository wires this to a Raspberry Pi camera
(`visual/frame_features.py`): grab a downscaled frame each second, extract,
and discard the frame — a third 1 Hz stream aligned with the audio deposit
and the light logger.
