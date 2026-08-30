"""Self-noise gating and GPS attribution for walk mode.

Footsteps and wind sit inside every zone's descriptors on a walk; the
gating must be able to say what a zone sounds like *without* the walker.
The GPS layer turns zone times into places, distances and speeds — and
gives cadence an independent cross-check.
"""
import numpy as np

from tests.test_walk import _walk_features


# ------------------------------------------------------------- footsteps

def _steppy_features(step_db=12.0):
    """Steps at 2 Hz whose impulses also raise the 8 Hz fast level."""
    F = _walk_features(nsec=300, change=150, steps_hz=2.0)
    tf = np.asarray(F["t_fast"], float)
    phase = (tf * 2.0) % 1.0
    F["fast_db"] = (np.asarray(F["fast_db"], float)
                    + np.where(phase < 0.12, step_db, 0.0)).astype(np.float32)
    return F


def test_step_mask_covers_the_impulses():
    from ambiscape.walk import step_mask
    F = _steppy_features()
    mask = step_mask(F)
    tf = np.asarray(F["t_fast"], float)
    on_step = (tf * 2.0) % 1.0 < 0.10
    assert mask.shape == tf.shape
    assert mask[on_step].mean() > 0.7          # steps are masked
    assert mask.mean() < 0.6                   # but not the whole walk


def test_gated_leq_removes_the_steps():
    from ambiscape.walk import analyze_walk
    F = _steppy_features(step_db=15.0)
    zones = analyze_walk(F)["zones"]
    for z in zones:
        assert z["leq_gated_dbfs"] is not None
        assert z["leq_gated_dbfs"] < z["leq_dbfs"] - 3.0
        assert 0.0 < z["step_time_fraction"] < 0.6


def test_no_gait_no_gating():
    from ambiscape.walk import analyze_walk
    F = _walk_features(nsec=300, change=150, steps_hz=0.0)
    zones = analyze_walk(F)["zones"]
    for z in zones:
        assert z["leq_gated_dbfs"] is None


# ------------------------------------------------------------------ wind

def test_wind_mask_flags_diffuse_lf_gusts():
    from ambiscape.walk import wind_mask
    F = _walk_features(nsec=300, change=150)
    # a gust: seconds 50-70 go LF-heavy AND diffuse at once
    F["oct_pow"][50:70, :2] *= 30.0
    F["diffuse"][50:70] = 0.95
    mask = wind_mask(F)
    assert mask[55:65].mean() > 0.7
    assert mask[100:140].mean() < 0.2


# ------------------------------------------------------------------- gps

GPX = """<?xml version="1.0"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
 <trk><trkseg>
  <trkpt lat="59.9400" lon="10.7200"><ele>90</ele>
    <time>2026-08-24T09:36:00Z</time></trkpt>
  <trkpt lat="59.9409" lon="10.7200"><ele>91</ele>
    <time>2026-08-24T09:37:00Z</time></trkpt>
  <trkpt lat="59.9418" lon="10.7200"><ele>92</ele>
    <time>2026-08-24T09:38:00Z</time></trkpt>
 </trkseg></trk>
</gpx>"""


def test_load_gpx(tmp_path):
    from ambiscape.geo import load_gpx
    p = tmp_path / "walk.gpx"
    p.write_text(GPX)
    tr = load_gpx(p)
    assert len(tr["t"]) == 3
    assert tr["lat"][0] == 59.94 and tr["lon"][2] == 10.72
    assert tr["t"][1] - tr["t"][0] == 60.0


def test_zone_geo_distance_and_speed(tmp_path):
    # 0.001 deg lat per minute is ~111 m/min ~ 1.85 m/s straight north
    from ambiscape.geo import load_gpx, zone_geo
    p = tmp_path / "walk.gpx"
    p.write_text(GPX)
    tr = load_gpx(p)
    g = zone_geo(tr, t0=tr["t"][0], t1=tr["t"][0] + 120.0)
    assert abs(g["distance_m"] - 200.0) < 10.0
    assert abs(g["speed_ms"] - 1.67) < 0.15
    assert abs(g["lat"] - 59.9409) < 0.0005


def test_write_walk_with_track(tmp_path):
    from ambiscape.geo import load_gpx
    from ambiscape.walk import write_walk
    p = tmp_path / "walk.gpx"
    p.write_text(GPX.replace("T09:37:00Z", "T09:38:30Z")
                    .replace("T09:38:00Z", "T09:41:00Z"))
    track = load_gpx(p)
    F = _walk_features(nsec=300, change=150, steps_hz=1.8)
    out = tmp_path / "analysis"
    out.mkdir()
    # session second 0 corresponds to the first track point
    r = write_walk(F, tmp_path, out, track=track, epoch0=track["t"][0])
    z0 = r["zones"][0]
    assert z0["distance_m"] and z0["distance_m"] > 0
    assert z0["speed_ms"] and 0.1 < z0["speed_ms"] < 3.0
    assert abs(z0["lat"] - 59.94) < 0.01
    head = (out / "walk_zones.tsv").read_text().splitlines()[0]
    assert "distance_m" in head and "speed_ms" in head
    assert (out / "route_map.png").exists()
