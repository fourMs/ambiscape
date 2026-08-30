"""Walk mode: segment-first analysis of a moving recording.

Synthetic feature dicts again — the walk analysis must run from the cached
features, like every other post-analyze command. The gait fixtures put a
2 Hz impulse train into the 50 Hz envelope: the cadence detector has to
find it, and has to stay quiet on shaped noise.
"""
import numpy as np
import pytest

from ambiscape.walk import (analyze_walk, cadence_track, classify_regime,
                            write_walk)


def _walk_features(nsec: int = 300, change: int = 150, steps_hz: float = 0.0,
                   seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    # 1 Hz layer: LF-heavy first half, HF-heavy second half
    lf = 10.0 ** np.array([-3, -3, -3.5, -4, -4.5, -5, -5.5, -6, -6.5, -7])
    hf = lf[::-1].copy()
    oct_pow = np.tile(lf, (nsec, 1)) * rng.lognormal(0, 0.05, (nsec, 10))
    oct_pow[change:] = hf * oct_pow[change:].sum(1, keepdims=True) / hf.sum()
    centroid = np.where(np.arange(nsec) < change, 300.0, 2300.0)
    F = {
        "t": np.arange(nsec, dtype=float),
        "oct_pow": oct_pow.astype(np.float32),
        "centroid": centroid + rng.normal(0, 5, nsec),
        "flatness": np.full(nsec, 0.02) + rng.normal(0, 0.001, nsec),
        "diffuse": np.where(np.arange(nsec) < change, 0.3, 0.7)
                   + rng.normal(0, 0.02, nsec),
        "rms_w": np.full(nsec, 1e-2, np.float32),
        "az": np.zeros(nsec), "el": np.zeros(nsec),
    }
    # 8 Hz fast level: quiet bed, louder second half
    fd = 0.125
    tf = np.arange(0, nsec, fd)
    fast = np.where(tf < change, -45.0, -38.0) + rng.normal(0, 0.8, len(tf))
    F["t_fast"], F["fast_db"], F["fast_dt"] = tf, fast.astype(np.float32), fd
    # 50 Hz envelope: shaped noise, plus an impulse train when steps_hz > 0
    hd = 0.02
    th = np.arange(0, nsec, hd)
    env = np.abs(rng.normal(1.0, 0.1, len(th)))
    if steps_hz:
        phase = (th * steps_hz) % 1.0
        env += 4.0 * (phase < 0.06)
    F["t_hi"], F["env_hi"], F["hi_dt"] = th, env.astype(np.float32), hd
    return F


def test_cadence_detects_a_2hz_step_train():
    F = _walk_features(steps_hz=2.0)
    tr = cadence_track(F)
    good = tr["salience"] > 6.0
    assert good.mean() > 0.7
    assert abs(np.median(tr["cadence_hz"][good]) - 2.0) < 0.15


def test_cadence_stays_quiet_on_noise():
    F = _walk_features(steps_hz=0.0)
    tr = cadence_track(F)
    assert np.median(tr["salience"]) < 5.0


def test_zones_follow_the_spectral_boundary():
    F = _walk_features(300, change=150)
    r = analyze_walk(F)
    assert len(r["zones"]) == 2
    z0, z1 = r["zones"]
    assert z0["lf_fraction"] > 0.7 and z1["lf_fraction"] < 0.3
    assert z1["leq_dbfs"] > z0["leq_dbfs"]
    for key in ("t0", "t1", "duration_s", "leq_dbfs", "l10_l90_db",
                "centroid_hz", "diffuseness", "events_per_min",
                "steps_per_min", "step_salience", "regime"):
        assert key in z0


def test_regime_rules():
    assert classify_regime(10.0, 8.0) == "own-steps"
    assert classify_regime(15.0, 3.0) == "external-events"
    assert classify_regime(4.0, 3.0) == "bed-only"
    assert classify_regime(9.0, 4.0) == "mixed"


def test_write_walk_outputs(tmp_path):
    F = _walk_features(300, change=150, steps_hz=1.8)
    out = tmp_path / "analysis"
    out.mkdir()
    r = write_walk(F, tmp_path, out)
    assert (out / "walk_zones.tsv").exists()
    assert (out / "route_profile.png").exists()
    md = (out / "walk.md").read_text()
    assert "| Zone" in md
    tsv = (out / "walk_zones.tsv").read_text().splitlines()
    assert len(tsv) == 1 + len(r["zones"])
