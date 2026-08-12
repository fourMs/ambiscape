"""Sound–motion entrainment on synthetic entrained and independent pairs."""
import datetime as dt
import importlib.util
import json

import pytest

import numpy as np

import ambiscape as asc
from ambiscape import entrain, features
from ambiscape.circstats import circ_corr
from tests.conftest import FS, diffuse_noise, write_bwf

MOD_HZ = 0.7          # shared modulation rate (audio envelope and QoM)
SWAY_PERIOD = 40.0    # s: both source azimuth and sway direction follow this
SWAY_AMP = 40.0       # deg
CARRIER_HZ = 2.5      # body oscillation carrier
DATE, TIME = "2026-07-17", "20:00:00"
N_SURR = 99           # surrogates in tests (default 200 is for real runs)


def _entrained_audio(dur_s, seed=3):
    """Modulated noise from a slowly swinging azimuth (ground truth)."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    sig = 0.15 * (1 + 0.8 * np.sin(2 * np.pi * MOD_HZ * t)) \
        * rng.standard_normal(n)
    az = np.radians(SWAY_AMP * np.sin(2 * np.pi * t / SWAY_PERIOD))
    data = np.stack([sig, sig * np.sin(az), np.zeros(n),
                     sig * np.cos(az)], axis=1)
    return data + diffuse_noise(n, level=0.005, seed=seed + 1)


def _entrained_motion_csv(path, dur_s, fs=100.0, seed=4):
    """Accelerometer CSV whose QoM and sway direction track the audio."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, dur_s, 1.0 / fs)
    amp = 1 + 0.8 * np.sin(2 * np.pi * MOD_HZ * t)
    th = np.radians(SWAY_AMP * np.sin(2 * np.pi * t / SWAY_PERIOD))
    osc = 0.4 * amp * np.sin(2 * np.pi * CARRIER_HZ * t)
    ax = osc * np.cos(th) + 0.01 * rng.standard_normal(len(t))
    ay = osc * np.sin(th) + 0.01 * rng.standard_normal(len(t))
    azc = 9.81 + 0.01 * rng.standard_normal(len(t))
    base = dt.datetime.fromisoformat(f"{DATE}T{TIME}")
    rows = ["time,acc_x,acc_y,acc_z"]
    for i in range(len(t)):
        stamp = (base + dt.timedelta(seconds=float(t[i]))).isoformat()
        rows.append(f"{stamp},{ax[i]:.6f},{ay[i]:.6f},{azc[i]:.6f}")
    path.write_text("\n".join(rows))
    return path


def _analyzed_session(folder, data):
    write_bwf(folder / "scene.wav", data, date=DATE, time=TIME)
    sess = asc.open_session(folder)
    out = folder / "analysis"
    features.extract_session(sess, out / "features", verbose=False)
    return sess, out


needs_mm = pytest.mark.skipif(
    importlib.util.find_spec("micromotion") is None,
    reason="directional_correlation goes through circ_corr, which "
           "re-exports micromotion since ambiscape 0.40.0")


@needs_mm
def test_entrained_pair_locks(tmp_path):
    dur = 160.0
    sess, out = _analyzed_session(tmp_path, _entrained_audio(dur))
    motion = _entrained_motion_csv(tmp_path / "motion.csv", dur)
    (out / "summary.json").write_text(json.dumps({"leq_dbfs": -20.0}))
    doc = entrain.analyze_entrainment(sess, motion, out_dir=out,
                                      n_surrogates=N_SURR)
    # level and QoM share the 0.7 Hz modulation: strong, significant r
    assert doc["temporal"]["r"] > 0.5
    assert doc["temporal"]["p"] < 0.05
    # the shared band is phase-locked and significant
    band = next(b for b in doc["plv"] if b["band_hz"] == [0.5, 1.0])
    assert band["plv"] > 0.5
    assert band["p"] < 0.05
    assert band["plv"] > band["null95"]
    # source azimuth and sway direction swing together
    dc = doc["directional"]
    assert dc["rho"] is not None and dc["rho"] > 0.3
    assert dc["p"] < 0.05
    # outputs and the multimodal summary join
    assert (out / "entrain.png").exists()
    assert (out / "entrain.json").exists()
    s = json.loads((out / "summary.json").read_text())
    assert s["leq_dbfs"] == -20.0                 # untouched
    assert s["ent_r_level_qom"] == doc["temporal"]["r"]
    assert s["ent_plv_max"] is not None


@needs_mm
def test_independent_pair_is_not_significant(tmp_path):
    dur = 120.0
    sess, out = _analyzed_session(tmp_path,
                                  diffuse_noise(int(dur * FS), level=0.05,
                                                seed=20))
    # independent smooth motion, plain-seconds timestamps (relative clock)
    rng = np.random.default_rng(21)
    fs = 50.0
    t = np.arange(0.0, dur, 1.0 / fs)
    from scipy import signal
    sos = signal.butter(2, 3.0, "low", fs=fs, output="sos")
    a = signal.sosfilt(sos, rng.standard_normal((len(t), 3)), axis=0)
    rows = ["t,ax,ay,az"] + [
        f"{t[i]:.3f},{a[i, 0]:.6f},{a[i, 1]:.6f},{9.81 + a[i, 2]:.6f}"
        for i in range(len(t))]
    (tmp_path / "motion.csv").write_text("\n".join(rows))
    doc = entrain.analyze_entrainment(sess, tmp_path / "motion.csv",
                                      out_dir=out, n_surrogates=N_SURR)
    # nothing couples: correlations weak, none of the tests significant
    assert abs(doc["temporal"]["r"]) < 0.3
    assert doc["temporal"]["p"] > 0.05
    band = next(b for b in doc["plv"] if b["band_hz"] == [0.5, 1.0])
    assert band["p"] > 0.05
    assert doc["directional"]["p"] > 0.05
    assert doc["overlap_s"] > 100                 # relative clock aligned


def test_load_motion_tsv_and_units(tmp_path):
    rows = ["Timestamp\tAcc_X (g)\tAcc_Y (g)\tAcc_Z (g)"] + [
        f"{i * 0.02:.2f}\t0.001\t-0.002\t1.0" for i in range(100)]
    p = tmp_path / "motion.tsv"
    p.write_text("\n".join(rows))
    t, acc, meta = entrain.load_motion(p)
    assert meta["date0"] is None                  # numeric seconds
    assert acc.shape == (100, 3)
    assert np.allclose(np.diff(t), 0.02)
    assert meta["columns"][0] == "Acc_X (g)"


def test_load_motion_iso_timestamps(tmp_path):
    rows = ["time,x,y,z"] + [
        f"2026-07-17T20:00:{i / 50:09.6f},0,0,1" for i in range(50)]
    p = tmp_path / "m.csv"
    p.write_text("\n".join(rows))
    t, acc, meta = entrain.load_motion(p)
    assert meta["date0"] == dt.date(2026, 7, 17)
    assert abs(t[0] - 20 * 3600) < 1e-6           # seconds since midnight


@needs_mm
def test_circ_corr_rotation_invariance():
    """`["r"]` since 0.40.0, when circ_corr became a re-export of micromotion.

    Skipped where micromotion is absent, so ambiscape's own suite still passes
    when it is installed alone --- which is the point of the lazy import.
    """
    rng = np.random.default_rng(5)
    a = rng.uniform(-np.pi, np.pi, 500)
    wob = a + rng.normal(0, 0.2, 500)
    assert circ_corr(a, wob + 1.0)["r"] > 0.8     # rotated frame, still locked
    assert abs(circ_corr(a, rng.uniform(-np.pi, np.pi, 500))["r"]) < 0.15


def test_plv_units():
    # a *wandering* shared oscillation: a strictly periodic pair would be
    # indistinguishable from its own circular-shift surrogates (p ~ 1),
    # which is exactly the conservatism the surrogate null is for
    dt_ = 0.125
    t = np.arange(0, 120, dt_)
    rng = np.random.default_rng(7)
    f_inst = 0.7 + 0.08 * np.cumsum(rng.standard_normal(len(t))) \
        / np.sqrt(len(t))
    phase = 2 * np.pi * np.cumsum(f_inst) * dt_
    x = np.sin(phase)
    y = np.sin(phase + 0.9)                       # constant lag: locked
    bands = entrain.plv(x, y, dt_, n_surrogates=N_SURR)
    b = next(b for b in bands if b["band_hz"] == [0.5, 1.0])
    assert b["plv"] > 0.9 and b["p"] < 0.05
    rng = np.random.default_rng(6)
    b2 = entrain.plv(x, rng.standard_normal(len(t)), dt_,
                     bands=((0.5, 1.0),), n_surrogates=N_SURR)[0]
    assert b2["p"] > 0.05
