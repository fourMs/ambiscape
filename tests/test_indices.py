"""Tests for the 0.5 measurement set: room criteria, ecoacoustic indices,
intermittency/emergence, decay metrics, spatial descriptors."""
import numpy as np
import pytest

from ambiscape import analysis, ecology, iso, spatial


# ------------------------------------------------------------ room criteria

OCT = (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000)


def _nr_curve(nr):
    A = (55.4, 35.5, 22.0, 12.0, 4.8, 0.0, -3.5, -6.1, -8.0)
    B = (0.681, 0.790, 0.870, 0.930, 0.974, 1.000, 1.015, 1.025, 1.030)
    return np.array([a + b * nr for a, b in zip(A, B)])


def test_nr_rating_on_curve():
    rc = iso.room_criteria(dict(zip(OCT, _nr_curve(30.0))))
    assert rc["NR"] == pytest.approx(30.0, abs=0.5)


def test_nr_rating_single_band_governs():
    spec = _nr_curve(25.0)
    spec[1] = 35.5 + 0.790 * 42          # 63 Hz band pushed to NR 42
    rc = iso.room_criteria(dict(zip(OCT, spec)))
    assert rc["NR"] == pytest.approx(42.0, abs=0.5)
    assert rc["NR_governing_hz"] == 63


def test_nc_rating_reasonable():
    # spectrum exactly on NC-30 tabulated values
    nc30 = {63: 57, 125: 48, 250: 41, 500: 35, 1000: 31, 2000: 29,
            4000: 28, 8000: 27}
    rc = iso.room_criteria({**{31.5: 60.0}, **{float(k): float(v)
                                               for k, v in nc30.items()}})
    assert rc["NC"] == pytest.approx(30.0, abs=1.0)


def test_rc_rumble_classification():
    rcv = 30.0
    spec = {f: rcv + 5 * np.log2(1000 / f) for f in OCT}    # on the RC line
    rc = iso.room_criteria(spec)
    assert rc["RC"] == pytest.approx(rcv, abs=1.0)
    assert rc["RC_class"] == "N"
    spec[63] += 12                                          # add rumble
    assert iso.room_criteria(spec)["RC_class"] == "R"


# ------------------------------------------------------- ecoacoustic indices

def _F_spec(power_hz=None, nsec=600, seed=0, level=1e-8):
    logf = np.geomspace(25, 20000, 97)
    fc = np.sqrt(logf[:-1] * logf[1:])
    rng = np.random.default_rng(seed)
    ls = level * (1 + 0.05 * rng.standard_normal((nsec, 96))) ** 2
    if power_hz:
        m = (fc >= power_hz[0]) & (fc <= power_hz[1])
        ls[:, m] *= 1000.0
    return {"logspec": ls, "logf": logf.astype(np.float32),
            "rms_w": np.sqrt(ls.sum(1)).astype(np.float32),
            "t": np.arange(nsec, dtype=float)}


def test_ndsi_extremes():
    assert ecology.indices(_F_spec((2000, 8000)))["ndsi"] > 0.8
    assert ecology.indices(_F_spec((1000, 2000)))["ndsi"] < -0.6


def test_adi_uniform_vs_single_band():
    hi = ecology.indices(_F_spec(None, level=1e-4))["adi"]     # all occupied
    F1 = _F_spec(None, level=1e-12)
    fc = np.sqrt(F1["logf"][:-1] * F1["logf"][1:])
    F1["logspec"][:, (fc >= 2000) & (fc <= 3000)] *= 1e8       # 80 dB contrast
    lo = ecology.indices(F1)["adi"]
    assert hi > lo + 0.2


def test_aci_fluctuating_beats_steady():
    Fs = _F_spec(None)
    Ff = _F_spec(None)
    Ff["logspec"] *= (1 + 0.9 * np.sin(np.arange(600))[:, None]) ** 2
    assert ecology.indices(Ff)["aci"] > ecology.indices(Fs)["aci"]


def test_entropy_tone_vs_noise():
    tone = _F_spec((1000, 1100))
    tone["logspec"][:, :] *= 1e-3
    m = (np.sqrt(tone["logf"][:-1] * tone["logf"][1:]) >= 1000) \
        & (np.sqrt(tone["logf"][:-1] * tone["logf"][1:]) <= 1100)
    tone["logspec"][:, m] *= 1e6
    assert ecology.indices(_F_spec(None))["acoustic_entropy"] > \
        ecology.indices(tone)["acoustic_entropy"]


# -------------------------------------------------- intermittency, emergence

def test_intermittency_ratio():
    dt = 0.125
    lvl = np.full(8000, -50.0)
    assert analysis.intermittency_ratio(lvl, dt) == pytest.approx(0.0)
    lvl2 = lvl.copy()
    lvl2[:800] = -20.0                     # loud 10 %, dominates energy
    ir = analysis.intermittency_ratio(lvl2, dt)
    assert ir > 95.0


# ------------------------------------------------------------ decay metrics

def test_decay_metrics_exponential():
    fs = 48000
    T60 = 0.6
    t = np.arange(int(1.5 * fs)) / fs
    rng = np.random.default_rng(6)
    x = np.zeros(2 * fs)
    tail = rng.standard_normal(len(t)) * 10 ** (-3 * t / T60)
    x[fs // 2:fs // 2 + len(t)] += tail
    dm = analysis.decay_metrics(x, fs)
    band = dm["500-1000"]
    assert band["T60"] == pytest.approx(T60, rel=0.25)
    assert band["EDT"] == pytest.approx(T60, rel=0.35)
    # analytic C50 for exponential decay: 10log10(e^(13.8*0.05/T60) - 1)
    c50_true = 10 * np.log10(np.exp(13.8 * 0.05 / T60) - 1)
    assert band["C50"] == pytest.approx(c50_true, abs=2.0)
    assert 0.4 < band["D50"] < 0.9


# --------------------------------------------------------- spatial additions

def _F_dir(az, el, nsec=400, seed=0):
    rng = np.random.default_rng(seed)
    return {"az": np.asarray(az, np.float32),
            "el": np.asarray(el, np.float32),
            "rms_w": np.ones(nsec, np.float32),
            "t": np.arange(nsec, dtype=float)}


def test_directional_entropy_extremes():
    n = 400
    rng = np.random.default_rng(1)
    one = _F_dir(10 + rng.normal(0, 2, n), np.zeros(n))
    uni = _F_dir(rng.uniform(-180, 180, n), np.zeros(n))
    assert spatial.directional_entropy(one) < 0.45
    assert spatial.directional_entropy(uni) > 0.9


def test_horizon_fractions():
    n = 300
    el = np.concatenate([np.full(100, 30.0), np.full(100, 0.0),
                         np.full(100, -30.0)])
    hf = spatial.horizon_fractions(_F_dir(np.zeros(n), el, nsec=n))
    assert hf["above"] == pytest.approx(1 / 3, abs=0.02)
    assert hf["below"] == pytest.approx(1 / 3, abs=0.02)


def test_fg_bg_overlap():
    n = 400
    rng = np.random.default_rng(2)
    az = rng.normal(0, 5, n)
    F = _F_dir(az, np.zeros(n))
    F["rms_w"] = np.linspace(0.1, 1.0, n).astype(np.float32)
    assert spatial.fg_bg_az_overlap(F) > 0.7          # same direction
    az2 = az.copy()
    az2[F["rms_w"] >= np.percentile(F["rms_w"], 75)] += 180.0
    F2 = _F_dir(az2, np.zeros(n))
    F2["rms_w"] = F["rms_w"]
    assert spatial.fg_bg_az_overlap(F2) < 0.3         # opposite directions


def test_azimuth_organization_short_take():
    """A take shorter than one 60 s window still yields one R value
    (regression: empty Rs made run_session's percentile raise, issue #11)."""
    n = 45
    F = _F_dir(np.full(n, 20.0), np.zeros(n), nsec=n)
    ts, Rs = spatial.azimuth_organization(F)
    assert len(Rs) == 1 and len(ts) == 1
    assert np.isfinite(Rs[0]) and Rs[0] > 0.9   # one dominant direction


# ------------------------------------------------------- short-audio guards

def test_aci_below_one_chunk_is_none_not_zero():
    """ACI sums over a whole 300 s chunk, so a shorter recording has no
    comparable value; the previous 0.0 was indistinguishable from a
    measured minimum on clip corpora (5-60 s sessions)."""
    assert ecology.aci(_F_spec(None, nsec=60)) is None
    assert ecology.indices(_F_spec(None, nsec=60))["aci"] is None
    assert ecology.indices(_F_spec(None, nsec=600))["aci"] is not None


def test_pick_segments_one_window_flags_coincidence():
    """A session exactly one segment long has a single window to offer:
    it is returned once, naming the kinds that coincide on it."""
    dt = 0.125
    t = np.arange(int(30 / dt)) * dt
    F = {"t_fast": t, "fast_db": np.full(len(t), -40.0)}
    picks = analysis.pick_segments(F, seg_s=30.0)
    assert len(picks) == 1
    assert picks[0]["kind"] == "quietest"
    assert set(picks[0]["also"]) == {"most_active", "typical"}


def test_pick_segments_keeps_distinct_windows_distinct():
    dt = 0.125
    n = int(600 / dt)
    t = np.arange(n) * dt
    rng = np.random.default_rng(4)
    fast = np.full(n, -60.0) + 0.5 * rng.standard_normal(n)
    fast[n // 2:] = -30.0 + 8 * rng.standard_normal(n // 2)   # loud, varying
    picks = analysis.pick_segments({"t_fast": t, "fast_db": fast},
                                   seg_s=60.0)
    assert [p["kind"] for p in picks[:3]] == ["quietest", "most_active",
                                              "typical"]
    assert len({p["t0"] for p in picks}) == len(picks)


def test_decay_metrics_trimmed_ir_drops_unmeasured_ranges():
    """A pre-trimmed IR has no noise floor, so the dynamic-range guard
    cannot fire. T20/T30 must be withheld when the decay never reached
    -25/-35 dB inside the file (77 of 270 MIT survey IRs returned a T30
    longer than the file itself)."""
    fs = 48000
    T60 = 1.2
    t = np.arange(int(0.3 * fs)) / fs        # cut ~15 dB into the decay
    rng = np.random.default_rng(7)
    x = np.concatenate([np.zeros(fs // 2),
                        rng.standard_normal(len(t)) * 10 ** (-3 * t / T60)])
    band = analysis.decay_metrics(x, fs)["500-1000"]
    assert "T60" in band and "EDT" in band
    assert "T20" not in band and "T30" not in band
