"""Spaced-array tests: fractional-delay ground truth for a linear array.

A plane wave from cone angle theta reaches mic ``i`` of a linear array at
``t_i = -(p_i . v) / c``; the fixtures synthesise band-limited noise with
exactly those fractional delays (FFT phase shifts), so TDOAs, bearings and
the coherence-based diffuseness proxy are tested against known values.
"""
import json

import numpy as np
import pytest

from ambiscape import array as A

FS = 16000
C = 343.0
SPACING = 0.05                       # SINS-style 4-mic linear array
GEOM = {"mics": [[k * SPACING, 0.0] for k in range(4)], "c": C}
POS = np.asarray(GEOM["mics"], float)


def _band_noise(n, seed, lo=200.0, hi=6000.0):
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    X[(f < lo) | (f > hi)] = 0.0
    x = np.fft.irfft(X, n)
    return x / (x.std() + 1e-12)


def _delayed(sig, tau):
    """Fractional delay by ``tau`` seconds via FFT phase shift."""
    n = len(sig)
    f = np.fft.rfftfreq(n, 1 / FS)
    return np.fft.irfft(np.fft.rfft(sig) * np.exp(-2j * np.pi * f * tau), n)


def plane_arrival(theta_deg, dur=4.0, seed=0, snr_amp=0.01):
    """4-channel array signal from cone angle theta (from the +x axis)."""
    v = np.array([np.cos(np.radians(theta_deg)),
                  np.sin(np.radians(theta_deg))])
    n = int(dur * FS)
    s = _band_noise(n, seed)
    rng = np.random.default_rng(seed + 100)
    return np.stack([_delayed(s, -float(p @ v) / C)
                     + snr_amp * rng.standard_normal(n) for p in POS], 1)


def true_tau(theta_deg, i, j):
    v = np.array([np.cos(np.radians(theta_deg)),
                  np.sin(np.radians(theta_deg))])
    return -(POS[i] - POS[j]) @ v / C


# ---------------------------------------------------------------- geometry


def test_load_geometry_forms(tmp_path):
    g = A.load_geometry(GEOM)
    assert g["linear"] and g["c"] == C
    assert np.allclose(g["axis"], [1.0, 0.0])
    p = tmp_path / "geom.json"
    p.write_text(json.dumps(GEOM))
    assert np.allclose(A.load_geometry(p)["pos"], POS)
    flat = A.load_geometry([0.0, 0.05, 0.1, 0.15])       # on-axis list
    assert np.allclose(flat["pos"], POS)
    square = A.load_geometry([[0, 0], [1, 0], [0, 1], [1, 1]])
    assert not square["linear"]
    with pytest.raises(ValueError):
        A.load_geometry([[0.0, 0.0]])                    # one mic


def test_tdoa_rejects_channel_mismatch():
    with pytest.raises(ValueError):
        A.tdoa(np.zeros((FS, 3)), FS, GEOM)


def test_bearing_requires_linear_array():
    td = A.tdoa(np.random.default_rng(0).standard_normal((FS, 4)), FS,
                [[0, 0], [1, 0], [0, 1], [1, 1]])
    with pytest.raises(ValueError):
        A.bearing(td)


# ---------------------------------------------------------------- TDOA


def test_tdoa_recovers_known_delays():
    theta = 60.0
    td = A.tdoa(plane_arrival(theta, seed=1), FS, GEOM)
    for p, (i, j) in enumerate(td["pairs"]):
        got = float(np.nanmedian(td["tau_s"][:, p]))
        assert abs(got - true_tau(theta, i, j)) < 25e-6   # < half a sample


def test_tdoa_sign_convention():
    # endfire from +x: the wavefront reaches the last mic (largest x) first,
    # so t_0 - t_3 > 0 for pair (0, 3)
    td = A.tdoa(plane_arrival(0.0, seed=2), FS, GEOM)
    p03 = td["pairs"].index((0, 3))
    assert np.nanmedian(td["tau_s"][:, p03]) > 0
    assert abs(np.nanmedian(td["tau_s"][:, p03])
               - 3 * SPACING / C) < 25e-6


# ---------------------------------------------------------------- bearing


@pytest.mark.parametrize("theta", [40.0, 90.0, 120.0])
def test_bearing_recovers_angle(theta):
    b = A.bearing(A.tdoa(plane_arrival(theta, seed=3), FS, GEOM))
    assert abs(float(np.median(b["bearing_deg"])) - theta) < 3.0
    assert float(np.median(b["confidence"])) > 0.3


def test_bearing_front_back_folds():
    # the mirror source (-60 deg = the other side of the axis) reads as 60
    b = A.bearing(A.tdoa(plane_arrival(-60.0, seed=4), FS, GEOM))
    assert abs(float(np.median(b["bearing_deg"])) - 60.0) < 3.0


def test_diffuse_noise_gives_low_confidence():
    rng = np.random.default_rng(5)
    b = A.bearing(A.tdoa(rng.standard_normal((4 * FS, 4)), FS, GEOM))
    assert float(np.median(b["confidence"])) < 0.15
    direct = A.bearing(A.tdoa(plane_arrival(60.0, seed=6), FS, GEOM))
    assert (np.median(direct["confidence"])
            > 3 * np.median(b["confidence"]))


# ---------------------------------------------------------------- coherence


def test_direct_field_low_gamma():
    cp = A.coherence_profile(plane_arrival(70.0, seed=7), FS, GEOM)
    assert float(np.nanmedian(cp["gamma_array"])) < 0.25


def test_decorrelated_noise_high_gamma():
    rng = np.random.default_rng(8)
    cp = A.coherence_profile(rng.standard_normal((4 * FS, 4)), FS, GEOM)
    assert float(np.nanmedian(cp["gamma_array"])) > 0.75


def test_isotropic_field_tracks_diffuse_curve():
    # many independent plane waves approximate a (2-D) diffuse field: the
    # measured coherence should fall towards the analytic curve and gamma
    # should read diffuse
    rng = np.random.default_rng(9)
    n = 4 * FS
    data = np.zeros((n, 4))
    for k, az in enumerate(np.linspace(0, 360, 36, endpoint=False)):
        data += plane_arrival(az + rng.uniform(-5, 5), seed=200 + k,
                              snr_amp=0.0)
    cp = A.coherence_profile(data, FS, GEOM)
    assert float(np.nanmedian(cp["gamma_array"])) > 0.6
    # widest pair, informative band: coherence well below the direct case
    p = int(np.argmax(cp["d_m"]))
    band = cp["msc_diffuse"][p] < 0.5
    assert float(np.median(cp["msc"][:, p, band])) < 0.4


def test_diffuse_curve_is_sinc_squared():
    rng = np.random.default_rng(12)
    cp = A.coherence_profile(rng.standard_normal((FS, 4)), FS, GEOM)
    for p, d in enumerate(cp["d_m"]):
        ref = np.sinc(2 * cp["f"] * d / C) ** 2
        assert np.allclose(cp["msc_diffuse"][p], ref)


# ---------------------------------------------------------------- triangulation


def _bearing_stream(node_pos, axis_deg, source, n=12, noise_deg=0.5,
                    conf=0.9, seed=0):
    v = np.asarray(source, float) - np.asarray(node_pos, float)
    world = np.degrees(np.arctan2(v[1], v[0]))
    theta = abs((world - axis_deg + 180) % 360 - 180)     # cone angle
    rng = np.random.default_rng(seed)
    return {"t": np.arange(n, dtype=float),
            "bearing_deg": theta + noise_deg * rng.standard_normal(n),
            "confidence": np.full(n, conf),
            "clipped": np.zeros(n, bool)}


PLAN = {"nodes": [{"name": "a", "pos": [0.0, 0.0], "axis_deg": 0.0},
                  {"name": "b", "pos": [4.0, 0.0], "axis_deg": 90.0}]}


def test_triangulate_recovers_position():
    src = (1.5, 2.0)
    streams = [_bearing_stream((0, 0), 0.0, src, seed=1),
               _bearing_stream((4, 0), 90.0, src, seed=2)]
    tri = A.triangulate(streams, PLAN)
    assert len(tri["t"]) >= 10
    err = np.linalg.norm(tri["xy"] - np.array(src), axis=1)
    assert float(np.median(err)) < 0.3
    assert not tri["ambiguous"].all()          # geometry breaks the mirror
    assert (tri["residual_m"] < 0.5).all()


def test_triangulate_flags_mirror_symmetric_geometry():
    # parallel axes on one line: (x, y) and (x, -y) fit equally well, so
    # the front-back ambiguity cannot be resolved and must be flagged
    plan = {"nodes": [{"name": "a", "pos": [0.0, 0.0], "axis_deg": 0.0},
                      {"name": "b", "pos": [4.0, 0.0], "axis_deg": 0.0}]}
    src = (1.5, 2.0)
    streams = [_bearing_stream((0, 0), 0.0, src, seed=3),
               _bearing_stream((4, 0), 0.0, src, seed=4)]
    tri = A.triangulate(streams, plan)
    assert len(tri["t"]) and tri["ambiguous"].all()


def test_triangulate_skips_low_confidence():
    src = (1.5, 2.0)
    streams = [_bearing_stream((0, 0), 0.0, src, conf=0.05, seed=5),
               _bearing_stream((4, 0), 90.0, src, seed=6)]
    tri = A.triangulate(streams, PLAN, min_conf=0.2)
    assert len(tri["t"]) == 0


def test_triangulate_figure(tmp_path):
    src = (1.5, 2.0)
    streams = [_bearing_stream((0, 0), 0.0, src, seed=1),
               _bearing_stream((4, 0), 90.0, src, seed=2)]
    tri = A.triangulate(streams, PLAN)
    out = A.triangulate_figure(tri, PLAN, tmp_path / "tri.png")
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------- end to end


def test_run_array_cli(tmp_path, capsys):
    import soundfile as sf
    from ambiscape.cli import main
    wav = tmp_path / "node.wav"
    sf.write(str(wav), plane_arrival(60.0, dur=3.0, seed=10).astype(
        np.float32), FS, subtype="FLOAT")
    gpath = tmp_path / "geom.json"
    gpath.write_text(json.dumps(GEOM))
    assert main(["array", str(wav), "--geometry", str(gpath)]) == 0
    out = tmp_path / "analysis"
    doc = json.loads((out / "array.json").read_text())
    assert doc["n_mics"] == 4 and doc["linear"]
    assert abs(doc["bearing"]["median_deg"] - 60.0) < 3.0
    assert doc["gamma_array_median"] < 0.25
    assert (out / "array_bearing.png").exists()
    assert (out / "array_coherence.png").exists()
    text = capsys.readouterr().out
    assert "bearing median" in text and "gamma_array" in text


def test_cli_spacing_shortcut(tmp_path):
    import soundfile as sf
    from ambiscape.cli import main
    wav = tmp_path / "node.wav"
    sf.write(str(wav), plane_arrival(90.0, dur=2.0, seed=11).astype(
        np.float32), FS, subtype="FLOAT")
    assert main(["array", str(wav), "--spacing", str(SPACING)]) == 0
    doc = json.loads((tmp_path / "analysis" / "array.json").read_text())
    assert abs(doc["bearing"]["median_deg"] - 90.0) < 3.0
    # exactly one of --geometry/--spacing must be given
    assert main(["array", str(wav)]) == 1


def test_near_source_index_separates_a_source_from_a_diffuse_field():
    """One delayed source correlates across channels; independent noise does not."""
    import numpy as np
    from ambiscape.array import near_source_index
    rng = np.random.default_rng(0)
    fs, n = 16000, 16000 * 3
    src = rng.standard_normal(n)
    coherent = np.stack([np.roll(src, k) + 0.05 * rng.standard_normal(n)
                         for k in range(4)], axis=1)
    diffuse = rng.standard_normal((n, 4))
    assert near_source_index(coherent, fs) > 0.8
    assert near_source_index(diffuse, fs) < 0.2


def test_near_source_index_is_blind_to_gain():
    """The point of it: a ratio between channels of one device.

    A node twice as sensitive as its neighbour must read the same, or the
    measure is no better than the levels it was introduced to replace.
    """
    import numpy as np
    from ambiscape.array import near_source_index
    rng = np.random.default_rng(1)
    fs, n = 16000, 16000 * 3
    src = rng.standard_normal(n)
    x = np.stack([np.roll(src, k) + 0.1 * rng.standard_normal(n)
                  for k in range(4)], axis=1)
    assert near_source_index(x, fs) == pytest.approx(
        near_source_index(x * 8.0, fs), abs=1e-9)


def test_near_source_index_declines_what_it_cannot_measure():
    """Too short for a Welch segment, or single-channel, returns nan not 0."""
    import numpy as np
    from ambiscape.array import near_source_index
    assert np.isnan(near_source_index(np.zeros((10, 4)), 16000))
    assert np.isnan(near_source_index(np.zeros((48000, 1)), 16000))
