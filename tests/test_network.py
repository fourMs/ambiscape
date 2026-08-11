"""Acoustic-network tests: three fake rooms with known coupling and lag.

Ground truth at the envelope level: node A carries envelope e1; node B
carries e1 delayed by 0.5 s plus a second envelope e2; node C carries e2
delayed plus its own noise. So A–B and B–C are true edges with known lag
direction (A leads B, B leads C), A–C is absent, and B is the hub. An
independence null (three unrelated envelopes) must recover no edges.
"""
import json

import numpy as np
import pytest

import ambiscape as asc
from ambiscape import features
from ambiscape import network as N

from tests.conftest import FS, write_bwf, plane_wave, diffuse_noise

RATE = 8.0
DT = 1.0 / RATE
SHIFT = 4                       # samples of delay = 0.5 s


def _env(n, seed, smooth=4):
    """Smooth zero-mean dB envelope, unit shape, std 12 dB."""
    rng = np.random.default_rng(seed)
    pad = 8 * smooth
    e = np.convolve(rng.standard_normal(n + 2 * pad),
                    np.hanning(2 * smooth + 1), "same")[pad:pad + n]
    return 12 * (e - e.mean()) / e.std()


def _trio(n=28800, t0=43200.0):
    """(t, X) for the coupled trio described in the module docstring."""
    rng = np.random.default_rng(9)
    b1, b2 = _env(n + 8, 1), _env(n + 8, 2)
    A = b1[SHIFT + 4:SHIFT + 4 + n] + 1.0 * rng.standard_normal(n)
    B = b1[4:4 + n] + b2[SHIFT + 4:SHIFT + 4 + n]
    C = b2[4:4 + n] + 1.0 * rng.standard_normal(n)
    t = t0 + DT * np.arange(n)
    return t, np.stack([A, B, C])


def _null(n=28800, t0=43200.0):
    t = t0 + DT * np.arange(n)
    return t, np.stack([_env(n, s) for s in (11, 12, 13)])


# ------------------------------------------------------------- coupling


def test_coupling_recovers_true_edges():
    t, X = _trio()
    res = N.pairwise_coupling(t, X, win_s=120.0, max_lag_s=4.0)
    C = N._nanmed(res["coupling"])
    assert C[0, 1] > 0.55 and C[1, 2] > 0.55      # true edges
    assert C[0, 2] < 0.3                          # absent edge
    assert np.allclose(C, C.T, equal_nan=True)    # symmetric
    assert np.isnan(np.diag(res["coupling"][0])).all()


def test_lag_direction_and_magnitude():
    t, X = _trio()
    res = N.pairwise_coupling(t, X, win_s=120.0, max_lag_s=4.0)
    lab = np.nanmedian(res["lag_s"][:, 0, 1])     # A leads B by 0.5 s
    lbc = np.nanmedian(res["lag_s"][:, 1, 2])     # B leads C by 0.5 s
    assert abs(lab - SHIFT * DT) < 0.15
    assert abs(lbc - SHIFT * DT) < 0.15
    assert abs(np.nanmedian(res["lag_s"][:, 1, 0]) + lab) < 1e-9   # antisym


def test_independence_null_recovers_no_edges():
    t, X = _null()
    res = N.pairwise_coupling(t, X, win_s=120.0, max_lag_s=4.0)
    C = N._nanmed(res["coupling"])
    iu = np.triu_indices(3, 1)
    assert (C[iu] < 0.35).all()
    gm = N.graph_measures(res["coupling"], threshold=0.35)
    assert np.nanmedian(gm["density"]) < 0.2


def test_coupling_nan_where_coverage_fails():
    t, X = _trio(n=1920)                          # two 120 s windows
    X[2, :900] = np.nan                           # node C absent in window 0
    res = N.pairwise_coupling(t, X, win_s=120.0, max_lag_s=4.0)
    assert np.isnan(res["coupling"][0, 0, 2])
    assert np.isfinite(res["coupling"][0, 0, 1])
    assert np.isfinite(res["coupling"][1, 0, 2])


# ------------------------------------------------------------- graphs


def test_graph_measures_hand_built():
    nan = np.nan
    w0 = [[nan, 0.8, 0.1], [0.8, nan, 0.5], [0.1, 0.5, nan]]
    w1 = [[nan, 0.8, 0.8], [0.8, nan, 0.8], [0.8, 0.8, nan]]
    gm = N.graph_measures(np.array([w0, w1]), threshold=0.35)
    assert np.allclose(gm["strength"][0], [0.9, 1.3, 0.6])
    assert gm["density"][0] == pytest.approx(2 / 3)
    assert gm["transitivity"][0] == 0.0           # open triangle at A-B-C
    assert gm["density"][1] == 1.0
    assert gm["transitivity"][1] == 1.0


def test_graph_measures_empty_window_is_nan():
    gm = N.graph_measures(np.full((1, 3, 3), np.nan), threshold=0.35)
    assert np.isnan(gm["density"][0])
    assert np.isnan(gm["strength"][0]).all()
    assert np.isnan(gm["transitivity"][0])


def test_hourly_measures_find_the_hub():
    t, X = _trio()                                # one hour from 12:00
    res = N.pairwise_coupling(t, X, win_s=120.0, max_lag_s=4.0)
    hourly = N.hourly_measures(res["win_t"], res["coupling"], threshold=0.35)
    assert list(hourly["hours"]) == [12]
    assert hourly["hub"][0] == 1                  # B couples to both sides
    assert hourly["n_windows"][0] == 30
    assert N.representative_hours(hourly) == [12]


# ------------------------------------------------------------- grid


def test_node_grid_aligns_and_marks_gaps():
    n = 160
    f = lambda t0: {"t_fast": t0 + DT * np.arange(n),
                    "fast_dba": np.full(n, -30.0)}
    import datetime as dt
    nodes = [{"name": "a", "date": dt.date(2026, 7, 20), "F": f(72000.0)},
             {"name": "b", "date": dt.date(2026, 7, 20), "F": f(72010.0)}]
    t, X = N.node_grid(nodes)
    assert t[0] == 72000.0 and len(t) == n + 80
    assert np.isnan(X[1, :80]).all() and np.isfinite(X[1, 80:]).all()
    assert np.isnan(X[0, n:]).all()
    # a node dated the next day shifts by a whole day onto the same axis
    nodes[1]["date"] = dt.date(2026, 7, 21)
    nodes[1]["F"] = f(72000.0 - 86400.0 + 10.0)
    t2, X2 = N.node_grid(nodes)
    assert np.allclose(t2, t) and np.isnan(X2[1, :80]).all()


# ------------------------------------------------------------- end to end


def _node_session(folder, env_db, az, seed):
    """Write a 90 s AmbiX WAV whose level follows ``env_db`` (8 Hz)."""
    folder.mkdir(parents=True, exist_ok=True)
    n = int(90 * FS)
    rng = np.random.default_rng(seed)
    env = np.interp(np.arange(n) / FS, DT * np.arange(len(env_db)), env_db)
    mono = 0.03 * rng.standard_normal(n) * 10 ** (env / 20)
    data = plane_wave(mono, az) + diffuse_noise(n, level=0.001, seed=seed)
    write_bwf(folder / "take.wav", data, date="2026-07-20", time="09:00:00")
    sess = asc.open_session(folder)
    features.extract_session(sess, folder / "analysis" / "features",
                             verbose=False)
    return folder


@pytest.fixture(scope="module")
def house(tmp_path_factory):
    root = tmp_path_factory.mktemp("net") / "house"
    m = int(90 * RATE) + 8
    e1, e2 = _env(m, 21, smooth=2) / 2, _env(m, 22, smooth=2) / 2
    _node_session(root / "kitchen", e1[SHIFT:SHIFT + m - 8], 30.0, 1)
    _node_session(root / "hall", 0.8 * e1[:m - 8], 120.0, 2)  # lags kitchen
    _node_session(root / "bedroom", e2[SHIFT:SHIFT + m - 8], -60.0, 3)
    return root


def test_run_network_end_to_end(house):
    doc = N.run_network(house, win_s=30.0, max_lag_s=2.0, threshold=0.35)
    assert doc["nodes"] == ["bedroom", "hall", "kitchen"]
    assert (house / "analysis" / "network.json").exists()
    assert (house / "analysis" / "network.png").exists()
    C = np.array(doc["coupling_median"], float)
    hall, kitchen, bedroom = 1, 2, 0
    assert C[hall, kitchen] > 0.6                  # the true edge
    assert C[bedroom, hall] < 0.35 and C[bedroom, kitchen] < 0.35
    assert doc["hub_node"] in ("hall", "kitchen")
    # kitchen leads hall by ~0.5 s
    lag = np.array(doc["lag_median_s"], float)[kitchen, hall]
    assert 0.2 < lag < 0.8
    assert doc["strongest_pair"]["nodes"] == ["kitchen", "hall"]
    assert 0 < doc["density_median"] <= 1


def test_run_network_writes_catalog_rows(house):
    from ambiscape import catalog
    summary = json.loads((house / "analysis" / "summary.json").read_text())
    assert summary["net_n_nodes"] == 3
    assert summary["net_hub_node"] in ("hall", "kitchen")
    assert 0 <= summary["net_density_median"] <= 1
    col = catalog.collect(house.parent)
    assert "house" in col and "net_hub_node" in col["house"]


def test_run_network_cli(house, capsys):
    from ambiscape.cli import main
    assert main(["network", str(house), "--win", "30",
                 "--max-lag", "2"]) == 0
    out = capsys.readouterr().out
    assert "3 nodes" in out and "hub" in out


def test_load_network_needs_two_nodes(tmp_path):
    (tmp_path / "only" / "analysis" / "features").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        N.load_network(tmp_path)


# ------------------------------------------------------- following a source


def _walk_grid(rooms=4, dwell_s=30.0, rate=RATE, gains=None, baseline_s=60.0):
    """A source that visits each room in turn, on a known schedule.

    Every node sits at its own floor, then a source moves through the rooms
    dwelling ``dwell_s`` in each. ``gains`` gives each node a different
    sensitivity, so a method that compares raw levels between nodes gets the
    wrong answer and one that compares each node with its own past does not.
    """
    dt = 1.0 / rate
    n_base = int(baseline_s * rate)
    n_dwell = int(dwell_s * rate)
    n = n_base + rooms * n_dwell
    t = dt * np.arange(n)
    gains = gains if gains is not None else np.zeros(rooms)
    floors = -60.0 + np.asarray(gains, float)
    X = np.repeat(floors[:, None], n, axis=1)
    for r in range(rooms):
        a = n_base + r * n_dwell
        X[r, a:a + n_dwell] += 30.0            # the source is here
        for other in range(rooms):             # leaks a little everywhere
            if other != r:
                X[other, a:a + n_dwell] += 5.0
    return t, X, float(t[n_base]), float(t[-1])


def test_follow_source_recovers_a_known_walk():
    t, X, t0, t1 = _walk_grid()
    names = ["living", "hall", "bath", "bed"]
    r = N.follow_source(t, X, t0, t1, names=names, baseline_s=60.0,
                        slice_s=10.0)
    assert [s["name"] for s in r["itinerary"]] == names
    assert r["n_visited"] == 4
    assert r["n_changes"] == 3
    for s, want in zip(r["itinerary"], names):
        assert s["t1"] - s["t0"] == pytest.approx(30.0, abs=10.0)


def test_follow_source_is_immune_to_per_node_gain():
    """The point of the rise: a 20 dB gain spread must not change the answer."""
    names = ["living", "hall", "bath", "bed"]
    flat = N.follow_source(*_walk_grid()[:2], 60.0, 179.0, names=names,
                           baseline_s=60.0)
    skew = _walk_grid(gains=[+12.0, -8.0, +3.0, -5.0])
    tilted = N.follow_source(skew[0], skew[1], skew[2], skew[3], names=names,
                             baseline_s=60.0)
    assert ([s["name"] for s in tilted["itinerary"]]
            == [s["name"] for s in flat["itinerary"]] == names)


def test_follow_source_reports_nothing_when_nothing_is_audible():
    """A silent interval must give an empty itinerary, not a guessed room."""
    t, X, t0, t1 = _walk_grid()
    X = np.repeat(X[:, :1], X.shape[1], axis=1)      # flat: no source at all
    r = N.follow_source(t, X, t0, t1, names=["a", "b", "c", "d"],
                        baseline_s=60.0)
    assert r["itinerary"] == []
    assert r["n_visited"] == 0 and r["n_changes"] == 0


def test_follow_source_validates_its_inputs():
    t, X, t0, t1 = _walk_grid()
    with pytest.raises(ValueError):
        N.follow_source(t, X[:, :-3], t0, t1)
    with pytest.raises(ValueError):
        N.follow_source(t, X, t0, t1, names=["only", "two"])
