"""Cross-session comparison tests: two synthetic visits to one 'room'.

Session ON carries a strong 200 Hz tonal line for its whole length; session
OFF is the same room without it. Ground truth: the line stands proud of the
minimum spectrum in ON and not in OFF; the low-band floor is higher in ON;
the two sessions, dated a day apart, land on one clock-aligned row.
"""
import datetime as dt
import json

import numpy as np
import pytest

import ambiscape as asc
from ambiscape import compare as C
from ambiscape import features

from tests.conftest import FS, write_bwf, plane_wave, diffuse_noise

LINE_HZ = 200.0


def _analyzed_session(folder, date, time, with_line, cycle_s=None):
    """Write a synthetic AmbiX WAV, extract features, and hand-write a
    summary.json + a two-interval states.json (first half 'machine_on')."""
    folder.mkdir(parents=True, exist_ok=True)
    dur = 180.0
    n = int(dur * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(0 if with_line else 1)
    mono = 0.02 * rng.standard_normal(n)
    if with_line:
        amp = np.full(n, 0.15)
        if cycle_s:                       # gate the tone into an on/off cycle
            amp *= (np.sin(2 * np.pi * t / cycle_s) > 0).astype(float)
        mono = mono + amp * np.sin(2 * np.pi * LINE_HZ * t)
    data = plane_wave(mono, 120.0) + diffuse_noise(n, level=0.01)
    write_bwf(folder / "take.wav", data, date=date, time=time)
    sess = asc.open_session(folder)
    out = folder / "analysis"
    features.extract_session(sess, out / "features", verbose=False)
    (out / "summary.json").write_text(json.dumps({
        "date": date, "duration_min": dur / 60, "laeq_dbfs": -40.0,
        "L90": -55.0, "events_per_min": 1.0, "ndsi": -0.2}))
    half = dur / 2
    (out / "states.json").write_text(json.dumps({"states": {
        "machine_on": {"intervals_s": [[0.0, half]], "duration_min": half / 60,
                       "L90": -50.0},
        "machine_off": {"intervals_s": [[half, dur]], "duration_min": half / 60,
                        "L90": -58.0}}}))
    return folder


@pytest.fixture(scope="module")
def two_sessions(tmp_path_factory):
    root = tmp_path_factory.mktemp("compare")
    on = _analyzed_session(root / "2026-07-15-room", "2026-07-15", "22:00:00",
                           with_line=True)
    off = _analyzed_session(root / "2026-07-16-room", "2026-07-16", "22:00:00",
                            with_line=False)
    return [on, off]


def test_load_comparison(two_sessions):
    sess = C.load_comparison(two_sessions)
    assert [s["name"] for s in sess] == ["2026-07-15-room", "2026-07-16-room"]
    assert sess[0]["date"] == dt.date(2026, 7, 15)
    assert "logspec" in sess[0]["F"] and sess[0]["states"] is not None


def test_load_comparison_needs_features(tmp_path):
    (tmp_path / "empty" / "analysis").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        C.load_comparison([tmp_path / "empty"])


def test_laeq_timeline_shape(two_sessions):
    sess = C.load_comparison(two_sessions)
    t, la = C.laeq_timeline(sess[0]["F"], bin_s=60.0)
    assert len(t) == len(la) == 3            # 180 s -> 3 one-minute bins
    assert np.isfinite(la).all()


def _fake(date, h0, h1):
    """A minimal session dict for clock_rows: only date + t-span are read."""
    return {"name": f"{date}-{int(h0)}", "date": dt.date.fromisoformat(date),
            "F": {"t": np.array([h0 * 3600, h1 * 3600])}}


def test_clock_rows_separate_visits():
    # two visits a full day apart at the same wall time -> two rows
    sess = [_fake("2026-07-15", 22, 25), _fake("2026-07-16", 22, 25)]
    rows = C.clock_rows(sess)
    assert len(rows) == 2


def test_clock_rows_bridge_night_into_day():
    # a night session flowing into the next morning's day session -> one row,
    # the day session shifted +24 h so the two lie end to end
    night = _fake("2026-07-18", 22.9, 34.0)     # 22:54 -> 10:02 next day
    day = _fake("2026-07-19", 11.0, 21.7)        # 11:00 -> 21:43, own axis
    rows = C.clock_rows([night, day])
    assert len(rows) == 1
    shifts = dict(rows[0])
    assert shifts[0] == 0.0 and shifts[1] == 24.0


def test_clock_rows_undated_gets_own_row(two_sessions):
    sess = C.load_comparison(two_sessions)
    sess[1]["date"] = None
    rows = C.clock_rows(sess)
    assert len(rows) == 2


def test_line_prominence_detects_the_machine(two_sessions):
    sess = C.load_comparison(two_sessions)
    on = C.line_prominence(sess[0], [LINE_HZ])[LINE_HZ]
    off = C.line_prominence(sess[1], [LINE_HZ])[LINE_HZ]
    assert abs(on["peak_hz"] - LINE_HZ) < 20
    assert on["prominence_db"] > off["prominence_db"] + 6   # clearly proud
    assert off["prominence_db"] < 3


def test_floor_difference_localizes_the_line(two_sessions):
    sess = C.load_comparison(two_sessions)
    # ON session hours 0..0.05 (its first 3 min) vs OFF session same window
    # both sessions start 22:00 -> query their first 3 min on the clock axis
    freqs, diff = C.floor_difference(sess[0], (22.0, 22.06),
                                     sess[1], (22.0, 22.06))
    band = (freqs > LINE_HZ - 20) & (freqs < LINE_HZ + 20)
    assert diff[band].max() > 6                 # excess concentrated at 200 Hz
    assert diff[band].max() > np.median(diff) + 6


def test_band_level_tracks_the_tone(two_sessions):
    sess = C.load_comparison(two_sessions)
    on = C.band_level(sess[0]["F"], 150, 260).mean()
    off = C.band_level(sess[1]["F"], 150, 260).mean()
    assert on > off + 3


def test_duty_cycle_recovers_a_gated_source(tmp_path):
    folder = _analyzed_session(tmp_path / "2026-07-17-room", "2026-07-17",
                               "12:00:00", with_line=True, cycle_s=20.0)
    sess = C.load_comparison([folder,
                              _analyzed_session(tmp_path / "2026-07-18-room",
                                                "2026-07-18", "12:00:00",
                                                with_line=False)])
    # session starts 12:00:00 -> its t-axis runs 43200..43380 s
    d = C.duty_cycle(sess[0]["F"], 43200.0, 43380.0, f0=150, f1=260,
                     period_range=(10.0, 60.0), smooth_s=5)
    assert abs(d["period_min"] - 20.0 / 60) < 0.1     # ~0.33 min period
    assert 25 < d["duty_pct"] < 75
    assert d["acf_peak"] > 0.2


def test_run_compare_writes_figures(two_sessions, tmp_path):
    out = tmp_path / "cmp"
    doc = C.run_compare(two_sessions, out, lines=[LINE_HZ],
                        band=(2000.0, 8000.0))
    assert (out / "compare.json").exists()
    for fig in ("timelines", "ltas", "roses", "band"):
        assert (out / doc["figures"][fig]).exists() or \
            (out / f"compare_{fig}.png").exists()
    on = doc["line_prominence"]["2026-07-15-room"][str(LINE_HZ)]
    off = doc["line_prominence"]["2026-07-16-room"][str(LINE_HZ)]
    assert on["prominence_db"] > off["prominence_db"] + 6
    assert set(doc["pooled"]) == {"2026-07-15-room", "2026-07-16-room"}


# ---------------------------------------------------------------- cross-node


def _legacy_npz(path, start, nsec, level_db, fill_tail=0):
    """A per-take feature cache in the pre-fix on-disk layout.

    ``fill_tail`` extra fast frames past the last whole second are left at
    exactly 0.0 dB — the preallocated full-scale fill the old extractor
    wrote at the end of every take (a false click at each take boundary).
    """
    fast = np.full(nsec * 8 + fill_tail, level_db, np.float32)
    if fill_tail:
        fast[-fill_tail:] = 0.0
    np.savez(path, start=np.float64(start), fs=np.int64(16000),
             fast_dt=np.float64(0.125), hi_dt=np.float64(0.02),
             fast_db=fast, fast_dba=fast,
             env_hi=np.full(nsec * 50 + fill_tail, 1e-6, np.float32),
             rms_w=np.full(nsec, 10 ** (level_db / 20), np.float32),
             peak=np.zeros(nsec, np.float32),
             oct_pow=np.zeros((nsec, 10), np.float32),
             centroid=np.zeros(nsec, np.float32),
             flatness=np.zeros(nsec, np.float32),
             logspec=np.zeros((nsec, 96), np.float32),
             I_band=np.zeros((nsec, 10, 3), np.float32),
             az=np.zeros(nsec, np.float32),
             el=np.zeros(nsec, np.float32),
             diffuse=np.zeros(nsec, np.float32),
             minspec=np.zeros((-(-nsec // 60), 5), np.float32),
             freqs=np.linspace(100, 8000, 5).astype(np.float32),
             logf=np.geomspace(25, 20000, 97).astype(np.float32))
    return path


def _day_seconds(F, nsec_day):
    """Pipeline-style 1 Hz day array from the concatenated fast track."""
    arr = np.full(nsec_day, np.nan, np.float32)
    sec = F["t_fast"].astype(np.int64)
    p = 10 ** (F["fast_dba"] / 10)
    pw = np.bincount(sec, weights=p, minlength=nsec_day)
    ct = np.bincount(sec, minlength=nsec_day)
    has = ct > 0
    arr[has] = 10 * np.log10(pw[has] / ct[has])
    return arr


def test_xnode_boundary_click_masked_and_silence_empty(tmp_path):
    """Two silent nodes, takes ending in legacy full-scale fill frames:
    the boundary click never reaches the heatmap, and a silent day earns
    no loudest dots even though one node's floor sits 2 dB higher."""
    from ambiscape import features as feat
    day = {}
    for node, lvl in (("n1", -60.0), ("n2", -58.0)):   # gain offset only
        d = tmp_path / node
        d.mkdir()
        paths = [_legacy_npz(d / f"take{i}.npz", start=i * 1200,
                             nsec=1190, level_db=lvl, fill_tail=4)
                 for i in range(3)]
        F = feat.load_features(paths)
        assert float(F["fast_dba"].max()) < -20        # fill frames dropped
        assert len(F["fast_dba"]) == 3 * 1190 * 8
        day[node] = _day_seconds(F, 3600)
    names, A, H = C.xnode_day_matrix(day, bin_s=300)
    assert np.nanmax(H) < 3.0                # no bright boundary bins
    floors = {n: C.xnode_floor(day[n]) for n in names}
    loud = C.xnode_loudest(names, A, floors_db=floors)
    assert all(v is None for v in loud)      # silence: strip stays empty


def test_xnode_margin_and_floor_rules():
    day = {"a": np.full(3600, -60.0), "b": np.full(3600, -58.0)}
    # flat gain offset: b is fractionally "louder" everywhere, no dots
    names, A, H = C.xnode_day_matrix(day, bin_s=300)
    assert all(v is None for v in C.xnode_loudest(names, A))
    # one genuine event on a (bin 2) + one near-floor excursion (bin 4);
    # both clear the 3 dB margin over b, which sits flat at -58 dB
    day["a"][600:900] = -40.0
    day["a"][1200:1500] = -54.5
    names, A, H = C.xnode_day_matrix(day, bin_s=300)
    floors = {n: C.xnode_floor(day[n]) for n in names}
    loud = C.xnode_loudest(names, A, floors_db=floors, margin_db=3.0)
    assert loud[2] == "a" and loud.count("a") == 2 and "b" not in loud
    # floor_suspect raises the required clearance by 3 dB: the -54.5 dB
    # excursion still beats b, but no longer clears a's handicapped floor
    floors["a"] = C.xnode_floor(day["a"], floor_suspect=True)
    assert floors["a"] == pytest.approx(C.xnode_floor(day["a"]) + 3.0)
    loud = C.xnode_loudest(names, A, floors_db=floors)
    assert loud[2] == "a" and loud[4] is None


def test_xnode_loudest_ranks_on_level_not_on_own_baseline():
    """A quiet node with a peaky day must not take a louder node's bins.

    Ranking on the display normalization H (level minus that node's own
    day median) rewards the largest excursion above a node's own
    baseline, which is the peakiest node and not the loudest one. Here
    'quiet' sits 10 dB below 'loud' all day and still swings further
    above its own median, so an H-ranked rule hands it the evening.
    """
    quiet = np.full(3600, -70.0)
    loud = np.full(3600, -60.0)
    quiet[1800:2100] = -50.0                 # +20 dB over its own median
    loud[1800:2100] = -45.0                  # +15 dB over its own median
    day = {"loud": loud, "quiet": quiet}
    names, A, H = C.xnode_day_matrix(day, bin_s=300)

    # what an H-ranked rule sees: the quiet node wins the evening bin
    i, j = names.index("quiet"), names.index("loud")
    assert H[i, 6] - H[j, 6] > 3.0

    # what the rule does now: the louder node takes it, on level
    loudest = C.xnode_loudest(names, A)
    assert loudest[6] == "loud"
    assert "quiet" not in loudest


def test_xnode_gain_offsets_make_nodes_comparable():
    """Two recorders hearing one field differ only by gain, so neither wins."""
    field = np.full(3600, -60.0)
    field[1800:2100] = -40.0                 # an event both nodes hear
    day = {"hot": field + 6.0, "cold": field.copy()}
    names, A, _ = C.xnode_day_matrix(day, bin_s=300)
    floors = {n: C.xnode_floor(day[n]) for n in names}

    # uncorrected, the higher-gain recorder wins wherever the floor rule
    # lets it speak, which is an artefact of its gain and nothing else
    assert "hot" in C.xnode_loudest(names, A, floors_db=floors)

    # the floors themselves estimate the offset, and correcting by it
    # leaves two nodes that agree, so no bin is awarded
    offs = C.xnode_gain_offsets(floors)
    assert offs["hot"] == pytest.approx(3.0, abs=0.5)
    assert offs["cold"] == pytest.approx(-3.0, abs=0.5)
    corrected = C.xnode_loudest(names, A, floors_db=floors,
                                gain_offsets_db=offs)
    assert all(v is None for v in corrected)


def test_xnode_figure_writes(tmp_path):
    day = {"a": np.full(3600, -60.0), "b": np.full(3600, -58.0)}
    day["a"][600:900] = -40.0
    day["b"][:300] = np.nan                  # a no-data bin
    names, A, H = C.xnode_day_matrix(day, bin_s=300)
    loud = C.xnode_loudest(names, A)
    p = C.xnode_figure(names, H, loud, tmp_path / "x.png",
                       title="test day", labels={"a": "node a (living)"})
    assert p.exists() and p.stat().st_size > 10_000


# ------------------------------------------------- xnode figure layout

def _xnode_fig(n_names=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ambiscape import compare as C
    fig = plt.figure(figsize=(12.8, 2.3 + 0.6 * n_names))
    axes = C._xnode_axes(fig, n_names)
    return fig, axes


def test_xnode_panels_are_the_same_width():
    """The colorbar must not steal width from the heatmap alone: with it
    inside ax[0], the two panels stop lining up and the strip no longer
    reads against the hours above it."""
    fig, (heat, strip, cax) = _xnode_fig()
    fig.canvas.draw()
    a, b = heat.get_position(), strip.get_position()
    assert a.x0 == pytest.approx(b.x0, abs=1e-6)
    assert a.x1 == pytest.approx(b.x1, abs=1e-6)
    assert cax.get_position().x0 > a.x1      # colourbar sits outside both


def test_xnode_hours_are_marked_every_four():
    from ambiscape import compare as C
    assert list(C._hour_ticks()) == [0, 4, 8, 12, 16, 20, 24]


# ------------------------------------- a flat node cannot be the loudest room

def test_loudest_ignores_a_node_with_no_dynamics_of_its_own():
    """A node whose level never clears its own floor is not reporting room
    activity, whatever its absolute level. In the SINS network the bedroom
    node rose 1.9 dB above its own floor at the 90th percentile, against
    18-22 dB for rooms in use.

    Per-bin floor clearance does not catch it. A stationary node with a few
    dropouts has its *estimated* floor pulled down by them, so every ordinary
    bin clears that floor by ~10 dB and the node reads as permanently in
    activity. Eligibility has to be judged over the day, and with a statistic
    the dropouts cannot move — hence the interquartile range.
    """
    import numpy as np
    from ambiscape import compare as C
    nbin = 96
    rng = np.random.default_rng(3)
    live = -55 + 30 * rng.random(nbin)          # a room in use
    flat = -30 + 0.5 * rng.standard_normal(nbin)   # loud, stationary
    flat[::10] = -42                            # dropouts drag the floor down
    A = np.vstack([live, flat])
    floors = {1: float(np.percentile(live, 5)),
              2: float(np.percentile(flat, 5))}
    assert np.median(flat) - floors[2] > 8, "synthetic should clear its floor"
    won = C.xnode_loudest([1, 2], A, floors_db=floors)
    assert 2 not in won, "a node with no dynamics of its own won a bin"
    assert 1 in won, "the live node should still win where it is defensible"
