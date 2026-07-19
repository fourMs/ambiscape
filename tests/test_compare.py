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
