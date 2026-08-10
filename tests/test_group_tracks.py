"""One source, interrupted, is several tracks. Grouping counts lines.

`tonal_tracks` answers how long a line was continuously present. A machine
that pauses, or changes speed between programme phases and comes back, is
correct as several tracks and misleading as a description of the source.
"""
from ambiscape.tonality import group_tracks


def _t(f, t0, t1, prom=12.0):
    return {"f_median_hz": f, "t0_min": t0, "t1_min": t1,
            "minutes": t1 - t0 + 1, "prominence_db": prom, "drift_cents": 0.0}


def test_a_pump_seen_five_times_is_one_line():
    """The Ghent dishwasher: one circulation pump, five track segments."""
    tracks = [_t(275.4, 48, 115), _t(275.4, 28, 45), _t(275.4, 120, 137),
              _t(275.4, 15, 24), _t(276.1, 140, 150)]
    lines = group_tracks(tracks)
    assert len(lines) == 1
    line = lines[0]
    assert line["n_segments"] == 5
    assert line["minutes"] == sum(t["minutes"] for t in tracks)
    assert line["t0_min"] == 15 and line["t1_min"] == 150


def test_two_genuinely_different_lines_stay_apart():
    lines = group_tracks([_t(275.4, 0, 20), _t(7810.5, 0, 20)])
    assert [l["f_median_hz"] for l in lines] == [275.4, 7810.5]


def test_the_tolerance_decides_and_is_in_cents():
    """A semitone apart: separate by default, one line at a wide tolerance."""
    a, b = _t(200.0, 0, 10), _t(211.9, 20, 30)      # ~100 cents apart
    assert len(group_tracks([a, b])) == 2
    assert len(group_tracks([a, b], tol_cents=150.0)) == 1


def test_prominence_is_weighted_by_segment_length():
    """A long quiet segment should not be averaged away by a short loud one."""
    lines = group_tracks([_t(300.0, 0, 99, prom=10.0),
                          _t(300.0, 200, 201, prom=40.0)])
    assert len(lines) == 1
    assert 10.0 < lines[0]["prominence_db"] < 11.0


def test_no_tracks_is_not_an_error():
    assert group_tracks([]) == []
