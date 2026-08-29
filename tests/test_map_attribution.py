"""Detected events inherit the hand annotation that covers them.

Without attribution every detected event plots as an anonymous
"incidental figure", even when the annotator has already named the span
it falls in — a walk's gravel stretch, a traffic corridor. With spans in
hand, the map should colour those events by the named object's Schafer
function and say how many it attributed.
"""
import numpy as np

from ambiscape.taxonomy import map_objects


def _F_with_bursts(nsec=200, bursts=(100.0, 150.0)):
    dt = 0.125
    tf = np.arange(0.0, nsec, dt)
    fast = np.full(len(tf), -50.0)
    for tb in bursts:
        fast[(tf >= tb) & (tf < tb + 1.0)] = -32.0
    return {"t": np.arange(float(nsec)), "t_fast": tf,
            "fast_db": fast.astype(np.float32)}


def _ann(kind="signal"):
    return {"objects": [
        {"name": "traffic corridor", "kind": kind,
         "mass": "complex", "facture": "sustained",
         "spans": [["00:01:30", "00:01:50"]]},   # covers 90-110 s
    ]}


def test_event_inside_a_named_span_inherits_kind_and_name():
    objs, stats = map_objects(ann=_ann("signal"), F=_F_with_bursts())
    at_100 = [o for o in objs if abs(o["spans"][0][0] - 100.0) < 3.0]
    assert at_100 and at_100[0]["kind"] == "signal"
    assert at_100[0]["name"].startswith("traffic corridor")
    assert stats["n_attributed"] == 1


def test_event_outside_named_spans_stays_incidental():
    objs, stats = map_objects(ann=_ann("signal"), F=_F_with_bursts())
    at_150 = [o for o in objs if abs(o["spans"][0][0] - 150.0) < 3.0]
    assert at_150 and at_150[0]["kind"] == "figure"


def test_keynote_spans_do_not_capture_events():
    # a keynote is a bed, not a source of figures: events over a bed stay
    # incidental rather than being painted keynote-blue
    objs, stats = map_objects(ann=_ann("keynote"), F=_F_with_bursts())
    at_100 = [o for o in objs if abs(o["spans"][0][0] - 100.0) < 3.0]
    assert at_100 and at_100[0]["kind"] == "figure"
    assert stats["n_attributed"] == 0


def test_timeline_style_marks_soundmark_attribute():
    from ambiscape.taxonomy import MAGENTA, KIND_COLOR, _timeline_style
    fill, edge, lw = _timeline_style({"kind": "signal",
                                      "soundmark": "community"})
    assert fill == KIND_COLOR["signal"]
    assert edge == MAGENTA and lw > 0
    fill2, edge2, lw2 = _timeline_style({"kind": "signal"})
    assert edge2 == "none" and lw2 == 0
    fill3, _e, _l = _timeline_style({"kind": "soundmark"})
    assert fill3 == MAGENTA
