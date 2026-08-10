"""A descriptor must not report a number the recording cannot support.

The case that prompted this: the complexity index averages over 300 s
chunks and returned 0.0 for any shorter session --- indistinguishable from
a measurement, and produced by every clip corpus in the field, since those
are built from four to thirty seconds of audio.
"""
import numpy as np
import pytest

from ambiscape import resolve, timescales as ts


@pytest.fixture(scope="module")
def short_features(tmp_path_factory):
    """A real 60 s session: long enough to analyse, too short for the index."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from conftest import (BELL_A, FS, bell_track, diffuse_noise, plane_wave,
                          write_bwf)
    import ambiscape as asc
    from ambiscape import features

    d = tmp_path_factory.mktemp("short")
    dur = 60.0
    a = bell_track(dur, dur - 10, BELL_A, seed=3)
    write_bwf(d / "short.wav",
              plane_wave(a, BELL_A["az"]) + diffuse_noise(int(dur * FS), 0.01))
    out = d / "analysis"
    features.extract_session(asc.open_session(d), out / "features",
                             verbose=False)
    return features.load_features(sorted((out / "features").glob("*.npz")))


# --------------------------------------------------------------- registry

def test_every_summary_key_is_registered_or_exempt(bell_features):
    """The guard rots the moment a descriptor is added without a window.

    A new key is neither trustworthy at all lengths nor known to be
    fragile until someone decides which, so the test refuses the choice
    being skipped rather than guessing on the author's behalf.
    """
    _sess, _out, F = bell_features
    s = resolve.full_summary(F, check_windows=False)
    missing = ts.unregistered(s)
    assert not missing, (
        "these summary keys have no timescale window and no exemption: "
        f"{missing}. Add them to WINDOWS with a window and a source, or to "
        "EXEMPT with the reason they are safe at any length."
    )


def test_windows_declare_their_provenance():
    """Every bound says whether it was measured or asserted."""
    for w in ts.WINDOWS.values():
        assert w.source in ("measured", "asserted")
        assert w.why, f"{w.key} has no reason recorded"


def test_bands_are_contiguous_and_ordered():
    lo = 0.0
    for name in ts.BANDS:          # every rung, however many there are now
        b = ts.BANDS[name]
        assert b["t_lo"] == lo, f"{name} does not start where the last ended"
        lo = b["t_hi"]
    assert lo == float("inf")


def test_band_of():
    assert ts.band_of(0.1) == "micro"
    assert ts.band_of(1.0) == "meso"        # the sound object, and meso motion
    assert ts.band_of(60.0) == "macro"
    # the ladder reaches past a day now: an hour is a building's rhythm,
    # not the same kind of thing as a minute
    assert ts.band_of(3600.0) == "cyclic"
    assert ts.band_of(24 * 3600.0) == "circadian"


# ------------------------------------------------------------------ guard

def test_hard_window_nulls_rather_than_returning_zero():
    """The regression: a chunkless complexity index must be None, not 0.0."""
    s, low = ts.check({"aci": 0.0, "leq_dbfs": -40.0}, duration_s=60.0)
    assert s["aci"] is None
    assert s["leq_dbfs"] == -40.0            # exempt, untouched
    assert [e["key"] for e in low] == ["aci"]
    assert low[0]["needs_s"] == 300.0 and low[0]["had_s"] == 60.0


def test_soft_window_keeps_the_value_and_flags_it():
    s, low = ts.check({"L50": -55.0}, duration_s=30.0)
    assert s["L50"] == -55.0, "a soft window keeps the value"
    assert low and low[0]["kind"] == "soft"


def test_long_session_is_untouched():
    before = {"aci": 58.6, "L50": -55.0, "mech_periodicity_hz": 0.3}
    s, low = ts.check(dict(before), duration_s=148.7 * 3600)
    assert s == before
    assert low == []


def test_full_summary_flags_a_short_session(short_features):
    """End to end: a one-minute session cannot support the index."""
    s = resolve.full_summary(short_features)
    assert s.get("aci") is None
    keys = {e["key"] for e in s.get("low_confidence", [])}
    assert "aci" in keys
    assert all(e["needs_s"] > 60 for e in s["low_confidence"])


def test_full_summary_can_be_asked_for_the_raw_set(short_features):
    raw = resolve.full_summary(short_features, check_windows=False)
    assert "low_confidence" not in raw


# ----------------------------------------------------------------- figure

def test_families_match_the_registry():
    """The figure's grouping must not drift from the windows it summarises.

    The figure shows seven families rather than 27 descriptors, which is a
    readability choice; it becomes a lie the moment a family's threshold
    stops matching the descriptors it stands for.
    """
    thresholds = {w.min_s for w in ts.WINDOWS.values()}
    for _label, min_s, _source in ts.FAMILIES:
        assert min_s in thresholds, (
            f"family threshold {min_s} s matches no descriptor window; "
            "either a window moved or the family is stale"
        )


def test_phenomena_and_conventions_are_ordered_and_sourced():
    """Ranges run forwards; a fixed-duration corpus is a point, not an error.

    StillStanding365 deposits sessions of one length, so t0 == t1. The
    figure gives such an entry a visible mark rather than a zero-width bar.
    """
    for name, t0, t1, why, source in ts.PHENOMENA + ts.CONVENTIONS:
        assert t1 >= t0, f"{name}: range runs backwards"
        assert source in ("measured", "asserted")
        assert why


def test_figure_renders(tmp_path):
    out = ts.figure(tmp_path / "timescales.png")
    assert out.exists() if hasattr(out, "exists") else True
    p = tmp_path / "timescales.png"
    assert p.stat().st_size > 20_000
