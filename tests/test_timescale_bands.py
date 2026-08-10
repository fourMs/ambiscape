"""The ladder must reach above `macro`, and there must be only one of it.

The registry's top band ran from 5 s to infinity, which put a 30-second
descriptor and a three-hour one in the same category and left no way to say
"this is a circadian quantity". The cycle finder in `analysis` then grew its
own, longer ladder, and a toolbox with two different sets of timescale bands
is a toolbox that will eventually contradict itself.
"""
from ambiscape import analysis, timescales


def test_the_ladder_reaches_past_a_day():
    assert timescales.band_of(60.0) == "macro"
    assert timescales.band_of(45 * 60.0) == "cyclic"
    assert timescales.band_of(24 * 3600.0) == "circadian"
    assert timescales.band_of(30 * 86400.0) == "seasonal"
    assert timescales.band_of(5 * 365 * 86400.0) == "archival"


def test_macro_no_longer_swallows_everything_above_five_seconds():
    """A three-hour descriptor is not the same kind of thing as a 30 s one."""
    assert timescales.band_of(30.0) == "macro"
    assert timescales.band_of(3 * 3600.0) != "macro"


def test_the_cycle_finder_uses_the_registry_ladder():
    """One ladder, not two. Any period must get the same name from both."""
    for t in (1.0, 30.0, 45 * 60.0, 24 * 3600.0, 30 * 86400.0):
        assert analysis.cycle_band(t) == timescales.band_of(t), t


def test_every_band_still_carries_its_descriptive_columns():
    """The figure and the ch3 table read these; a new band without them
    would render as a blank row rather than fail loudly."""
    for name, b in timescales.BANDS.items():
        assert b["space_room"] and b["memory"], name
