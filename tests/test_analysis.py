"""Onset and span on a bare series: where something begins, and where it stops.

`series_onset` answers the first question and `series_span` both. The tests
pin the property that matters for the pair --- that the span's opening index is
exactly what the onset returns, so a prefix and a suffix measured with them sit
on one convention --- and the censoring cases, where a series was cut before the
event began or after it ended.
"""
import numpy as np
import pytest

import ambiscape as asc

# --------------------------------------------------- series_span (first, last)


def test_series_span_first_index_matches_series_onset():
    """The span's opening index must be exactly what series_onset gives."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = np.abs(rng.standard_normal(400)) * 0.05
        a = int(rng.integers(80, 300))
        x[a:a + 40] += rng.uniform(1.0, 8.0)
        for rise in (0.25, 0.5, 0.75):
            first, _ = asc.analysis.series_span(x, rise)
            assert first == asc.analysis.series_onset(x, rise)


def test_series_span_brackets_the_event():
    """A single burst in a quiet series is bracketed by the two indices."""
    x = np.full(300, 0.01)
    x[120:160] = 5.0
    first, last = asc.analysis.series_span(x, 0.5)
    assert first == 120 and last == 159


def test_series_span_marks_a_clip_cut_at_its_start():
    """A clip that begins already in the event opens at index 0."""
    x = np.full(300, 0.01)
    x[:60] = 5.0                       # the trim cut into the event
    first, _ = asc.analysis.series_span(x, 0.5)
    assert first == 0                  # caller must read this as censored


def test_series_span_marks_a_clip_cut_at_its_end():
    """A clip still in the event at the last frame closes at the last index."""
    x = np.full(300, 0.01)
    x[240:] = 5.0
    _, last = asc.analysis.series_span(x, 0.5)
    assert last == len(x) - 1


def test_series_span_has_no_range_on_a_flat_series():
    """A flat series has no floor-to-peak range, so neither end is defined.

    This is the reason a caller cannot treat (None, None) as "no event": it
    also means "nothing to measure against".
    """
    assert asc.analysis.series_span(np.full(200, 5.0), 0.5) == (None, None)


def test_series_span_declines_on_a_flat_or_short_series():
    assert asc.analysis.series_span(np.ones(50), 0.5) == (None, None)
    assert asc.analysis.series_span([1.0, 2.0], 0.5) == (None, None)
