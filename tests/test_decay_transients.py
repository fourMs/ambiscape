"""Blind decay from transients: a known exponential decay must come back, and silence must not.

Synthetic clicks convolved with an exponentially decaying noise tail of known T60 in a quiet
room; the transient picker must find the clicks and the per-band medians must land near the
true T60. A recording with no transients gives empty estimates rather than an error.
"""
import numpy as np
import pytest

import ambiscape as asc

FS = 16000


def _room_clicks(t60, n_clicks=12, dur=40.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(int(dur * FS))
    tail_len = int(1.5 * FS)
    tail = rng.standard_normal(tail_len) * np.exp(-6.91 * np.arange(tail_len) / FS / t60)
    for k in range(n_clicks):
        i = int((2.0 + k * 3.0) * FS)
        x[i:i + tail_len] += 0.8 * tail * (0.7 + 0.6 * rng.random())
    return x + 1e-4 * rng.standard_normal(len(x))


def test_candidates_find_the_clicks():
    x = _room_clicks(0.5)
    c = asc.transient_candidates(x, FS, n_max=20)
    assert 8 <= len(c) <= 12
    assert all(abs((t - 2.0) % 3.0) < 0.05 or abs((t - 2.0) % 3.0 - 3.0) < 0.05 for t in c)


@pytest.mark.parametrize("t60", [0.4, 1.0])
def test_decay_from_transients_recovers_t60(t60):
    out = asc.decay_from_transients(_room_clicks(t60), FS, bands=((500, 1000), (1000, 2000), (2000, 4000)))
    assert out["n_mid"] >= 8
    assert abs(out["T60_mid"] - t60) < 0.25 * t60 + 0.05
    assert out["T60_iqr"]["1000-2000"] < 0.3


def test_silence_gives_no_estimates():
    x = 1e-4 * np.random.default_rng(0).standard_normal(20 * FS)
    out = asc.decay_from_transients(x, FS)
    assert out["candidates"] == [] and out["estimates"] == [] and np.isnan(out["T60_mid"])
