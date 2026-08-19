"""Synthetic AmbiX fixtures: ground truth without real recordings.

The generators live in `ambiscape.examples` now, so that the documentation can build a
session a reader can run against. Imported rather than copied: a fixture that drifts
from the shipped example tests something no user can reproduce.

``write_bwf`` writes a 4-channel PCM WAV with the BWF ``bext`` chunk the session
scanner parses (date, time, zTRK channel tags). The signal generators produce plane
waves (known azimuth/elevation), diffuse noise (uncorrelated channels), and a two-bell
"carillon" with known cycle period, strike phases, partial ratios and swing FM -- the
ground truth the analysis modules are tested against.
"""
from __future__ import annotations

from pathlib import Path  # noqa: F401

import numpy as np
import pytest

from ambiscape.examples import (  # noqa: F401
    BELL_A, BELL_B, CYCLE, FS, bell_track, diffuse_noise, plane_wave, write_bwf,
)


@pytest.fixture(scope="session")
def bell_session(tmp_path_factory):
    """A 300 s session: two locked synthetic bells for 240 s, then quiet.

    Ground truth: cycle 3.0 s; bell A strikes at phases 0 and 0.35, bell B
    at 0.5; partials at f0*(2.4, 4, 6) for f0 = 480 / 600 Hz; FM 3 / 2
    cents; azimuths 30 / 60 deg.
    """
    folder = tmp_path_factory.mktemp("bells")
    dur, active = 300.0, 240.0
    n = int(dur * FS)
    a = bell_track(dur, active, BELL_A, seed=1)
    b = bell_track(dur, active, BELL_B, seed=2)
    data = (plane_wave(a, BELL_A["az"]) + plane_wave(b, BELL_B["az"])
            + diffuse_noise(n, level=0.01))
    write_bwf(folder / "bells.wav", data)
    return folder


@pytest.fixture(scope="session")
def bell_features(bell_session):
    """Session + cached features (analysis/features/*.npz) for the bells."""
    import ambiscape as asc
    from ambiscape import features
    sess = asc.open_session(bell_session)
    out = bell_session / "analysis"
    features.extract_session(sess, out / "features", verbose=False)
    F = features.load_features(sorted((out / "features").glob("*.npz")))
    return sess, out, F
