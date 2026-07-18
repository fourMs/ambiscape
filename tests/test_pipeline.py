"""End-to-end tests on the synthetic two-bell session (known ground truth:
cycle 3.0 s, strike phases A@0/0.35 B@0.5, partials f0*(2.4, 4, 6) for
f0 = 480/600 Hz, FM 3/2 cents, azimuths 30/60 deg, active 0-240 s)."""
import numpy as np
import pytest

from tests.conftest import BELL_A, BELL_B, CYCLE


def test_analyze_summary(bell_features):
    _sess, _out, F = bell_features
    from ambiscape import analysis
    s = analysis.summarize(F)
    assert s["duration_min"] == pytest.approx(5.0, abs=0.1)
    assert s["diffuseness_median"] < 0.6          # two plane-wave sources


def test_detect_partials(bell_features):
    _sess, _out, F = bell_features
    from ambiscape.rhythm import detect_partials, _activity_masks
    active, quiet, _ = _activity_masks(F)
    assert active[:4].all() and quiet[4:].any()
    pfreq, _rise = detect_partials(F, active, quiet)
    truth = sorted([BELL_A["f0"] * r for r in BELL_A["ratios"]]
                   + [BELL_B["f0"] * r for r in BELL_B["ratios"]])
    for f in truth:
        assert np.min(np.abs(pfreq - f)) < 15, f"partial {f} not found"


def test_rhythm_pipeline_recovers_ground_truth(bell_features):
    sess, out, _F = bell_features
    from ambiscape.rhythm import run_session
    res = run_session(sess, out, verbose=False)
    assert len(res["sources"]) == 2
    for src in res["sources"]:
        assert src["period_s"] == pytest.approx(CYCLE, abs=0.02)
    # bell A's two strike phases 0.35 apart appear in one source's clusters
    phases = [sorted(s["phase_clusters"]) for s in res["sources"]]
    gaps = [min(abs(a - b) for a in ph for b in ph if a != b)
            for ph in phases if len(ph) > 1]
    assert any(abs(g - 0.35) < 0.06 for g in gaps)
    # the two sources are phase-locked
    lock = list(res["phase_lock"].values())[0]
    assert lock["R"] > 0.95
    # azimuths in the right sector (per-strike DOA on clean plane waves)
    azs = sorted(abs(s["azimuth_deg"]) for s in res["sources"])
    assert azs[0] == pytest.approx(30, abs=10)
    assert azs[1] == pytest.approx(60, abs=10)


def test_partial_fm_recovers_cents(bell_features):
    sess, _out, _F = bell_features
    from ambiscape.rhythm import partial_fm
    r = partial_fm(sess.takes[0], BELL_A["f0"] * 4, CYCLE, 240.0)
    assert r["fm_cents_at_cycle"] == pytest.approx(BELL_A["fm_cents"],
                                                   rel=0.5)
    assert r["fm_cents_at_cycle"] > 4 * r["fm_cents_control"]


def test_masking_from_bells(bell_features):
    _sess, _out, F = bell_features
    from ambiscape.background import masking_index
    t = F["t"] - F["t"][0]
    mi = masking_index(F, t < 240, t > 250)
    assert mi["floor_elevation_max_db"] > 6     # partial bands elevated


def test_spatial_run(bell_features):
    sess, out, _F = bell_features
    from ambiscape.spatial import run_session
    doc = run_session(sess, out)
    assert doc["azimuth_R_median"] > 0.8        # two static sources
    assert (out / "spatial.png").exists()


def test_tonality_run(bell_features):
    sess, out, _F = bell_features
    from ambiscape.tonality import run_session
    doc = run_session(sess, out)
    assert doc["inharmonicity_median"] > 0.2    # bell ratios, not harmonic
    fs = [tr["f_median_hz"] for tr in doc["tracks"]]
    assert any(abs(f - BELL_A["f0"] * 2.4) < 20 for f in fs)


def test_modspec_run(bell_features):
    sess, out, _F = bell_features
    from ambiscape.modulation import run_session
    prof = run_session(sess, out)
    # strikes: 3 per 3 s cycle -> dominant micro modulation near 1/3-1 Hz
    assert 0.2 < prof["scales"]["micro"]["peak_freq_hz"] < 1.5
