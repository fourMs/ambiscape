"""Tests for the case-study toolbox: machine states, source fingerprints,
civic grid scans, and segment export — synthetic ground truth throughout."""
import numpy as np
import pytest
import soundfile as sf

from ambiscape import background, schedule, states
from ambiscape.io import export_segment, open_session, stereo_preview

from .conftest import FS, plane_wave, write_bwf


# ------------------------------------------------------------------- states

def _square_F(period_s=1440, duty=0.5, nsec=7200, rise_db=8.0,
              band=(250.0, 1000.0)):
    """Fake cached features: a machine adds rise_db in `band` when on."""
    logf = np.geomspace(25, 20000, 97)
    fc = np.sqrt(logf[:-1] * logf[1:])
    rng = np.random.default_rng(5)
    ls = 1e-8 * (1 + 0.05 * rng.standard_normal((nsec, 96))) ** 2
    on = (np.arange(nsec) % period_s) < duty * period_s
    m = (fc >= band[0]) & (fc <= band[1])
    ls[np.ix_(on, m)] *= 10 ** (rise_db / 10)
    return {"logspec": ls, "logf": logf.astype(np.float32),
            "t": np.arange(nsec, dtype=float)}, on


def test_band_level_tracks_state():
    F, on = _square_F()
    lvl = states.band_level(F, (250, 1000))
    assert lvl.shape == (7200,)
    assert np.median(lvl[on]) - np.median(lvl[~on]) == pytest.approx(8.0,
                                                                     abs=1.0)


def test_state_segments_recover_square_wave():
    F, _on = _square_F()
    segs = states.state_segments(states.band_level(F, (250, 1000)))
    ons = [s for s in segs if s["state"] == "on"]
    offs = [s for s in segs if s["state"] == "off"]
    assert len(ons) == 5 and len(offs) == 5
    assert all(s["dur_s"] == pytest.approx(720, abs=30) for s in segs)
    assert ons[0]["t0_s"] == pytest.approx(0, abs=15)
    assert offs[0]["t0_s"] == pytest.approx(720, abs=15)
    # the drone is steady: within-state level SD far below the on/off step
    assert all(s["sd_db"] < 1.0 for s in segs)
    assert ons[0]["median_db"] - offs[0]["median_db"] == pytest.approx(
        8.0, abs=1.0)


def test_switch_points_and_duty_cycle():
    F, _on = _square_F()
    segs = states.state_segments(states.band_level(F, (250, 1000)))
    sw = states.switch_points(segs)
    assert [s["direction"] for s in sw[:2]] == ["off", "on"]
    assert sw[0]["t_s"] == pytest.approx(720, abs=15)
    dc = states.duty_cycle(segs)
    assert dc["period_s"] == pytest.approx(1440, abs=30)
    assert dc["duty"] == pytest.approx(0.5, abs=0.05)
    assert dc["n_cycles"] == 5


def test_state_segments_min_duration_merges_glitches():
    F, _on = _square_F()
    lvl = states.band_level(F, (250, 1000))
    lvl[1000:1003] -= 20.0                      # 3 s dropout inside an "off"
    segs = states.state_segments(lvl, min_dur_s=30)
    assert len(segs) == 10                      # glitch merged away


# --------------------------------------------------------------- fingerprint

def test_source_fingerprint_finds_hump_and_comb():
    freqs = np.linspace(0, 8000, 4097)
    rng = np.random.default_rng(7)
    base = 1e-9 * (1 + 0.02 * rng.standard_normal((40, len(freqs)))) ** 2
    minspec = base.copy()
    active = np.zeros(40, bool)
    active[:20] = True
    hump = (freqs >= 250) & (freqs <= 1000)
    minspec[np.ix_(active, hump)] *= 10 ** 0.6          # +6 dB turbulence
    for k in range(2, 7):                                # 260..780 Hz comb
        line = np.abs(freqs - 130.0 * k) < 4
        minspec[np.ix_(active, line)] *= 10 ** 1.6       # +16 dB partials
    F = {"minspec": minspec, "freqs": freqs}
    fp = background.source_fingerprint(F, active, ~active)
    assert fp["rise_max_db"] == pytest.approx(6.0, abs=2.0)  # the hump
    assert 250 <= fp["rise_max_hz"] <= 1000
    pk_f = [p["f_hz"] for p in fp["peaks"]]
    assert any(abs(f - 260) < 10 for f in pk_f)
    assert fp["comb"]["f0_hz"] == pytest.approx(130.0, rel=0.05)
    assert fp["comb"]["harmonicity"] > 0.8
    # rise curve is ~0 outside the source's band
    m_out = (fp["freqs"] > 2000) & (fp["freqs"] < 8000)
    assert np.median(fp["rise_db"][m_out]) == pytest.approx(0.0, abs=1.0)


# ----------------------------------------------------------------- grid scan

def test_grid_scan_detects_only_ticks_with_strikes():
    nsec, t0 = 5400, 54000.0
    logf = np.geomspace(25, 20000, 97)
    fc = np.sqrt(logf[:-1] * logf[1:])
    rng = np.random.default_rng(9)
    ls = 1e-8 * (1 + 0.05 * rng.standard_normal((nsec, 96))) ** 2
    band = (fc >= 300) & (fc <= 1500)
    hit_ticks = (1, 3, 5)
    for k in hit_ticks:
        i = 900 * k + 120
        ls[np.ix_(range(i, i + 8), band)] *= 10 ** 1.2   # 8 s, +12 dB
    F = {"logspec": ls, "logf": logf.astype(np.float32),
         "t": t0 + np.arange(nsec, dtype=float)}
    scans = schedule.grid_scan(F, 900.0, band=(300, 1500), win_s=300,
                               min_rise_db=6.0)
    det = {int(round((s["t_tick"] - t0) / 900)): s for s in scans
           if s["detected"]}
    assert set(det) == set(hit_ticks)
    for k in hit_ticks:
        assert det[k]["offset_s"] == pytest.approx(120, abs=10)
        assert det[k]["rise_db"] > 8


# ------------------------------------------------------------------- export

def test_export_segment_bit_exact(tmp_path):
    t = np.arange(10 * FS) / FS
    data = plane_wave(0.4 * np.sin(2 * np.pi * 440 * t), az_deg=90.0)
    write_bwf(tmp_path / "a.wav", data, time="12:00:00")
    sess = open_session(tmp_path)
    out = export_segment(sess, sess.takes[0].start + 2.0, 3.0,
                         tmp_path / "seg.wav")
    orig, _fs = sf.read(tmp_path / "a.wav", dtype="int16", always_2d=True)
    seg, fs = sf.read(out, dtype="int16", always_2d=True)
    assert fs == FS and seg.shape == (3 * FS, 4)
    assert np.array_equal(seg, orig[2 * FS:5 * FS])
    assert sf.info(str(out)).subtype == sf.info(str(tmp_path / "a.wav")).subtype


def test_stereo_preview_side_cardioids():
    t = np.arange(FS) / FS
    x = plane_wave(0.3 * np.sin(2 * np.pi * 300 * t), az_deg=90.0)  # hard left
    st = stereo_preview(x)
    assert st.shape == (FS, 2)
    rms = np.sqrt((st ** 2).mean(0))
    assert rms[0] > 20 * rms[1]                  # left cardioid gets it all
