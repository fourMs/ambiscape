"""Sensor-noise-floor guard: a pinned high-band floor is flagged as
recorder self-noise; a floor that breathes is not.

The evidence pattern follows the SINS sensor-network corpus (node 1, one
week of a living room): the 4–8 kHz floor flat to < 1 dB over the whole
span while the bands below 1 kHz vary by several dB, with daytime activity
raising the floor of a minority of chunks without unpinning the quiet-time
minimum.
"""
import numpy as np

from ambiscape import analysis


def _F(nsec=21600, flat_high=False, activity=False, seed=0):
    """Synthetic 1 s octave-band powers (10 bands, 31.5 Hz–16 kHz).

    All bands carry a slow ±5 dB diurnal-style swing plus jitter — a floor
    that breathes. With ``flat_high``, the 4 and 8 kHz bands are replaced
    by a pinned floor (sigma 0.02 dB: flat to well under 1 dB over the
    span, the SINS signature). With ``activity``, a fifth of the session
    raises the high bands by 10 dB in bursts (television, dishes) — the
    low-tail spread statistic must see through it.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(nsec)
    swing = 10 ** (0.5 * np.sin(2 * np.pi * t / 3600))          # +-5 dB
    op = np.empty((nsec, 10))
    for b in range(10):
        jitter = 10 ** (0.1 * rng.standard_normal(nsec))
        op[:, b] = 1e-6 * swing * jitter
    if flat_high:
        for b in (7, 8):                                        # 4, 8 kHz
            op[:, b] = 1e-7 * 10 ** (0.002 * rng.standard_normal(nsec))
    if activity:
        burst = rng.random(nsec // 300) < 0.2                   # 20 % of chunks
        mask = np.repeat(burst, 300)[:nsec]
        op[mask, 6:] *= 10.0                                    # +10 dB
    return {"oct_pow": op, "fs": 48000}


def test_flat_high_floor_flagged():
    r = analysis.floor_suspicion(_F(flat_high=True))
    assert r["floor_suspect"] is True
    # 4 and 8 kHz octaves: band edges 4000/sqrt(2) .. 8000*sqrt(2)
    assert r["floor_suspect_lo_hz"] == 2828
    assert r["floor_suspect_hi_hz"] == 11314
    assert r["floor_spread_db"] < 1.0                           # SINS: < 1 dB


def test_activity_does_not_hide_pinned_floor():
    # SINS pattern: evenings raise the high-band floor, nights re-pin it
    r = analysis.floor_suspicion(_F(flat_high=True, activity=True))
    assert r["floor_suspect"] is True
    assert r["floor_suspect_lo_hz"] == 2828


def test_fluctuating_floor_not_flagged():
    r = analysis.floor_suspicion(_F())
    assert r["floor_suspect"] is False
    assert r["floor_suspect_lo_hz"] is None
    assert r["floor_spread_db"] is None


def test_short_session_never_flagged():
    # under six 300 s chunks there is no evidence either way
    r = analysis.floor_suspicion(_F(nsec=900, flat_high=True))
    assert r["floor_suspect"] is False


def test_band_above_nyquist_not_flagged():
    # 16 kHz material: the 16 kHz octave is empty, the 8 kHz octave still
    # has content below Nyquist and stays eligible
    F = _F(flat_high=True)
    F["fs"] = 16000
    F["oct_pow"][:, 9] = 0.0
    r = analysis.floor_suspicion(F)
    assert r["floor_suspect"] is True
    assert r["floor_suspect_hi_hz"] == 8000                     # clamped at Nyquist


def test_summarize_carries_flag(bell_features):
    _sess, _out, F = bell_features
    s = analysis.summarize(F)
    assert s["floor_suspect"] is False                          # 300 s session


def test_readme_warning_line(tmp_path):
    from types import SimpleNamespace
    from ambiscape import report

    sess = SimpleNamespace(folder=tmp_path, name="floor-test", takes=[])
    out = tmp_path / "analysis"
    out.mkdir()
    summary = {"duration_min": 360.0, "L90": -58.8,
               "floor_suspect": True, "floor_suspect_lo_hz": 2828,
               "floor_suspect_hi_hz": 11314, "floor_spread_db": 0.6}
    report.write_readme(sess, summary, out)
    txt = (tmp_path / "README.md").read_text()
    assert "2.8–11.3 kHz" in txt
    assert "recorder self-noise" in txt
    assert "reflect the instrument, not the room" in txt

    summary["floor_suspect"] = False
    report.write_readme(sess, summary, out)
    assert "self-noise" not in (tmp_path / "README.md").read_text()
