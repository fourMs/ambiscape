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
    assert "reflect the recorder, not the room" in txt

    summary["floor_suspect"] = False
    report.write_readme(sess, summary, out)
    assert "self-noise" not in (tmp_path / "README.md").read_text()


# ------------------------------------------------- how much of it is floor

def _F_occupied(nsec=21600, active_frac=0.5, seed=1):
    """A room used for part of the session: a quiet floor with activity
    raising the broadband level well above it for ``active_frac`` of the
    time."""
    rng = np.random.default_rng(seed)
    op = 1e-7 * 10 ** (0.05 * rng.standard_normal((nsec, 10)))   # near floor
    n_active = int(active_frac * nsec)
    if n_active:
        op[:n_active] *= 10 ** (1.5 + 0.3 * rng.standard_normal((n_active, 1)))
    return {"oct_pow": op, "fs": 48000}


def test_floor_occupancy_separates_an_empty_room_from_a_used_one():
    """`floor_suspicion` fires for every node in the SINS corpus, so it
    cannot tell a self-noise band from a room that is empty all week. The
    fraction of a session sitting within a few dB of its own floor can:
    that is a fact about the room, not a fault in the recorder."""
    empty = analysis.floor_occupancy(_F_occupied(active_frac=0.02))
    used = analysis.floor_occupancy(_F_occupied(active_frac=0.6))
    assert empty["at_floor_fraction"] > 0.9
    assert used["at_floor_fraction"] < 0.5
    assert empty["at_floor_fraction"] > used["at_floor_fraction"]


def test_floor_occupancy_is_level_invariant():
    """A quiet room and a loud one are not distinguished by gain: the
    measure is each session against its own floor."""
    F = _F_occupied(active_frac=0.3)
    loud = {"oct_pow": F["oct_pow"] * 100.0, "fs": F["fs"]}
    a = analysis.floor_occupancy(F)["at_floor_fraction"]
    b = analysis.floor_occupancy(loud)["at_floor_fraction"]
    assert a == b


# ------------------------------------------------- picking a capsule to trust

def test_quietest_channel_picks_the_one_with_the_most_room_to_measure_in():
    """A multi-capsule node shares one gain chain, so its channels should
    agree. In the SINS network node 9's four capsules reached the same peak
    within 0.7 dB while channel 0's floor sat 5.5 dB higher — half the usable
    range, on the channel the analysis happened to read."""
    rng = np.random.default_rng(5)
    n = 16000 * 30
    room = 10 ** (-58 / 20) * rng.standard_normal(n)      # shared room sound
    x = np.stack([room + 10 ** (-55 / 20) * rng.standard_normal(n),   # noisy
                  room + 10 ** (-64 / 20) * rng.standard_normal(n),
                  room + 10 ** (-63 / 20) * rng.standard_normal(n),
                  room + 10 ** (-65 / 20) * rng.standard_normal(n)],  # best
                 axis=1)
    ch, floors = analysis.quietest_channel(x, 16000)
    assert ch == 3
    assert len(floors) == 4
    assert floors[3] < floors[0] - 2      # the noisy channel is clearly worse


def test_quietest_channel_on_mono_returns_the_only_channel():
    ch, floors = analysis.quietest_channel(np.zeros(16000), 16000)
    assert ch == 0 and len(floors) == 1
