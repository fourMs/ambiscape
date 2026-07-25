"""Escapement tick-regularity analysis on a synthetic clockwork."""
import numpy as np

from ambiscape import escapement
from tests.conftest import FS, write_bwf

BEAT = 0.82           # s between clicks (tick->tock)
JITTER_MS = 3.0       # white timing noise per click
TOCK_DELAY = 0.03     # tock sits 30 ms late in the cycle: beat error 15 ms


def _tick_train(dur_s, fs=FS, seed=7):
    """Alternating tick/tock clicks: 5 ms 3-6 kHz noise bursts.

    Ground truth: beat BEAT s, tick amp 1.0 / tock amp 0.6, tock late by
    TOCK_DELAY, white jitter JITTER_MS per click.
    """
    rng = np.random.default_rng(seed)
    n = int(dur_s * fs)
    out = 0.002 * rng.standard_normal(n)          # noise floor
    click_n = int(0.005 * fs)
    t_env = np.exp(-np.arange(click_n) / (0.0015 * fs))
    k = 0
    truth = []
    while True:
        base = k * BEAT + (TOCK_DELAY if k % 2 else 0.0)
        t = base + rng.normal(0, JITTER_MS / 1000.0)
        i = int(t * fs)
        if i + click_n >= n - fs:
            break
        burst = rng.standard_normal(click_n) * t_env
        b = np.fft.rfft(burst)
        f = np.fft.rfftfreq(click_n, 1 / fs)
        b[(f < 3000) | (f > 6000)] = 0
        burst = np.fft.irfft(b, click_n)
        amp = 1.0 if k % 2 == 0 else 0.6
        out[i:i + click_n] += 0.4 * amp * burst / (np.abs(burst).max() + 1e-9)
        truth.append(t)
        k += 1
    return out, np.array(truth)


def test_detects_ticks_and_regularity(tmp_path):
    x, truth = _tick_train(120.0)
    write_bwf(tmp_path / "ticks.wav", np.stack([x] * 4, axis=1))
    import ambiscape as asc
    sess = asc.open_session(tmp_path)
    out = tmp_path / "analysis"
    out.mkdir()
    doc = escapement.run_session(sess, out, band=(2000.0, 9000.0))
    # beat period recovered within 2 ms (the alternation delay cancels in
    # the mean inter-onset interval)
    assert abs(doc["beat_period_s"] - BEAT) < 0.002
    # nearly every click found
    assert doc["n_ticks"] > 0.9 * len(truth)
    assert doc["coverage"] > 0.9
    # jitter of the right order (few ms; alternation cancelled)
    assert 1.0 < doc["fast_sd_ms"] < 8.0
    # the tick/tock beat error (= TOCK_DELAY) is seen, within 8 ms
    assert abs(doc["beat_error_ms"] - 1000 * TOCK_DELAY) < 8.0
    # alternation amplitude difference is seen (log-compressed envelope,
    # so only a qualitative indicator)
    assert doc["amp_alternation_db"] > 0.5
    assert (out / "escapement.png").exists()
    assert (out / "escapement.json").exists()


def test_matched_times(tmp_path):
    x, truth = _tick_train(60.0, seed=11)
    write_bwf(tmp_path / "ticks.wav", np.stack([x] * 4, axis=1))
    import ambiscape as asc
    sess = asc.open_session(tmp_path)
    out = tmp_path / "analysis"
    out.mkdir()
    doc = escapement.run_session(sess, out, band=(2000.0, 9000.0))
    det = np.array(doc["tick_times_s"])
    # every truth click has a detection within 20 ms (allow a few misses)
    d = np.abs(det[None, :] - truth[:, None]).min(axis=1)
    assert np.median(d) < 0.02
    assert (d < 0.02).mean() > 0.9
