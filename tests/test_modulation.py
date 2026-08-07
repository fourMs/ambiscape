"""Modulation profile: synthetic AM recovery per scale, and the shared
normalisation / band-overlap continuity of the three-scale spectrum."""
import numpy as np
import pytest

from ambiscape.modulation import BANDS, EXT, profile

RNG = np.random.default_rng(7)
HI_DT = 0.025      # 40 Hz envelope stream; 5 frames per 125 ms fast frame


def _streams_from_power(p_hi, dur):
    """Derive cache-like streams from one 40 Hz linear-power envelope by
    block averaging, the way the extractor does (fast = 125 ms mean power,
    rms_w = 1 s RMS)."""
    fast_pow = p_hi.reshape(-1, 5).mean(1)                  # 8 Hz
    sec_pow = p_hi.reshape(dur, -1).mean(1)                 # 1 Hz
    return {
        "t": np.arange(dur, dtype=float),
        "t_fast": np.arange(len(fast_pow)) * 0.125,
        "fast_db": 10 * np.log10(fast_pow),
        "rms_w": np.sqrt(sec_pow).astype(np.float32),
        "env_hi": p_hi.astype(np.float32),
        "hi_dt": HI_DT,
        "t_hi": np.arange(len(p_hi)) * HI_DT,
    }


def _am_profile(fm, dur):
    th = np.arange(0, dur, HI_DT)
    p = 1e-4 * (1 + 0.5 * np.sin(2 * np.pi * fm * th)) ** 2
    p *= 1 + 0.02 * RNG.standard_normal(len(th))
    return profile(_streams_from_power(np.abs(p), dur))


@pytest.mark.parametrize("scale,fm,dur", [
    ("micro", 2.0, 1800),       # 0.5 s beat
    ("meso", 0.05, 3600),       # 20 s wave
    ("macro", 0.002, 7200),     # 500 s duty cycle
])
def test_known_am_recovered_per_scale(scale, fm, dur):
    prof = _am_profile(fm, dur)
    st = prof["scales"][scale]
    assert st["peak_freq_hz"] == pytest.approx(fm, rel=0.15)
    assert st["peak_prominence_db"] > 10


def test_meso_am_not_flat():
    """A 0.05 Hz AM must give the meso spectrum real structure (the flat-line
    failure mode of tail-contaminated caches was > 30 dB mean, < 1 dB range)."""
    prof = _am_profile(0.05, 3600)
    P = np.array(prof["spectra"]["meso"]["power"])
    assert 10 * np.log10(P.max() / P.min()) > 20


def test_scales_overlap_and_agree():
    """Shared normalisation: the three spectra are unit-mean power-envelope
    PSDs computed EXT past their nominal edges, so adjacent scales overlap
    in frequency and agree in level where they meet."""
    dur = 7200
    th = np.arange(0, dur, HI_DT)
    # broadband envelope modulation: smoothed positive noise
    n = RNG.standard_normal(len(th))
    k = int(2.0 / HI_DT)
    m = np.convolve(n, np.ones(k) / k, mode="same")
    p = 1e-4 * np.exp(0.8 * m)
    prof = profile(_streams_from_power(p, dur))

    curves = {s: (np.array(sp["freq_hz"]),
                  10 * np.log10(np.array(sp["power"]) + 1e-20))
              for s, sp in prof["spectra"].items()}
    for a, b, edge in [("macro", "meso", BANDS["meso"][0]),
                       ("meso", "micro", BANDS["micro"][0])]:
        fa, da = curves[a]
        fb, db = curves[b]
        lo, hi = edge / EXT, edge * EXT
        mb = (fb >= lo) & (fb <= hi)
        # the overlap exists ...
        assert fa.max() >= hi * 0.99 and fb.min() <= lo * 1.01
        assert mb.sum() >= 4
        # ... and the two estimates of the same PSD agree there
        ia = np.interp(np.log(fb[mb]), np.log(fa), da)
        assert np.median(np.abs(ia - db[mb])) < 3.0


def test_micro_fallback_uses_linear_power():
    """Without env_hi, micro falls back to the fast stream converted to
    linear power — same normalisation as the other scales, flagged in the
    output."""
    dur = 1800
    th = np.arange(0, dur, HI_DT)
    p = 1e-4 * (1 + 0.5 * np.sin(2 * np.pi * 1.0 * th)) ** 2
    F = _streams_from_power(p, dur)
    for key in ("env_hi", "hi_dt", "t_hi"):
        del F[key]
    prof = profile(F)
    assert "micro_limited" in prof
    st = prof["scales"]["micro"]
    assert st["peak_freq_hz"] == pytest.approx(1.0, rel=0.15)
