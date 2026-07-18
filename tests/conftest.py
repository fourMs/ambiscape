"""Synthetic AmbiX fixtures: ground truth without real recordings.

``write_bwf`` writes a 4-channel PCM WAV with the BWF ``bext`` chunk the
session scanner parses (date, time, zTRK channel tags). Signal generators
produce plane waves (known azimuth/elevation), diffuse noise (uncorrelated
channels), and a two-bell "carillon" with known cycle period, strike
phases, partial ratios, and swing FM — the ground truth the analysis
modules are tested against.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

FS = 48000


def write_bwf(path, data: np.ndarray, fs=FS, date="2026-07-17",
              time="20:00:00", order="ambix"):
    """Minimal BWF writer: bext + fmt (PCM16) + data chunks."""
    x = np.clip(data, -1, 1)
    pcm = (x * 32767).astype("<i2").tobytes()
    trk = {"ambix": ("W", "Y", "Z", "X"), "fuma": ("W", "X", "Y", "Z")}[order]
    desc = "".join(f"zTRK{i+1}={c}\r\n" for i, c in enumerate(trk))
    bext = bytearray(602)
    bext[0:256] = desc.encode().ljust(256, b"\0")[:256]
    bext[256:288] = b"ambiscape-tests".ljust(32, b"\0")
    bext[320:330] = date.encode()
    bext[330:338] = time.encode()
    h, m, s = (int(v) for v in time.split(":"))
    struct.pack_into("<Q", bext, 338, (h * 3600 + m * 60 + s) * fs)
    nch = data.shape[1]
    fmt = struct.pack("<HHIIHH", 1, nch, fs, fs * nch * 2, nch * 2, 16)
    chunks = (b"bext" + struct.pack("<I", len(bext)) + bytes(bext)
              + b"fmt " + struct.pack("<I", len(fmt)) + fmt
              + b"data" + struct.pack("<I", len(pcm)) + pcm)
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", 4 + len(chunks))
                           + b"WAVE" + chunks)
    return Path(path)


def plane_wave(sig: np.ndarray, az_deg: float, el_deg=0.0) -> np.ndarray:
    """AmbiX ACN/SN3D encode of a mono signal from one direction."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([sig,
                     sig * np.sin(az) * np.cos(el),
                     sig * np.sin(el),
                     sig * np.cos(az) * np.cos(el)], axis=1)


def diffuse_noise(n: int, level=0.05, seed=0) -> np.ndarray:
    """Uncorrelated noise in all four channels: diffuseness near 1."""
    rng = np.random.default_rng(seed)
    return level * rng.standard_normal((n, 4))


# ---------------------------------------------------------------- bells

BELL_A = dict(f0=480.0, ratios=(2.4, 4.0, 6.0), az=30.0,
              strike_phases=(0.0, 0.35), fm_cents=3.0)
BELL_B = dict(f0=600.0, ratios=(2.4, 4.0, 6.0), az=60.0,
              strike_phases=(0.5,), fm_cents=2.0)
CYCLE = 3.0  # s


def bell_track(dur_s: float, active_s: float, spec: dict, cycle=CYCLE,
               fs=FS, seed=1) -> np.ndarray:
    """Mono synthetic swinging bell: retriggered decaying partials on
    continuous FM oscillators (Doppler at the cycle rate)."""
    t = np.arange(int(dur_s * fs)) / fs
    rng = np.random.default_rng(seed)
    # amplitude envelope: strikes at cycle*k + phase*cycle while active
    env = np.zeros(len(t))
    for ph in spec["strike_phases"]:
        ts = np.arange(ph * cycle, active_s, cycle)
        ts += rng.normal(0, 0.01, len(ts))          # 10 ms jitter
        for s in ts:
            i = int(s * fs)
            if i < len(env):
                seg = np.arange(len(t) - i) / fs
                env[i:] = np.maximum(env[i:], np.exp(-seg / 1.2))
    out = np.zeros(len(t))
    for k, r in enumerate(spec["ratios"]):
        fp = spec["f0"] * r
        cents = spec["fm_cents"] * np.sin(2 * np.pi * t / cycle)
        finst = fp * 2 ** (cents / 1200.0)
        phase = 2 * np.pi * np.cumsum(finst) / fs
        out += (0.5 ** k) * env * np.sin(phase)
    return 0.2 * out


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
