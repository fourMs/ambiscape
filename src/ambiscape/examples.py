"""A synthetic session, so the documentation runs with no recordings of your own.

Every example in the guides points at a folder like ``2026-07-15-Haarlem``, which is a
session the reader does not have. This module writes one they do.

    >>> import ambiscape
    >>> root = ambiscape.examples.demo_session("/tmp/demo-session")
    >>> sess = ambiscape.open_session(root)

Then anything in the guides works against ``root``, including the command line::

    ambiscape analyze /tmp/demo-session

A soundscape is a property of a PLACE, and a synthetic session has no place in it. So
this is built to exercise the analysis rather than to resemble anywhere: a diffuse
background that should read as diffuseness near 1, and two swinging bells at KNOWN
azimuths, 30 and 60 degrees, which the spatial analysis should recover and which give the
anglegram something to show. Two takes an hour apart, so the session has a timeline and
the longitudinal views have two points to join.

Use it to learn what each command produces, to check an installation, and to see whether
a change to the spatial code still recovers the two bearings. Do not use it as material
for any claim about a real acoustic environment: there is no room here, no
reverberation, no traffic and no birds, and the ecological and source-domain indices will
return numbers that mean nothing.

The test suite builds its fixtures from these same generators, which is why they are
worth trusting to run and not worth citing.
"""

from __future__ import annotations

import pathlib
import struct

import numpy as np

__all__ = [
    "FS", "write_bwf", "plane_wave", "diffuse_noise", "bell_track",
    "BELL_A", "BELL_B", "CYCLE", "demo_session",
]

FS = 48000


def write_bwf(path, data: np.ndarray, fs=FS, date="2026-07-17",
              time="20:00:00", order="ambix"):
    """Minimal BWF writer: bext + fmt (PCM16) + data chunks.

    The ``bext`` chunk carries the date, the time and the zTRK channel tags that the
    session scanner reads to work out when a take started and which channel is which.
    A plain WAV would be read too, but its start time would come from the file's mtime,
    so a demo written today would claim to have been recorded today.
    """
    x = np.clip(data, -1, 1)
    pcm = (x * 32767).astype("<i2").tobytes()
    trk = {"ambix": ("W", "Y", "Z", "X"), "fuma": ("W", "X", "Y", "Z")}[order]
    desc = "".join(f"zTRK{i+1}={c}\r\n" for i, c in enumerate(trk))
    bext = bytearray(602)
    bext[0:256] = desc.encode().ljust(256, b"\0")[:256]
    bext[256:288] = b"ambiscape-examples".ljust(32, b"\0")
    bext[320:330] = date.encode()
    bext[330:338] = time.encode()
    h, m, s = (int(v) for v in time.split(":"))
    struct.pack_into("<Q", bext, 338, (h * 3600 + m * 60 + s) * fs)
    nch = data.shape[1]
    fmt = struct.pack("<HHIIHH", 1, nch, fs, fs * nch * 2, nch * 2, 16)
    chunks = (b"bext" + struct.pack("<I", len(bext)) + bytes(bext)
              + b"fmt " + struct.pack("<I", len(fmt)) + fmt
              + b"data" + struct.pack("<I", len(pcm)) + pcm)
    pathlib.Path(path).write_bytes(b"RIFF" + struct.pack("<I", 4 + len(chunks))
                                   + b"WAVE" + chunks)
    return pathlib.Path(path)


def plane_wave(sig: np.ndarray, az_deg: float, el_deg=0.0) -> np.ndarray:
    """AmbiX ACN/SN3D encode of a mono signal arriving from one direction."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([sig,
                     sig * np.sin(az) * np.cos(el),
                     sig * np.sin(el),
                     sig * np.cos(az) * np.cos(el)], axis=1)


def diffuse_noise(n: int, level=0.05, seed=0) -> np.ndarray:
    """Uncorrelated noise in all four channels, so diffuseness reads near 1."""
    rng = np.random.default_rng(seed)
    return level * rng.standard_normal((n, 4))


BELL_A = dict(f0=480.0, ratios=(2.4, 4.0, 6.0), az=30.0,
              strike_phases=(0.0, 0.35), fm_cents=3.0)
BELL_B = dict(f0=600.0, ratios=(2.4, 4.0, 6.0), az=60.0,
              strike_phases=(0.5,), fm_cents=2.0)
CYCLE = 3.0  # s


def bell_track(dur_s: float, active_s: float, spec: dict, cycle=CYCLE,
               fs=FS, seed=1) -> np.ndarray:
    """Mono synthetic swinging bell: retriggered decaying partials on continuous FM
    oscillators, so the Doppler of the swing appears at the cycle rate."""
    t = np.arange(int(dur_s * fs)) / fs
    rng = np.random.default_rng(seed)
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


def demo_session(path, dur_s: float = 60.0, active_s: float = 45.0) -> pathlib.Path:
    """Write a two-take synthetic session under ``path`` and return that folder.

    Take 1 is background alone. Take 2, an hour later, adds two bells at 30 and 60
    degrees azimuth over the same background. Four-channel AmbiX at 48 kHz, about 23 MB
    at the default duration; existing files are overwritten, so calling it twice is safe.

    Args:
        path: Folder to write into. Created if it does not exist.
        dur_s (float): Seconds per take.
        active_s (float): Seconds of the second take in which the bells ring, so that
            the take contains a transition rather than one steady state.

    Returns:
        pathlib.Path: The session root, ready for :func:`ambiscape.open_session`.
    """
    root = pathlib.Path(path).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    n = int(dur_s * FS)

    quiet = diffuse_noise(n, seed=0)
    write_bwf(root / "01-background.wav", quiet, date="2026-07-17", time="20:00:00")

    bells = (plane_wave(bell_track(dur_s, active_s, BELL_A, seed=1), BELL_A["az"])
             + plane_wave(bell_track(dur_s, active_s, BELL_B, seed=2), BELL_B["az"]))
    write_bwf(root / "02-bells.wav", bells + diffuse_noise(n, seed=3),
              date="2026-07-17", time="21:00:00")
    return root
