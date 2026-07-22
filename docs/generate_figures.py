#!/usr/bin/env python3
"""Regenerate the documentation illustrations.

Synthesises one rich first-order AmbiX session (swinging bells at known
azimuths, an elevated source, a broadband lateral source, mains hum, and a
quiet tail), runs the real analysis commands over it, and copies one
representative PNG per figure type into ``docs/img/``. Everything comes from
the same tested code paths the toolbox ships, so the figures never drift from
the behaviour they illustrate.

Usage:  python docs/generate_figures.py
Deps:   ambiscape[music]  (librosa) for the music/carillon figures.
"""
from __future__ import annotations

import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

FS = 48000
DUR = 120.0
ACTIVE = 90.0            # bells stop here; quiet tail gives an on/off state
HERE = Path(__file__).resolve().parent
IMG = HERE / "img"


# --------------------------------------------------------------- synthesis
def write_bwf(path, data, fs=FS, date="2026-07-17", time="09:00:00",
              order="ambix"):
    x = np.clip(data, -1, 1)
    pcm = (x * 32767).astype("<i2").tobytes()
    trk = {"ambix": ("W", "Y", "Z", "X")}[order]
    desc = "".join(f"zTRK{i+1}={c}\r\n" for i, c in enumerate(trk))
    bext = bytearray(602)
    bext[0:256] = desc.encode().ljust(256, b"\0")[:256]
    bext[256:288] = b"ambiscape-docs".ljust(32, b"\0")
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


def plane_wave(sig, az_deg, el_deg=0.0):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([sig, sig * np.sin(az) * np.cos(el), sig * np.sin(el),
                     sig * np.cos(az) * np.cos(el)], axis=1)


def bell(dur_s, active_s, f0, ratios, phases, az, cycle=3.0, fm_cents=3.0,
         seed=1):
    t = np.arange(int(dur_s * FS)) / FS
    rng = np.random.default_rng(seed)
    env = np.zeros(len(t))
    for ph in phases:
        for s in np.arange(ph * cycle, active_s, cycle) + rng.normal(0, 0.01):
            i = int(s * FS)
            if 0 <= i < len(env):
                seg = np.arange(len(t) - i) / FS
                env[i:] = np.maximum(env[i:], np.exp(-seg / 1.2))
    out = np.zeros(len(t))
    for k, r in enumerate(ratios):
        cents = fm_cents * np.sin(2 * np.pi * t / cycle)
        phase = 2 * np.pi * np.cumsum(f0 * r * 2 ** (cents / 1200.0)) / FS
        out += (0.5 ** k) * env * np.sin(phase)
    return plane_wave(0.2 * out, az)


def synth_session(folder, seed=0):
    n = int(DUR * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    data = np.zeros((n, 4))
    data += bell(DUR, ACTIVE, 480.0, (2.4, 4.0, 6.0), (0.0, 0.35), 30.0, seed=1)
    data += bell(DUR, ACTIVE, 600.0, (2.4, 4.0, 6.0), (0.5,), 60.0, seed=2)
    # broadband lateral source (left), amplitude-modulated ~ speech cadence
    br = rng.standard_normal(n) * (0.4 + 0.6 * (np.sin(2 * np.pi * 0.5 * t) > 0))
    data += plane_wave(0.05 * br, az_deg=-40.0)
    # elevated source (building services) for horizon fractions
    data += plane_wave(0.03 * rng.standard_normal(n), az_deg=120.0, el_deg=35.0)
    # mains hum on the omni (50 Hz + a little 150 Hz), for ENF
    hum = 0.02 * np.sin(2 * np.pi * 50 * t) + 0.006 * np.sin(2 * np.pi * 150 * t)
    data += plane_wave(hum, az_deg=0.0)
    data += 0.01 * rng.standard_normal((n, 4))            # diffuse floor
    folder.mkdir(parents=True, exist_ok=True)
    write_bwf(folder / "scene.wav", data)
    return folder


# --------------------------------------------------------------- driving
def run(*args, cwd=None):
    print("  $ ambiscape", *args)
    r = subprocess.run(["ambiscape", *args], cwd=cwd,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("    ! failed:", (r.stderr or r.stdout).strip().splitlines()[-1:])
    return r.returncode == 0


# figure filename in the analysis dir -> doc image name
COLLECT = {
    "overview.png": "overview.png",
    "ltas_percentiles.png": "ltas_percentiles.png",
    "directogram.png": "directogram.png",
    "spatial.png": "spatial.png",
    "modulation_profile.png": "modulation_profile.png",
    "tonality.png": "tonality.png",
    "timbre.png": "timbre.png",
    "rhythm_overview.png": "rhythm_overview.png",
    "music.png": "music.png",
    "carillon.png": "carillon.png",
    "enf.png": "enf.png",
    "schaeffer_map.png": "schaeffer_map.png",
    "schafer_timeline.png": "schafer_timeline.png",
}

ANNOTATIONS = {
    "session": "docs-demo",
    "objects": [
        {"label": "bell A", "facture": "impulse", "mass": "tonic",
         "kind": "soundmark", "source": "anthropophony",
         "t_start": 0, "t_end": 90},
        {"label": "bell B", "facture": "impulse", "mass": "tonic-complex",
         "kind": "signal", "source": "anthropophony",
         "t_start": 0, "t_end": 90},
        {"label": "mains hum", "facture": "sustained", "mass": "tonic",
         "kind": "keynote", "source": "anthropophony", "t_start": 0, "t_end": 120},
        {"label": "voices", "facture": "iteration", "mass": "complex",
         "kind": "figure", "source": "anthropophony", "t_start": 10, "t_end": 80},
        {"label": "diffuse floor", "facture": "unlimited", "mass": "noise",
         "kind": "keynote", "source": "geophony", "t_start": 0, "t_end": 120},
    ],
}


def main():
    IMG.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="ambiscape-figs-"))
    sess = synth_session(tmp / "scene")
    analysis = sess / "analysis"
    print("synthesised", sess)

    run("analyze", str(sess))
    for cmd in ("spatial", "modspec", "tonality", "timbre", "rhythm",
                "music", "carillon", "enf"):
        run(cmd, str(sess))

    import json
    (sess / "annotations.json").write_text(json.dumps(ANNOTATIONS, indent=2))
    run("taxonomy", str(sess))

    # gather every produced PNG (analysis dir + session dir), copy the wanted
    produced = {p.name: p for p in list(analysis.rglob("*.png"))
                + list(sess.rglob("*.png"))}
    got, missing = [], []
    for src_name, dst_name in COLLECT.items():
        if src_name in produced:
            shutil.copy(produced[src_name], IMG / dst_name)
            got.append(dst_name)
        else:
            missing.append(src_name)

    print(f"\ncopied {len(got)} figures to {IMG}:")
    for g in sorted(got):
        print("  ", g)
    if missing:
        print("MISSING (command may have failed or needs extra deps):")
        for m in missing:
            print("  ", m)
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
