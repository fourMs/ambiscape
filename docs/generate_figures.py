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


def synth_network(root, seed=7):
    """Three coupled node sessions (a SINS-style house) for the network figure.

    Envelope ground truth mirrors the network test fixture: the kitchen's
    activity envelope reappears in the hall 0.5 s later at reduced strength,
    and the bedroom carries an unrelated envelope. In the last third of the
    deployment every room falls back to its own independent quiet floor, so
    the density timeline shows the house decoupling when activity stops. The
    node WAVs are written at 16 kHz to keep regeneration cheap; the network
    reads only the 8 Hz level streams, so nothing downstream changes.
    """
    rate, dur, shift, fs = 8.0, 1800.0, 4, 16000   # 0.5 s lag = 4 samples
    m = int(dur * rate) + shift

    def env(sd, smooth=2):
        rng = np.random.default_rng(sd)
        pad = 8 * smooth
        e = np.convolve(rng.standard_normal(m + 2 * pad),
                        np.hanning(2 * smooth + 1), "same")[pad:pad + m]
        return 12 * (e - e.mean()) / e.std()

    e1, e2 = env(seed), env(seed + 1)
    rooms = (("kitchen", e1[shift:m], 30.0),           # leads
             ("hall", 0.8 * e1[:m - shift], 120.0),    # lags kitchen by 0.5 s
             ("bedroom", e2[shift:m], -60.0))          # uncoupled
    quiet = np.arange(m - shift) > 2 * (m - shift) / 3
    for k, (name, e_db, az) in enumerate(rooms):
        e_db = np.where(quiet, 0.2 * env(seed + 20 + k)[:m - shift] - 14.0,
                        e_db)                          # activity stops
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        n = int(dur * fs)
        rng = np.random.default_rng(seed + 10 + k)
        amp = np.interp(np.arange(n) / fs, np.arange(len(e_db)) / rate,
                        10 ** (e_db / 20))
        data = (plane_wave(0.03 * rng.standard_normal(n) * amp, az)
                + 0.001 * rng.standard_normal((n, 4)))
        write_bwf(folder / "take.wav", data, fs=fs, time="07:00:00")
    return root


def synth_array(folder, seed=3, fs=16000, dur=60.0, spacing=0.05, c=343.0):
    """One SINS-style node WAV for the array figures: a band-noise source
    sweeping 20°–160° past a four-mic linear array, silent over 25–35 s so
    only the decorrelated floor remains (the confidence dip and the diffuse
    reading), plus that floor throughout. Returns (wav_path, geometry_path).
    """
    import json as _json
    import soundfile as sf
    folder.mkdir(parents=True, exist_ok=True)
    pos = np.array([[k * spacing, 0.0] for k in range(4)])
    n = int(dur * fs)
    rng = np.random.default_rng(seed)
    src = rng.standard_normal(n)
    S = np.fft.rfft(src)
    fr = np.fft.rfftfreq(n, 1 / fs)
    S[(fr < 200) | (fr > 6000)] = 0
    src = np.fft.irfft(S, n)
    src /= src.std()
    t = np.arange(n) / fs
    src *= np.clip(np.minimum(np.abs(t - 25.0), np.abs(t - 35.0)), 0, 1) \
        ** 2 * ((t < 25.0) | (t > 35.0))          # smooth silent interval
    data = np.zeros((n, 4))
    block = int(0.25 * fs)                        # piecewise-constant angle
    fb = np.fft.rfftfreq(block, 1 / fs)
    for b0 in range(0, n - block + 1, block):
        theta = np.radians(20.0 + 140.0 * b0 / n)
        v = np.array([np.cos(theta), np.sin(theta)])
        Sb = np.fft.rfft(src[b0:b0 + block])
        for m in range(4):
            tau = -float(pos[m] @ v) / c
            data[b0:b0 + block, m] += np.fft.irfft(
                Sb * np.exp(-2j * np.pi * fb * tau), block)
    data += 0.1 * rng.standard_normal((n, 4))     # decorrelated floor
    wav = folder / "node.wav"
    sf.write(str(wav), (0.2 * data).astype(np.float32), fs, subtype="FLOAT")
    geom = folder / "geometry.json"
    geom.write_text(_json.dumps({"mics": pos.tolist(), "c": c}))
    return wav, geom


def array_triangulation(tmp):
    """Three-node triangulation of a source walking across a floor plan
    (the library-call side of the array module). Three nodes, because with
    two the mirror rays of a bearing pair also intersect exactly and many
    fixes are honestly flagged ambiguous; a third ray breaks the tie."""
    from ambiscape import array as arr
    plan = {"nodes": [{"name": "living", "pos": [0.0, 0.0], "axis_deg": 0.0},
                      {"name": "kitchen", "pos": [4.0, 0.0],
                       "axis_deg": 90.0},
                      {"name": "hall", "pos": [2.0, 3.5],
                       "axis_deg": -30.0}]}
    rng = np.random.default_rng(11)
    tt = np.arange(60.0)
    path = np.stack([0.8 + (3.2 - 0.8) * tt / tt[-1],
                     2.8 - (2.8 - 1.2) * tt / tt[-1]], 1)
    streams = []
    for nd in plan["nodes"]:
        v = path - np.asarray(nd["pos"])
        world = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        theta = np.abs((world - nd["axis_deg"] + 180) % 360 - 180)
        streams.append({"t": tt,
                        "bearing_deg": theta + 0.8 * rng.standard_normal(60),
                        "confidence": np.full(60, 0.85),
                        "clipped": np.zeros(60, bool)})
    tri = arr.triangulate(streams, plan)
    out = tmp / "array_triangulate.png"
    arr.triangulate_figure(tri, plan, out, title="three nodes")
    return out


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
    "survey.png": "survey.png",
    "entrain.png": "entrain.png",
    "network.png": "network.png",
    "array_bearing.png": "array_bearing.png",
    "array_coherence.png": "array_coherence.png",
    "array_triangulate.png": "array_triangulate.png",
}

# ISO 12913-2 Method-A responses (5-point) for the survey circumplex: a
# calm-leaning place with one dissenting respondent, so the ellipse is visible.
SURVEY_ROWS = [
    (4, 2, 2, 1, 2, 3, 4, 4), (4, 3, 2, 2, 2, 2, 3, 4),
    (5, 2, 1, 1, 1, 3, 4, 5), (3, 3, 3, 2, 2, 2, 3, 3),
    (4, 2, 2, 1, 2, 4, 4, 4), (2, 2, 4, 4, 4, 2, 2, 2),
    (4, 3, 2, 2, 1, 3, 4, 4), (5, 2, 2, 1, 2, 3, 4, 4),
]


def write_survey_responses(path):
    head = ("respondent,pleasant,vibrant,eventful,chaotic,"
            "annoying,monotonous,uneventful,calm")
    lines = [head] + [f"r{i+1:02d}," + ",".join(str(v) for v in row)
                      for i, row in enumerate(SURVEY_ROWS)]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def write_motion_csv(path, dur=DUR, fs_m=50.0, seed=5):
    """Accelerometer series whose sway tracks the scene's 0.5 Hz
    amplitude-modulated lateral source, so entrainment has something to find."""
    t = np.arange(int(dur * fs_m)) / fs_m
    rng = np.random.default_rng(seed)
    envelope = (np.sin(2 * np.pi * 0.5 * t) > 0).astype(float)
    ax = 0.6 * envelope * np.sin(2 * np.pi * 0.5 * t) + 0.05 * rng.standard_normal(len(t))
    ay = 0.3 * envelope * np.cos(2 * np.pi * 0.5 * t) + 0.05 * rng.standard_normal(len(t))
    az = 9.81 + 0.05 * rng.standard_normal(len(t))
    lines = ["time,acc_x,acc_y,acc_z"]
    lines += [f"{ti:.3f},{x:.4f},{y:.4f},{z:.4f}"
              for ti, x, y, z in zip(t, ax, ay, az)]
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)

ANNOTATIONS = {
    "session": "docs-demo",
    "objects": [
        {"name": "bell A", "facture": "impulse", "mass": "tonic",
         "kind": "soundmark", "source": "anthrophony", "spans": [[0, 90]]},
        {"name": "bell B", "facture": "impulse", "mass": "tonic-complex",
         "kind": "signal", "source": "anthrophony", "spans": [[0, 90]]},
        {"name": "mains hum", "facture": "sustained", "mass": "tonic",
         "kind": "keynote", "source": "anthrophony", "spans": [[0, 120]]},
        {"name": "voices", "facture": "iteration", "mass": "complex",
         "kind": "figure", "source": "anthrophony", "spans": [[10, 80]]},
        {"name": "diffuse floor", "facture": "unlimited", "mass": "noise",
         "kind": "keynote", "source": "geophony", "spans": [[0, 120]]},
    ],
}


def main():
    IMG.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="ambiscape-figs-"))
    sess = synth_session(tmp / "scene")
    analysis = sess / "analysis"
    print("synthesised", sess)

    run("analyze", str(sess))
    for cmd in ("spatial", "modspec", "tonality", "timbre",
                "music", "carillon", "enf"):
        run(cmd, str(sess))

    import json
    (sess / "annotations.json").write_text(json.dumps(ANNOTATIONS, indent=2))
    run("taxonomy", str(sess))

    responses = write_survey_responses(tmp / "responses.csv")
    run("survey", str(sess), "--responses", str(responses))
    motion = write_motion_csv(tmp / "motion.csv")
    run("entrain", str(sess), "--motion", str(motion), "--surrogates", "100")

    # rhythm needs clean sustained partials (like the test fixture), which the
    # busy scene above masks -- so render it from a bells-only sub-session.
    rsess = tmp / "bells"
    rsess.mkdir(parents=True, exist_ok=True)
    rdur, ractive = 300.0, 240.0          # rhythm needs many cycles to lock on
    nb = int(rdur * FS)
    bells = (bell(rdur, ractive, 480.0, (2.4, 4.0, 6.0), (0.0, 0.35), 30.0, seed=1)
             + bell(rdur, ractive, 600.0, (2.4, 4.0, 6.0), (0.5,), 60.0, seed=2)
             + 0.01 * np.random.default_rng(0).standard_normal((nb, 4)))
    write_bwf(rsess / "bells.wav", bells)
    run("analyze", str(rsess))
    run("rhythm", str(rsess))

    # multi-recorder network: three coupled node sessions of one house
    house = synth_network(tmp / "house")
    for room in ("kitchen", "hall", "bedroom"):
        run("analyze", str(house / room), "--no-resolve")
    run("network", str(house), "--win", "30", "--max-lag", "2")

    # gather produced PNGs: everything from the rich scene, rhythm from bells
    produced = {p.name: p for p in sess.rglob("*.png")}
    for p in rsess.rglob("*.png"):
        if p.name == "rhythm_overview.png":
            produced[p.name] = p
    net_png = house / "analysis" / "network.png"
    if net_png.exists():
        produced["network.png"] = net_png
    # spaced-mic array: one node WAV through the CLI, triangulation as the
    # library call it is
    wav, geom = synth_array(tmp / "arraynode")
    run("array", str(wav), "--geometry", str(geom))
    for name in ("array_bearing.png", "array_coherence.png"):
        p = tmp / "arraynode" / "analysis" / name
        if p.exists():
            produced[name] = p
    tri_png = array_triangulation(tmp)
    if tri_png.exists():
        produced["array_triangulate.png"] = tri_png
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
