"""Streaming per-second feature extraction from soundscape recordings.

Designed for arbitrarily long recordings: files are read in 60-s blocks and
never held in memory. Per second: broadband and A-weighted fast levels
(125 ms), octave-band powers, spectral centroid/flatness, a 96-band
log-frequency spectrogram row, per-octave pseudo-intensity vectors, broadband
DOA (azimuth, elevation) and diffuseness. Per minute: full-resolution mean PSD
(for narrowband hum tracking and fingerprinting).

The level and spectral features run on a single *mono reference*: the W
channel for AmbiX (ACN W,Y,Z,X, as written by the Zoom H3-VR), the L/R mean
for stereo, or the lone channel for mono. Direction depends on the mode:

- **ambix** (>= 4 ch): full 3-D pseudo-intensity — azimuth, elevation,
  diffuseness, and a per-octave intensity vector.
- **stereo** (2 ch): a *lateral* left/right cue only. Azimuth is the
  energy balance mapped to +-90 deg (+ = left, 0 = centre; no front/back or
  elevation), and "diffuseness" is one minus the inter-channel coherence (a
  point source at the centre reads coherent/near-zero, a decorrelated
  ambient field reads diffuse/near-one). Elevation is undefined (NaN).
- **mono** (1 ch): no direction at all — azimuth, elevation, diffuseness
  and the intensity vector are NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .io import Session, Take

NFFT = 8192
HOP = 4800  # 0.1 s at 48 kHz
OCT_CENTERS = (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)
DOA_BAND = (80.0, 3000.0)
LOGF_RANGE = (25.0, 20000.0)
N_LOGBANDS = 96
FAST = 0.125  # fast level window (s)
HI_ENV = 0.02  # high-rate broadband envelope frame (s), for micro-rhythm


def a_weighting_sos(fs: int):
    """IEC 61672 A-weighting as SOS (bilinear transform of the analog filter)."""
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    a1000 = 1.9997
    nums = [(2 * np.pi * f4) ** 2 * 10 ** (a1000 / 20), 0, 0, 0, 0]
    dens = np.polymul(
        np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                   [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2]),
        np.polymul([1, 2 * np.pi * f3], [1, 2 * np.pi * f2]),
    )
    b, a = signal.bilinear(nums, dens, fs)
    return signal.tf2sos(b, a)


def extract_take(take: Take, verbose: bool = False) -> dict:
    """Run the streaming extractor over one file; returns feature arrays."""
    fs = take.samplerate
    hop = int(HOP * fs / 48000)
    nfft = NFFT
    win = np.hanning(nfft).astype(np.float32)
    wsum2 = float((win ** 2).sum())
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    oct_idx = [np.where((freqs >= c / np.sqrt(2)) & (freqs < c * np.sqrt(2)))[0]
               for c in OCT_CENTERS]
    logf = np.geomspace(*LOGF_RANGE, N_LOGBANDS + 1)
    log_idx = np.clip(np.searchsorted(logf, freqs) - 1, -1, N_LOGBANDS - 1)
    doa_mask = (freqs >= DOA_BAND[0]) & (freqs <= DOA_BAND[1])
    spec_mask = (freqs >= 50) & (freqs <= 16000)
    a_sos = a_weighting_sos(fs)
    a_state = np.zeros((a_sos.shape[0], 2), dtype=np.float64)

    nsec = int(take.frames // fs)
    nfast = int(take.frames // int(FAST * fs))
    ffs = int(FAST * fs)
    hfs = int(HI_ENV * fs)
    nhi = int(take.frames // hfs)
    F = {
        "fast_db": np.zeros(nfast, np.float32),
        "fast_dba": np.zeros(nfast, np.float32),
        "env_hi": np.zeros(nhi, np.float32),
        "rms_w": np.zeros(nsec, np.float32),
        "peak": np.zeros(nsec, np.float32),
        "oct_pow": np.zeros((nsec, len(OCT_CENTERS)), np.float32),
        "centroid": np.zeros(nsec, np.float32),
        "flatness": np.zeros(nsec, np.float32),
        "logspec": np.zeros((nsec, N_LOGBANDS), np.float32),
        "I_band": np.zeros((nsec, len(OCT_CENTERS), 3), np.float32),
        "az": np.zeros(nsec, np.float32),
        "el": np.zeros(nsec, np.float32),
        "diffuse": np.zeros(nsec, np.float32),
    }
    mode = getattr(take, "mode", "ambix")
    if mode != "ambix":                 # direction is partial (stereo) or absent
        F["el"][:] = np.nan
        if mode in ("mono", "binaural"):  # no valid DOA (binaural L/R = HRTF)
            F["az"][:] = np.nan
            F["diffuse"][:] = np.nan
            F["I_band"][:] = np.nan
        else:                            # stereo: lateral az + coherence only
            F["I_band"][:] = np.nan
    nmin = -(-nsec // 60) if nsec else 0
    minspec = np.zeros((nmin, len(freqs)), np.float64)
    mincnt = np.zeros(nmin, np.int64)
    eps = 1e-20

    nch = take.channels
    carry = np.zeros((0, nch), np.float32)
    sec_base = 0
    fast_base = 0
    hi_base = 0
    with sf.SoundFile(str(take.audio_path)) as f:
        while True:
            block = f.read(60 * fs, dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                break
            data = np.concatenate([carry, block]) if carry.shape[0] else block
            navail = data.shape[0]
            # mono reference: W (ambix), L/R mean (stereo), the channel (mono)
            if mode == "ambix":
                ref = data[:, take.wyzx[0]]
            elif mode in ("stereo", "binaural"):
                ref = 0.5 * (data[:, 0] + data[:, 1])
            else:
                ref = data[:, 0]
            nsec_blk = min(navail // fs, nsec - sec_base)
            nwin = (navail - nfft) // hop + 1 if navail >= nfft else 0
            if nsec_blk <= 0:
                break

            # fast levels on the mono reference (contiguous 125 ms frames)
            nfast_blk = min((nsec_blk * fs) // ffs, nfast - fast_base)
            wseg = ref[: nfast_blk * ffs].reshape(nfast_blk, ffs)
            F["fast_db"][fast_base:fast_base + nfast_blk] = 10 * np.log10(
                (wseg.astype(np.float64) ** 2).mean(1) + eps)
            wa, a_state = signal.sosfilt(a_sos, ref[: nfast_blk * ffs]
                                         .astype(np.float64), zi=a_state)
            F["fast_dba"][fast_base:fast_base + nfast_blk] = 10 * np.log10(
                (wa.reshape(nfast_blk, ffs) ** 2).mean(1) + eps)
            fast_base += nfast_blk

            # 20 ms broadband envelope (linear power, for modulation)
            nhi_blk = min((nsec_blk * fs) // hfs, nhi - hi_base)
            hseg = ref[: nhi_blk * hfs].reshape(nhi_blk, hfs)
            F["env_hi"][hi_base:hi_base + nhi_blk] = \
                (hseg.astype(np.float64) ** 2).mean(1)
            hi_base += nhi_blk

            if nwin > 0:
                idx = np.arange(nfft)[None, :] + hop * np.arange(nwin)[:, None]
                Wf = np.fft.rfft(ref[idx] * win)
                Pw = (Wf.real ** 2 + Wf.imag ** 2) / wsum2
                centers = (idx[:, 0] + nfft // 2) / fs
                if mode == "ambix":
                    iW, iY, iZ, iX = take.wyzx
                    Yf = np.fft.rfft(data[:, iY][idx] * win)
                    Zf = np.fft.rfft(data[:, iZ][idx] * win)
                    Xf = np.fft.rfft(data[:, iX][idx] * win)
                    IX = (Wf.conj() * Xf).real / wsum2
                    IY = (Wf.conj() * Yf).real / wsum2
                    IZ = (Wf.conj() * Zf).real / wsum2
                    Ev = (Xf.real ** 2 + Xf.imag ** 2 + Yf.real ** 2
                          + Yf.imag ** 2 + Zf.real ** 2 + Zf.imag ** 2) / wsum2
                elif mode == "stereo":
                    Lf = np.fft.rfft(data[:, 0][idx] * win)
                    Rf = np.fft.rfft(data[:, 1][idx] * win)
                    PL = (Lf.real ** 2 + Lf.imag ** 2) / wsum2
                    PR = (Rf.real ** 2 + Rf.imag ** 2) / wsum2
                    CLR = (Lf.conj() * Rf) / wsum2      # complex cross-spectrum

            for s in range(nsec_blk):
                g = sec_base + s
                seg = data[s * fs:(s + 1) * fs]
                F["rms_w"][g] = np.sqrt((ref[s * fs:(s + 1) * fs]
                                         .astype(np.float64) ** 2).mean())
                F["peak"][g] = float(np.abs(seg).max())
                if nwin == 0:
                    continue
                sel = np.where((centers >= s) & (centers < s + 1))[0]
                if len(sel) == 0:
                    continue
                pw = Pw[sel].mean(0)
                for b, bi in enumerate(oct_idx):
                    F["oct_pow"][g, b] = pw[bi].sum()
                p = pw[spec_mask]
                F["centroid"][g] = float((freqs[spec_mask] * p).sum() / (p.sum() + eps))
                F["flatness"][g] = float(np.exp(np.log(p + eps).mean()) / (p.mean() + eps))
                np.add.at(F["logspec"][g], log_idx[log_idx >= 0], pw[log_idx >= 0])
                if mode == "ambix":
                    ix, iy, iz = (IX[sel].mean(0), IY[sel].mean(0),
                                  IZ[sel].mean(0))
                    ev = Ev[sel].mean(0)
                    for b, bi in enumerate(oct_idx):
                        F["I_band"][g, b] = (ix[bi].sum(), iy[bi].sum(),
                                             iz[bi].sum())
                    Ix, Iy, Iz = (ix[doa_mask].sum(), iy[doa_mask].sum(),
                                  iz[doa_mask].sum())
                    F["az"][g] = np.degrees(np.arctan2(Iy, Ix))
                    F["el"][g] = np.degrees(np.arctan2(Iz, np.hypot(Ix, Iy)))
                    etot = (pw[doa_mask].sum() + ev[doa_mask].sum()) / 2
                    inorm = float(np.sqrt(Ix ** 2 + Iy ** 2 + Iz ** 2))
                    F["diffuse"][g] = 1.0 - min(1.0, inorm / (etot + eps))
                elif mode == "stereo":
                    pl, pr = PL[sel].mean(0), PR[sel].mean(0)
                    clr = CLR[sel].mean(0)
                    sL, sR = float(pl[doa_mask].sum()), float(pr[doa_mask].sum())
                    # lateral balance -> +-90 deg (+ = left), coherence -> width
                    F["az"][g] = 90.0 * (sL - sR) / (sL + sR + eps)
                    coh = abs(clr[doa_mask].sum()) / (np.sqrt(sL * sR) + eps)
                    F["diffuse"][g] = 1.0 - min(1.0, float(coh))
                    for b, bi in enumerate(oct_idx):
                        F["I_band"][g, b] = (0.0, float(pl[bi].sum()
                                                        - pr[bi].sum()), 0.0)
                minspec[g // 60] += pw
                mincnt[g // 60] += 1

            carry = data[nsec_blk * fs:].copy()
            sec_base += nsec_blk
            if sec_base >= nsec:
                break

    minspec[mincnt > 0] /= mincnt[mincnt > 0, None]
    F["minspec"] = minspec.astype(np.float32)
    F["freqs"] = freqs.astype(np.float32)
    F["logf"] = logf.astype(np.float32)
    F["start"] = np.float64(take.start)
    F["fs"] = np.int64(fs)
    F["fast_dt"] = np.float64(FAST)
    F["hi_dt"] = np.float64(HI_ENV)
    F["mode"] = np.str_(getattr(take, "mode", "ambix"))
    F["channels"] = np.int64(take.channels)
    return F


def extract_session(sess: Session, out_dir: str | Path, verbose=True) -> list[Path]:
    """Extract features for every take; save one npz per take. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for tk in sess.takes:
        out = out_dir / (tk.path.stem + ".npz")
        if not out.exists():
            F = extract_take(tk)
            np.savez_compressed(out, **F)
            if verbose:
                print(f"  extracted {tk.path.name} ({tk.duration:.0f}s)", flush=True)
        paths.append(out)
    return paths


def load_features(npz_paths: list[str | Path]) -> dict:
    """Concatenate per-take feature files onto one absolute time axis."""
    parts = [np.load(str(p)) for p in npz_paths]
    parts.sort(key=lambda p: float(p["start"]))
    out = {}
    out["t"] = np.concatenate([p["start"] + np.arange(len(p["rms_w"]))
                               for p in parts])
    fd = float(parts[0]["fast_dt"])
    out["t_fast"] = np.concatenate([p["start"] + fd * np.arange(len(p["fast_db"]))
                                    for p in parts])
    for k in ("fast_db", "fast_dba", "rms_w", "peak", "oct_pow", "centroid",
              "flatness", "logspec", "I_band", "az", "el", "diffuse"):
        out[k] = np.concatenate([p[k] for p in parts])
    if all("env_hi" in p for p in parts):    # absent in pre-0.2 caches
        hd = float(parts[0]["hi_dt"])
        out["hi_dt"] = hd
        out["t_hi"] = np.concatenate(
            [p["start"] + hd * np.arange(len(p["env_hi"])) for p in parts])
        out["env_hi"] = np.concatenate([p["env_hi"] for p in parts])
    out["min_t"] = np.concatenate([p["start"] + 60 * np.arange(p["minspec"].shape[0])
                                   for p in parts])
    out["minspec"] = np.concatenate([p["minspec"] for p in parts])
    out["freqs"] = parts[0]["freqs"]
    out["logf"] = parts[0]["logf"]
    if "mode" in parts[0]:                   # absent in pre-0.13 caches
        out["mode"] = str(parts[0]["mode"])
        out["channels"] = int(parts[0]["channels"])
    return out
