"""Background-bed rendering: remove the foreground, keep the keynote.

Schafer's layers as a signal operation: estimate the per-frequency
*background surface* of a recording (the per-block median magnitude
followed by a running minimum across blocks — minimum statistics — so that
even minutes-long events cannot raise their own ceiling) and cap every time-frequency cell at that surface
plus ``margin_db``. Everything that *emerges* — bells, voices, doors — is
pulled down into the bed; everything already at background level passes
untouched, phases included.

The margin chooses the layer boundary: ~3 dB gives the strict continuous
bed (city hum, room tone), ~12 dB keeps quiet iterative keynotes (a clock's
ticking) while still flattening chimes and speech. Note that loud passages
are *capped*, not reconstructed: the render contains their energy pushed
down to background level, not the true background that was masked beneath
them.

Reads the first take's audio directly (like ``carillon``); all channels are
processed with per-channel gains, so stereo and ambiX renders stay
listenable, though exact spatial coherence under heavy capping is not
guaranteed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter

EPS = 1e-12

NPER = 4096
HOP = 2048


def background_bed(x: np.ndarray, fs: int, margin_db: float = 12.0,
                   block_s: float = 10.0, pctl: float = 50.0,
                   minwin_s: float = 120.0):
    """Cap one channel's spectrogram at its background surface + margin.

    Returns (rendered_channel, exceed_before, exceed_after) where the
    exceed values are the fraction of time-frequency cells more than
    ``margin_db + 3`` dB above the background surface (i.e. meaningfully
    above the cap), before and after capping.
    """
    _, _, Z = signal.stft(x.astype(np.float32), fs=fs, nperseg=NPER,
                          noverlap=NPER - HOP, padded=True)
    mag = np.abs(Z).astype(np.float32)
    nfr = mag.shape[1]
    fpb = max(1, int(round(block_s * fs / HOP)))
    nb = max(1, nfr // fpb)
    # per-bin block percentile, then a running minimum across blocks: the
    # minimum-statistics step stops long events from lifting their own bed
    blocks = np.stack([np.percentile(mag[:, i * fpb:(i + 1) * fpb],
                                     pctl, axis=1) for i in range(nb)],
                      axis=1)
    minblocks = max(1, min(int(round(minwin_s / block_s)) | 1, nb))
    blocks = minimum_filter(blocks, size=(1, minblocks))
    idx = np.clip(np.arange(nfr) // fpb, 0, nb - 1)
    B = blocks[:, idx]
    lim = B * 10 ** (margin_db / 20)
    thr = B * 10 ** ((margin_db + 3) / 20)
    exceed = mag > thr
    gain = np.minimum(1.0, lim / (mag + EPS))
    # light time-frequency smoothing keeps musical noise down
    gain = signal.convolve2d(gain, np.ones((3, 3)) / 9, mode="same",
                             boundary="symm").astype(np.float32)
    # smoothing must not lift capped cells back above the surface: re-cap
    gain = np.minimum(gain, lim / (mag + EPS))
    gain = np.minimum(gain, 1.0)
    exceed_after = (mag * gain) > thr
    _, y = signal.istft(Z * gain, fs=fs, nperseg=NPER, noverlap=NPER - HOP)
    return (y[:len(x)].astype(np.float32), float(exceed.mean()),
            float(exceed_after.mean()))


def run_session(sess, out_dir, margin_db: float = 12.0, t0: float = 0.0,
                dur: float | None = None, block_s: float = 10.0,
                pctl: float = 50.0, minwin_s: float = 120.0,
                out_path=None) -> dict:
    """CLI driver: render the first take's background bed to a WAV.

    Writes ``<out_dir>/background_<stem>_<margin>dB.wav`` (24-bit, same
    rate and channel count as the take) plus ``background_render.json``
    with the parameters and suppression metrics.
    """
    import json
    out_dir = Path(out_dir)
    take = sess.takes[0]
    fs = take.samplerate
    with sf.SoundFile(str(take.audio_path)) as f:
        f.seek(int(t0 * fs))
        n = f.frames - int(t0 * fs) if dur is None else int(dur * fs)
        x = f.read(n, dtype="float32", always_2d=True)
    chans, eb, ea = [], [], []
    for c in range(x.shape[1]):
        y, before, after = background_bed(
            x[:, c], fs, margin_db=margin_db, block_s=block_s, pctl=pctl,
            minwin_s=minwin_s)
        chans.append(y)
        eb.append(before)
        ea.append(after)
    y = np.stack(chans, axis=1)
    peak = float(np.abs(y).max()) + EPS
    if peak > 0.99:
        y *= 0.99 / peak
    if out_path is None:
        out_path = out_dir / (
            f"background_{take.path.stem}_{margin_db:g}dB.wav")
    out_path = Path(out_path)
    sf.write(str(out_path), y, fs, subtype="PCM_24")
    doc = {
        "out_path": str(out_path),
        "take": take.path.name,
        "margin_db": margin_db, "block_s": block_s, "pctl": pctl,
        "minwin_s": minwin_s, "t0_s": t0,
        "rendered_s": round(len(y) / fs, 1),
        "exceed_fraction_before": round(float(np.mean(eb)), 4),
        "exceed_fraction_after": round(float(np.mean(ea)), 4),
        "_method_note": (
            "per-bin block percentile background with a running minimum "
            "(minimum statistics), capped at background + margin_db; "
            "capped regions contain foreground energy pushed to background "
            "level, not the masked true background."),
    }
    (out_dir / "background_render.json").write_text(
        json.dumps(doc, indent=2, default=float))
    return doc
