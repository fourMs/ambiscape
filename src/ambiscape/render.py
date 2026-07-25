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


def _render_array(x, fs, margin_db, block_s, pctl, minwin_s):
    """All channels of one array through background_bed."""
    chans, eb, ea = [], [], []
    for c in range(x.shape[1]):
        y, before, after = background_bed(
            x[:, c], fs, margin_db=margin_db, block_s=block_s, pctl=pctl,
            minwin_s=minwin_s)
        chans.append(y)
        eb.append(before)
        ea.append(after)
    return np.stack(chans, axis=1), float(np.mean(eb)), float(np.mean(ea))


def run_session(sess, out_dir, margin_db: float = 12.0, t0: float = 0.0,
                dur: float | None = None, block_s: float = 10.0,
                pctl: float = 50.0, minwin_s: float = 120.0,
                chunk_s: float | None = 1200.0, out_path=None) -> dict:
    """CLI driver: render the first take's background bed to a WAV.

    Writes ``<out_dir>/background_<stem>_<margin>dB.wav`` (24-bit, same
    rate and channel count as the take) plus ``background_render.json``
    with the parameters and suppression metrics. Takes longer than
    ``chunk_s`` are processed in chunks with ``minwin_s`` of context on
    each side and 50 ms crossfades at the joins, keeping memory bounded
    (a one-shot render of an hour-plus take can need tens of GB);
    ``chunk_s=None`` forces one-shot.
    """
    import json
    out_dir = Path(out_dir)
    take = sess.takes[0]
    fs = take.samplerate
    with sf.SoundFile(str(take.audio_path)) as f:
        f.seek(int(t0 * fs))
        n = f.frames - int(t0 * fs) if dur is None else int(dur * fs)
        total = n
        if chunk_s is None or total <= int(chunk_s * fs):
            x = f.read(total, dtype="float32", always_2d=True)
            y, ebm, eam = _render_array(x, fs, margin_db, block_s, pctl,
                                        minwin_s)
            eb, ea = [ebm], [eam]
        else:
            ctx = int(minwin_s * fs)
            step = int(chunk_s * fs)
            xf = int(0.05 * fs)                      # 50 ms crossfade
            fade_in = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
            pieces, eb, ea = [], [], []
            start = int(t0 * fs)
            pos = 0
            while pos < total:
                a = max(0, pos - ctx)
                b = min(total, pos + step + ctx)
                f.seek(start + a)
                x = f.read(b - a, dtype="float32", always_2d=True)
                yc, ebm, eam = _render_array(x, fs, margin_db, block_s,
                                             pctl, minwin_s)
                lo = pos - a                          # trim left context
                hi = lo + min(step, total - pos)
                keep = yc[max(0, lo - (xf if pieces else 0)):hi]
                if pieces:                            # crossfade the join
                    prev = pieces[-1]
                    prev[-xf:] = (prev[-xf:] * (1 - fade_in)
                                  + keep[:xf] * fade_in)
                    keep = keep[xf:]
                pieces.append(keep)
                eb.append(ebm)
                ea.append(eam)
                pos += step
            y = np.concatenate(pieces, axis=0)
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
        "chunk_s": chunk_s,
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


def characteristic_excerpt(sess, F: dict, out_dir, dur_s: float = 60.0,
                           hop_s: float = 5.0,
                           event_weight_db: float = 20.0,
                           out_path=None) -> dict:
    """Export the excerpt that best characterizes the session's background.

    Scans ``dur_s`` windows on a ``hop_s`` grid and scores each by (a) the
    mean per-band distance between the window's median log-spectrum and the
    session's background spectral profile (10th percentile per band), and
    (b) the fraction of eventful seconds (broadband level more than 6 dB
    over the session median), weighted by ``event_weight_db``. The winner
    is exported **bit-exact from the original take** via
    :func:`ambiscape.io.export_segment` — a soundscape thumbnail with no
    processing artifacts. Needs cached features (a prior analyze run).
    """
    import json
    from .io import export_segment

    out_dir = Path(out_dir)
    logspec = np.asarray(F["logspec"], np.float64)      # nsec x nband
    with np.errstate(divide="ignore"):
        spec_db = 10 * np.log10(logspec + EPS)
    profile = np.percentile(spec_db, 10, axis=0)        # background profile
    level = 10 * np.log10(np.asarray(F["rms_w"], np.float64) ** 2 + EPS)
    eventful = level > np.median(level) + 6.0
    t = np.asarray(F["t"], np.float64)

    n = len(t)
    w = int(round(dur_s))
    if n <= w:
        starts = [0]
    else:
        starts = list(range(0, n - w + 1, max(1, int(round(hop_s)))))
    best = None
    for i0 in starts:
        win = slice(i0, i0 + min(w, n))
        dist = float(np.mean(np.abs(
            np.median(spec_db[win], axis=0) - profile)))
        ev = float(np.mean(eventful[win]))
        score = dist + event_weight_db * ev
        if best is None or score < best[0]:
            best = (score, i0, dist, ev)
    _, i0, dist, ev = best
    t0 = float(t[i0])
    if out_path is None:
        take = sess.takes[0]
        out_path = out_dir / f"excerpt_{take.path.stem}_{int(dur_s)}s.wav"
    out_path = Path(out_path)
    export_segment(sess, t0, float(min(dur_s, t[-1] - t0 + 1.0)),
                   str(out_path))
    doc = {
        "out_path": str(out_path), "t0_s": round(t0, 1),
        "t0_in_take_s": round(t0 - sess.takes[0].start, 1),
        "dur_s": dur_s, "clock": sess.clock(t0),
        "spectral_distance_db": round(dist, 2),
        "eventful_fraction": round(ev, 3),
        "_method_note": (
            "bit-exact excerpt (no processing): the window whose median "
            "spectrum is closest to the session's 10th-percentile "
            "background profile, penalized by eventful seconds."),
    }
    (out_dir / "background_excerpt.json").write_text(
        json.dumps(doc, indent=2, default=float))
    return doc
