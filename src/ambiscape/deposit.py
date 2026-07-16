"""Non-identifying feature export in the StillStanding365 deposit schema.

Writes one TSV per take with the columns used by the StillStanding365 Zenodo
deposit (``audio/{day}.tsv``): per-second ``Time``, ``level_dbfs``,
``centroid_hz``, ``low_frac`` (< 250 Hz), ``high_frac`` (> 2 kHz). A 1 Hz
loudness/spectral envelope is far below speech timescales and carries no
intelligible content, so these files are safe to publish where raw audio is
not.

Method notes vs. the original ``extract_audio.py``: levels here come from the
W (omni) channel at native rate (the original used an ffmpeg 4-channel
downmix at 8 kHz — offsets of a few tenths of a dB are expected), and band
fractions are power fractions from the cached log-spectrogram (the original
used magnitude fractions of an 8 kHz FFT). Trends and dynamics are directly
comparable; absolute fraction values differ slightly by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import db


def export_take_tsv(npz_path: str | Path, out_dir: str | Path) -> Path:
    p = np.load(str(npz_path))
    logf = p["logf"]
    fc = np.sqrt(logf[:-1] * logf[1:])
    S = p["logspec"]
    tot = S.sum(1) + 1e-20
    low = S[:, fc < 250].sum(1) / tot
    high = S[:, fc > 2000].sum(1) / tot
    level = db(p["rms_w"].astype(np.float64) ** 2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (Path(npz_path).stem + ".tsv")
    with open(out, "w") as f:
        f.write("Time\tlevel_dbfs\tcentroid_hz\tlow_frac\thigh_frac\n")
        for i in range(len(level)):
            f.write(f"{i}\t{level[i]:.1f}\t{p['centroid'][i]:.0f}\t"
                    f"{low[i]:.3f}\t{high[i]:.3f}\n")
    return out


def export_session(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    fdir = folder / "analysis" / "features"
    outs = []
    for npz in sorted(fdir.glob("*.npz")):
        outs.append(export_take_tsv(npz, folder / "deposit"))
    return outs
