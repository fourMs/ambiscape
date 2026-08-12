"""Bridge to musiscape: music analysis on an ambiscape session.

**The analysis moved to musiscape on 2026-08-12.** It had lived here while
musiscape --- the music toolbox --- imported six of its symbols across three
modules, and no library code in this package ever used it; one CLI subcommand
did. `tempogram`, `chromagram`, `dominant_period`, `pulse_clarity`,
`fifths_center`, `tonal_center_spread` and `tartyp_profile` are now
:mod:`musiscape.music`, and their circular statistics come from
:mod:`micromotion.circular`, which owns them.

What stays here is what could not travel: the two functions that know what an
ambiscape :class:`Session` is. `load_w` pulls a take's mono reference through
the same downmix the rest of this pipeline uses, and `run_session` drives the
figure and summary for a session folder. They are an *adapter*, in the same
sense as `musicalgestures._soundscape` is an adapter the other way --- MGT
owns pixels, ambiscape owns samples, musiscape owns music, and each crossing
is one small module that says so.

musiscape is an **optional** dependency. ambiscape does not require it, and
this module raises a plain instruction if it is missing rather than failing
somewhere deeper. Anyone who only wants the analysis should call musiscape
directly and never come through here.
"""
from __future__ import annotations

import soundfile as sf


def _require_musiscape():
    """The music analysis, or a message saying where it went."""
    try:
        from musiscape import music
    except ImportError as e:                                  # pragma: no cover
        raise ImportError(
            "ambiscape.music is a bridge; the analysis lives in musiscape "
            "since 2026-08-12. Install it with `pip install musiscape`, or "
            "call musiscape.music directly on your own audio."
        ) from e
    return music


def load_w(take, t0=0.0, dur=None, sr=22050):
    """Mono reference of a take (W / L-R mean / channel), resampled to ``sr``.

    Reads the take's decoded audio (so a transcoded ``.m4a`` works too) and
    downmixes per the take's mode --- MIR runs on the same mono reference the
    rest of the pipeline uses. This is the half of the old module that knows
    about takes, which is why it stayed.
    """
    import librosa
    fs = take.samplerate
    with sf.SoundFile(str(take.audio_path)) as f:
        f.seek(int(t0 * fs))
        n = f.frames - int(t0 * fs) if dur is None else int(dur * fs)
        x = f.read(n, dtype="float32", always_2d=True)
    return librosa.resample(take.mono_ref(x), orig_sr=fs, target_sr=sr), sr


def run_session(sess, out_dir, t0=0.0, dur=None) -> dict:
    """Tempogram and chromagram figure plus summary for a session's first take.

    The session handling is this module's; every number in the summary comes
    from :mod:`musiscape.music`.
    """
    import json
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    m = _require_musiscape()
    y, sr = load_w(sess.takes[0], t0, dur)
    tg, bpms = m.tempogram(y, sr)
    chroma = m.chromagram(y, sr)
    summary = {
        "pulse": m.pulse_clarity(y, sr),
        "fifths": m.fifths_center(chroma.mean(axis=1)),
        "tartyp": m.tartyp_profile(y, sr),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6))
    ax[0].imshow(tg, aspect="auto", origin="lower")
    ax[0].set_ylabel("tempo (BPM)")
    ax[1].imshow(chroma, aspect="auto", origin="lower")
    ax[1].set_ylabel("pitch class")
    fig.tight_layout()
    fig.savefig(out / "music.png", dpi=130)
    plt.close(fig)
    (out / "music.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
