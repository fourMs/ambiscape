"""Anthropophony measures: human speech and activity by acoustic structure.

Beyond the privacy VAD gate (:func:`ambiscape.ml.speech_gate`), a soundscape
wants a *descriptor* of human presence. Speech and conversation carry two
signatures the cached features hold: energy in the voice band (~250-2000 Hz)
and, above all, amplitude modulation at the 3-8 Hz syllabic rate -- the
strongest model-free speech cue. This module reads the features (no audio
pass, no ML) and returns:

- **voice-band fraction** from the octave powers;
- **syllabic modulation** -- the share of 50 Hz-envelope modulation energy in
  3-8 Hz (conversation/announcement cadence);
- **activity fraction** -- seconds where the voice band rises above its own
  running background.

Caveats: proxies, not detection. Music, radio and TV also fill the voice band
and modulate syllabically; confirm actual speech with
:func:`ambiscape.ml.speech_fraction` (silero-VAD, ``[ml]`` extra). The privacy
stance is the deposit module's: publish features, not audio.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import percentile_filter

from .features import OCT_CENTERS

EPS = 1e-12
VOICE_BAND = (250.0, 2000.0)
SYLLABIC = (3.0, 8.0)


def _voice_band(F):
    c = np.asarray(OCT_CENTERS, float)
    m = (c >= VOICE_BAND[0]) & (c <= VOICE_BAND[1])
    op = np.asarray(F["oct_pow"], float)
    return op[:, m].sum(1), op.sum(1)


def voice_band_fraction(F: dict) -> float:
    band, tot = _voice_band(F)
    return float(band.sum() / (tot.sum() + EPS))


def syllabic_modulation(F: dict, band=SYLLABIC) -> float:
    """Fraction of broadband-envelope modulation energy at the syllabic rate."""
    env = np.asarray(F.get("env_hi"), float)
    dt = float(F.get("hi_dt", 0.02))
    if env.size < 16:
        return 0.0
    x = env - env.mean()
    fr = np.fft.rfftfreq(len(x), dt)
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    ac = P[1:].sum() + EPS
    sel = (fr >= band[0]) & (fr <= band[1])
    return float(P[sel].sum() / ac)


def activity_fraction(F: dict, k_db: float = 3.0) -> float:
    """Seconds where the voice band rises above its running 10th-pct floor."""
    band, _ = _voice_band(F)
    env_db = 10 * np.log10(band + EPS)
    n = max(3, min(len(env_db), 121)) | 1
    bg = percentile_filter(env_db, 10, size=n, mode="nearest")
    return float((env_db > bg + k_db).mean())


def summarize_anthropophony(F: dict) -> dict:
    """Anthropophony descriptors for the analyze summary."""
    vb = voice_band_fraction(F)
    syl = syllabic_modulation(F)
    act = activity_fraction(F)
    index = float(np.clip(vb * (syl / 0.3) * (0.5 + act), 0.0, 1.0))
    return {
        "anthro_voiceband_fraction": round(vb, 3),
        "anthro_syllabic_mod": round(syl, 3),
        "anthro_activity_fraction": round(act, 3),
        "anthropophony_index": round(index, 3),
    }
