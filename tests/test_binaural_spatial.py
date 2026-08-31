"""Binaural spatial path: ITD-based lateral azimuth + interaural coherence.

HRTF ear signals carry no valid intensity-vector DOA, but they do carry
interaural cues. The binaural mode derives a lateral azimuth from the
interaural time difference (GCC-PHAT over the DOA band, Woodworth-limited)
and a diffuseness from interaural magnitude coherence, which is invariant
to per-channel linear filtering and therefore survives the HRTF.

Level differences must NOT drive the binaural azimuth: an ILD without a
delay is HRTF colouring, not direction. That is the behaviour that
separates this path from the stereo balance cue.
"""
import numpy as np
import soundfile as sf

import ambiscape as asc
from ambiscape import features

FS = 48000


def _extract(tmp_path, left, right):
    data = np.stack([left, right], axis=1).astype(np.float32)
    p = tmp_path / "260721_090000_ears.wav"
    sf.write(p, data, FS, subtype="PCM_24")
    sess = asc.open_recording(p, mode="binaural")
    return features.extract_take(sess.takes[0])


def _noise(seconds, seed):
    rng = np.random.default_rng(seed)
    return 0.1 * rng.standard_normal(int(seconds * FS)).astype(np.float32)


def test_left_leading_source_gives_positive_azimuth(tmp_path):
    # right ear delayed by 14 samples (~292 us) = source on the left;
    # Woodworth-limited arcsin puts that near +26 deg
    s = _noise(4, 0)
    left = s
    right = np.concatenate([np.zeros(14, np.float32), s[:-14]])
    F = _extract(tmp_path, left, right)
    az = F["az"][np.isfinite(F["az"])]
    assert az.size >= 3
    assert 10.0 < float(np.median(az)) < 50.0


def test_right_leading_source_gives_negative_azimuth(tmp_path):
    s = _noise(4, 1)
    left = np.concatenate([np.zeros(14, np.float32), s[:-14]])
    right = s
    F = _extract(tmp_path, left, right)
    az = F["az"][np.isfinite(F["az"])]
    assert az.size >= 3
    assert -50.0 < float(np.median(az)) < -10.0


def test_identical_ears_are_frontal_and_coherent(tmp_path):
    s = _noise(4, 2)
    F = _extract(tmp_path, s, s.copy())
    az = F["az"][np.isfinite(F["az"])]
    dif = F["diffuse"][np.isfinite(F["diffuse"])]
    assert abs(float(np.median(az))) < 5.0
    assert float(np.median(dif)) < 0.1


def test_ild_without_itd_does_not_move_azimuth(tmp_path):
    # a 12 dB level difference with zero delay is HRTF colouring, not
    # direction: the stereo balance cue would call this ~+60 deg left,
    # the binaural ITD cue must stay frontal
    s = _noise(4, 3)
    F = _extract(tmp_path, s, (s * 0.25).copy())
    az = F["az"][np.isfinite(F["az"])]
    assert abs(float(np.median(az))) < 5.0


def test_filtered_ear_stays_coherent(tmp_path):
    # one ear lowpassed (a crude HRTF): magnitude coherence is invariant
    # to linear filtering, so the point source must not look diffuse
    from scipy.signal import butter, lfilter
    s = _noise(4, 4)
    b, a = butter(2, 4000 / (FS / 2))
    F = _extract(tmp_path, s, lfilter(b, a, s).astype(np.float32))
    dif = F["diffuse"][np.isfinite(F["diffuse"])]
    assert float(np.median(dif)) < 0.25


def test_independent_ears_are_diffuse(tmp_path):
    F = _extract(tmp_path, _noise(4, 5), _noise(4, 6))
    dif = F["diffuse"][np.isfinite(F["diffuse"])]
    assert float(np.median(dif)) > 0.6


def test_elevation_and_intensity_bands_stay_nan(tmp_path):
    s = _noise(3, 7)
    F = _extract(tmp_path, s, s.copy())
    assert not np.isfinite(F["el"]).any()
    assert not np.isfinite(F["I_band"]).any()
