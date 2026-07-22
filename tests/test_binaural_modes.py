"""iso.binaural must handle every input mode and both B-format conventions.

Regression cover for the FuMa channel-order bug: a left-incident source
encoded as FuMa (W, X, Y, Z) must still land in the left ear once ``order``
is honoured, and must NOT keep its left bias when misread as AmbiX.
"""
import numpy as np

from ambiscape import iso

FS = 48000


def _left_source(n=4096):
    """W, Y, Z, X (SN3D) of a plane wave arriving from the left (+Y)."""
    rng = np.random.default_rng(0)
    s = rng.standard_normal(n).astype(np.float32)
    w = (s / np.sqrt(2.0)).astype(np.float32)   # SN3D W scaling
    y = s                                        # +Y = left
    z = np.zeros(n, np.float32)
    x = np.zeros(n, np.float32)
    return w, y, z, x


def _e(a):
    return float(np.mean(np.asarray(a, np.float64) ** 2))


def test_ambix_left_source_louder_in_left_ear():
    w, y, z, x = _left_source()
    ambix = np.stack([w, y, z, x], axis=1)       # W, Y, Z, X
    ears, method = iso.binaural(ambix, FS, order="ambix", mode="ambix")
    assert ears.shape[1] == 2
    assert _e(ears[:, 0]) > _e(ears[:, 1])       # left > right


def test_fuma_left_source_decoded_correctly():
    w, y, z, x = _left_source()
    fuma = np.stack([w, x, y, z], axis=1)        # same source, FuMa W, X, Y, Z
    ears, _ = iso.binaural(fuma, FS, order="fuma", mode="ambix")
    assert _e(ears[:, 0]) > _e(ears[:, 1])       # remap restores left bias


def test_fuma_block_misread_as_ambix_loses_lateralization():
    # The bug the `order` argument fixes: reading a FuMa block as AmbiX puts
    # X (front) where Y (left) is expected, so the left bias disappears.
    w, y, z, x = _left_source()
    fuma = np.stack([w, x, y, z], axis=1)
    ears, _ = iso.binaural(fuma, FS, order="ambix", mode="ambix")
    el, er = _e(ears[:, 0]), _e(ears[:, 1])
    assert abs(el - er) <= 0.05 * max(el, er)    # ~symmetric


def test_mono_duplicated_to_both_ears():
    rng = np.random.default_rng(1)
    m = rng.standard_normal((2048, 1)).astype(np.float32)
    ears, method = iso.binaural(m, FS, mode="mono")
    assert method == "mono-duplicated"
    assert np.allclose(ears[:, 0], ears[:, 1])


def test_stereo_and_binaural_pass_through():
    rng = np.random.default_rng(2)
    st = rng.standard_normal((2048, 2)).astype(np.float32)
    for mode in ("stereo", "binaural"):
        ears, method = iso.binaural(st, FS, mode=mode)
        assert method == "stereo-passthrough"
        assert np.allclose(ears, st[:, :2])


def test_defaults_are_safe_for_short_blocks():
    # Called with defaults (order=ambix, mode=ambix) on a 2-col block, the
    # column-count guard must still route to pass-through, not index col 3.
    rng = np.random.default_rng(3)
    st = rng.standard_normal((1024, 2)).astype(np.float32)
    ears, method = iso.binaural(st, FS)
    assert method == "stereo-passthrough"
    assert ears.shape == (1024, 2)
