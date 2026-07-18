"""Vision feature tests on synthetic frames (numpy-only, no camera)."""
import numpy as np
import pytest

from ambiscape import vision


def _frame(h=60, w=90, color=(0.0, 0.0, 0.0)):
    f = np.zeros((h, w, 3), float)
    f[:] = color
    return f


def test_brightness_extremes():
    assert vision.frame_features(np.zeros((60, 90, 3)))["brightness"] == \
        pytest.approx(0.0, abs=1e-6)
    assert vision.frame_features(np.ones((60, 90, 3)))["brightness"] == \
        pytest.approx(1.0, abs=1e-6)


def test_uint8_input_is_scaled():
    f = (np.ones((60, 90, 3)) * 255).astype(np.uint8)
    assert vision.frame_features(f)["brightness"] == pytest.approx(1.0, abs=0.01)


def test_warm_cool_and_balance():
    red = vision.frame_features(_frame(color=(1, 0, 0)))
    blue = vision.frame_features(_frame(color=(0, 0, 1)))
    assert red["warm_cool_ratio"] > blue["warm_cool_ratio"]
    assert red["r_frac"] > 0.9 and blue["b_frac"] > 0.9


def test_colourfulness_grey_low_saturated_high():
    grey = vision.frame_features(_frame(color=(0.5, 0.5, 0.5)))
    f = _frame()
    f[:, :45] = (1, 0, 0)
    f[:, 45:] = (0, 0, 1)
    sat = vision.frame_features(f)
    assert grey["saturation"] < 0.05
    assert grey["colourfulness"] < 0.05
    assert sat["colourfulness"] > grey["colourfulness"] + 0.3


def test_spatial_entropy_uniform_vs_spot():
    uniform = vision.frame_features(_frame(color=(0.5, 0.5, 0.5)))
    spot = np.zeros((60, 90, 3))
    spot[:20, :30] = 1.0                      # bright top-left grid cell
    s = vision.frame_features(spot)
    assert uniform["spatial_entropy"] > 0.95
    assert s["spatial_entropy"] < 0.6
    assert s["bright_centroid_x"] < 0.4 and s["bright_centroid_y"] < 0.4


def test_frame_delta_motion():
    a = np.zeros((60, 90, 3))
    b = np.ones((60, 90, 3))
    assert vision.frame_delta(a, a) == pytest.approx(0.0, abs=1e-6)
    assert vision.frame_delta(a, b) == pytest.approx(1.0, abs=0.01)


def test_summarize_vision_keys():
    rows = [vision.frame_features(_frame(color=(c, c, c)))
            for c in (0.2, 0.5, 0.8)]
    s = vision.summarize_vision(rows, motion=[0.0, 0.1, 0.05])
    for k in ("vis_brightness_median", "vis_brightness_range",
              "vis_colourfulness_median", "vis_spatial_entropy_median",
              "vis_motion_mean", "n_frames"):
        assert k in s
    assert s["n_frames"] == 3
    assert s["vis_brightness_range"] == pytest.approx(0.48, abs=0.03)  # p90-p10


def test_summarize_vision_empty():
    assert vision.summarize_vision([])["n_frames"] == 0
