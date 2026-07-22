"""vision.py wired for multimodal use: per-frame analysis, timeline
alignment, a video/image-folder driver, and the audio-summary merge."""
import json

import numpy as np
import pytest

from ambiscape import vision


def _frames(n=6, h=32, w=48, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 200, (h, w, 3), dtype=np.uint8)
    out = []
    for k in range(n):
        f = np.clip(base.astype(int) + k * 8
                    + rng.integers(0, 20, (h, w, 3)), 0, 255).astype(np.uint8)
        out.append(f)
    return out


def test_analyze_frames_summary_and_motion():
    res = vision.analyze_frames(_frames(6))
    assert res["summary"]["n_frames"] == 6
    assert len(res["frames"]) == 6
    assert "motion" not in res["frames"][0]        # no motion on the first frame
    assert "motion" in res["frames"][1]
    assert "vis_brightness_median" in res["summary"]
    assert "vis_motion_mean" in res["summary"]


def test_analyze_frames_times_align_to_session_clock():
    res = vision.analyze_frames(_frames(4), times=[0.0, 2.0, 4.0, 6.0])
    assert [r["t"] for r in res["frames"]] == [0.0, 2.0, 4.0, 6.0]


def test_run_video_from_image_folder_and_merge(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    for i, f in enumerate(_frames(5)):
        Image.fromarray(f).save(tmp_path / f"frame_{i:03d}.png")
    # a pre-existing audio summary to merge into
    summ = tmp_path / "summary.json"
    summ.write_text(json.dumps({"leq_dbfs": -30.0}))

    out = tmp_path / "analysis"
    s = vision.run_video(str(tmp_path), out, fps=2.0, merge=str(summ))
    assert (out / "vision.json").exists()
    assert (out / "vision.png").exists()
    assert s["n_frames"] == 5

    merged = json.loads(summ.read_text())
    assert merged["leq_dbfs"] == -30.0             # audio kept
    assert "vis_brightness_median" in merged       # visual folded in
