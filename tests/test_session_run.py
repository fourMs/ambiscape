"""The per-session chain: session.json, the generic post-analysis steps, the
report template, and `ambiscape run` end to end on a synthetic session."""
import json
import subprocess
import sys

import numpy as np
import pytest

from ambiscape import analysis, ml, report, resolve, runner
from ambiscape.io import open_session


def test_energy_concentration_finds_the_one_loud_frame():
    rng = np.random.default_rng(0)
    la = -60 + rng.standard_normal(10_000)
    la[123] = 0.0
    F = {"fast_dba": la, "t_fast": np.arange(la.size) * 0.125, "peak": np.array([1.0])}
    e = analysis.energy_concentration(F, handling_s=5.0)
    assert e["frames_half_energy"] == 1
    assert e["laeq_dbfs"] == pytest.approx(e["laeq_trim5_dbfs"] + 20, abs=0.5)
    assert e["laeq_without_ends_dbfs"] == pytest.approx(e["laeq_dbfs"], abs=0.5)  # frame 123 is 15 s in
    assert e["loudest_s"][0] == pytest.approx(123 * 0.125)


def test_handling_states_split_the_ends():
    F = {"t": np.arange(1000.0)}
    st = resolve.handling_states(F, handling_s=60.0)
    assert st["handling"] == [(0.0, 60.0), (940.0, 1000.0)]
    assert st["settled"] == [(60.0, 940.0)]


def test_tag_groups_pool_and_place_on_the_clock(tmp_path, bell_session):
    sess = open_session(bell_session)
    tags = tmp_path / "tags"; tags.mkdir()
    names = np.array(["Speech", "Dog", "Bark", "Silence"])
    t = np.arange(0, 20, 2.0) + 2.0
    P = np.zeros((t.size, 4), np.float16)
    P[3, 1] = 0.9                                   # Dog at window 3
    P[5, 2] = 0.4                                   # Bark at window 5
    np.savez(tags / f"{sess.takes[0].path.stem}.npz", t=t, P=P, names=names)
    doc, arr = ml.tag_groups(sess, tags, threshold=0.3)
    assert doc["counts"]["dog"] == 2 and doc["counts"]["speech"] == 0
    assert arr["t"][0] == pytest.approx(sess.takes[0].start + 2.0)
    assert doc["events"]["dog"]["first"] == sess.clock(sess.takes[0].start + t[3])[-8:]


def test_fill_template(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"x": -1.25, "rows": [{"c": "12:00"}], "t": "| a |"}))
    out = report.fill_template("v {{a.json:x}} f {{a.json:x:+.1f}} c {{a.json:rows.0.c}}\n{{a.json:t}}", tmp_path)
    assert out == "v −1.25 f −1.2 c 12:00\n| a |"
    with pytest.raises(KeyError):
        report.fill_template("{{a.json:missing}}", tmp_path)


def test_init_writes_a_skeleton(tmp_path):
    p = runner.init_config(tmp_path)
    cfg = json.loads(p.read_text())
    for k in ("title", "place", "lat", "lon", "notes", "handling_s", "listen"):
        assert k in cfg
    with pytest.raises(FileExistsError):
        runner.init_config(tmp_path)


def test_python_dash_m_runs():
    r = subprocess.run([sys.executable, "-m", "ambiscape", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "run" in r.stdout


def test_run_post_and_report(tmp_path, bell_session):
    import shutil
    folder = tmp_path / "s"
    shutil.copytree(bell_session, folder, ignore=shutil.ignore_patterns("analysis"))
    runner.init_config(folder)
    cfg = json.loads((folder / "session.json").read_text())
    cfg.update(title="Bells", lat="", listen=[["20:01:00", 2, "bells"]])
    (folder / "session.json").write_text(json.dumps(cfg))
    (folder / "REPORT.template.md").write_text("LAeq {{summary.json:laeq_dbfs}}; settled {{split.json:states.settled.duration_min}}\n{{tables.json:takes}}\n")
    runner.run(folder, stages=["analyze", "post", "report"], ml=False)
    a = folder / "analysis"
    for f in ("summary.json", "energy.json", "split.json", "tables.json"):
        assert (a / f).exists(), f
    assert not (a / "tag_groups.json").exists()     # no tags without ml
    assert len(list((folder / "listen").glob("*bells*.wav"))) == 1
    rep = (folder / "REPORT.md").read_text()
    assert rep.startswith("LAeq −") and "| Take |" in rep
