"""Web Audio resynthesis page: recipe distilled from analysis outputs."""
import json

import numpy as np
import soundfile as sf

from ambiscape import resynth


def _session_with_line(tmp_path, fs=16000, dur=120, f0=500.0):
    """Noise bed + one prominent tonal line, plus minimal module JSONs."""
    import ambiscape as asc
    from ambiscape import features as afeat
    rng = np.random.default_rng(7)
    t = np.arange(dur * fs) / fs
    x = 0.02 * rng.standard_normal(dur * fs) + 0.1 * np.sin(2 * np.pi * f0 * t)
    sf.write(tmp_path / "20260724_120000_take.wav", x.astype(np.float32), fs)
    sess = asc.open_session(tmp_path)
    out = tmp_path / "analysis"
    F = afeat.load_features(afeat.extract_session(sess, out / "features",
                                                  verbose=False))
    (out / "tonality.json").write_text(json.dumps({
        "tracks": [{"f_median_hz": f0, "t0_min": 0, "t1_min": 2,
                    "minutes": 2, "prominence_db": 18.0}]}))
    (out / "summary.json").write_text(json.dumps({
        "duration_min": 2.0, "laeq_dbfs": -30.0, "events_per_min": 1.5,
        "event_median_dur_s": 0.4, "centroid_median_hz": 600.0}))
    return sess, F, out


def test_recipe_reflects_analysis(tmp_path):
    sess, F, out = _session_with_line(tmp_path)
    recipe = resynth.build_recipe(sess, F, out)
    bed = recipe["layers"]["bed"]
    assert len(bed["octave_hz"]) == len(bed["gain_db"]) == 10
    assert all(np.isfinite(g) for g in bed["gain_db"])
    tones = recipe["layers"]["machine"]["tones"]
    assert any(abs(t["freq_hz"] - 500.0) / 500.0 < 0.02 for t in tones)
    ev = recipe["layers"]["events"]
    assert ev["per_min"] >= 0 and ev["dur_s"] > 0


def test_page_is_self_contained(tmp_path):
    sess, F, out = _session_with_line(tmp_path)
    recipe = resynth.build_recipe(sess, F, out)
    page = resynth.write_page(recipe, tmp_path / "resynthesis" / "index.html")
    html = page.read_text()
    i0 = html.index('type="application/json"')
    j0 = html.index(">", i0) + 1
    embedded = json.loads(html[j0:html.index("</script>", j0)])
    assert embedded["layers"]["machine"]["tones"]
    for layer in ("bed", "machine", "events", "space"):
        assert f'data-layer="{layer}"' in html
    assert "http://" not in html and "https://" not in html
