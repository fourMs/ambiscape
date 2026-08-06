"""Sweep-based impulse-response measurement and auralization, tested
against synthetic rooms: a known exponential decay must come back out of
the sweep -> playback -> deconvolution chain with its T60 intact."""
import json

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import oaconvolve

from ambiscape import impulse

FS = 48000


def _synth_ir(T60=0.6, dur=1.2, fs=FS, seed=0, direct=5.0):
    """White-noise tail with exact exponential decay and a direct spike."""
    t = np.arange(int(dur * fs)) / fs
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(len(t)) * 10 ** (-3 * t / T60)
    ir[:5] = 0.0
    ir[5] = direct
    return ir


@pytest.fixture(scope="module")
def round_trip():
    """Sweep -> simulated room -> deconvolution, shared across tests."""
    sweep, inv, meta = impulse.exp_sweep(duration=4.0, fs=FS)
    ir_true = _synth_ir()
    rec = oaconvolve(sweep, ir_true)
    rec = np.concatenate([np.zeros(FS // 2), rec, np.zeros(FS // 2)])
    rec += 1e-5 * np.random.default_rng(1).standard_normal(len(rec))
    h = impulse.deconvolve(rec, inv)
    ir, direct = impulse.extract_ir(h, FS, pre_ms=5.0)
    return dict(sweep=sweep, inv=inv, meta=meta, ir_true=ir_true,
                ir=ir, direct=direct)


# ------------------------------------------------------------------- sweep

def test_sweep_shape_headroom_fades():
    sweep, inv, meta = impulse.exp_sweep(duration=2.0, fs=FS)
    assert len(sweep) == len(inv) == 2 * FS
    assert np.abs(sweep).max() == pytest.approx(0.5, abs=1e-3)   # -6 dBFS
    assert abs(sweep[0]) < 1e-9 and abs(sweep[-1]) < 1e-3        # fades
    assert meta["f0_hz"] == 40.0 and meta["f1_hz"] == 18000.0


def test_sweep_times_inverse_is_unit_impulse():
    sweep, inv, _ = impulse.exp_sweep(duration=2.0, fs=FS)
    d = oaconvolve(sweep, inv)
    assert int(np.abs(d).argmax()) == len(sweep) - 1
    assert np.abs(d).max() == pytest.approx(1.0, abs=1e-6)
    # nearly all energy sits in the main lobe: an impulse, not a smear
    # (total energy ~1/0.75 = the sweep band's share of the Nyquist range)
    pk = len(sweep) - 1
    assert (d[pk - 50:pk + 50] ** 2).sum() > 0.8 * (d ** 2).sum()


def test_write_sweep_sidecar_regenerates_inverse(tmp_path):
    r = impulse.write_sweep(tmp_path / "sweep.wav", duration=1.0)
    assert r["sweep"].exists() and r["inverse"].exists()
    meta = json.loads(r["params"].read_text())
    inv_wav, fs = sf.read(str(r["inverse"]), dtype="float64")
    inv_regen = impulse.inverse_from_meta(meta)
    assert fs == meta["fs"] == FS
    assert np.allclose(inv_wav, inv_regen.astype(np.float32), atol=1e-7)


# ----------------------------------------------------------- deconvolution

def test_round_trip_recovers_injected_ir(round_trip):
    ir, direct, ir_true = (round_trip["ir"], round_trip["direct"],
                           round_trip["ir_true"])
    n = 40000
    a, b = ir_true[5:5 + n], ir[direct:direct + n, 0]
    corr = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
    off = ir[direct + 1:direct + 1 + n, 0]
    corr_off = (a * off).sum() / np.sqrt((a * a).sum() * (off * off).sum())
    assert corr > 0.8            # sweep band is 40-18k of a 24k Nyquist
    assert corr_off < 0.5        # and it is sample-aligned, not smeared


def test_round_trip_recovers_t60(round_trip):
    m = impulse.ir_metrics(round_trip["ir"], FS)
    for band in ("500", "1000", "2000"):
        assert m[band]["T60"] == pytest.approx(0.6, rel=0.15)
        assert m[band]["T30"] == pytest.approx(0.6, rel=0.15)


# ---------------------------------------------------------------- metrics

def test_ir_metrics_on_trimmed_exponential():
    T60 = 0.6
    ir = _synth_ir(T60=T60, direct=1.0)          # peak at t~0: trimmed IR
    m = impulse.ir_metrics(ir, FS)
    band = m["500"]
    assert band["T60"] == pytest.approx(T60, rel=0.25)
    assert band["EDT"] == pytest.approx(T60, rel=0.35)
    c50_true = 10 * np.log10(np.exp(13.8 * 0.05 / T60) - 1)
    assert band["C50"] == pytest.approx(c50_true, abs=2.0)
    assert band["C80"] > band["C50"]
    assert 0.4 < band["D50"] < 0.9
    assert band["T30"] == pytest.approx(T60, rel=0.25)
    assert band["T20"] == pytest.approx(T60, rel=0.3)


def test_sti_extremes_and_ordering():
    click = np.zeros(2000)
    click[100] = 1.0
    dry = impulse.sti(click, FS)
    short = impulse.sti(_synth_ir(T60=0.4), FS)
    long_ = impulse.sti(_synth_ir(T60=2.5, dur=3.0), FS)
    assert dry["sti"] > 0.98                     # anechoic: perfect MTF
    assert long_["sti"] < short["sti"] < dry["sti"]
    assert long_["sti"] < 0.55                   # 2.5 s hall smears speech
    assert set(dry["mti"]) == {str(c) for c in impulse.OCTAVE_CENTERS}


def test_iacc_early():
    mono = _synth_ir()
    same = np.stack([mono, mono], axis=1)
    assert impulse.iacc_early(same, FS) == pytest.approx(1.0, abs=1e-6)
    other = _synth_ir(seed=9)
    both = np.stack([mono, other], axis=1)
    both[5] = 1.0                                # shared direct spike only
    assert impulse.iacc_early(both, FS) < 0.5
    assert impulse.iacc_early(mono, FS) is None  # mono: no ears
    assert impulse.iacc_early(np.zeros((100, 4)), FS) is None


# ------------------------------------------------------------ auralization

def test_partitioned_convolve_matches_reference():
    rng = np.random.default_rng(2)
    x, h = rng.standard_normal(20000), rng.standard_normal(30000)
    y = impulse.partitioned_convolve(x, h, block=4096)
    ref = oaconvolve(x, h)
    assert y.shape == ref.shape
    assert np.abs(y - ref).max() < 1e-9 * np.abs(ref).max() + 1e-12
    h_short = rng.standard_normal(100)           # shorter than one block
    assert np.allclose(impulse.partitioned_convolve(x, h_short),
                       oaconvolve(x, h_short))


def test_auralize_length_channels_normalization():
    rng = np.random.default_rng(3)
    dry = 0.7 * rng.standard_normal(FS)
    ir = np.stack([_synth_ir(seed=4), _synth_ir(seed=5)], axis=1)
    wet, gain_db = impulse.auralize(dry, FS, ir, FS)
    assert wet.shape == (len(dry) + len(ir) - 1, 2)   # mono fans out
    assert np.abs(wet).max() == pytest.approx(np.abs(dry).max(), rel=1e-6)
    delta = np.zeros(64)
    delta[0] = 2.0                                # unit room, gain 2
    raw, g = impulse.auralize(dry, FS, delta, FS, normalize=None)
    assert g == 0.0
    assert np.allclose(raw[:len(dry), 0], 2.0 * dry)  # raw gain preserved


def test_auralize_resamples_ir():
    rng = np.random.default_rng(6)
    dry = rng.standard_normal(FS // 2)
    ir = _synth_ir(dur=0.5, fs=24000)
    wet, _ = impulse.auralize(dry, FS, ir, 24000)
    assert wet.shape[0] == len(dry) + 2 * len(ir) - 1   # IR now at 48 kHz
    assert np.isfinite(wet).all()


# -------------------------------------------------------------------- CLI

def test_cli_sweep_impulse_auralize(tmp_path, capsys):
    from ambiscape.cli import main
    sw_path = tmp_path / "sweep.wav"
    assert main(["sweep", "--duration", "2", "-o", str(sw_path)]) == 0
    sweep, fs = sf.read(str(sw_path), dtype="float64")
    ir_true = _synth_ir(T60=0.4, dur=0.8)
    rec = oaconvolve(sweep, ir_true)
    rec_path = tmp_path / "recorded.wav"
    sf.write(str(rec_path), rec.astype(np.float32), fs, subtype="FLOAT")
    # sidecar sweep.json sits next to the recording: found automatically
    assert main(["impulse", str(rec_path)]) == 0
    doc = json.loads((tmp_path / "impulse.json").read_text())
    assert (tmp_path / "ir.wav").exists()
    assert doc["bands"]["1000"]["T60"] == pytest.approx(0.4, rel=0.25)
    assert 0.0 < doc["sti"] <= 1.0
    ir_saved, _ = sf.read(str(tmp_path / "ir.wav"), dtype="float64")
    assert np.abs(ir_saved).max() == pytest.approx(0.5, abs=1e-3)
    dry_path = tmp_path / "dry.wav"
    dry = 0.5 * np.random.default_rng(7).standard_normal(fs // 2)
    sf.write(str(dry_path), dry.astype(np.float32), fs, subtype="FLOAT")
    assert main(["auralize", str(dry_path), "--ir",
                 str(tmp_path / "ir.wav")]) == 0
    wet, wfs = sf.read(str(tmp_path / "dry_wet.wav"), dtype="float64")
    assert wfs == fs and len(wet) > len(dry)
    out = capsys.readouterr().out
    assert "STI" in out and "wrote" in out
