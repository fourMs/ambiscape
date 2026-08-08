"""Sweep-based impulse response measurement and auralisation.

The measurement chain is Farina's exponential sine sweep (ESS) method:

1. :func:`exp_sweep` generates a logarithmic sweep plus its *matched
   inverse filter* (the time-reversed sweep with a −6 dB/octave amplitude
   envelope, scaled so sweep ⊛ inverse peaks at exactly 1). Play the sweep
   in the room, record it.
2. :func:`deconvolve` convolves the recording with the inverse filter.
   Harmonic-distortion products land *before* the linear impulse response
   (the point of the ESS method), so trimming everything earlier than a few
   milliseconds before the direct-sound peak (:func:`extract_ir`) removes
   both pre-ringing and loudspeaker distortion.
3. :func:`ir_metrics`, :func:`sti`, :func:`iacc_early` and :func:`iacc_e3`
   characterise the room from the IR; :func:`auralize` convolves dry
   material with it (uniformly partitioned FFT convolution).

For anything compared against the concert-hall literature use
:func:`iacc_e3`, not :func:`iacc_early`: published hall values are IACC_E3,
the mean of the 500, 1000 and 2000 Hz octave bands, and the broadband
figure is a different quantity that low-frequency content moves around.

Headroom: sweeps are written at peak −6 dBFS (``amplitude=0.5``) so a
playback chain with a mild bass boost or resonance does not clip; the
deconvolution normalisation is documented per function.

STI here is the *indirect* method of IEC 60268-16: modulation transfer
functions computed from the measured IR (Schroeder's integral), which
assumes the measurement itself is noise-free and the room is linear and
time-invariant. Ambient noise and masking corrections are NOT applied, so
the value is an upper bound describing reverberant smearing only — an
occupied or noisy room will have a lower effective STI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EPS = 1e-20

#: Octave-band centre frequencies used for IR metrics and STI (Hz).
OCTAVE_CENTERS = (125, 250, 500, 1000, 2000, 4000, 8000)

#: The three octave bands averaged into IACC_E3 by the hall literature (Hz).
IACC_E3_CENTERS = (500, 1000, 2000)

#: Below this |IACC| the sign of the signed peak is noise, not anti-phase:
#: two decorrelated ears produce a near-zero peak whose sign is arbitrary.
#: A reporting threshold only — nothing in ISO 3382-1 defines it.
IACC_SIGN_FLOOR = 0.5

# IEC 60268-16:2011 male-speech octave weights (alpha) and adjacent-band
# redundancy corrections (beta) for the seven bands 125 Hz .. 8 kHz.
_STI_ALPHA = (0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173)
_STI_BETA = (0.085, 0.078, 0.065, 0.011, 0.047, 0.095)
#: The 14 third-octave-spaced modulation frequencies of the STI matrix (Hz).
_STI_FMOD = (0.63, 0.80, 1.0, 1.25, 1.6, 2.0, 2.5,
             3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5)


def _octave_edges(centers, fs):
    """(lo, hi) integer band edges (centre / x sqrt2) below Nyquist."""
    out = []
    for c in centers:
        lo, hi = int(round(c / np.sqrt(2))), int(round(c * np.sqrt(2)))
        if hi < fs / 2:
            out.append((c, lo, hi))
    return out


# ------------------------------------------------------------------ sweep

def exp_sweep(duration=10.0, f0=40.0, f1=18000.0, fs=48000,
              fade_in=0.1, fade_out=0.02, amplitude=0.5):
    """Exponential sine sweep and matched inverse filter (Farina 2000).

    The sweep spends equal time per octave from ``f0`` to ``f1`` over
    ``duration`` seconds, with raised-cosine fades (``fade_in`` seconds at
    the start, ``fade_out`` at the end) so the loudspeaker is not stepped,
    and peak amplitude ``amplitude`` (default 0.5 = −6 dBFS of headroom
    against playback-chain resonances). The inverse filter is the
    time-reversed sweep weighted by the exponential envelope that whitens
    the pink energy distribution, scaled so that ``sweep ⊛ inverse`` is a
    unit-peak impulse at index ``len(sweep) − 1``.

    Returns ``(sweep, inverse, meta)`` where ``meta`` is a JSON-ready dict
    of the generation parameters (enough to regenerate the inverse with
    :func:`inverse_from_meta`).
    """
    from scipy.signal import oaconvolve
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    L = duration / np.log(f1 / f0)
    sweep = np.sin(2 * np.pi * f0 * L * (np.exp(t / L) - 1.0))
    ni, no = int(round(fade_in * fs)), int(round(fade_out * fs))
    if ni:
        sweep[:ni] *= 0.5 * (1 - np.cos(np.pi * np.arange(ni) / ni))
    if no:
        sweep[n - no:] *= 0.5 * (1 - np.cos(np.pi * np.arange(no) / no))[::-1]
    sweep *= amplitude
    # time-reverse, then attenuate 6 dB/octave along the reversed time axis
    # (which runs high -> low frequency), undoing the sweep's pink energy
    inverse = sweep[::-1] * np.exp(-t / L)
    inverse /= np.abs(oaconvolve(sweep, inverse)).max()
    meta = {"kind": "ambiscape-sweep", "duration_s": duration,
            "f0_hz": f0, "f1_hz": f1, "fs": fs, "fade_in_s": fade_in,
            "fade_out_s": fade_out, "amplitude": amplitude}
    return sweep, inverse, meta


def inverse_from_meta(meta: dict) -> np.ndarray:
    """Regenerate the matched inverse filter from a sweep's sidecar dict."""
    _, inverse, _ = exp_sweep(
        duration=meta["duration_s"], f0=meta["f0_hz"], f1=meta["f1_hz"],
        fs=meta["fs"], fade_in=meta.get("fade_in_s", 0.1),
        fade_out=meta.get("fade_out_s", 0.02),
        amplitude=meta.get("amplitude", 0.5))
    return inverse


def write_sweep(out_path, duration=10.0, f0=40.0, f1=18000.0, fs=48000,
                amplitude=0.5) -> dict:
    """Write ``<out>.wav`` (the sweep), ``<out>_inverse.wav`` and a
    ``<out>.json`` parameter sidecar. Returns the paths + meta."""
    import soundfile as sf
    out_path = Path(out_path)
    sweep, inverse, meta = exp_sweep(duration=duration, f0=f0, f1=f1,
                                     fs=fs, amplitude=amplitude)
    inv_path = out_path.with_name(out_path.stem + "_inverse.wav")
    json_path = out_path.with_suffix(".json")
    sf.write(str(out_path), sweep.astype(np.float32), fs, subtype="FLOAT")
    sf.write(str(inv_path), inverse.astype(np.float32), fs, subtype="FLOAT")
    json_path.write_text(json.dumps(meta, indent=2))
    return {"sweep": out_path, "inverse": inv_path, "params": json_path,
            "meta": meta}


# ----------------------------------------------------------- deconvolution

def deconvolve(rec: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """Linear convolution of a recorded sweep with the inverse filter.

    ``rec`` is (n,) or (n, ch); returns the full (n + len(inverse) − 1, ch)
    deconvolution buffer. With the inverse from :func:`exp_sweep`, feeding
    the pristine sweep back in yields a unit impulse, so the amplitude of
    the result is the recording's own level referenced to that unit — no
    further scaling is applied here.
    """
    from scipy.signal import oaconvolve
    rec = np.atleast_2d(np.asarray(rec, np.float64).T).T
    return np.stack([oaconvolve(rec[:, c], inverse)
                     for c in range(rec.shape[1])], axis=1)


def extract_ir(h: np.ndarray, fs: int, pre_ms=5.0, dur=None):
    """Trim a deconvolution buffer to the impulse response.

    The direct sound is the largest absolute peak across channels; the IR
    keeps ``pre_ms`` milliseconds before it (so the onset is intact) and
    discards everything earlier — deconvolution pre-ringing and the
    harmonic-distortion images, which the ESS method places before the
    linear response. ``dur`` caps the kept tail in seconds (default: to the
    end of the buffer). Returns ``(ir, direct_index)`` with
    ``direct_index`` the peak position inside ``ir``.
    """
    h = np.atleast_2d(np.asarray(h, np.float64).T).T
    pk = int(np.abs(h).max(axis=1).argmax())
    i0 = max(0, pk - int(round(pre_ms * fs / 1000)))
    i1 = len(h) if dur is None else min(len(h), pk + int(round(dur * fs)))
    return h[i0:i1], pk - i0


# ---------------------------------------------------------------- metrics

def ir_metrics(ir: np.ndarray, fs: int, centers=OCTAVE_CENTERS) -> dict:
    """Octave-band T60/T20/T30, EDT, C50/C80, D50 from an impulse response.

    A thin wrapper over :func:`ambiscape.analysis.decay_metrics` (the same
    truncated-Schroeder machinery used for clap-based estimates): a trimmed
    IR starts at its peak, so half a second of silent pre-roll is prepended
    to satisfy that function's noise-floor estimation, and the edge-labelled
    bands are relabelled by octave centre. Multichannel IRs are analysed on
    channel 0 (the omni/W channel of a B-format IR).
    Returns ``{centre_hz: {"T60", "T20", "T30", "EDT", "C50", "C80",
    "D50", "dr_db"}}`` (T20/T30 only when the dynamic range supports them
    and the decay was observed that far before the file ends — a
    pre-trimmed archive IR commonly yields T60 and EDT but neither).
    """
    from .analysis import decay_metrics
    x = np.asarray(ir, np.float64)
    if x.ndim > 1:
        x = x[:, 0]
    x = x / (np.abs(x).max() + EPS)     # metrics are level-invariant
    if int(np.abs(x).argmax()) < fs // 4:
        x = np.concatenate([np.zeros(fs // 2), x])
    edges = _octave_edges(centers, fs)
    dm = decay_metrics(x, fs, bands=tuple((lo, hi) for _, lo, hi in edges))
    return {str(c): dm[f"{lo}-{hi}"] for c, lo, hi in edges
            if f"{lo}-{hi}" in dm}


def sti(ir: np.ndarray, fs: int) -> dict:
    """Speech Transmission Index from an IR (IEC 60268-16 indirect method).

    Per octave band (125 Hz – 8 kHz), the modulation transfer function at
    the 14 standard modulation frequencies is Schroeder's integral

        m(fm) = |∫ h²(t) e^{−j2πfm t} dt| / ∫ h²(t) dt,

    converted to an effective SNR (clipped to ±15 dB), a transmission
    index, and band MTIs, combined with the male-speech alpha/beta weights
    of IEC 60268-16:2011.

    Assumptions (documented, not corrected for): the measurement is
    noise-free (no ambient-noise term — the MTF denominator is signal
    energy only), no level-dependent auditory masking, no absolute-speech-
    level term. The result is an upper bound: reverberant smearing only.

    Returns ``{"sti": float, "mti": {centre_hz: float}}``. Multichannel
    IRs use channel 0.
    """
    from scipy import signal as sg
    x = np.asarray(ir, np.float64)
    if x.ndim > 1:
        x = x[:, 0]
    t = np.arange(len(x)) / fs
    edges = _octave_edges(OCTAVE_CENTERS, fs)
    mti = {}
    for c, lo, hi in edges:
        sos = sg.butter(4, [lo, hi], "bandpass", fs=fs, output="sos")
        p = sg.sosfilt(sos, x) ** 2
        e = p.sum() + EPS
        ti = []
        for fm in _STI_FMOD:
            m = np.abs(np.sum(p * np.exp(-2j * np.pi * fm * t))) / e
            snr = np.clip(10 * np.log10(m / max(1 - m, EPS)), -15.0, 15.0)
            ti.append((snr + 15.0) / 30.0)
        mti[c] = float(np.mean(ti))
    m = [mti[c] for c, _, _ in edges]
    if len(m) < len(OCTAVE_CENTERS):        # fs too low for the 8 kHz band
        return {"sti": None, "mti": {str(c): round(v, 3)
                                     for c, v in mti.items()}}
    val = (sum(a * v for a, v in zip(_STI_ALPHA, m))
           - sum(b * np.sqrt(m[k] * m[k + 1])
                 for k, b in enumerate(_STI_BETA)))
    return {"sti": round(float(np.clip(val, 0.0, 1.0)), 3),
            "mti": {str(c): round(v, 3) for c, v in mti.items()}}


def iacc_early(ir: np.ndarray, fs: int, window_ms=80.0, max_lag_ms=1.0):
    """Early interaural cross-correlation of a stereo/binaural IR.

    IACC_E per ISO 3382-1: the maximum of the normalised cross-correlation
    between the two channels over the first ``window_ms`` (default 0–80 ms
    after the direct sound), searched over lags of ±``max_lag_ms``
    (default 1 ms). Broadband (no octave filtering). 1 = the two ears hear
    identical signals; low values = spatially decorrelated early sound.
    Returns None unless the IR has exactly two channels.
    """
    h = np.atleast_2d(np.asarray(ir, np.float64).T).T
    if h.shape[1] != 2:
        return None
    pk = int(np.abs(h).max(axis=1).argmax())
    n = int(round(window_ms * fs / 1000))
    seg = h[pk:pk + n]
    lag = int(round(max_lag_ms * fs / 1000))
    l, r = seg[:, 0], seg[:, 1]
    denom = np.sqrt((l ** 2).sum() * (r ** 2).sum()) + EPS
    cc = np.correlate(l, r, "full")[len(seg) - 1 - lag:len(seg) + lag]
    return round(float(np.abs(cc).max() / denom), 3)


def iacc_e3(ir: np.ndarray, fs: int, window_ms=80.0, max_lag_ms=1.0,
            centers=IACC_E3_CENTERS):
    """Octave-band early IACC and the IACC_E3 average.

    :func:`iacc_early` is broadband; the concert-hall literature reports
    IACC_E3, the mean of the 500, 1000 and 2000 Hz octave bands. The two
    are not the same quantity — low-frequency content moves the broadband
    value — so comparing a broadband number against published hall values
    compares different things. Use this one for any such comparison.

    Per band, ISO 3382-1: the maximum of the *modulus* of the normalised
    interaural cross-correlation over lags of ±``max_lag_ms``, within the
    first ``window_ms`` after the direct sound.

    ``iacc_signed`` carries the signed correlation at that same lag, which
    is the one thing the modulus discards: a negative value means the ears
    receive anti-phase sound, perceptually very different from the strong
    correlation an IACC near 1 otherwise implies. It is a diagnostic, not
    an ISO quantity.

    Returns ``{"iacc_e3", "iacc": {centre: v}, "iacc_signed": {centre: v}}``,
    with ``iacc_e3`` None when the sample rate cannot carry all three bands.
    Returns None unless the IR has exactly two channels.
    """
    from scipy import signal as sg
    h = np.atleast_2d(np.asarray(ir, np.float64).T).T
    if h.shape[1] != 2:
        return None
    pk = int(np.abs(h).max(axis=1).argmax())
    seg = h[pk:pk + int(round(window_ms * fs / 1000))]
    lag = int(round(max_lag_ms * fs / 1000))
    edges = _octave_edges(centers, fs)
    iacc, signed = {}, {}
    for c, lo, hi in edges:
        sos = sg.butter(4, [lo, hi], "bandpass", fs=fs, output="sos")
        left, right = sg.sosfilt(sos, seg[:, 0]), sg.sosfilt(sos, seg[:, 1])
        denom = np.sqrt((left ** 2).sum() * (right ** 2).sum()) + EPS
        cc = (np.correlate(left, right, "full")
              [len(seg) - 1 - lag:len(seg) + lag] / denom)
        peak = cc[int(np.abs(cc).argmax())]
        iacc[str(c)] = round(float(abs(peak)), 3)
        signed[str(c)] = round(float(peak), 3)
    e3 = (round(float(np.mean(list(iacc.values()))), 3)
          if len(edges) == len(centers) else None)
    return {"iacc_e3": e3, "iacc": iacc, "iacc_signed": signed}


# ------------------------------------------------------------- auralization

def partitioned_convolve(x: np.ndarray, h: np.ndarray,
                         block=8192) -> np.ndarray:
    """Uniformly partitioned FFT convolution (overlap-save, mono in/out).

    The IR is split into ``block``-sample partitions whose spectra sit in a
    frequency-domain delay line; the input streams through in ``block``
    hops with 2×``block`` FFTs, so memory and per-block cost stay constant
    for arbitrarily long IRs. Output equals full linear convolution:
    length ``len(x) + len(h) − 1``.
    """
    x = np.asarray(x, np.float64)
    h = np.asarray(h, np.float64)
    B, N = int(block), 2 * int(block)
    P = max(1, -(-len(h) // B))
    H = np.stack([np.fft.rfft(h[p * B:(p + 1) * B], N) for p in range(P)])
    n_out = len(x) + len(h) - 1
    n_blocks = -(-n_out // B)
    xpad = np.zeros((n_blocks + 1) * B)
    xpad[B:B + len(x)] = x
    fdl = np.zeros((P, N // 2 + 1), complex)
    out = np.empty(n_blocks * B)
    for k in range(n_blocks):
        fdl = np.roll(fdl, 1, axis=0)
        fdl[0] = np.fft.rfft(xpad[k * B:k * B + N])
        out[k * B:(k + 1) * B] = np.fft.irfft((fdl * H).sum(axis=0), N)[B:]
    return out[:n_out]


def auralize(dry: np.ndarray, fs: int, ir: np.ndarray, fs_ir: int,
             block=8192, normalize="match"):
    """Convolve dry audio with a room impulse response.

    Sample rates: if ``fs_ir != fs`` the IR is resampled (polyphase) to the
    dry material's rate — the dry audio is never resampled.

    Channel policy: equal channel counts convolve pairwise; a mono dry
    signal fans out through each IR channel (mono source, N-channel room);
    a mono IR is applied to each dry channel; any other mismatch mono-sums
    the dry signal first and fans out through the IR channels.

    Normalisation policy: raw convolution gain is arbitrary (it scales
    with the IR's level), so ``normalize="match"`` (default) rescales the
    wet result so its absolute peak equals the dry input's peak — the
    output is clip-safe iff the input was, and A/B comparisons sit at
    comparable levels. ``normalize=None`` keeps the raw convolution.

    Returns ``(wet, gain_db)`` where ``wet`` has length
    ``len(dry) + len(ir) − 1`` and ``gain_db`` is the applied make-up gain.
    """
    from scipy.signal import resample_poly
    from math import gcd
    dry = np.atleast_2d(np.asarray(dry, np.float64).T).T
    ir = np.atleast_2d(np.asarray(ir, np.float64).T).T
    if fs_ir != fs:
        g = gcd(int(fs), int(fs_ir))
        ir = np.stack([resample_poly(ir[:, c], fs // g, fs_ir // g)
                       for c in range(ir.shape[1])], axis=1)
    cd, ci = dry.shape[1], ir.shape[1]
    if cd == ci:
        pairs = [(c, c) for c in range(cd)]
    elif cd == 1:
        pairs = [(0, c) for c in range(ci)]
    elif ci == 1:
        pairs = [(c, 0) for c in range(cd)]
    else:
        dry = dry.mean(axis=1, keepdims=True)
        pairs = [(0, c) for c in range(ci)]
    wet = np.stack([partitioned_convolve(dry[:, a], ir[:, b], block=block)
                    for a, b in pairs], axis=1)
    gain = 1.0
    if normalize == "match":
        gain = (np.abs(dry).max() + EPS) / (np.abs(wet).max() + EPS)
        wet *= gain
    return wet, round(float(20 * np.log10(gain + EPS)), 2)


# ------------------------------------------------------------------ runner

def measure(recording, inverse=None, params=None, out_path=None,
            pre_ms=5.0, dur=None) -> dict:
    """Full measurement pass: recorded sweep → ir.wav + metrics dict.

    ``inverse`` is the matched inverse filter WAV from :func:`write_sweep`;
    alternatively ``params`` names the sweep's JSON sidecar and the inverse
    is regenerated bit-identically from it. With neither given, a
    ``sweep.json`` next to the recording is tried. The saved ``ir.wav`` is
    float32, rescaled to peak 0.5 (−6 dBFS; the applied gain is logged in
    ``impulse.json``, and all reported metrics are level-invariant).
    """
    import soundfile as sf
    recording = Path(recording)
    rec, fs = sf.read(str(recording), dtype="float64", always_2d=True)
    if inverse is not None:
        inv, fs_inv = sf.read(str(inverse), dtype="float64")
        src = str(inverse)
    else:
        p = Path(params) if params else recording.parent / "sweep.json"
        if not p.exists():
            raise FileNotFoundError(
                "no inverse filter: give --inverse <wav> or --params "
                f"<json> (looked for {p})")
        meta = json.loads(p.read_text())
        inv, fs_inv, src = inverse_from_meta(meta), meta["fs"], str(p)
    if fs_inv != fs:
        raise ValueError(f"inverse fs {fs_inv} != recording fs {fs} — "
                         "measure and deconvolve at one rate")
    h = deconvolve(rec, inv)
    pk = int(np.abs(h).max(axis=1).argmax())    # direct sound in the buffer
    ir, direct = extract_ir(h, fs, pre_ms=pre_ms, dur=dur)
    peak = np.abs(ir).max() + EPS
    out_path = Path(out_path) if out_path else recording.parent / "ir.wav"
    sf.write(str(out_path), (ir * (0.5 / peak)).astype(np.float32), fs,
             subtype="FLOAT")
    doc = {"recording": recording.name, "inverse": src, "fs": fs,
           "channels": int(ir.shape[1]), "ir_s": round(len(ir) / fs, 3),
           # deconvolution delays everything by len(inv) − 1 samples
           "direct_in_recording_s": round((pk - len(inv) + 1) / fs, 3),
           "pre_ms": pre_ms, "peak_dbfs": -6.02,
           "gain_db": round(float(20 * np.log10(0.5 / peak)), 2),
           "bands": ir_metrics(ir, fs), **sti(ir, fs),
           "iacc_early": iacc_early(ir, fs), **(iacc_e3(ir, fs) or {}),
           "ir_path": str(out_path)}
    (out_path.parent / "impulse.json").write_text(
        json.dumps(doc, indent=2, default=float))
    return doc
