# Impulse response & auralization

Clap-based T60 ([`decay_time`](acoustics.md)) is fine for incidental
impulses, but a deliberate measurement deserves a deliberate excitation. If the only
recording of a room is a camera's, see the substitution warning in the
[clap-based guide](acoustics.md): T60 survives it, the early field does not.
The `impulse` module implements Farina's exponential sine sweep (ESS)
method: tens of decibels more dynamic range than any clap, and the
loudspeaker's harmonic distortion is pushed *ahead* of the linear
response, where it can simply be trimmed off. The resulting impulse
response (IR) is both a measurement (T60, clarity, STI) and an instrument:
convolve any dry sound with it and it plays in that room.

## 1. Generate a sweep

```bash
ambiscape sweep --duration 10 --f0 40 --f1 18000 -o sweep.wav
```

(`--fs` sets the sample rate, default 48000; `--amplitude` the peak,
default 0.5.) The command writes three files:

- `sweep.wav` — the exponential sweep, equal time per octave, with
  raised-cosine fades (100 ms in, 20 ms out) so the loudspeaker is not
  stepped, at peak **−6 dBFS** (`--amplitude 0.5`). The headroom is
  deliberate: playback chains have bass boosts and resonances, and a sweep
  that clips anywhere in the chain injects distortion the method can no
  longer separate out.
- `sweep_inverse.wav` — the matched inverse filter: the time-reversed
  sweep with a −6 dB/octave amplitude envelope, scaled so that
  sweep ⊛ inverse is a unit-peak impulse.
- `sweep.json` — the generation parameters. Keep it with the recording;
  the inverse can be regenerated bit-identically from it, so the inverse
  WAV never needs to travel.

Longer sweeps buy signal-to-noise (3 dB per doubling); 10 s is a good
default for rooms, 20–30 s for very reverberant or noisy spaces. Play the
sweep at a healthy level, record with any rig (mono, stereo, binaural, or
B-format — every channel is deconvolved), and leave a tail of silence at
least as long as the reverberation.

## 2. Deconvolve to an impulse response

```bash
ambiscape impulse recorded.wav --inverse sweep_inverse.wav
ambiscape impulse recorded.wav --params sweep.json      # equivalent
ambiscape impulse recorded.wav     # finds sweep.json next to the recording
```

The recording is convolved with the inverse filter; everything earlier
than 5 ms before the direct-sound peak (`--pre-ms`) is trimmed —
deconvolution pre-ringing and the harmonic-distortion images both live
there — and `--dur` caps the kept IR tail in seconds (default: to the
end of the recording). The IR is written as float32 `ir.wav`, rescaled to peak −6 dBFS
(the applied gain is logged in `impulse.json`; every reported metric is
level-invariant). Alongside it, `impulse.json` holds:

- **octave-band decay** (125 Hz – 8 kHz): T60, and where the dynamic range
  supports the ISO 3382 fixed ranges, T20 (−5…−25 dB) and T30 (−5…−35 dB);
  EDT; clarity C50/C80; definition D50; and the usable dynamic range
  `dr_db` behind each fit. These reuse the toolbox's
  truncated-Schroeder machinery (`analysis.decay_metrics`) — the same
  noise-floor and re-attack safeguards as the clap path, here fed a proper
  excitation. On a *trimmed* IR, one with no recording before its peak,
  `dr_db` is measured from the quietest part of the decay rather than from
  the silence `ir_metrics` prepends to let the estimator run (fixed in
  0.28.1; before that it read the padding and reported around 190 dB).
- **STI** (IEC 60268-16, *indirect method*): modulation transfer functions
  from Schroeder's integral of the squared IR at the 14 standard
  modulation frequencies, male-speech weights.

    !!! warning "STI assumptions"
        The indirect method assumes the measurement chain is noise-free
        and the room linear and time-invariant. No ambient-noise or
        masking correction is applied, so the value is an **upper bound**
        describing reverberant smearing only — an occupied or noisy room
        will have a lower effective STI. Treat it as a room descriptor,
        not an occupancy-condition prediction.

- **IACC (early)** for two-channel IRs: the maximum of the *modulus* of
  the interaural cross-correlation over 0–80 ms after the direct sound,
  lags ±1 ms, broadband. Taking the modulus is ISO 3382-1's definition,
  not a deviation from it. Meaningful for binaural (ear-signal) IRs; for
  an ordinary spaced stereo pair it is a channel-similarity figure, not a
  perceptual one. Multichannel decay/STI metrics are computed on channel 0
  (the omni/W channel of a B-format IR).

- **IACC_E3** (`iacc_e3`, since 0.28.0): the same quantity per octave
  band, plus the mean of the 500, 1000 and 2000 Hz bands.

    !!! warning "Use this one against the literature"

        Published concert-hall values are IACC_E3, not broadband. The two
        are different quantities — low-frequency content moves the
        broadband figure — so comparing `iacc_early` against hall values
        compares different things.

    The result also carries `iacc_signed`: the signed correlation at the
    same lag as the reported IACC. The modulus discards exactly one case,
    ears receiving *anti-phase* sound, which otherwise read as strongly
    correlated. The CLI mentions it only above `IACC_SIGN_FLOOR` (0.5),
    because the sign of a near-zero peak is noise — decorrelated ears
    would otherwise be flagged anti-phase on every band. A diagnostic,
    not an ISO quantity.

## 3. Auralize

```bash
ambiscape auralize dry.wav --ir ir.wav -o wet.wav
```

convolves dry (ideally anechoic) material with the measured IR using
uniformly partitioned FFT convolution — constant memory however long the
IR, exact linear convolution (`impulse.partitioned_convolve` is tested
against `scipy.signal.oaconvolve` to machine precision).

- **Sample rates**: a rate-mismatched IR is resampled (polyphase) to the
  dry material's rate; the dry audio is never resampled.
- **Channels**: equal counts convolve pairwise; mono dry material fans out
  through each IR channel (one source, N-channel room); a mono IR applies
  to each dry channel; any other mismatch mono-sums the dry input first.
- **Normalization**: raw convolution gain is arbitrary (it scales with
  the IR's level), so by default the wet result is rescaled so its peak
  equals the dry input's peak — clip-safe iff the input was, and A/B
  comparisons sit at comparable levels. `--no-normalize` keeps the raw
  convolution; the applied make-up gain is printed either way.

## Python API

```python
from ambiscape import impulse

sweep, inverse, meta = impulse.exp_sweep(duration=10)
h = impulse.deconvolve(recording, inverse)      # (n, ch)
ir, direct = impulse.extract_ir(h, fs)
impulse.ir_metrics(ir, fs)                      # {"500": {"T60": ...}, ...}
impulse.sti(ir, fs)                             # {"sti": ..., "mti": {...}}
impulse.iacc_early(ir, fs)                      # broadband, stereo/binaural
impulse.iacc_e3(ir, fs)                         # {"iacc_e3", "iacc", ...}
wet, gain_db = impulse.auralize(dry, fs, ir, fs_ir)
```

or `impulse.measure(recording, params="sweep.json")` for the whole
deconvolve-trim-save-analyse pass the CLI runs.
