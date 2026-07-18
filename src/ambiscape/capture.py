"""Always-on capture daemon: continuous feature extraction on the edge.

For a study that runs for months---a year in one room---recording raw
ambisonic audio is impractical (terabytes) and a privacy hazard (speech).
The alternative is to extract *features* continuously on a small always-on
computer (a Raspberry Pi) and keep only those. This module is that daemon.

It works by reusing the tested file extractor rather than re-implementing a
live stream: each block (an hour, say) is captured to a short-lived WAV on
disk, run through :func:`ambiscape.features.extract_take`, saved as a feature
``.npz``, and the audio is then **deleted**. Audio therefore exists only
transiently (one block at a time), which is both robust---a crash leaves the
block on disk to reprocess---and privacy-preserving. At each midnight the day
is rolled up into a per-day ``summary.json`` (a catalog- and
longitudinal-ready scene) and, optionally, a non-identifying deposit.

Layout written under ``root``::

    root/2024-03-21/features/100000.npz   (one per block)
    root/2024-03-21/analysis/summary.json (rolled up at day end)

The audio source and the wall clock are injectable, so the orchestration is
tested without any hardware; the default source uses ``sounddevice`` (the
optional ``[capture]`` extra) to read a 4-channel interface. A soundfield
mic that outputs A-format is converted to B-format with
:func:`aformat_to_bformat` and a mic-specific matrix before extraction.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .io import Take


def capture_available() -> bool:
    """True if a live-capture backend (``sounddevice``) is importable."""
    try:
        import sounddevice  # noqa: F401
        return True
    except Exception:
        return False


def list_devices():
    """List audio input devices (for choosing ``--device``)."""
    import sounddevice as sd
    return sd.query_devices()


def aformat_to_bformat(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Convert A-format capsule signals to B-format via a 4x4 ``matrix``.

    ``x`` is (samples, 4) capsule audio; ``matrix`` maps capsules to
    (W, Y, Z, X) per the microphone maker's calibration. Returns
    (samples, 4) AmbiX-ordered B-format.
    """
    return np.asarray(x, float) @ np.asarray(matrix, float).T


def _device_source(out_path, seconds, fs, channels, device=None):
    """Stream ``seconds`` of live audio to ``out_path`` (low, constant RAM)."""
    import sounddevice as sd
    remaining = int(seconds * fs)
    with sf.SoundFile(str(out_path), "w", fs, channels, subtype="PCM_24") as f:
        with sd.InputStream(samplerate=fs, channels=channels, device=device,
                            dtype="float32") as stream:
            while remaining > 0:
                block, _overflow = stream.read(min(fs, remaining))
                f.write(block)
                remaining -= len(block)


class CaptureDaemon:
    """Continuous block-capture → feature-extraction → daily-rollup loop.

    Parameters
    ----------
    root : path
        Output directory; per-day subfolders are created under it.
    fs, channels : int
        Capture format (default 48 kHz, 4-channel AmbiX).
    block_seconds : float
        Seconds per capture block (default 3600 = hourly).
    device : optional
        ``sounddevice`` input-device id for the default source.
    order : str
        Channel convention of the captured audio, ``"ambix"`` or ``"fuma"``.
    aformat_matrix : optional 4x4 array
        If given, captured audio is treated as A-format and converted to
        B-format (AmbiX) before extraction.
    keep_audio : bool
        Keep the per-block WAVs instead of deleting them (default False).
    deposit : bool
        Also write a non-identifying 1 Hz deposit per day (default True).
    source, now : callables
        Injectable audio source ``source(out_path, seconds, fs, channels)``
        and clock ``now() -> datetime``; the defaults use the sound device
        and the system clock.
    retry_wait_s : float
        Seconds to wait after a capture error before retrying.
    """

    def __init__(self, root, fs: int = 48000, channels: int = 4,
                 block_seconds: float = 3600.0, device=None, order="ambix",
                 aformat_matrix=None, keep_audio=False, deposit=True,
                 source=None, now=None, retry_wait_s: float = 5.0,
                 log=None):
        self.root = Path(root)
        self.fs, self.channels = fs, channels
        self.block_seconds = block_seconds
        self.device, self.order = device, order
        self.aformat_matrix = (np.asarray(aformat_matrix, float)
                               if aformat_matrix is not None else None)
        self.keep_audio, self.deposit = keep_audio, deposit
        self.retry_wait_s = retry_wait_s
        self._now = now or _dt.datetime.now
        self._source = source or (
            lambda p, s, fs, ch: _device_source(p, s, fs, ch, self.device))
        self._log = log or self._default_log
        self._stop = False
        self._current_day = None

    # -------------------------------------------------------------- logging
    def _default_log(self, msg: str):
        self.root.mkdir(parents=True, exist_ok=True)
        # real wall clock, independent of the (possibly injected) block clock
        stamp = _dt.datetime.now().isoformat(timespec="seconds")
        with open(self.root / "capture.log", "a") as f:
            f.write(f"{stamp}  {msg}\n")

    def stop(self):
        """Request a graceful stop after the current block."""
        self._stop = True

    # -------------------------------------------------------------- one block
    def _process_block(self, wav: Path, start: _dt.datetime, daydir: Path):
        if self.aformat_matrix is not None:
            data, fs = sf.read(str(wav), always_2d=True)
            sf.write(str(wav), aformat_to_bformat(data, self.aformat_matrix),
                     fs, subtype="PCM_24")
        info = sf.info(str(wav))
        sod = start.hour * 3600 + start.minute * 60 + start.second
        take = Take(path=wav, start=float(sod),
                    duration=info.frames / info.samplerate,
                    frames=info.frames, samplerate=info.samplerate,
                    channels=info.channels, date=start.date().isoformat(),
                    clock=start.strftime("%H:%M:%S"),
                    order="ambix" if self.aformat_matrix is not None
                    else self.order)
        from .features import extract_take
        F = extract_take(take)
        out = daydir / "features" / (start.strftime("%H%M%S") + ".npz")
        np.savez_compressed(out, **F)
        return out

    # -------------------------------------------------------------- rollup
    def _rollup_day(self, daydir: Path):
        from .features import load_features
        from .resolve import full_summary
        paths = sorted((daydir / "features").glob("*.npz"))
        if not paths:
            return
        try:
            F = load_features(paths)
            summary = full_summary(F)
            summary["date"] = daydir.name
            summary["n_blocks"] = len(paths)
            (daydir / "analysis").mkdir(parents=True, exist_ok=True)
            (daydir / "analysis" / "summary.json").write_text(
                json.dumps(summary, indent=2))
            if self.deposit:
                from .deposit import export_session
                try:
                    export_session(daydir)
                except Exception as e:            # deposit is best-effort
                    self._log(f"deposit failed for {daydir.name}: {e}")
            self._log(f"rolled up {daydir.name} ({len(paths)} blocks)")
        except Exception as e:
            self._log(f"rollup failed for {daydir.name}: {e}")

    # -------------------------------------------------------------- main loop
    def run(self, max_blocks: int | None = None):
        """Capture blocks until stopped (or ``max_blocks`` reached)."""
        import time
        self._log(f"capture start: {self.channels}ch @ {self.fs} Hz, "
                  f"{self.block_seconds:.0f}s blocks -> {self.root}")
        n = 0
        while not self._stop and (max_blocks is None or n < max_blocks):
            start = self._now()
            day = start.date().isoformat()
            if self._current_day is not None and day != self._current_day:
                self._rollup_day(self.root / self._current_day)
            self._current_day = day
            daydir = self.root / day
            (daydir / "features").mkdir(parents=True, exist_ok=True)
            wav = daydir / "features" / (start.strftime("%H%M%S") + ".wav")
            try:
                self._source(str(wav), self.block_seconds, self.fs,
                             self.channels)
                self._process_block(wav, start, daydir)
            except Exception as e:
                self._log(f"block {start.isoformat()} failed: {e}")
                if self.retry_wait_s:
                    time.sleep(self.retry_wait_s)
            finally:
                if wav.exists() and not self.keep_audio:
                    wav.unlink()
            n += 1
        self._log("capture stopping")

    def finish(self):
        """Roll up the day in progress (call on graceful shutdown)."""
        if self._current_day is not None:
            self._rollup_day(self.root / self._current_day)
