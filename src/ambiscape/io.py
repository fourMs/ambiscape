"""Session discovery and metadata for soundscape recordings.

A *session* is a folder of audio files from one recording occasion. Files
whose timestamps chain end-to-start (recorder 2 GB splits) are treated as one
continuous *take*; otherwise they are separate takes on a common absolute
timeline (seconds since the session's first midnight).

Inputs need not be four-channel AmbiX. A file's channel count picks a
processing *mode*: ``ambix`` (>= 4 ch, first-order B-format, full 3-D
direction), ``stereo`` (2 ch, a lateral left/right cue and a coherence-based
width), or ``mono`` (1 ch, no direction). Containers libsndfile cannot open
(a phone's AAC ``.m4a``, say) are transcoded to WAV with ffmpeg on ingest,
and a recording's start time is taken from its BWF timestamp if present, else
a ``YYMMDD_HHMMSS`` / ``YYYYMMDD_HHMMSS`` stamp in the filename, else the
file's modification time.
"""
from __future__ import annotations

import datetime as _dt
import re
import struct
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

AMBIX_CHANNELS = ("W", "Y", "Z", "X")  # Zoom H3-VR AmbiX (ACN/SN3D)

# Audio containers libsndfile cannot open (AAC/Apple/etc.); decoded to WAV
# with ffmpeg on ingest. WAV/FLAC/OGG/MP3/AIFF etc. are read natively.
_NEEDS_TRANSCODE = {".m4a", ".aac", ".mp4", ".m4b", ".mov", ".3gp", ".wma",
                    ".opus", ".webm",
                    # 360-camera containers (Insta360, GoPro): plain MP4s
                    # with the first audio stream in the expected place
                    ".insv", ".lrv", ".glv", ".360"}
_AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".w64",
                   ".rf64", ".caf", ".au"} | _NEEDS_TRANSCODE
_DECODE_DIR = ".ambiscape_decoded"

# codecs whose decoded audio is deeper than 16 bit and worth a 24-bit cache
_DEEP_PCM = ("pcm_s24", "pcm_s32", "pcm_f32", "pcm_f64", "pcm_s64")

# leading YYYYMMDD_HHMMSS or YYMMDD_HHMMSS in a filename (phone / recorder)
_TS_LONG = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})")
_TS_SHORT = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})")


def channel_mode(channels: int) -> str:
    """Map a channel count to a processing mode.

    ``mono`` (1), ``stereo`` (2), ``ambix`` (>= 4, first-order B-format).
    Three channels are treated as ``stereo`` on the first two (rare; a
    best-effort fallback).
    """
    if channels >= 4:
        return "ambix"
    if channels == 1:
        return "mono"
    return "stereo"


def read_bext(path: str | Path) -> dict:
    """Parse the BWF 'bext' chunk (pure python RIFF walk)."""
    out = {}
    with open(path, "rb") as f:
        riff, _size, wave = struct.unpack("<4sI4s", f.read(12))
        if riff != b"RIFF" or wave != b"WAVE":
            raise ValueError(f"{path}: not a RIFF/WAVE file")
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, csize = struct.unpack("<4sI", hdr)
            if cid == b"bext":
                data = f.read(min(csize, 604))
                out["description"] = data[0:256].split(b"\0")[0].decode("ascii", "replace")
                out["originator"] = data[256:288].split(b"\0")[0].decode("ascii", "replace")
                out["date"] = data[320:330].decode("ascii", "replace").strip("\0 ")
                out["time"] = data[330:338].decode("ascii", "replace").strip("\0 ")
                out["time_reference"] = struct.unpack("<Q", data[338:346])[0]
                break
            f.seek(csize + (csize & 1), 1)
    return out


def channel_order(bext_description: str) -> str:
    """Detect B-format convention from the H3-VR's zTRK tags in the bext
    description: 'ambix' (W,Y,Z,X) or 'fuma' (W,X,Y,Z). Defaults to 'ambix'
    when no tags are present."""
    trk = {}
    for line in bext_description.replace("\r", "\n").split("\n"):
        if line.startswith("zTRK") and "=" in line:
            k, v = line.split("=", 1)
            trk[int(k[4:])] = v.strip().upper()
    seq = [trk.get(i) for i in (1, 2, 3, 4)]
    if seq == ["W", "X", "Y", "Z"]:
        return "fuma"
    return "ambix"


@dataclass
class Take:
    path: Path
    start: float          # seconds since session day 0 midnight
    duration: float
    frames: int
    samplerate: int
    channels: int
    date: str
    clock: str
    order: str = "ambix"  # 'ambix' (W,Y,Z,X) or 'fuma' (W,X,Y,Z)
    mode: str = "ambix"   # 'ambix' | 'stereo' | 'mono' (from channel count)
    audio_path: Path | None = None  # readable WAV (== path unless transcoded)

    def __post_init__(self):
        if self.audio_path is None:
            self.audio_path = self.path

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def wyzx(self) -> tuple[int, int, int, int]:
        """Column indices of (W, Y, Z, X) for this take's convention."""
        return (0, 2, 3, 1) if self.order == "fuma" else (0, 1, 2, 3)

    def mono_ref(self, data):
        """Mono reference column from an (n, ch) block for this take's mode:
        the W channel (ambix), the L/R mean (stereo/binaural), or the lone
        channel (mono). The single signal every level/spectral/MIR feature
        runs on."""
        if self.mode in ("stereo", "binaural"):
            return 0.5 * (data[:, 0] + data[:, 1])
        if self.mode == "mono":
            return data[:, 0]
        return data[:, self.wyzx[0]]


@dataclass
class Session:
    folder: Path
    takes: list[Take] = field(default_factory=list)
    day0: _dt.date | None = None

    @property
    def duration(self) -> float:
        return sum(t.duration for t in self.takes)

    @property
    def name(self) -> str:
        return getattr(self, "_name", None) or self.folder.name

    def clock(self, t: float) -> str:
        """Absolute seconds -> 'DD Mon HH:MM:SS' string."""
        base = _dt.datetime.combine(self.day0, _dt.time())
        return (base + _dt.timedelta(seconds=t)).strftime("%d %b %H:%M:%S")


def _filename_datetime(path: Path) -> _dt.datetime | None:
    """Parse a leading ``YYYYMMDD_HHMMSS`` / ``YYMMDD_HHMMSS`` timestamp."""
    m = _TS_LONG.search(path.name)
    if m:
        y, mo, d, hh, mm, ss = (int(g) for g in m.groups())
        try:
            return _dt.datetime(y, mo, d, hh, mm, ss)
        except ValueError:
            pass
    m = _TS_SHORT.search(path.name)
    if m:
        yy, mo, d, hh, mm, ss = (int(g) for g in m.groups())
        try:
            return _dt.datetime(2000 + yy, mo, d, hh, mm, ss)
        except ValueError:
            pass
    return None


def _pick_audio_stream(path: Path) -> tuple[int, bool]:
    """Best decodable audio stream of a container: (a:N index, deep?).

    Most channels wins (a GoPro MAX's 4-channel ambisonic PCM track beats
    its stereo AAC); undecodable codecs (e.g. Apple's APAC, which ffprobe
    reports as ``none``) are skipped; ties break to the lower index; a:0
    is the floor when probing fails. Note two camera quirks documented in
    the lab's spatial-audio study (Riaz, Guo & Jensenius): Insta360
    4-channel streams are NOT B-format ambisonics (spatial metrics from
    them are unreliable), and Garmin VIRB 360 AmbiX has an empty Z channel
    (elevation is meaningless; prefer in-camera originals over VIRB Edit
    exports, whose FOA track is bugged).
    """
    import json
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name,channels",
             "-of", "json", str(path)],
            capture_output=True, text=True, check=True)
        streams = json.loads(out.stdout).get("streams", [])
    except (FileNotFoundError, subprocess.CalledProcessError,
            ValueError):
        return 0, False
    best, best_ch = 0, -1
    deep = False
    for n, s in enumerate(streams):
        codec = s.get("codec_name") or "none"
        if codec in ("none", "unknown"):
            continue
        ch = int(s.get("channels", 0))
        if ch > best_ch:
            best, best_ch = n, ch
            deep = codec.startswith(_DEEP_PCM)
    return best, deep


def _ensure_readable(path: Path) -> Path:
    """A libsndfile-readable path for ``path``.

    WAV/FLAC/MP3/etc. pass through; compressed containers libsndfile cannot
    open (AAC ``.m4a`` and friends) are decoded once to a cached WAV under
    ``<folder>/.ambiscape_decoded/`` with ffmpeg (native rate and channels).
    Containers with several audio streams decode their best one — most channels
    among decodable codecs (see ``_pick_audio_stream``); sources deeper than
    16-bit are cached at 24-bit. The cache is reused while newer than the source.
    """
    if path.suffix.lower() not in _NEEDS_TRANSCODE:
        return path
    # Stream selection must happen before the cache-freshness check: the
    # cache filename only encodes the selection when it deviates from the
    # old default (a:0, 16-bit), so a stale pre-selection cache from an
    # older ambiscape (always a:0/16-bit) is never mistaken for a fresh
    # cache of a different stream/bit depth.
    stream, deep = _pick_audio_stream(path)
    cache = path.parent / _DECODE_DIR
    cache.mkdir(exist_ok=True)
    wav = cache / (path.stem
                   + (f".a{stream}s24" if (stream > 0 or deep) else "")
                   + ".wav")
    if wav.exists() and wav.stat().st_mtime >= path.stat().st_mtime:
        return wav
    codec = "pcm_s24le" if deep else "pcm_s16le"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                        "-map", f"a:{stream}", "-c:a", codec, str(wav)],
                       check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            f"cannot decode {path.name}: ffmpeg is required for "
            f"{path.suffix} input ({e})") from e
    return wav


def _probe_recording(path: Path) -> dict:
    """Readable audio path, soundfile info, and (date, time, order) for a
    recording — from BWF if present, else a filename timestamp, else mtime."""
    audio_path = _ensure_readable(path)
    info = sf.info(str(audio_path))
    order = "ambix"
    date = time = None
    if path.suffix.lower() not in _NEEDS_TRANSCODE:
        try:
            bx = read_bext(path)
            iso = (bx.get("date") or "").replace(":", "-")
            _dt.date.fromisoformat(iso)                 # validate
            if bx.get("time", "").strip():
                date, time = iso, bx["time"]
                order = channel_order(bx.get("description", ""))
        except (ValueError, KeyError, OSError):
            pass
    if date is None:
        dtm = (_filename_datetime(path)
               or _dt.datetime.fromtimestamp(path.stat().st_mtime))
        date, time = dtm.date().isoformat(), dtm.strftime("%H:%M:%S")
    return {"audio_path": audio_path, "info": info, "date": date,
            "time": time, "order": order}


def resolve_mode(channels: int, override: str | None) -> str:
    """Channel-count mode, unless a *compatible* explicit override is given.

    ``binaural`` (2-channel ear signals from in-ear mics or a dummy head)
    cannot be told from ``stereo`` by channel count, so it is only ever
    selected explicitly -- via ``calibration.json`` ``"mode"`` or
    ``open_recording(mode=...)``. An override that contradicts the channel
    count (e.g. ``"binaural"`` on a 4-channel file) is ignored.
    """
    auto = channel_mode(channels)
    if not override:
        return auto
    ok = {"mono": channels == 1, "stereo": channels == 2,
          "binaural": channels == 2, "ambix": channels >= 4}
    return override if ok.get(override, False) else auto


def _make_take(path: Path, day0: _dt.date, clock_offset: float = 0.0,
               mode_override: str | None = None) -> Take:
    """Build a Take from any supported recording, relative to ``day0``."""
    m = _probe_recording(path)
    info, date = m["info"], _dt.date.fromisoformat(m["date"])
    hh, mm, ss = (int(x) for x in m["time"].split(":"))
    start = ((date - day0).days * 86400 + hh * 3600 + mm * 60 + ss
             + clock_offset)
    return Take(
        path=path, audio_path=m["audio_path"], start=float(start),
        duration=info.frames / info.samplerate, frames=info.frames,
        samplerate=info.samplerate, channels=info.channels,
        date=m["date"], clock=m["time"], order=m["order"],
        mode=resolve_mode(info.channels, mode_override))


def open_session(folder: str | Path) -> Session:
    """Scan a session folder of one or more recordings.

    Any supported audio (WAV/FLAC/MP3 natively, AAC ``.m4a`` etc. via
    ffmpeg) is accepted; channel count sets each take's mode (ambix / stereo
    / mono). Start times come from BWF timestamps, filename stamps, or file
    mtimes (see the module docstring). If ``calibration.json`` contains
    ``clock_offset_s``, that many seconds are added to every take's start
    time — the fix for a recorder whose clock was found to be off (positive
    offset = clock was slow). ``clock_offsets_s`` maps individual filenames
    to additional per-take offsets for multi-device sessions.
    """
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in _AUDIO_SUFFIXES and p.is_file())
    if not paths:
        raise FileNotFoundError(f"no audio files in {folder}")
    clock_offset = 0.0
    clock_offsets = {}
    mode_override = None
    cal = folder / "calibration.json"
    if cal.exists():
        import json
        c = json.loads(cal.read_text())
        clock_offset = float(c.get("clock_offset_s", 0.0))
        clock_offsets = {str(k): float(v)
                         for k, v in c.get("clock_offsets_s", {}).items()}
        mode_override = c.get("mode")            # e.g. "binaural" for ear signals
    unmatched = set(clock_offsets) - {p.name for p in paths}
    if unmatched:
        warnings.warn(f"calibration.json clock_offsets_s names no session "
                      f"file: {sorted(unmatched)}", stacklevel=2)

    sess = Session(folder=folder)
    metas = [(p, _probe_recording(p)) for p in paths]
    sess.day0 = min(_dt.date.fromisoformat(m["date"]) for _, m in metas)
    for p, _ in metas:
        sess.takes.append(_make_take(
            p, sess.day0,
            clock_offset + clock_offsets.get(p.name, 0.0),
            mode_override))
    sess.takes.sort(key=lambda t: t.start)
    return sess


def open_clips(folder: str | Path) -> Session:
    """Open a folder of dataset clips as takes on a synthetic clock.

    Corpus clips — a DCASE STARSS fold, a folder of contributed excerpts —
    typically carry no BWF ``bext`` chunk and no filename timestamp, so
    :func:`open_session` would fall back to file modification times: download
    times, which say nothing about capture and usually pile every clip onto
    one meaningless, overlapping stretch of timeline. Here the clips are
    instead chained end-to-end in sorted filename order from midnight of a
    nominal day 0 (1970-01-01). Positions on the session timeline are
    deterministic and reproducible across machines, but clock-of-day
    readings carry no meaning for these takes. ``calibration.json`` is not
    consulted (there is no real clock to correct).
    """
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in _AUDIO_SUFFIXES and p.is_file())
    if not paths:
        raise FileNotFoundError(f"no audio files in {folder}")
    sess = Session(folder=folder, day0=_dt.date(1970, 1, 1))
    cursor = 0.0
    for p in paths:
        m = _probe_recording(p)
        info = m["info"]
        c = int(cursor)
        sess.takes.append(Take(
            path=p, audio_path=m["audio_path"], start=cursor,
            duration=info.frames / info.samplerate, frames=info.frames,
            samplerate=info.samplerate, channels=info.channels,
            date=sess.day0.isoformat(),
            clock=f"{c // 3600:02d}:{c % 3600 // 60:02d}:{c % 60:02d}",
            order=m["order"], mode=channel_mode(info.channels)))
        cursor = sess.takes[-1].end
    return sess


def open_recording(path: str | Path, mode: str | None = None) -> Session:
    """Open a single recording as a one-take session ("scene").

    ``mode`` forces the processing mode when the channel count is ambiguous --
    chiefly ``"binaural"`` for a 2-channel ear-signal recording, which would
    otherwise be read as ``stereo``. Ignored if it contradicts the channel
    count.

    The folder-as-session model of :func:`open_session` assumes every file in
    a folder belongs to one recording occasion on a shared clock. A
    contributed corpus is often the opposite: one folder per recordist, each
    holding many independent one-off scenes from different places and dates.
    This opens exactly one file as its own session (day0 = that file's date),
    so each scene can go through the full pipeline on its own. Accepts any
    supported audio (transcoding compressed containers); the session name is
    the file stem.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    day0 = _dt.date.fromisoformat(_probe_recording(path)["date"])
    sess = Session(folder=path.parent, day0=day0)
    sess.takes.append(_make_take(path, day0, mode_override=mode))
    sess._name = path.stem
    return sess


def read_span(sess: Session, t0: float, dur: float, dtype="float32"):
    """Read [t0, t0+dur) seconds (session time) from whichever take covers it."""
    for tk in sess.takes:
        if tk.start <= t0 < tk.end:
            fs = tk.samplerate
            off = int((t0 - tk.start) * fs)
            n = min(int(dur * fs), tk.frames - off)
            with sf.SoundFile(str(tk.audio_path)) as f:
                f.seek(off)
                return f.read(n, dtype=dtype, always_2d=True), fs
    raise ValueError(f"t={t0} not covered by session {sess.name}")


def export_segment(sess: Session, t0: float, dur: float,
                   out_path: str | Path) -> Path:
    """Bit-exact excerpt [t0, t0+dur) to a WAV, in the take's channel count.

    Samples are copied in the readable source's own PCM subtype (no float
    round trip), so the excerpt is archival: the representative segments of a
    report stay citable against the raw takes (or, for a transcoded input,
    against its decoded WAV). The span must lie within one take (recorder
    2 GB splits chain seamlessly only in ``read_span``'s float path).
    Returns the output path.
    """
    out_path = Path(out_path)
    for tk in sess.takes:
        if tk.start <= t0 < tk.end:
            fs = tk.samplerate
            off = int((t0 - tk.start) * fs)
            n = min(int(dur * fs), tk.frames - off)
            with sf.SoundFile(str(tk.audio_path)) as f:
                subtype = f.subtype
                dtype = "int16" if subtype == "PCM_16" else "int32"
                f.seek(off)
                data = f.read(n, dtype=dtype, always_2d=True)
            sf.write(str(out_path), data, fs, subtype=subtype)
            return out_path
    raise ValueError(f"t={t0} not covered by session {sess.name}")


def stereo_preview(x, wyzx=(0, 1, 2, 3), az_deg: float = 90.0, mode="ambix"):
    """Two-channel decode of a block for listenable previews.

    ``ambix``: side-facing cardioids at ±``az_deg`` in the horizontal plane,
    ``0.5 * (W ± sin(az) * Y)`` (SN3D). ``stereo``: the first two channels
    pass through unchanged. ``mono``: the single channel is duplicated.
    Returns an (n, 2) float array — write it with ``soundfile``.
    """
    import numpy as np
    if mode == "mono" or x.shape[1] == 1:
        return np.repeat(x[:, :1], 2, axis=1)
    if mode == "stereo" or x.shape[1] < 4:
        return x[:, :2]
    W, Y = x[:, wyzx[0]], x[:, wyzx[1]]
    g = float(np.sin(np.radians(az_deg)))
    return np.stack([0.5 * (W + g * Y), 0.5 * (W - g * Y)], axis=1)
