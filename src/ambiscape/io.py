"""Session discovery and BWF metadata for AmbiX recordings.

A *session* is a folder of WAV files from one recording occasion. Files whose
BWF timestamps chain end-to-start (recorder 2 GB splits) are treated as one
continuous *take*; otherwise they are separate takes on a common absolute
timeline (seconds since the session's first midnight).
"""
from __future__ import annotations

import datetime as _dt
import struct
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

AMBIX_CHANNELS = ("W", "Y", "Z", "X")  # Zoom H3-VR AmbiX (ACN/SN3D)


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

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def wyzx(self) -> tuple[int, int, int, int]:
        """Column indices of (W, Y, Z, X) for this take's convention."""
        return (0, 2, 3, 1) if self.order == "fuma" else (0, 1, 2, 3)


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
        return self.folder.name

    def clock(self, t: float) -> str:
        """Absolute seconds -> 'DD Mon HH:MM:SS' string."""
        base = _dt.datetime.combine(self.day0, _dt.time())
        return (base + _dt.timedelta(seconds=t)).strftime("%d %b %H:%M:%S")


def open_session(folder: str | Path) -> Session:
    """Scan a session folder.

    If ``calibration.json`` contains ``clock_offset_s``, that many seconds are
    added to every take's start time — the fix for a recorder whose clock was
    found to be off (positive offset = clock was slow). All clock-labeled
    outputs (figures, annotations, reports) then agree on corrected time.
    """
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() == ".wav" and p.is_file())
    if not paths:
        raise FileNotFoundError(f"no WAV files in {folder}")
    clock_offset = 0.0
    cal = folder / "calibration.json"
    if cal.exists():
        import json
        clock_offset = float(json.loads(cal.read_text())
                             .get("clock_offset_s", 0.0))
    sess = Session(folder=folder)
    for p in paths:
        info = sf.info(str(p))
        bx = read_bext(p)
        date = _dt.date.fromisoformat(bx["date"].replace(":", "-"))
        if sess.day0 is None:
            sess.day0 = date
        hh, mm, ss = (int(x) for x in bx["time"].split(":"))
        start = ((date - sess.day0).days * 86400 + hh * 3600 + mm * 60 + ss
                 + clock_offset)
        sess.takes.append(Take(
            path=p, start=float(start), duration=info.frames / info.samplerate,
            frames=info.frames, samplerate=info.samplerate,
            channels=info.channels, date=bx["date"], clock=bx["time"],
            order=channel_order(bx.get("description", "")),
        ))
    sess.takes.sort(key=lambda t: t.start)
    return sess


def read_span(sess: Session, t0: float, dur: float, dtype="float32"):
    """Read [t0, t0+dur) seconds (session time) from whichever take covers it."""
    for tk in sess.takes:
        if tk.start <= t0 < tk.end:
            fs = tk.samplerate
            off = int((t0 - tk.start) * fs)
            n = min(int(dur * fs), tk.frames - off)
            with sf.SoundFile(str(tk.path)) as f:
                f.seek(off)
                return f.read(n, dtype=dtype, always_2d=True), fs
    raise ValueError(f"t={t0} not covered by session {sess.name}")
