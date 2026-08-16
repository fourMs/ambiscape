"""An exported segment must carry the clock it was cut from.

ambiscape reads a start time from a BWF timestamp, else a leading
``YYYYMMDD_HHMMSS`` in the filename, else the file's modification time. Its
own exports wrote none of the three, so a folder of segments read back as a
session had every one of them at whatever time they happened to be written.
The toolbox could not read what it had just written.

That convention is the seam the fourMs packages meet at, so what matters
here is the round trip: export, re-open, and land on the same wall clock.
"""
import datetime as dt

import numpy as np
import pytest
import soundfile as sf

from ambiscape.io import export_segment, open_session

from .conftest import FS, plane_wave, write_bwf


@pytest.fixture
def session(tmp_path):
    t = np.arange(30 * FS) / FS
    data = plane_wave(0.4 * np.sin(2 * np.pi * 440 * t), az_deg=90.0)
    write_bwf(tmp_path / "a.wav", data, date="2026-07-17", time="20:00:00")
    return open_session(tmp_path), tmp_path


def test_export_names_the_segment_with_its_wall_clock(session):
    sess, tmp = session

    out = export_segment(sess, sess.takes[0].start + 5.0, 3.0,
                         tmp / "out" / "seg.wav")

    assert out.exists()
    assert out.name.startswith("20260717_200005"), out.name


def test_exported_segments_reopen_on_the_same_clock(session):
    """The round trip: what ambiscape writes, ambiscape reads back."""
    sess, tmp = session
    out_dir = tmp / "segments"

    for offset in (2.0, 22.0):
        export_segment(sess, sess.takes[0].start + offset, 5.0,
                       out_dir / "seg.wav")

    again = open_session(out_dir)
    starts = sorted(tk.start for tk in again.takes)

    assert len(again.takes) == 2
    assert again.day0 == dt.date(2026, 7, 17)
    assert starts[1] - starts[0] == pytest.approx(20.0, abs=1.0)


def test_export_is_still_bit_exact(session):
    """Naming must not cost the archival guarantee."""
    sess, tmp = session

    out = export_segment(sess, sess.takes[0].start + 2.0, 3.0,
                         tmp / "out" / "seg.wav")

    orig, _ = sf.read(tmp / "a.wav", dtype="int16", always_2d=True)
    seg, fs = sf.read(out, dtype="int16", always_2d=True)
    assert fs == FS and seg.shape == (3 * FS, 4)
    assert np.array_equal(seg, orig[2 * FS:5 * FS])


def test_export_keeps_the_given_name_when_asked(session):
    """A caller that needs an exact path can still have one."""
    sess, tmp = session

    out = export_segment(sess, sess.takes[0].start + 2.0, 3.0,
                         tmp / "out" / "seg.wav", stamp=False)

    assert out.name == "seg.wav"
