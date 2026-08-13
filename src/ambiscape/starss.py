"""DCASE STARSS clip collections: annotation reading and DOA validation.

The STARSS datasets (Sony-TAu Realistic Spatial Soundscapes, the DCASE
sound-event localisation and detection task data) distribute real recorded
scenes as first-order ambisonic WAV clips (24 kHz, 16 bit, ACN/SN3D) with a
per-clip annotation CSV. Each headerless CSV row labels one active source in
one 100 ms frame::

    frame, class, source, azimuth, elevation[, distance]

with the frame number an integer index of 100 ms intervals, class an index
into the 13 STARSS sound-event classes (:data:`CLASSES`), source an
integer distinguishing simultaneous instances of a class, azimuth in
[-180, 180] degrees increasing counter-clockwise (0 = front, +90 = left),
elevation in [-90, 90] degrees, and distance in cm (STARSS23; the STARSS22
metadata has no distance column, so both five- and six-column rows are
accepted). The counter-clockwise azimuth convention matches ambiscape's own
pseudo-intensity azimuth (``atan2(Iy, Ix)`` over ACN/SN3D W, Y, X), so
labelled and estimated azimuths compare directly.

:func:`run_validation` drives ``ambiscape doavalidate``: every clip in a
folder (opened with :func:`ambiscape.io.open_clips` — STARSS clips carry no
BWF timestamps) is paired with its annotation CSV by file stem, the clip's
per-frame energy azimuth is compared with the labelled azimuth on
*single-source frames only*, and circular error statistics are reported
overall and per class, with an error-rose / per-class figure.

Multi-source frames are excluded by design: the energy azimuth is one
broadband direction per frame, and with two or more simultaneous sources the
pseudo-intensity vector points at an energy-weighted mixture of them, so its
deviation from either label measures the mixture, not the estimator.
"""
from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from .circstats import circular_sd, mean_resultant

EPS = 1e-20

#: The 13 STARSS sound-event classes, indexed by the CSV class column.
CLASSES = (
    "female speech", "male speech", "clapping", "telephone", "laughter",
    "domestic sounds", "footsteps", "door", "music", "musical instrument",
    "water tap", "bell", "knock",
)

FRAME_S = 0.1  # the STARSS label grid: 100 ms frames


def read_annotations(path: str | Path) -> list[dict]:
    """Parse one STARSS annotation CSV into a list of row dicts.

    Rows are headerless ``frame, class, source, azimuth, elevation`` with an
    optional trailing ``distance`` (cm). Returns dicts with keys ``frame``,
    ``class_id``, ``class_name``, ``source``, ``azimuth``, ``elevation``,
    ``distance_cm`` (None when absent). Blank lines are skipped; any other
    column count raises ``ValueError``.
    """
    rows = []
    for ln, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) not in (5, 6):
            raise ValueError(f"{path}:{ln}: expected 5 or 6 columns "
                             f"(frame, class, source, azimuth, elevation"
                             f"[, distance]), got {len(parts)}")
        try:
            frame, cls, src = int(parts[0]), int(parts[1]), int(parts[2])
            az, el = float(parts[3]), float(parts[4])
        except ValueError as e:
            raise ValueError(f"{path}:{ln}: non-numeric field: {line!r}") from e
        rows.append({
            "frame": frame,
            "class_id": cls,
            "class_name": CLASSES[cls] if 0 <= cls < len(CLASSES) else str(cls),
            "source": src,
            "azimuth": az,
            "elevation": el,
            "distance_cm": float(parts[5]) if len(parts) == 6 else None,
        })
    return rows


def single_source_frames(rows: list[dict]) -> tuple[dict, int]:
    """Frames labelled with exactly one active source.

    Returns ``(frame -> row, n_multi)`` where ``n_multi`` counts the frames
    excluded for carrying two or more simultaneous labels (see the module
    docstring for why those cannot test a single energy direction).
    """
    counts = Counter(r["frame"] for r in rows)
    singles = {r["frame"]: r for r in rows if counts[r["frame"]] == 1}
    n_multi = sum(1 for f, c in counts.items() if c > 1)
    return singles, n_multi


def frame_azimuths(path: str | Path, frame_s: float = FRAME_S,
                   wyzx=(0, 1, 2, 3), band=(80.0, 3000.0),
                   method: str = "intensity", sub_s: float = 0.01):
    """Per-frame azimuth of a FOA clip, on the label grid.

    Streams the clip in blocks, band-passes all channels to ``band`` (the
    corpus DOA band, capped below Nyquist), and returns ``(az_deg, energy,
    diffuseness)`` with one value per complete ``frame_s`` frame; a trailing
    partial frame is dropped, as it has no label. Azimuth is in degrees,
    counter-clockwise positive, 0 = front, which is the STARSS convention.

    Two estimators, neither with a parameter fitted to any corpus:

    ``method="intensity"``
        The pseudo-intensity azimuth over the whole frame,
        ``atan2(sum W*Y, sum W*X)``. Every sample counts equally, so a frame
        that is mostly reverberant tail is dominated by the tail.

    ``method="energy"``
        The same azimuth computed on ``sub_s`` sub-frames and combined as a
        circular mean weighted by each sub-frame's energy. The direct sound
        of an event carries more energy than the reverberation after it, so
        this asks where the *loudest* part of the frame came from rather than
        where the frame came from on average. It is the natural second
        estimator and it is not obviously better: weighting by energy also
        weights toward whichever source is loudest when two overlap.

    ``diffuseness`` is ``1 - |I| / E`` per frame, with ``I`` the
    pseudo-intensity vector and ``E`` the total energy in the standard
    convention. It runs from 0 for a single plane wave to 1 for an isotropic
    field, and is returned so a caller can withhold an estimate where the
    field carries no usable direction --- a choice this function deliberately
    does not make on the caller's behalf.
    """
    import soundfile as sf
    from scipy.signal import butter, sosfilt

    with sf.SoundFile(str(path)) as f:
        fs, nch = f.samplerate, f.channels
        if nch < 4:
            raise ValueError(f"{path}: DOA validation needs 4-channel FOA, "
                             f"got {nch} channel(s)")
        spf = max(int(round(frame_s * fs)), 1)
        hi = min(band[1], 0.45 * fs)
        sos = butter(4, [band[0], hi], "bandpass", fs=fs, output="sos")
        zi = np.zeros((sos.shape[0], 2, nch))
        iw, iy, iz, ix = wyzx[0], wyzx[1], wyzx[2], wyzx[3]
        spsub = max(int(round(sub_s * fs)), 1)
        azs, ens, dfs = [], [], []
        while True:
            blk = f.read(spf * 600, dtype="float64", always_2d=True)
            if not len(blk):
                break
            blk, zi = sosfilt(sos, blk, axis=0, zi=zi)
            n = len(blk) // spf
            if n == 0:
                break
            blk = blk[:n * spf]
            W = blk[:, iw].reshape(n, spf)
            Y = blk[:, iy].reshape(n, spf)
            X = blk[:, ix].reshape(n, spf)
            Z = blk[:, iz].reshape(n, spf)
            ix_ = (W * X).sum(1)
            iy_ = (W * Y).sum(1)
            iz_ = (W * Z).sum(1)
            # Standard diffuseness: the intensity vector shrinks relative to
            # the energy as the field becomes isotropic.
            e = (W ** 2).sum(1) + (X ** 2 + Y ** 2 + Z ** 2).sum(1) / 3.0
            mag = np.sqrt(ix_ ** 2 + iy_ ** 2 + iz_ ** 2)
            dfs.append(np.clip(1.0 - mag / (e / 2.0 + EPS), 0.0, 1.0))
            ens.append((W ** 2).sum(1))
            if method == "intensity":
                azs.append(np.degrees(np.arctan2(iy_, ix_)))
            elif method == "energy":
                k = spf // spsub
                if k < 2:
                    azs.append(np.degrees(np.arctan2(iy_, ix_)))
                else:
                    m = k * spsub
                    Ws = W[:, :m].reshape(n, k, spsub)
                    Ys = Y[:, :m].reshape(n, k, spsub)
                    Xs = X[:, :m].reshape(n, k, spsub)
                    a = np.arctan2((Ws * Ys).sum(2), (Ws * Xs).sum(2))
                    w = (Ws ** 2).sum(2)
                    v = (w * np.exp(1j * a)).sum(1)
                    azs.append(np.degrees(np.angle(v)))
            else:
                raise ValueError(f"unknown method {method!r}; "
                                 "expected 'intensity' or 'energy'")
    if not azs:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    return (np.concatenate(azs), np.concatenate(ens), np.concatenate(dfs))


def wrap_deg(d):
    """Wrap angle difference(s) to (-180, 180] degrees."""
    return -((180.0 - np.asarray(d, float)) % 360.0 - 180.0)


def validate_clip(wav_path: str | Path, csv_path: str | Path,
                  frame_s: float = FRAME_S, wyzx=(0, 1, 2, 3),
                  method: str = "intensity") -> dict:
    """Compare one clip's energy azimuth with its labels, frame by frame.

    Only single-source frames are scored (see the module docstring).
    Returns ``records`` (one dict per scored frame: frame, class_name,
    label azimuth, estimated azimuth, signed circular ``error_deg``),
    ``n_frames_labelled`` (distinct labelled frames), and ``n_multi``
    (frames excluded as multi-source).
    """
    rows = read_annotations(csv_path)
    singles, n_multi = single_source_frames(rows)
    az_est, _en, diff = frame_azimuths(wav_path, frame_s=frame_s, wyzx=wyzx,
                                       method=method)
    records = []
    for frame, row in sorted(singles.items()):
        if not 0 <= frame < len(az_est):
            continue        # label beyond the audio (annotation overrun)
        err = float(wrap_deg(az_est[frame] - row["azimuth"]))
        records.append({
            "frame": frame,
            "diffuseness": round(float(diff[frame]), 3),
            "class_name": row["class_name"],
            "label_az_deg": row["azimuth"],
            "est_az_deg": round(float(az_est[frame]), 1),
            "error_deg": round(err, 1),
        })
    return {"records": records,
            "n_frames_labelled": len({r["frame"] for r in rows}),
            "n_multi": n_multi}


def error_stats(records: list[dict]) -> dict:
    """Circular error statistics over scored frames.

    Median and IQR of the absolute circular error, circular bias (mean of
    the signed error) with circular SD, the fraction of frames within 20
    degrees, and a per-class breakdown (n, median, IQR).
    """
    err = np.array([r["error_deg"] for r in records], float)
    ae = np.abs(err)
    mu, R = mean_resultant(np.radians(err))

    def _mi(a):
        return (round(float(np.median(a)), 1),
                round(float(np.percentile(a, 75) - np.percentile(a, 25)), 1))

    med, iqr = _mi(ae)
    per_class = {}
    for name in sorted({r["class_name"] for r in records}):
        sub = np.array([abs(r["error_deg"]) for r in records
                        if r["class_name"] == name])
        m, q = _mi(sub)
        per_class[name] = {"n": int(len(sub)), "median_abs_deg": m,
                           "iqr_deg": q}
    return {
        "n_frames": int(len(err)),
        "median_abs_deg": med,
        "iqr_deg": iqr,
        "bias_deg": round(float(np.degrees(mu)), 1),
        "circ_sd_deg": round(float(np.degrees(circular_sd(R))), 1),
        "within_20deg": round(float((ae <= 20).mean()), 2),
        "per_class": per_class,
    }


def validate_collection(folder: str | Path, ann_dir: str | Path,
                        frame_s: float = FRAME_S) -> dict:
    """Validate every clip in ``folder`` against CSVs in ``ann_dir``.

    Clips are opened with :func:`ambiscape.io.open_clips` (synthetic clock;
    STARSS clips carry no BWF timestamps) and paired with annotations by
    file stem (``fold4_room23_mix001.wav`` ↔ ``fold4_room23_mix001.csv``).
    Clips without a matching CSV are skipped with a warning. Returns overall
    statistics, per-clip statistics, and the pooled per-frame records.
    """
    from .io import open_clips

    ann_dir = Path(ann_dir)
    sess = open_clips(folder)
    all_records, per_clip = [], {}
    n_multi = n_labelled = 0
    for tk in sess.takes:
        csv = ann_dir / (tk.path.stem + ".csv")
        if not csv.exists():
            warnings.warn(f"no annotation CSV for {tk.path.name} in "
                          f"{ann_dir}", stacklevel=2)
            continue
        v = validate_clip(tk.audio_path, csv, frame_s=frame_s, wyzx=tk.wyzx)
        n_multi += v["n_multi"]
        n_labelled += v["n_frames_labelled"]
        if v["records"]:
            per_clip[tk.path.stem] = error_stats(v["records"])
        all_records.extend(v["records"])
    if not all_records:
        raise FileNotFoundError(
            f"no labelled single-source frames scored: check that {ann_dir} "
            f"holds CSVs matching the clip stems in {folder}")
    return {
        "overall": error_stats(all_records),
        "n_frames_labelled": n_labelled,
        "n_multi_excluded": n_multi,
        "per_clip": per_clip,
        "records": all_records,
    }


def _figure(doc: dict, out_png: Path, title: str):
    """Error rose (signed circular error) + per-class absolute error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err = np.array([r["error_deg"] for r in doc["records"]], float)
    per_class = doc["overall"]["per_class"]

    fig = plt.figure(figsize=(12.8, 5.4), dpi=130)
    ax0 = fig.add_subplot(1, 2, 1, projection="polar")
    bins = np.radians(np.arange(-180, 181, 10))
    h, edges = np.histogram(np.radians(err), bins=bins)
    ax0.bar((edges[:-1] + edges[1:]) / 2, h, width=np.radians(10),
            color="#2a78d6", alpha=0.85, edgecolor="none")
    ax0.set_theta_zero_location("N")
    ax0.set_title("signed azimuth error (0 = agreement)", fontsize=9)
    ax0.tick_params(labelsize=7)

    ax1 = fig.add_subplot(1, 2, 2)
    names = list(per_class)
    med = [per_class[n]["median_abs_deg"] for n in names]
    iqr = [per_class[n]["iqr_deg"] for n in names]
    ypos = np.arange(len(names))
    ax1.barh(ypos, med, xerr=np.array(iqr) / 2, color="#2a78d6",
             alpha=0.85, error_kw={"ecolor": "#d66a2a", "lw": 1.2})
    ax1.set_yticks(ypos, [f"{n} (n={per_class[n]['n']})" for n in names],
                   fontsize=8)
    ax1.axvline(doc["overall"]["median_abs_deg"], color="#d66a2a", lw=1,
                ls="--", label="overall median")
    ax1.set(xlabel="median |error| (deg); bars = IQR/2",
            title="per-class absolute error")
    ax1.legend(fontsize=8)
    ax1.grid(axis="x", alpha=0.2)
    ax1.invert_yaxis()
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def run_validation(folder: str | Path, ann_dir: str | Path,
                   out_dir: str | Path | None = None,
                   frame_s: float = FRAME_S) -> dict:
    """CLI driver: validate, write ``doavalidate.json`` + ``doavalidate.png``.

    The JSON keeps the statistics but not the pooled per-frame records
    (which can run to hundreds of thousands of rows on a full fold).
    """
    import json

    folder = Path(folder)
    out = Path(out_dir) if out_dir else folder / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    doc = validate_collection(folder, ann_dir, frame_s=frame_s)
    slim = {k: v for k, v in doc.items() if k != "records"}
    (out / "doavalidate.json").write_text(json.dumps(slim, indent=2,
                                                     default=float))
    o = doc["overall"]
    _figure(doc, out / "doavalidate.png",
            f"{folder.name} — energy DOA vs STARSS labels: "
            f"n={o['n_frames']} single-source frames, "
            f"median |err| {o['median_abs_deg']}°")
    return slim
