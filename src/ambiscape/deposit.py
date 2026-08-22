"""What a session publishes: non-identifying feature TSVs, and upload metadata.

Two things leave a session folder. The first is a 1 Hz feature deposit in the
StillStanding365 schema, described below. The second is an audio excerpt bound
for a public repository, and since April 2025 Freesound will not accept one
without a category from its Broad Sound Taxonomy, so
:func:`freesound_sidecar` writes that category and the rest of the upload
metadata beside the WAV rather than leaving it to be typed into a web form.


Writes one TSV per take with the columns used by the StillStanding365 Zenodo
deposit (``audio/{day}.tsv``): per-second ``Time``, ``level_dbfs``,
``centroid_hz``, ``low_frac`` (< 250 Hz), ``high_frac`` (> 2 kHz). A 1 Hz
loudness/spectral envelope is far below speech timescales and carries no
intelligible content, so these files are safe to publish where raw audio is
not.

Method notes vs. the original ``extract_audio.py``: levels here come from the
W (omni) channel at native rate (the original used an ffmpeg 4-channel
downmix at 8 kHz — offsets of a few tenths of a dB are expected), and band
fractions are power fractions from the cached log-spectrogram (the original
used magnitude fractions of an 8 kHz FFT). Trends and dynamics are directly
comparable; absolute fraction values differ slightly by construction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import db


# --- Freesound's Broad Sound Taxonomy (mandatory on upload since April 2025) ---
#
# Five top-level categories, 28 subcategories of which five are residual
# "other" slots. The taxonomy classifies the FILE -- what kind of audio this is
# and what somebody would use it for -- not the sound, so it is deliberately
# not a fourth scheme in `ambiscape.taxonomy`, where Schaeffer, Schafer and
# soundscape ecology each ask a question of the same sound. For a session
# recorded in a room the answer here is nearly always `ss-i`, and a field with
# one value is no use as a descriptor. It is upload metadata and lives here.
#
# https://freesound.org/help/broad-sound-taxonomy/
BST_CATEGORIES: dict[str, str] = {
    "m-sp": "Music / Solo percussion",
    "m-si": "Music / Solo instrument",
    "m-m": "Music / Multiple instruments",
    "m-other": "Music / Other",
    "is-p": "Instrument samples / Percussion",
    "is-s": "Instrument samples / String",
    "is-w": "Instrument samples / Wind",
    "is-k": "Instrument samples / Piano, keyboard instruments",
    "is-e": "Instrument samples / Synths, electronic",
    "is-other": "Instrument samples / Other",
    "sp-s": "Speech / Solo speech",
    "sp-c": "Speech / Conversation, crowd",
    "sp-p": "Speech / Processed, synthetic",
    "sp-other": "Speech / Other",
    "fx-o": "Sound effects / Objects, house appliances",
    "fx-v": "Sound effects / Vehicles",
    "fx-m": "Sound effects / Other mechanisms, engines, machines",
    "fx-h": "Sound effects / Human sounds and actions",
    "fx-a": "Sound effects / Animals",
    "fx-n": "Sound effects / Natural elements and explosions",
    "fx-ex": "Sound effects / Experimental",
    "fx-el": "Sound effects / Electronic, design",
    "fx-other": "Sound effects / Other",
    "ss-n": "Soundscapes / Nature",
    "ss-i": "Soundscapes / Indoors",
    "ss-u": "Soundscapes / Urban",
    "ss-s": "Soundscapes / Synthetic, artificial",
    "ss-other": "Soundscapes / Other",
}

#: The categories a whole-room recording can plausibly take. A session is a
#: place rather than a sample, so anything outside this set is a typo far more
#: often than it is a decision.
SOUNDSCAPE_CATEGORIES = ("ss-n", "ss-i", "ss-u", "ss-s", "ss-other")


def validate_bst_category(code: str, soundscape_only: bool = False) -> str:
    """Return `code` if it is a Broad Sound Taxonomy subcategory, else raise.

    Set `soundscape_only` to reject the music, speech, instrument-sample and
    sound-effect branches: a ten-minute recording of a room is a soundscape,
    and a category from another branch in that position is a mistake worth
    catching before the upload rather than after it.
    """
    code = str(code).strip().lower()
    if code not in BST_CATEGORIES:
        raise ValueError(
            f"{code!r} is not a Broad Sound Taxonomy subcategory. "
            f"Valid codes: {', '.join(sorted(BST_CATEGORIES))}")
    if soundscape_only and code not in SOUNDSCAPE_CATEGORIES:
        raise ValueError(
            f"{code!r} is {BST_CATEGORIES[code]}, not a soundscape. "
            f"A whole-room recording takes one of: "
            f"{', '.join(SOUNDSCAPE_CATEGORIES)}")
    return code


def freesound_sidecar(wav_path: str | Path, bst_category: str,
                      licence: str = "CC BY 4.0",
                      tags: "list[str] | None" = None,
                      description: str | None = None,
                      speech_fraction: float | None = None,
                      soundscape_only: bool = True,
                      extra: dict | None = None) -> Path:
    """Write `<wav>.freesound.json`, the upload metadata for one excerpt.

    Freesound has required a Broad Sound Taxonomy category on every upload
    since April 2025, and it is also a search facet, so the category decides
    whether anyone finds the file. Recording it in a sidecar rather than
    choosing it in the upload form keeps the choice reproducible and reviewable
    across a pack of excerpts.

    `speech_fraction` is the result of the privacy gate (``ambiscape
    speechgate``, silero-vad). It is stored rather than enforced here, because
    what counts as an acceptable fraction is a judgement about the recording
    and not a property of the format; a value above 0.01 is written with a
    ``privacy_review`` flag so a pack cannot be uploaded without someone
    looking at it.
    """
    import json

    wav_path = Path(wav_path)
    code = validate_bst_category(bst_category, soundscape_only=soundscape_only)
    doc = {
        "filename": wav_path.name,
        "bst_category": code,
        "bst_category_name": BST_CATEGORIES[code],
        "licence": licence,
        "tags": list(tags) if tags else [],
        "description": description,
    }
    if speech_fraction is not None:
        doc["speech_fraction"] = round(float(speech_fraction), 4)
        doc["privacy_review"] = float(speech_fraction) > 0.01
    if extra:
        doc.update(extra)
    doc["_taxonomy_note"] = (
        "bst_category is Freesound's Broad Sound Taxonomy "
        "(https://freesound.org/help/broad-sound-taxonomy/), mandatory on "
        "upload since April 2025. It classifies the file, not the sound: it "
        "is unrelated to the Schaeffer, Schafer and soundscape-ecology "
        "labels in ambiscape.taxonomy.")
    out = wav_path.parent / (wav_path.name + ".freesound.json")
    out.write_text(json.dumps(doc, indent=2) + "\n")
    return out


def export_take_tsv(npz_path: str | Path, out_dir: str | Path) -> Path:
    """One take's per-second features as a plain TSV: level, centroid, low and high share.

    The deposit format, written so that a reader who has neither this package nor numpy can
    use the analysis: four columns, one row a second, no compression and no pickling. The
    `.npz` beside it keeps the full resolution.
    """
    p = np.load(str(npz_path))
    logf = p["logf"]
    fc = np.sqrt(logf[:-1] * logf[1:])
    S = p["logspec"]
    tot = S.sum(1) + 1e-20
    low = S[:, fc < 250].sum(1) / tot
    high = S[:, fc > 2000].sum(1) / tot
    level = db(p["rms_w"].astype(np.float64) ** 2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (Path(npz_path).stem + ".tsv")
    with open(out, "w") as f:
        f.write("Time\tlevel_dbfs\tcentroid_hz\tlow_frac\thigh_frac\n")
        for i in range(len(level)):
            f.write(f"{i}\t{level[i]:.1f}\t{p['centroid'][i]:.0f}\t"
                    f"{low[i]:.3f}\t{high[i]:.3f}\n")
    return out


def export_session(folder: str | Path) -> list[Path]:
    """Run `export_take_tsv` over every feature file in a session, into `deposit/`."""
    folder = Path(folder)
    fdir = folder / "analysis" / "features"
    outs = []
    for npz in sorted(fdir.glob("*.npz")):
        outs.append(export_take_tsv(npz, folder / "deposit"))
    return outs
