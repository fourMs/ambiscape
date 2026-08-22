"""Freesound upload metadata: the taxonomy code is checked, not just stored.

A Broad Sound Taxonomy category has been mandatory on Freesound upload since
April 2025 and is also a search facet, so a wrong code is not a cosmetic
problem: it decides whether the excerpt is findable at all. These tests pin
the two mistakes worth catching before an upload rather than after it -- a
code that does not exist, and a code from the wrong branch of the taxonomy.
"""
import json

import pytest

from ambiscape.deposit import (
    BST_CATEGORIES,
    SOUNDSCAPE_CATEGORIES,
    freesound_sidecar,
    validate_bst_category,
)


def test_taxonomy_is_complete():
    """Five top-level branches, 28 subcategories, five of them residual."""
    assert len(BST_CATEGORIES) == 28
    assert sum(1 for k in BST_CATEGORIES if k.endswith("-other")) == 5
    branches = {k.split("-")[0] for k in BST_CATEGORIES}
    assert branches == {"m", "is", "sp", "fx", "ss"}
    assert all(c in BST_CATEGORIES for c in SOUNDSCAPE_CATEGORIES)


def test_case_and_whitespace_are_forgiven():
    assert validate_bst_category(" SS-I ") == "ss-i"


def test_unknown_code_raises():
    with pytest.raises(ValueError, match="not a Broad Sound Taxonomy"):
        validate_bst_category("ss-indoor")


def test_wrong_branch_raises_when_soundscape_only():
    """A ten-minute recording of a room is not an instrument sample."""
    with pytest.raises(ValueError, match="not a soundscape"):
        validate_bst_category("fx-o", soundscape_only=True)
    assert validate_bst_category("fx-o") == "fx-o"


def test_sidecar_contents(tmp_path):
    wav = tmp_path / "excerpt_haarlem_600s.wav"
    wav.write_bytes(b"")
    out = freesound_sidecar(wav, "ss-i", tags=["ambisonic", "room-tone"],
                            description="Loft, ventilation running.",
                            extra={"session": "2026-07-15-Haarlem"})
    assert out.name == "excerpt_haarlem_600s.wav.freesound.json"
    doc = json.loads(out.read_text())
    assert doc["bst_category"] == "ss-i"
    assert doc["bst_category_name"] == "Soundscapes / Indoors"
    assert doc["licence"] == "CC BY 4.0"
    assert doc["tags"] == ["ambisonic", "room-tone"]
    assert doc["session"] == "2026-07-15-Haarlem"
    assert "speech_fraction" not in doc


def test_speech_fraction_sets_the_review_flag(tmp_path):
    """The gate is recorded, not enforced: the judgement stays with a person."""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    clean = json.loads(freesound_sidecar(wav, "ss-u",
                                         speech_fraction=0.0).read_text())
    assert clean["privacy_review"] is False
    speechy = json.loads(freesound_sidecar(wav, "ss-u",
                                           speech_fraction=0.07).read_text())
    assert speechy["speech_fraction"] == 0.07
    assert speechy["privacy_review"] is True
