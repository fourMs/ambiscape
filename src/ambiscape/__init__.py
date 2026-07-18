"""ambiscape — long-duration first-order ambisonic soundscape analysis.

Streaming companion to ambiviz (https://github.com/fisheggg/ambiviz):
ambiscape summarizes hours of AmbiX recordings (levels, spectra, events,
DOA/diffuseness timelines, reverberation, representative segments); ambiviz
renders rich spatial visuals (AEM, anglegram, directogram) on the short
excerpts ambiscape selects.
"""
from .io import open_session, read_bext, read_span
from .features import extract_take, extract_session, load_features
from .analysis import summarize, detect_events, decay_time, pick_segments
from . import figures, taxonomy, iso

__version__ = "0.2.0"
__all__ = [
    "open_session", "read_bext", "read_span",
    "extract_take", "extract_session", "load_features",
    "summarize", "detect_events", "decay_time", "pick_segments",
    "figures", "taxonomy",
]
