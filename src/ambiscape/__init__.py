"""ambiscape — long-duration first-order ambisonic soundscape analysis.

Streaming companion to ambiviz (https://github.com/fisheggg/ambiviz):
ambiscape summarizes hours of AmbiX recordings (levels, spectra, events,
DOA/diffuseness timelines, reverberation, representative segments); ambiviz
renders rich spatial visuals (AEM, anglegram, directogram) on the short
excerpts ambiscape selects.
"""
from .io import (open_session, read_bext, read_span, export_segment,
                 stereo_preview)
from .features import extract_take, extract_session, load_features
from .analysis import (summarize, detect_events, decay_time, decay_metrics,
                       intermittency_ratio, pick_segments)
from . import (biophony, catalog, ecology, enf, figures, resolve, states,
               taxonomy, iso)

__version__ = "0.9.0"
__all__ = [
    "open_session", "read_bext", "read_span", "export_segment",
    "stereo_preview",
    "extract_take", "extract_session", "load_features",
    "summarize", "detect_events", "decay_time", "decay_metrics",
    "intermittency_ratio", "pick_segments",
    "biophony", "catalog", "ecology", "enf", "figures", "resolve", "states",
    "taxonomy",
]
