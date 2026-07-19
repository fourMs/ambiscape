"""ambiscape — long-duration first-order ambisonic soundscape analysis.

Streaming companion to ambiviz (https://github.com/fisheggg/ambiviz):
ambiscape summarizes hours of AmbiX recordings (levels, spectra, events,
DOA/diffuseness timelines, reverberation, representative segments); ambiviz
renders rich spatial visuals (AEM, anglegram, directogram) on the short
excerpts ambiscape selects.
"""
from .io import (open_session, open_recording, read_bext, read_span,
                 export_segment, stereo_preview)
from .features import extract_take, extract_session, load_features
from .analysis import (summarize, detect_events, decay_time, decay_metrics,
                       intermittency_ratio, pick_segments)
from . import (biophony, capture, catalog, compare, ecology, enf, figures,
               longitudinal, resolve, states, taxonomy, vision, iso)

__version__ = "0.13.0"
__all__ = [
    "open_session", "open_recording", "read_bext", "read_span",
    "export_segment", "stereo_preview",
    "extract_take", "extract_session", "load_features",
    "summarize", "detect_events", "decay_time", "decay_metrics",
    "intermittency_ratio", "pick_segments",
    "biophony", "capture", "catalog", "compare", "ecology", "enf", "figures",
    "longitudinal", "resolve", "states", "taxonomy", "vision",
]
