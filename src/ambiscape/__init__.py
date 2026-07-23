"""ambiscape — a holistic toolbox for analysing soundscapes (sonic ambiences).

It works from any recording — mono, stereo, binaural, or first-order
ambisonic — using whatever spatial information each format carries, and streams
recordings of any length (minutes to whole nights) in constant memory. Level,
spectral, spatial, temporal, ecological, and source-domain descriptors are
brought together for a holistic view of a place's sound, meant to be useful to
researchers, artists, and students alike.

(Companion to ambiviz, https://github.com/fisheggg/ambiviz, which renders rich
spatial visuals — AEM, anglegram, directogram — on the short excerpts
ambiscape selects from hours of audio.)
"""
from .io import (open_session, open_recording, read_bext, read_span,
                 export_segment, stereo_preview)
from .features import extract_take, extract_session, load_features
from .analysis import (summarize, detect_events, decay_time, decay_metrics,
                       intermittency_ratio, pick_segments)
from . import (anthropophony, biophony, capture, catalog, compare, ecology,
               enf, figures, geophony, longitudinal, mechanical, resolve,
               states, taxonomy, vision, iso)

__version__ = "0.15.1"
__all__ = [
    "open_session", "open_recording", "read_bext", "read_span",
    "export_segment", "stereo_preview",
    "extract_take", "extract_session", "load_features",
    "summarize", "detect_events", "decay_time", "decay_metrics",
    "intermittency_ratio", "pick_segments",
    "biophony", "capture", "catalog", "compare", "ecology", "enf", "figures",
    "longitudinal", "resolve", "states", "taxonomy", "vision", "iso",
    "mechanical", "anthropophony", "geophony",
]
