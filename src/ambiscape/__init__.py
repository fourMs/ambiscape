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
from .io import (open_session, open_recording, open_clips, read_bext,
                 read_span, export_segment, stereo_preview)
from .features import extract_take, extract_session, load_features
from .analysis import (summarize, detect_events, decay_time, decay_metrics,
                       intermittency_ratio, pick_segments)
from . import (anthrophony, array, biophony, capture, catalog, compare,
               ecology, enf, figures, geophony, impulse, longitudinal,
               mechanical, network, resolve, starss, states, taxonomy,
               vision, iso)

__version__ = "0.26.0"
__all__ = [
    "open_session", "open_recording", "open_clips", "read_bext",
    "read_span", "export_segment", "stereo_preview",
    "extract_take", "extract_session", "load_features",
    "summarize", "detect_events", "decay_time", "decay_metrics",
    "intermittency_ratio", "pick_segments",
    "array", "biophony", "capture", "catalog", "compare", "ecology", "enf",
    "figures",
    "impulse", "longitudinal", "network", "resolve", "starss", "states",
    "taxonomy", "vision", "iso", "mechanical", "anthrophony", "geophony",
]


# Renamed in 0.22.0 to match the spelling used by Pijanowski and most of the soundscape-ecology
# literature. The old name still resolves so that existing imports and scripts keep working.
anthropophony = anthrophony
