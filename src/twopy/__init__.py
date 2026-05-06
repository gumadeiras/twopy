"""Top-level package for the twopy two-photon imaging analysis tool.

The package exposes project metadata and script-friendly public APIs.
Inputs: none.
Outputs: package constants and helper functions imported by users and scripts.
"""

from twopy.api import find_recordings
from twopy.conversion import convert_recording_to_twopy, load_source_conversion_inputs
from twopy.converted import load_converted_recording
from twopy.responses import (
    FrameWindow,
    WindowedRoiResponse,
    frame_windows_from_photodiode_alignment,
    make_frame_windows,
    split_traces_by_frame_windows,
)
from twopy.roi import (
    BackgroundCorrectedRoiTraces,
    BackgroundCorrectionMethod,
    RoiSet,
    RoiTraces,
    extract_background_corrected_roi_traces,
    extract_roi_traces,
    load_roi_set,
    make_roi_set,
    save_roi_set,
)
from twopy.stimulus import (
    StimulusSpecificColumnMapping,
    map_stimulus_specific_column,
)
from twopy.synchronization import (
    AlignedPhotodiodeEvent,
    PhotodiodeAlignment,
    PhotodiodeEvent,
    PhotodiodeEventSet,
    detect_photodiode_events,
    detect_recording_photodiode_events,
    pair_photodiode_events_to_imaging_frames,
)

__all__ = [
    "__version__",
    "AlignedPhotodiodeEvent",
    "BackgroundCorrectedRoiTraces",
    "BackgroundCorrectionMethod",
    "convert_recording_to_twopy",
    "detect_photodiode_events",
    "detect_recording_photodiode_events",
    "extract_background_corrected_roi_traces",
    "extract_roi_traces",
    "find_recordings",
    "frame_windows_from_photodiode_alignment",
    "FrameWindow",
    "load_converted_recording",
    "load_roi_set",
    "load_source_conversion_inputs",
    "make_frame_windows",
    "make_roi_set",
    "map_stimulus_specific_column",
    "pair_photodiode_events_to_imaging_frames",
    "PhotodiodeAlignment",
    "PhotodiodeEvent",
    "PhotodiodeEventSet",
    "RoiSet",
    "RoiTraces",
    "save_roi_set",
    "split_traces_by_frame_windows",
    "StimulusSpecificColumnMapping",
    "WindowedRoiResponse",
]

__version__ = "0.1.0"
