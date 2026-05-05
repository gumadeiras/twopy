"""Top-level package for the twopy two-photon imaging analysis tool.

The package exposes project metadata and script-friendly public APIs.
Inputs: none.
Outputs: package constants and helper functions imported by users and scripts.
"""

from twopy.api import find_recordings
from twopy.conversion import convert_recording_to_twopy, load_source_conversion_inputs
from twopy.converted import load_converted_recording
from twopy.roi import (
    RoiSet,
    RoiTraces,
    extract_roi_traces,
    load_roi_set,
    make_roi_set,
    save_roi_set,
)

__all__ = [
    "__version__",
    "convert_recording_to_twopy",
    "extract_roi_traces",
    "find_recordings",
    "load_converted_recording",
    "load_roi_set",
    "load_source_conversion_inputs",
    "make_roi_set",
    "RoiSet",
    "RoiTraces",
    "save_roi_set",
]

__version__ = "0.1.0"
