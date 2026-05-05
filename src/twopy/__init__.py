"""Top-level package for the twopy two-photon imaging analysis tool.

The package exposes project metadata and script-friendly public APIs.
Inputs: none.
Outputs: package constants and helper functions imported by users and scripts.
"""

from twopy.api import find_recordings
from twopy.conversion import convert_recording_to_twopy, load_source_conversion_inputs
from twopy.converted import load_converted_recording

__all__ = [
    "__version__",
    "convert_recording_to_twopy",
    "find_recordings",
    "load_converted_recording",
    "load_source_conversion_inputs",
]

__version__ = "0.1.0"
