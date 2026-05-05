"""Convert microscope source files into twopy-owned HDF5 files.

Inputs: one validated two-photon recording session directory.
Outputs: gzip-compressed HDF5 files containing twopy-owned datasets for later
analysis.

This package keeps the public conversion API small while separating source
loading, HDF5 writing, and data-object definitions into focused modules.
"""

from twopy.conversion.api import convert_recording_to_twopy
from twopy.conversion.source_loading import load_source_conversion_inputs
from twopy.conversion.types import (
    AcquisitionMetadata,
    AlignedMovieSource,
    ConvertedRecording,
    PhotodiodeSignals,
    SourceConversionInputs,
    StimulusParameters,
    StimulusTimeline,
)

__all__ = [
    "AcquisitionMetadata",
    "AlignedMovieSource",
    "ConvertedRecording",
    "PhotodiodeSignals",
    "SourceConversionInputs",
    "StimulusParameters",
    "StimulusTimeline",
    "convert_recording_to_twopy",
    "load_source_conversion_inputs",
]
