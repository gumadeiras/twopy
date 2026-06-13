"""ROI generation controls and actions for the napari ROIs tab.

Inputs: converted recording metadata, calibration rows, and user ROI mode
settings.
Outputs: Qt controls plus Qt-free helpers for creating generated ROI labels.
"""

from twopy.napari.plotting.roi_generation.actions import (
    GeneratedRoiLabels,
    generate_roi_labels,
)
from twopy.napari.plotting.roi_generation.controls import RoiGenerationControls
from twopy.napari.plotting.roi_generation.options import (
    RoiGenerationMode,
    RoiGenerationOptions,
    RoiGenerationUnits,
    default_roi_generation_options,
    roi_generation_edited_after_generation,
    roi_generation_metadata_from_options,
    roi_generation_options_from_metadata,
)

__all__ = [
    "GeneratedRoiLabels",
    "RoiGenerationControls",
    "RoiGenerationMode",
    "RoiGenerationOptions",
    "RoiGenerationUnits",
    "default_roi_generation_options",
    "generate_roi_labels",
    "roi_generation_edited_after_generation",
    "roi_generation_metadata_from_options",
    "roi_generation_options_from_metadata",
]
