"""Manual group-matching napari widgets and shared helpers.

Inputs: loaded napari recordings, FOV assignments, ROI selections, and group
matching CSV paths.
Outputs: staged Qt views for assigning FOVs and matching ROIs across
recordings.

This package keeps the public group-matching entrypoint stable while splitting
the implementation into window orchestration, FOV assignment, ROI assignment,
image helpers, response helpers, and styling helpers.
"""

from twopy.napari.group_matching.window import (
    FovAssignmentView,
    FovRecordingCard,
    GroupMatchingPanel,
    RoiAssignmentView,
    mean_image_thumbnail_pixmap,
    roi_labels_from_layer_data,
)

__all__ = [
    "FovAssignmentView",
    "FovRecordingCard",
    "GroupMatchingPanel",
    "RoiAssignmentView",
    "mean_image_thumbnail_pixmap",
    "roi_labels_from_layer_data",
]
