"""Convert prior saved ROI masks into twopy ROI sets for parity checks.

Inputs: read-only saved-analysis ROI label images and converted recording
metadata.
Outputs: normal ``RoiSet`` objects that twopy analysis functions can consume.

This module is only for parity work. It lets audit scripts reuse historical ROI
masks while keeping normal analysis inputs in twopy-owned objects.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twopy.converted import RecordingData
from twopy.parity.saved_analysis import SavedAnalysisLastRoi
from twopy.roi import RoiSet, make_roi_set

__all__ = [
    "SavedAnalysisRoiSet",
    "roi_set_from_saved_analysis",
]


@dataclass(frozen=True)
class SavedAnalysisRoiSet:
    """A twopy ROI set plus provenance from a saved ROI label image.

    Inputs: one saved ROI label image and one converted recording.
    Outputs: full-frame ``RoiSet`` masks, original numeric ROI labels, and a
    note describing whether the source mask was already full-frame or embedded
    from the alignment-valid crop.

    The provenance fields make parity comparisons auditable without changing
    the normal ``RoiSet`` contract.
    """

    roi_set: RoiSet
    source_label_values: tuple[int, ...]
    source_shape: tuple[int, int]
    spatial_source: str


def roi_set_from_saved_analysis(
    saved_analysis: SavedAnalysisLastRoi,
    recording: RecordingData,
) -> SavedAnalysisRoiSet:
    """Build a full-frame twopy ROI set from a saved ``lastRoi`` mask.

    Args:
        saved_analysis: Loaded saved-analysis object with ``roi_mask_initial``.
        recording: Converted recording that defines the movie shape and saved
            alignment-valid crop.

    Returns:
        ``SavedAnalysisRoiSet`` with masks in full movie coordinates.

    Raises:
        ValueError: If the label image shape is not the movie shape or the
            alignment-valid crop shape, or if ROI labels are not positive
            integer-like values.

    Historical ROI masks from the example saved outputs are crop-domain label
    images. twopy ROI masks are full-frame, so crop-shaped masks are embedded
    back into the converted movie coordinates before analysis.
    """
    label_image = _integer_label_image(saved_analysis.roi_mask_initial)
    full_frame_labels, spatial_source = _label_image_in_movie_coordinates(
        label_image,
        recording,
    )
    label_values = _positive_label_values(full_frame_labels)
    masks = np.stack(
        tuple(full_frame_labels == label_value for label_value in label_values),
    )
    roi_set = make_roi_set(
        masks,
        labels=tuple(f"saved_roi_{label_value:04d}" for label_value in label_values),
    )
    return SavedAnalysisRoiSet(
        roi_set=roi_set,
        source_label_values=label_values,
        source_shape=saved_analysis.roi_mask_initial.shape,
        spatial_source=spatial_source,
    )


def _integer_label_image(label_image: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Return a saved ROI label image as integer labels.

    Args:
        label_image: Two-dimensional numeric ROI label image.

    Returns:
        Integer label image with nonfinite values treated as background.
    """
    values = np.asarray(label_image, dtype=np.float64)
    if values.ndim != 2:
        msg = f"Saved ROI mask must be two-dimensional; got {values.shape}"
        raise ValueError(msg)

    finite = np.isfinite(values)
    rounded = np.rint(values)
    if not np.allclose(values[finite], rounded[finite]):
        msg = "Saved ROI mask labels must be integer-like values"
        raise ValueError(msg)
    labels = np.zeros(values.shape, dtype=np.int64)
    labels[finite] = rounded[finite].astype(np.int64)
    if np.any(labels < 0):
        msg = "Saved ROI mask labels cannot be negative"
        raise ValueError(msg)
    return labels


def _label_image_in_movie_coordinates(
    label_image: npt.NDArray[np.int64],
    recording: RecordingData,
) -> tuple[npt.NDArray[np.int64], str]:
    """Return a label image in full movie coordinates.

    Args:
        label_image: Integer saved ROI label image.
        recording: Converted recording that defines full-frame and crop shapes.

    Returns:
        Full-frame label image plus the source coordinate description.
    """
    movie_shape = recording.movie.shape[1:]
    if label_image.shape == movie_shape:
        return label_image, "full_frame"

    crop = recording.alignment_valid_crop
    if label_image.shape == crop.shape:
        full_frame = np.zeros(movie_shape, dtype=np.int64)
        full_frame[
            crop.axis0_start : crop.axis0_stop,
            crop.axis1_start : crop.axis1_stop,
        ] = label_image
        return full_frame, "alignment_valid_crop"

    msg = (
        "Saved ROI mask shape must match the movie frame or alignment-valid "
        f"crop; got {label_image.shape}, movie={movie_shape}, crop={crop.shape}"
    )
    raise ValueError(msg)


def _positive_label_values(label_image: npt.NDArray[np.int64]) -> tuple[int, ...]:
    """Return sorted positive ROI label values.

    Args:
        label_image: Full-frame integer ROI label image.

    Returns:
        Positive labels sorted by numeric value.
    """
    labels = tuple(int(value) for value in np.unique(label_image) if value > 0)
    if len(labels) == 0:
        msg = "Saved ROI mask contains no positive ROI labels"
        raise ValueError(msg)
    return labels
