"""ROI helpers for the twopy napari adapter.

Inputs: napari Labels layers, label images, ROI HDF5 files, and converted
recording metadata.
Outputs: twopy ``RoiSet`` objects, label images for display, and validated ROI
save/load paths.

This module is the only napari-facing layer that translates between editable
Labels data and core ROI masks.
"""

from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from twopy.converted import RecordingData
from twopy.napari.protocols import NapariLayerWithData
from twopy.roi import (
    RoiSet,
    load_roi_set,
    make_roi_set_from_label_image,
    roi_set_to_label_image,
    save_roi_set,
)

__all__ = [
    "roi_label_image_from_layer",
    "save_napari_label_rois",
]


def roi_label_image_from_layer(layer: object) -> npt.NDArray[np.int64]:
    """Read integer label data from a napari Labels layer.

    Args:
        layer: Napari Labels layer, or a test object with a ``data`` attribute.

    Returns:
        Two-dimensional integer label image.

    This small helper keeps scripts from reaching into napari layer internals
    in several different ways. The label image can be passed directly to
    ``save_napari_label_rois``.
    """
    data = cast(NapariLayerWithData, layer).data
    return np.asarray(data, dtype=np.int64)


def save_napari_label_rois(
    label_image: npt.ArrayLike,
    path: Path,
    *,
    labels: tuple[str, ...] | None = None,
    label_prefix: str = "roi",
) -> RoiSet:
    """Save a napari labels-layer image as a twopy ROI set.

    Args:
        label_image: Two-dimensional labels-layer data where ``0`` is
            background and positive integers are ROIs.
        path: Destination ROI HDF5 file.
        labels: Optional text labels matching positive ROI values.
        label_prefix: Prefix for generated ROI labels when ``labels`` is
            omitted.

    Returns:
        Saved ``RoiSet``.

    This keeps napari editing output in the same HDF5 format that scripts and
    analysis functions already load.
    """
    roi_set = make_roi_set_from_label_image(
        np.asarray(label_image),
        labels=labels,
        label_prefix=label_prefix,
    )
    save_roi_set(roi_set, path)
    return roi_set


def roi_label_image_for_display(
    roi_set: RoiSet | Path | None,
    recording: RecordingData,
) -> npt.NDArray[np.int64]:
    """Create the label image shown in napari for ROI drawing/editing.

    Args:
        roi_set: Optional existing ROI set or saved ROI HDF5 path.
        recording: Loaded converted recording.

    Returns:
        Integer label image with the movie spatial shape.
    """
    if roi_set is None:
        return np.zeros(recording.movie.shape[1:], dtype=np.int64)
    loaded_roi_set = resolve_roi_set(roi_set)
    validate_roi_shape(loaded_roi_set, recording)
    return roi_set_to_label_image(loaded_roi_set)


def resolve_roi_save_file(
    *,
    recording_data_path: Path,
    roi_set: RoiSet | Path | None,
    explicit_roi_save_file: Path | None,
) -> Path:
    """Resolve the default path used by the magicgui Save ROIs button.

    Args:
        recording_data_path: Converted recording manifest path.
        roi_set: Optional ROI object or path opened in napari.
        explicit_roi_save_file: Optional caller-provided save path.

    Returns:
        Path for saving ROI HDF5 files.
    """
    if explicit_roi_save_file is not None:
        return explicit_roi_save_file.expanduser()
    if isinstance(roi_set, Path):
        return roi_set.expanduser()
    return recording_data_path.expanduser().parent / "rois.h5"


def resolve_roi_set(roi_set: RoiSet | Path) -> RoiSet:
    """Return a ROI set object from either an object or HDF5 path.

    Args:
        roi_set: Existing ``RoiSet`` or saved ROI HDF5 path.

    Returns:
        Loaded ``RoiSet``.
    """
    if isinstance(roi_set, Path):
        return load_roi_set(roi_set)
    return roi_set


def validate_roi_shape(roi_set: RoiSet, recording: RecordingData) -> None:
    """Validate that ROI labels can be displayed over the recording.

    Args:
        roi_set: ROI masks to display.
        recording: Loaded converted recording.

    Returns:
        None.
    """
    if roi_set.masks.shape[1:] != recording.movie.shape[1:]:
        msg = (
            f"ROI mask shape {roi_set.masks.shape[1:]} does not match movie "
            f"shape {recording.movie.shape[1:]}"
        )
        raise ValueError(msg)
