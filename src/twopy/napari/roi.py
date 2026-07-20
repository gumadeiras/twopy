"""ROI helpers for the twopy napari adapter.

Inputs: napari Labels layers, label images, ROI HDF5 files, and converted
recording metadata.
Outputs: twopy ``RoiSet`` objects, label images for display, and validated ROI
save/load paths.

This module is the only napari-facing layer that translates between editable
Labels data and core ROI masks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from twopy.converted import RecordingData
from twopy.filenames import ROI_FILENAME
from twopy.hdf5_utils import Hdf5Scalar
from twopy.napari.display import spatial_crop_from_layer_metadata
from twopy.napari.protocols import NapariLayerWithData
from twopy.roi import (
    RoiSet,
    load_roi_generation_metadata,
    load_roi_set,
    make_roi_set_from_label_image,
    roi_set_to_label_image,
    save_roi_set,
)
from twopy.spatial import SpatialCrop, full_frame_crop

__all__ = [
    "load_roi_file_on_layer",
    "MergedRoiLabels",
    "merge_roi_label_values_on_layer",
    "next_roi_label_value",
    "remove_roi_label_values_from_layer",
    "roi_label_image_from_layer",
    "roi_label_image_from_layer_for_recording",
    "save_napari_label_rois",
    "select_next_roi_label_value",
    "set_roi_label_image_on_layer",
    "validate_roi_layer_matches_recording_crop",
]


class _LabelsLayerWithSelectedLabel(Protocol):
    """Small protocol for napari Labels layers with an active paint label."""

    selected_label: int


@dataclass(frozen=True)
class MergedRoiLabels:
    """Labels-layer values changed by one ROI merge.

    Args:
        target_label_value: Positive Labels value that remains after the merge.
        merged_label_values: Positive Labels values found in the layer and
            combined into ``target_label_value``.

    Returns:
        Immutable record of one successful ROI merge.

    The ROIs tab uses this result to keep its row list readable immediately
    after the label pixels change. The later live response update recomputes
    the scientific traces from the merged mask.
    """

    target_label_value: int
    merged_label_values: tuple[int, ...]


def roi_label_image_from_layer(layer: object) -> npt.NDArray[np.int64]:
    """Read integer label data from a napari Labels layer.

    Args:
        layer: Napari Labels layer, or a test object with a ``data`` attribute.

    Returns:
        Full-frame movie-coordinate integer label image.

    This small helper keeps scripts from reaching into napari layer internals
    in several different ways. The label image can be passed directly to
    ``save_napari_label_rois``. When twopy opened a cropped layer, crop metadata
    on the layer is used to restore full-frame coordinates.
    """
    data = cast(NapariLayerWithData, layer).data
    labels = np.asarray(data, dtype=np.int64)
    crop = spatial_crop_from_layer_metadata(layer)
    if crop is None:
        return labels
    full_frame = np.zeros(crop.original_shape, dtype=np.int64)
    full_frame[
        crop.axis0_start : crop.axis0_stop,
        crop.axis1_start : crop.axis1_stop,
    ] = labels
    return full_frame


def next_roi_label_value(layer: object | None) -> int:
    """Return the first unused positive ROI label value in a Labels layer.

    Args:
        layer: Napari Labels layer, or a test object with a ``data`` attribute.

    Returns:
        Smallest positive integer not present in the layer data.

    The napari paint tool uses ``selected_label`` as the value for new pixels.
    Picking the first unused value keeps hand-drawn ROIs from overwriting an
    existing cell when a recording is loaded or selected.
    """
    if layer is None or not hasattr(layer, "data"):
        return 1
    label_image = np.asarray(cast(NapariLayerWithData, layer).data)
    used_values = {
        int(label_value) for label_value in np.unique(label_image) if label_value > 0
    }
    label_value = 1
    while label_value in used_values:
        label_value += 1
    return label_value


def select_next_roi_label_value(layer: object | None) -> None:
    """Set a napari Labels layer to paint the next unused ROI value.

    Args:
        layer: Napari Labels layer, or a test object exposing ``data`` and
            ``selected_label``.

    Returns:
        None.

    Real napari Labels layers expose ``selected_label``. Some tests and
    script-level callers only pass array-like layer doubles, so the helper
    leaves objects without that attribute unchanged.
    """
    if layer is None or not hasattr(layer, "selected_label"):
        return
    labels_layer = cast(_LabelsLayerWithSelectedLabel, layer)
    labels_layer.selected_label = next_roi_label_value(layer)


def merge_roi_label_values_on_layer(
    layer: object | None,
    label_values: tuple[int, ...],
) -> MergedRoiLabels | None:
    """Merge selected ROI label values in a napari Labels layer.

    Args:
        layer: Active napari Labels layer, or a test object with settable
            ``data``.
        label_values: Positive integer Labels values to combine. The first
            value is kept as the merged ROI label.

    Returns:
        Merge details when at least two selected labels were found, otherwise
        ``None``.

    A merge is a direct label-image edit. Pixels from the selected ROIs become
    the first selected ROI value, so the mask remains inspectable as one Labels
    object before response traces are recomputed.
    """
    if layer is None:
        return None
    values = tuple(dict.fromkeys(value for value in label_values if value > 0))
    if len(values) < 2:
        return None

    labels_layer = cast(NapariLayerWithData, layer)
    label_image = np.asarray(labels_layer.data)
    target_value = values[0]
    selected_mask = np.isin(label_image, values)
    if not np.any(selected_mask):
        return None
    present_values = {int(value) for value in np.unique(label_image[selected_mask])}
    found_values = tuple(value for value in values if value in present_values)
    if len(found_values) < 2 or found_values[0] != target_value:
        return None

    updated = np.array(label_image, copy=True)
    updated[selected_mask] = target_value
    labels_layer.data = updated
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()
    return MergedRoiLabels(
        target_label_value=target_value,
        merged_label_values=found_values,
    )


def remove_roi_label_values_from_layer(
    layer: object | None,
    label_values: tuple[int, ...],
) -> int:
    """Remove selected ROI label values from a napari Labels layer.

    Args:
        layer: Active napari Labels layer, or a test object with settable
            ``data``.
        label_values: Positive integer Labels values to erase.

    Returns:
        Number of distinct ROI labels found and removed.

    The ROI tab selects ROIs by their napari integer label values. Deletion is
    just a label-image edit: selected values become background ``0`` while all
    other ROI pixels stay unchanged.
    """
    if layer is None:
        return 0
    values = tuple(sorted({value for value in label_values if value > 0}))
    if len(values) == 0:
        return 0

    labels_layer = cast(NapariLayerWithData, layer)
    label_image = np.asarray(labels_layer.data)
    removal_mask = np.isin(label_image, values)
    if not np.any(removal_mask):
        return 0

    updated = np.array(label_image, copy=True)
    removed_count = len(np.unique(label_image[removal_mask]))
    updated[removal_mask] = 0
    labels_layer.data = updated
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()
    return removed_count


def set_roi_label_image_on_layer(
    layer: object,
    recording: RecordingData,
    label_image: npt.NDArray[np.int64],
) -> None:
    """Write a movie-coordinate ROI label image into a napari Labels layer.

    Args:
        layer: Active napari Labels layer, or a test layer with settable data.
        recording: Loaded converted recording that defines the displayed crop.
        label_image: Full-frame movie-coordinate ROI labels.

    Returns:
        None.
    """
    display_labels = recording.alignment_valid_crop.crop_image(label_image)
    cast(NapariLayerWithData, layer).data = display_labels
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def load_roi_file_on_layer(
    path: Path,
    layer: object,
    recording: RecordingData,
) -> dict[str, Hdf5Scalar] | None:
    """Load one saved ROI HDF5 file into a napari Labels layer.

    Args:
        path: ROI HDF5 file to load.
        layer: Active napari Labels layer, or a test layer with settable data.
        recording: Loaded converted recording that defines valid ROI shape and
            displayed crop.

    Returns:
        Optional ROI-generation metadata from the saved file.

    Raises:
        ValueError: If the ROI file does not match the recording shape.
        OSError: If the ROI HDF5 file cannot be read.
    """
    roi_set = load_roi_set(path)
    validate_roi_shape(roi_set, recording)
    set_roi_label_image_on_layer(
        layer,
        recording,
        roi_set_to_label_image(roi_set),
    )
    return load_roi_generation_metadata(path)


def roi_label_image_from_layer_for_recording(
    layer: object,
    recording: RecordingData,
) -> npt.NDArray[np.int64]:
    """Return full-frame movie-coordinate ROI labels from a crop-native layer.

    Args:
        layer: Napari Labels layer, or a test layer with ``data``.
        recording: Loaded converted recording that defines the displayed crop.

    Returns:
        Full-frame movie-coordinate label image with zeros outside the crop.

    Raises:
        ValueError: If the Labels layer is not shaped like the recording's
            displayed alignment-valid crop.

    Napari ROI editing is intentionally crop-native: users see and edit only
    the pixels that analysis accepts by default. Persisted ROI files stay
    full-frame so core scripts and analysis have one stable mask shape.
    """
    validate_roi_layer_matches_recording_crop(layer, recording)
    display_labels = np.asarray(cast(NapariLayerWithData, layer).data, dtype=np.int64)
    crop = recording.alignment_valid_crop
    full_frame = np.zeros(crop.original_shape, dtype=np.int64)
    full_frame[
        crop.axis0_start : crop.axis0_stop,
        crop.axis1_start : crop.axis1_stop,
    ] = display_labels
    return full_frame


def validate_roi_layer_matches_recording_crop(
    layer: object,
    recording: RecordingData,
) -> None:
    """Validate that a napari Labels layer matches the displayed recording crop.

    Args:
        layer: Napari Labels layer, or a test layer with ``data``.
        recording: Loaded converted recording that defines the displayed crop.

    Returns:
        None.

    Raises:
        ValueError: If the Labels layer does not have the cropped display
            shape.
    """
    display_labels = np.asarray(cast(NapariLayerWithData, layer).data)
    crop = recording.alignment_valid_crop
    expected_shape = crop.shape
    if display_labels.shape != expected_shape:
        msg = (
            "ROI Labels layer must use the cropped recording view. "
            f"Expected display shape {expected_shape} from "
            f"{crop.source}. Got {display_labels.shape}."
        )
        raise ValueError(msg)


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
    *,
    spatial_crop: SpatialCrop | None = None,
) -> npt.NDArray[np.int64]:
    """Create the label image shown in napari for ROI drawing/editing.

    Args:
        roi_set: Optional existing ROI set or saved ROI HDF5 path.
        recording: Loaded converted recording.
        spatial_crop: Optional crop shown in napari. When omitted, the full
            movie frame is displayed.

    Returns:
        Integer label image in napari display coordinates.
    """
    crop = spatial_crop
    if crop is None:
        crop = full_frame_crop(recording.movie.shape[1:])
    if roi_set is None:
        return np.zeros(crop.shape, dtype=np.int64)
    loaded_roi_set = resolve_roi_set(roi_set)
    validate_roi_shape(loaded_roi_set, recording)
    return crop.crop_image(roi_set_to_label_image(loaded_roi_set))


def resolve_roi_save_file(
    *,
    recording_data_path: Path,
    roi_set: RoiSet | Path | None,
    explicit_roi_save_file: Path | None,
) -> Path:
    """Resolve the default path used when persisting napari ROIs.

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
    return recording_data_path.expanduser().parent / ROI_FILENAME


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
