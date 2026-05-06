"""Display-coordinate helpers for the twopy napari adapter.

Inputs: movie-coordinate images, label images, and spatial crops.
Outputs: napari-display arrays and full-frame movie-coordinate ROI labels.

twopy stores converted movie data as ``(frame, axis0, axis1)`` because those are
the HDF5 array axes. napari displays two-dimensional images as
``(row, column)``. These helpers keep that transpose explicit so ROI pixels
drawn in napari can be converted back to analysis coordinates without guessing.
"""

from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from twopy.spatial import SpatialCrop
from twopy.typing_guards import (
    int_or_none,
    int_tuple_or_none,
    string_key_mapping_or_none,
)

DISPLAY_CROP_METADATA_KEY = "twopy_display_spatial_crop"

__all__ = [
    "display_image_from_movie_image",
    "display_labels_from_movie_labels",
    "display_metadata_for_spatial_crop",
    "movie_labels_from_display_layer",
]


class _LayerWithData(Protocol):
    """Small protocol for napari-like layers with array data."""

    data: object


class _LayerWithMetadata(Protocol):
    """Small protocol for napari-like layers with metadata."""

    metadata: object


class _LayerWithOptions(Protocol):
    """Small protocol for fake test layers that store napari kwargs."""

    options: dict[str, object]


def display_image_from_movie_image(
    image: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Convert a movie-coordinate image or movie block for napari display.

    Args:
        image: Two-dimensional ``(axis0, axis1)`` image or three-dimensional
            ``(frame, axis0, axis1)`` movie block.

    Returns:
        View with the two spatial axes swapped for napari display.
    """
    values = np.asarray(image)
    if values.ndim not in (2, 3):
        msg = f"Expected a 2-D image or 3-D movie block; got {values.shape}"
        raise ValueError(msg)
    return np.swapaxes(values, -1, -2)


def display_labels_from_movie_labels(
    label_image: npt.NDArray[np.int64],
    spatial_crop: SpatialCrop,
) -> npt.NDArray[np.int64]:
    """Convert full-frame movie-coordinate labels into napari display labels.

    Args:
        label_image: Full-frame integer labels in movie coordinates.
        spatial_crop: Spatial crop displayed in napari.

    Returns:
        Display-coordinate label image for the crop.

    The viewer shows the same alignment-valid crop used by default response
    analysis. Cropping here prevents users from drawing ROIs in pixels that the
    default analysis path will reject later.
    """
    cropped_labels = spatial_crop.crop_image(label_image)
    return display_image_from_movie_image(cropped_labels)


def movie_labels_from_display_layer(layer: object) -> npt.NDArray[np.int64]:
    """Read napari display labels and return full-frame movie labels.

    Args:
        layer: napari Labels layer, or a test object with ``data`` and optional
            crop metadata.

    Returns:
        Full-frame integer label image in movie coordinates.
    """
    display_labels = np.asarray(cast(_LayerWithData, layer).data, dtype=np.int64)
    crop = _spatial_crop_from_layer_metadata(layer)
    if crop is None:
        return cast(npt.NDArray[np.int64], display_labels)

    movie_crop_labels = display_image_from_movie_image(display_labels)
    full_frame = np.zeros(crop.original_shape, dtype=np.int64)
    full_frame[
        crop.axis0_start : crop.axis0_stop,
        crop.axis1_start : crop.axis1_stop,
    ] = movie_crop_labels
    return full_frame


def display_metadata_for_spatial_crop(spatial_crop: SpatialCrop) -> dict[str, object]:
    """Return napari layer metadata that records the displayed movie crop.

    Args:
        spatial_crop: Crop shown in napari display coordinates.

    Returns:
        Plain dictionary that can be stored in ``layer.metadata``.
    """
    return {
        DISPLAY_CROP_METADATA_KEY: {
            "axis0_start": spatial_crop.axis0_start,
            "axis0_stop": spatial_crop.axis0_stop,
            "axis1_start": spatial_crop.axis1_start,
            "axis1_stop": spatial_crop.axis1_stop,
            "original_shape": spatial_crop.original_shape,
            "source": spatial_crop.source,
        },
    }


def _spatial_crop_from_layer_metadata(layer: object) -> SpatialCrop | None:
    """Return the crop recorded on a napari layer, if present.

    Args:
        layer: napari layer or test layer.

    Returns:
        ``SpatialCrop`` or ``None`` when the layer has no twopy crop metadata.
    """
    metadata = (
        cast(_LayerWithMetadata, layer).metadata if hasattr(layer, "metadata") else None
    )
    if metadata is None and hasattr(layer, "options"):
        metadata = cast(_LayerWithOptions, layer).options.get("metadata")
    metadata_dict = string_key_mapping_or_none(metadata)
    if metadata_dict is None:
        return None
    crop_data = metadata_dict.get(DISPLAY_CROP_METADATA_KEY)
    crop_dict = string_key_mapping_or_none(crop_data)
    if crop_dict is None:
        return None
    raw_shape = int_tuple_or_none(crop_dict.get("original_shape"), length=2)
    if raw_shape is None:
        return None
    axis0_start = int_or_none(crop_dict.get("axis0_start"))
    axis0_stop = int_or_none(crop_dict.get("axis0_stop"))
    axis1_start = int_or_none(crop_dict.get("axis1_start"))
    axis1_stop = int_or_none(crop_dict.get("axis1_stop"))
    source = crop_dict.get("source")
    if (
        axis0_start is None
        or axis0_stop is None
        or axis1_start is None
        or axis1_stop is None
        or not isinstance(source, str)
    ):
        return None
    return SpatialCrop(
        axis0_start=axis0_start,
        axis0_stop=axis0_stop,
        axis1_start=axis1_start,
        axis1_stop=axis1_stop,
        original_shape=(raw_shape[0], raw_shape[1]),
        source=source,
    )
