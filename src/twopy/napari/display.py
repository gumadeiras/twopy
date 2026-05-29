"""Display-crop metadata helpers for the twopy napari adapter.

Inputs: spatial crops stored on napari layers.
Outputs: plain metadata dictionaries and validated crop objects.

Converted twopy arrays use Python image order matching MATLAB display of the
source data before napari sees them. This module only records which crop is
shown, so ROI edits can be expanded back into full-frame storage coordinates
without guessing.
"""

from typing import Protocol, cast

from twopy.spatial import SpatialCrop
from twopy.typing_guards import (
    int_or_none,
    int_tuple_or_none,
    string_key_mapping_or_none,
)

DISPLAY_CROP_METADATA_KEY = "twopy_display_spatial_crop"

__all__ = [
    "display_metadata_for_spatial_crop",
    "spatial_crop_from_layer_metadata",
]


class _LayerWithMetadata(Protocol):
    """Small protocol for napari-like layers with metadata."""

    metadata: object


class _LayerWithOptions(Protocol):
    """Small protocol for fake test layers that store napari kwargs."""

    options: dict[str, object]


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


def spatial_crop_from_layer_metadata(layer: object) -> SpatialCrop | None:
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
