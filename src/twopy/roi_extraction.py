"""Extract ROI masks from images without GUI or workflow side effects.

Inputs: movie-coordinate mean images, spatial shapes, optional region masks, and
small extraction configs.
Outputs: validated ``RoiSet`` objects ready for trace extraction or saving.

This module owns native twopy ROI discovery logic. UI code may collect a region
mask or display the result, but the extraction algorithms stay pure array
functions so scripts, tests, and future workflows share the same behavior.
"""

import math
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Literal, assert_never

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from twopy.roi import RoiSet, make_roi_set

__all__ = [
    "RoiExtractionConfig",
    "RoiExtractionMethod",
    "extract_rois_from_image",
    "grid_roi_set_microns",
    "grid_roi_set",
    "grid_size_pixels_from_microns",
    "watershed_roi_set",
]

RoiExtractionMethod = Literal["grid", "watershed"]
_EIGHT_CONNECTED = np.ones((3, 3), dtype=np.bool_)
_NEIGHBOR_OFFSETS = tuple(
    (axis0_delta, axis1_delta)
    for axis0_delta in (-1, 0, 1)
    for axis1_delta in (-1, 0, 1)
    if axis0_delta != 0 or axis1_delta != 0
)


@dataclass(frozen=True)
class RoiExtractionConfig:
    """Configuration for one pure image-to-ROI extraction pass.

    Args:
        method: Native twopy extraction method.
        region_mask: Optional boolean image marking the accepted region. ROIs
            are kept when their center of mass falls inside this mask.
        grid_size_pixels: Square grid width for ``method="grid"``.
        watershed_min_pixels: Minimum watershed ROI size after optional region
            restriction.
        watershed_smoothing_sigma: Optional Gaussian smoothing sigma before
            watershed seeding.
        label_prefix: Prefix for generated ROI labels. When omitted, the method
            name is used.

    The config is intentionally small. New extraction methods should add their
    own explicit fields instead of taking opaque keyword dictionaries.
    """

    method: RoiExtractionMethod
    region_mask: npt.NDArray[np.bool_] | None = None
    grid_size_pixels: int = 16
    watershed_min_pixels: int = 1
    watershed_smoothing_sigma: float = 0.0
    label_prefix: str | None = None


def extract_rois_from_image(
    image: npt.ArrayLike,
    config: RoiExtractionConfig,
) -> RoiSet:
    """Extract ROIs from one two-dimensional image.

    Args:
        image: Two-dimensional movie-coordinate image. Grid extraction only
            uses the shape; watershed extraction uses intensities.
        config: Extraction method and parameters.

    Returns:
        ``RoiSet`` with one mask per discovered ROI.

    Raises:
        ValueError: If the image, config, or region mask is invalid.

    This dispatcher is deliberately thin: lifecycle decisions such as loading a
    converted recording, asking a user to draw a region, saving HDF5, or adding
    napari layers belong outside this module.
    """
    image_values = _validated_image(image)
    region_mask = _validated_region_mask(config.region_mask, image_values.shape)
    if config.method == "grid":
        masks = _grid_masks(image_values, config)
    elif config.method == "watershed":
        masks = _watershed_masks(image_values, config)
    else:
        assert_never(config.method)
    masks = _apply_region_mask_by_center(masks, region_mask)
    label_prefix = config.label_prefix or config.method
    return make_roi_set(masks, labels=_labels(label_prefix, masks.shape[0]))


def grid_roi_set(
    spatial_shape: tuple[int, int],
    *,
    grid_size_pixels: int = 16,
    region_mask: npt.ArrayLike | None = None,
    label_prefix: str = "grid",
) -> RoiSet:
    """Create square grid ROIs over a movie-coordinate spatial shape.

    Args:
        spatial_shape: ``(axis0, axis1)`` image shape.
        grid_size_pixels: Width and height of each grid cell in pixels. Edge
            cells are smaller when the image size is not divisible by this
            value.
        region_mask: Optional boolean mask accepting ROIs by center of mass.
        label_prefix: Prefix for generated ROI labels.

    Returns:
        ``RoiSet`` covering the image or accepted region with non-overlapping
        rectangular masks.
    """
    shape = _validated_spatial_shape(spatial_shape)
    image = np.zeros(shape, dtype=np.float64)
    config = RoiExtractionConfig(
        method="grid",
        region_mask=_region_array_or_none(region_mask),
        grid_size_pixels=grid_size_pixels,
        label_prefix=label_prefix,
    )
    return extract_rois_from_image(image, config)


def grid_roi_set_microns(
    spatial_shape: tuple[int, int],
    *,
    micron_grid_size: float,
    pixel_size_um: float,
    region_mask: npt.ArrayLike | None = None,
    label_prefix: str = "grid",
) -> RoiSet:
    """Create square grid ROIs from a physical grid size.

    Args:
        spatial_shape: ``(axis0, axis1)`` image shape.
        micron_grid_size: Desired square grid width in microns.
        pixel_size_um: Calibrated microns per image pixel.
        region_mask: Optional boolean mask accepting ROIs by center of mass.
        label_prefix: Prefix for generated ROI labels.

    Returns:
        ``RoiSet`` created with ``floor(micron_grid_size / pixel_size_um)``
        pixels per grid cell.

    This helper connects calibration to native grid extraction without making
    ROI extraction know where calibration data came from.
    """
    grid_size_pixels = grid_size_pixels_from_microns(
        micron_grid_size,
        pixel_size_um,
    )
    return grid_roi_set(
        spatial_shape,
        grid_size_pixels=grid_size_pixels,
        region_mask=region_mask,
        label_prefix=label_prefix,
    )


def grid_size_pixels_from_microns(
    micron_grid_size: float,
    pixel_size_um: float,
) -> int:
    """Convert a physical grid width to native twopy pixel grid size.

    Args:
        micron_grid_size: Desired square grid width in microns.
        pixel_size_um: Calibrated microns per image pixel.

    Returns:
        Integer grid size in image pixels.

    Raises:
        ValueError: If either input is non-positive, non-finite, or the physical
            grid is smaller than one image pixel.

    twopy uses the same floor rule as psycho5 for physical grid sizes, but it
    takes the calibrated microns-per-pixel value directly.
    """
    micron_size = _validated_positive_float("micron_grid_size", micron_grid_size)
    pixel_size = _validated_positive_float("pixel_size_um", pixel_size_um)
    grid_size = math.floor(micron_size / pixel_size)
    if grid_size <= 0:
        msg = (
            f"micron_grid_size {micron_size} is smaller than one pixel "
            f"at pixel_size_um {pixel_size}"
        )
        raise ValueError(msg)
    return grid_size


def watershed_roi_set(
    image: npt.ArrayLike,
    *,
    region_mask: npt.ArrayLike | None = None,
    min_pixels: int = 1,
    smoothing_sigma: float = 0.0,
    label_prefix: str = "watershed",
) -> RoiSet:
    """Create watershed-style ROIs from bright structures in one image.

    Args:
        image: Two-dimensional mean or summary image.
        region_mask: Optional boolean mask accepting ROIs by center of mass.
        min_pixels: Minimum accepted ROI size.
        smoothing_sigma: Optional Gaussian smoothing sigma before seeding.
        label_prefix: Prefix for generated ROI labels.

    Returns:
        ``RoiSet`` containing one mask per watershed basin.

    twopy seeds high-intensity local maxima and floods neighboring pixels in
    descending intensity order. This mirrors psycho5's intent of segmenting
    bright structures while keeping the implementation local and inspectable.
    """
    config = RoiExtractionConfig(
        method="watershed",
        region_mask=_region_array_or_none(region_mask),
        watershed_min_pixels=min_pixels,
        watershed_smoothing_sigma=smoothing_sigma,
        label_prefix=label_prefix,
    )
    return extract_rois_from_image(image, config)


def _grid_masks(
    image: npt.NDArray[np.float64],
    config: RoiExtractionConfig,
) -> npt.NDArray[np.bool_]:
    """Return square grid masks over an image shape.

    Args:
        image: Validated two-dimensional image.
        config: Extraction config with ``grid_size_pixels``.

    Returns:
        Boolean masks shaped ``(rois, axis0, axis1)``.
    """
    grid_size = config.grid_size_pixels
    if grid_size <= 0:
        msg = f"grid_size_pixels must be positive; got {grid_size}"
        raise ValueError(msg)

    label_image = np.zeros(image.shape, dtype=np.int64)
    axis0_length, axis1_length = image.shape
    label = 1
    for axis0_start in range(0, axis0_length, grid_size):
        axis0_stop = min(axis0_start + grid_size, axis0_length)
        for axis1_start in range(0, axis1_length, grid_size):
            axis1_stop = min(axis1_start + grid_size, axis1_length)
            label_image[axis0_start:axis0_stop, axis1_start:axis1_stop] = label
            label += 1

    return _masks_from_positive_labels(label_image)


def _watershed_masks(
    image: npt.NDArray[np.float64],
    config: RoiExtractionConfig,
) -> npt.NDArray[np.bool_]:
    """Return watershed-style masks from a bright-structure image.

    Args:
        image: Validated two-dimensional image.
        config: Extraction config with watershed parameters.

    Returns:
        Boolean masks shaped ``(rois, axis0, axis1)``.
    """
    min_pixels = config.watershed_min_pixels
    if min_pixels <= 0:
        msg = f"watershed_min_pixels must be positive; got {min_pixels}"
        raise ValueError(msg)
    sigma = config.watershed_smoothing_sigma
    if sigma < 0:
        msg = f"watershed_smoothing_sigma cannot be negative; got {sigma}"
        raise ValueError(msg)

    working = image - np.min(image)
    if sigma > 0:
        working = ndimage.gaussian_filter(working, sigma=sigma)
    labels = _flood_from_local_maxima(working)
    masks = _masks_from_positive_labels(labels)
    return _filter_masks_by_pixel_count(masks, min_pixels)


def _flood_from_local_maxima(
    image: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """Assign every image pixel to a local-maximum basin.

    Args:
        image: Nonnegative two-dimensional image.

    Returns:
        Integer label image with one positive label per local-maximum basin.
    """
    local_maxima = image == ndimage.maximum_filter(
        image,
        footprint=_EIGHT_CONNECTED,
        mode="nearest",
    )
    if float(np.max(image)) > float(np.min(image)):
        local_maxima &= image > float(np.min(image))
    seed_labels, seed_count = ndimage.label(local_maxima, structure=_EIGHT_CONNECTED)
    if seed_count == 0:
        msg = "Watershed extraction found no local maxima"
        raise ValueError(msg)

    labels = np.asarray(seed_labels, dtype=np.int64)
    visited = labels > 0
    heap: list[tuple[float, int, int, int]] = []
    for axis0, axis1 in np.argwhere(visited):
        label = int(labels[axis0, axis1])
        heappush(heap, (-float(image[axis0, axis1]), label, int(axis0), int(axis1)))

    while heap:
        _, label, axis0, axis1 = heappop(heap)
        for axis0_delta, axis1_delta in _NEIGHBOR_OFFSETS:
            neighbor0 = axis0 + axis0_delta
            neighbor1 = axis1 + axis1_delta
            if (
                neighbor0 < 0
                or neighbor1 < 0
                or neighbor0 >= image.shape[0]
                or neighbor1 >= image.shape[1]
            ):
                continue
            if visited[neighbor0, neighbor1]:
                continue
            visited[neighbor0, neighbor1] = True
            labels[neighbor0, neighbor1] = label
            heappush(
                heap,
                (
                    -float(image[neighbor0, neighbor1]),
                    label,
                    neighbor0,
                    neighbor1,
                ),
            )

    return labels


def _apply_region_mask_by_center(
    masks: npt.NDArray[np.bool_],
    region_mask: npt.NDArray[np.bool_] | None,
) -> npt.NDArray[np.bool_]:
    """Keep masks whose center of mass falls inside a region mask.

    Args:
        masks: Candidate ROI masks.
        region_mask: Optional accepted-region mask.

    Returns:
        Filtered masks.
    """
    if region_mask is None:
        return masks

    keep = np.array(
        [_mask_center_is_inside_region(mask, region_mask) for mask in masks],
        dtype=np.bool_,
    )
    if not np.any(keep):
        msg = "ROI extraction region excluded every ROI"
        raise ValueError(msg)
    return masks[keep]


def _mask_center_is_inside_region(
    mask: npt.NDArray[np.bool_],
    region_mask: npt.NDArray[np.bool_],
) -> bool:
    """Return whether one mask center lies inside the accepted region.

    Args:
        mask: One candidate ROI mask.
        region_mask: Accepted-region mask with the same shape.

    Returns:
        ``True`` when the nearest center-of-mass pixel is accepted.
    """
    coordinates = np.argwhere(mask)
    center = np.rint(coordinates.mean(axis=0)).astype(np.int64)
    center[0] = np.clip(center[0], 0, region_mask.shape[0] - 1)
    center[1] = np.clip(center[1], 0, region_mask.shape[1] - 1)
    return bool(region_mask[center[0], center[1]])


def _masks_from_positive_labels(
    label_image: npt.NDArray[np.int64],
) -> npt.NDArray[np.bool_]:
    """Convert positive labels to stacked ROI masks.

    Args:
        label_image: Two-dimensional integer label image.

    Returns:
        Boolean masks sorted by label value.
    """
    label_values = tuple(int(value) for value in np.unique(label_image) if value > 0)
    if len(label_values) == 0:
        msg = "ROI extraction produced no positive labels"
        raise ValueError(msg)
    return np.stack(tuple(label_image == label for label in label_values))


def _filter_masks_by_pixel_count(
    masks: npt.NDArray[np.bool_],
    min_pixels: int,
) -> npt.NDArray[np.bool_]:
    """Drop masks smaller than a minimum pixel count.

    Args:
        masks: Candidate ROI masks.
        min_pixels: Minimum accepted pixel count.

    Returns:
        Filtered masks.
    """
    pixel_counts = masks.reshape(masks.shape[0], -1).sum(axis=1)
    keep = pixel_counts >= min_pixels
    if not np.any(keep):
        msg = "ROI extraction produced no ROIs after minimum-size filtering"
        raise ValueError(msg)
    return masks[keep]


def _validated_image(image: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a finite two-dimensional image as float64.

    Args:
        image: Candidate extraction image.

    Returns:
        Validated image values.
    """
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2:
        msg = f"ROI extraction image must be two-dimensional; got {values.shape}"
        raise ValueError(msg)
    if values.size == 0:
        msg = "ROI extraction image cannot be empty"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "ROI extraction image must contain only finite values"
        raise ValueError(msg)
    return values


def _validated_spatial_shape(spatial_shape: tuple[int, int]) -> tuple[int, int]:
    """Validate a two-dimensional spatial shape.

    Args:
        spatial_shape: Candidate movie spatial shape.

    Returns:
        Validated shape tuple.
    """
    if len(spatial_shape) != 2 or spatial_shape[0] <= 0 or spatial_shape[1] <= 0:
        msg = f"ROI extraction spatial shape must be positive 2-D; got {spatial_shape}"
        raise ValueError(msg)
    return spatial_shape


def _validated_positive_float(key: str, value: float) -> float:
    """Return one finite positive float.

    Args:
        key: Field name, used for clear errors.
        value: Candidate numeric value.

    Returns:
        Validated float value.
    """
    if not math.isfinite(value) or value <= 0:
        msg = f"{key} must be a finite positive number; got {value!r}"
        raise ValueError(msg)
    return value


def _region_array_or_none(
    region_mask: npt.ArrayLike | None,
) -> npt.NDArray[np.bool_] | None:
    """Convert an optional region mask to a boolean array.

    Args:
        region_mask: Optional accepted-region mask.

    Returns:
        Boolean array or ``None``.
    """
    if region_mask is None:
        return None
    return np.asarray(region_mask, dtype=np.bool_)


def _validated_region_mask(
    region_mask: npt.NDArray[np.bool_] | None,
    image_shape: tuple[int, int],
) -> npt.NDArray[np.bool_] | None:
    """Validate an optional accepted-region mask.

    Args:
        region_mask: Optional boolean region mask.
        image_shape: Required mask shape.

    Returns:
        The validated mask or ``None``.
    """
    if region_mask is None:
        return None
    if region_mask.shape != image_shape:
        msg = (
            f"ROI extraction region mask shape {region_mask.shape} does not "
            f"match image shape {image_shape}"
        )
        raise ValueError(msg)
    if not np.any(region_mask):
        msg = "ROI extraction region mask cannot be empty"
        raise ValueError(msg)
    return region_mask


def _labels(label_prefix: str, count: int) -> tuple[str, ...]:
    """Create deterministic ROI labels.

    Args:
        label_prefix: Label prefix.
        count: Number of labels.

    Returns:
        Tuple of labels.
    """
    return tuple(f"{label_prefix}_{index + 1:04d}" for index in range(count))
