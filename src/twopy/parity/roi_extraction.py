"""psycho5 ROI extraction algorithms for comparison-only parity checks.

Inputs: small images, MATLAB-style watershed labels, and psycho5 grid
parameters.
Outputs: label images or ``RoiSet`` objects that mirror psycho5 ROI extraction
where the MATLAB primitive output is available.

Normal twopy ROI discovery lives in ``twopy.roi_extraction``. This module keeps
source-faithful psycho5 behavior isolated so parity scripts can compare native
twopy choices against historical MATLAB analysis without importing psycho5
concepts into the normal workflow.
"""

import numpy as np
import numpy.typing as npt

from twopy.roi import RoiSet, make_roi_set_from_label_image

__all__ = [
    "psycho5_grid_pixel_size",
    "psycho5_grid_roi_label_image",
    "psycho5_grid_roi_set",
    "psycho5_watershed_image_from_preseg",
    "psycho5_watershed_roi_set_from_preseg",
]


def psycho5_grid_pixel_size(
    *,
    micron_grid_size: float,
    zoom_factor: float,
    image_axis1_size: int,
    one_x_pixel_per_micron_256_axis1: float,
) -> int:
    """Return the pixel grid size used by psycho5 ``GridRoiExtraction``.

    Args:
        micron_grid_size: psycho5 ``micronGridSize`` argument.
        zoom_factor: ``imageDescription.acq.zoomFactor``.
        image_axis1_size: Number of pixels along movie axis 1.
        one_x_pixel_per_micron_256_axis1: Lab calibration value
            ``twoPhotonOneXPixelPerMicron256PixelXAxisRes`` from psycho5
            ``GetSystemConfiguration``.

    Returns:
        Integer grid width in pixels after psycho5's ``floor`` conversion.

    psycho5 derives pixel size from a 256-pixel x-axis calibration, zoom, and
    current image width. twopy's native grid API takes pixel size directly so
    scripts do not depend on lab-local MATLAB system configuration.
    """
    if image_axis1_size <= 0:
        msg = f"image_axis1_size must be positive. Got {image_axis1_size}"
        raise ValueError(msg)
    pixel_size = np.floor(
        micron_grid_size
        * one_x_pixel_per_micron_256_axis1
        * zoom_factor
        * (256 / image_axis1_size),
    )
    resolved = int(pixel_size)
    if resolved <= 0:
        msg = f"psycho5 grid pixel size must be positive. Got {resolved}"
        raise ValueError(msg)
    return resolved


def psycho5_grid_roi_label_image(
    spatial_shape: tuple[int, int],
    pixel_grid_size: int,
) -> npt.NDArray[np.int64]:
    """Return the label image produced by psycho5 grid extraction.

    Args:
        spatial_shape: Two-dimensional movie-coordinate image shape.
        pixel_grid_size: Pixel block size after psycho5's micron-to-pixel
            conversion.

    Returns:
        Integer label image where ``0`` is background and positive integers are
        ROIs.

    This follows psycho5's ``im2col(..., 'distinct')`` block order: blocks are
    numbered down axis 0 first, then across axis 1. Edge blocks are retained
    when the image shape is not divisible by the grid size, matching MATLAB's
    padded distinct-block behavior after psycho5 removes padded zeros.
    """
    axis0_length, axis1_length = _validated_spatial_shape(spatial_shape)
    if pixel_grid_size <= 0:
        msg = f"pixel_grid_size must be positive. Got {pixel_grid_size}"
        raise ValueError(msg)

    label_image = np.zeros((axis0_length, axis1_length), dtype=np.int64)
    label = 1
    for axis1_start in range(0, axis1_length, pixel_grid_size):
        axis1_stop = min(axis1_start + pixel_grid_size, axis1_length)
        for axis0_start in range(0, axis0_length, pixel_grid_size):
            axis0_stop = min(axis0_start + pixel_grid_size, axis0_length)
            label_image[axis0_start:axis0_stop, axis1_start:axis1_stop] = label
            label += 1
    return label_image


def psycho5_grid_roi_set(
    spatial_shape: tuple[int, int],
    pixel_grid_size: int,
) -> RoiSet:
    """Return a ``RoiSet`` from psycho5 grid ROI labels.

    Args:
        spatial_shape: Two-dimensional movie-coordinate image shape.
        pixel_grid_size: Pixel block size used by psycho5.

    Returns:
        ``RoiSet`` with labels named ``psycho5_grid_####``.
    """
    return make_roi_set_from_label_image(
        psycho5_grid_roi_label_image(spatial_shape, pixel_grid_size),
        label_prefix="psycho5_grid",
    )


def psycho5_watershed_image_from_preseg(
    image: npt.ArrayLike,
    preseg: npt.ArrayLike,
) -> npt.NDArray[np.int64]:
    """Apply psycho5 ``WatershedImage`` border assignment to MATLAB labels.

    Args:
        image: Original two-dimensional image passed to psycho5
            ``WatershedImage``.
        preseg: Label image produced by MATLAB ``watershed(-image, 8)`` before
            psycho5 fills border pixels.

    Returns:
        Integer label image after psycho5 assigns ``preseg == 0`` pixels to the
        nearest segment peak.

    This is the exact source-faithful part of psycho5 ``WatershedImage`` that
    can be represented without MATLAB. Exact end-to-end parity still requires
    MATLAB's own ``watershed`` output as ``preseg`` because SciPy/skimage do not
    promise identical basin labeling or plateau behavior.
    """
    image_values = _validated_image(image)
    preseg_labels = _validated_preseg(preseg, image_values.shape)
    max_label = int(np.max(preseg_labels))
    if max_label <= 0:
        msg = "psycho5 watershed preseg must contain positive labels"
        raise ValueError(msg)

    peak_coordinates = tuple(
        _psycho5_segment_peak(image_values, preseg_labels, label)
        for label in range(1, max_label + 1)
    )
    assigned_borders = _nearest_peak_labels(image_values.shape, peak_coordinates)
    return np.where(preseg_labels == 0, assigned_borders, preseg_labels).astype(
        np.int64,
    )


def psycho5_watershed_roi_set_from_preseg(
    image: npt.ArrayLike,
    preseg: npt.ArrayLike,
) -> RoiSet:
    """Return a ``RoiSet`` from psycho5 watershed labels.

    Args:
        image: Original image passed to psycho5.
        preseg: MATLAB ``watershed(-image, 8)`` labels before border fill.

    Returns:
        ``RoiSet`` with labels named ``psycho5_watershed_####``.
    """
    return make_roi_set_from_label_image(
        psycho5_watershed_image_from_preseg(image, preseg),
        label_prefix="psycho5_watershed",
    )


def _psycho5_segment_peak(
    image: npt.NDArray[np.float64],
    preseg: npt.NDArray[np.int64],
    label: int,
) -> tuple[int, int]:
    """Return psycho5's peak coordinate for one watershed segment.

    Args:
        image: Original image.
        preseg: MATLAB watershed labels.
        label: Positive segment label.

    Returns:
        ``(axis0, axis1)`` peak coordinate.

    psycho5 multiplies the image by one segment mask, then takes the first
    maximum in MATLAB column-major order. That means zero-valued pixels outside
    the segment can win when all in-segment values are negative. This function
    preserves that source behavior instead of correcting it.
    """
    segment_image = image * (preseg == label)
    max_value = np.max(segment_image)
    flat_matches = np.flatnonzero(
        np.ravel(segment_image == max_value, order="F"),
    )
    if flat_matches.size == 0:
        msg = f"psycho5 watershed segment {label} has no peak"
        raise ValueError(msg)
    axis0, axis1 = np.unravel_index(int(flat_matches[0]), image.shape, order="F")
    return int(axis0), int(axis1)


def _nearest_peak_labels(
    shape: tuple[int, int],
    peak_coordinates: tuple[tuple[int, int], ...],
) -> npt.NDArray[np.int64]:
    """Return each pixel's nearest psycho5 segment-peak label.

    Args:
        shape: Image shape.
        peak_coordinates: One peak coordinate per positive segment label.

    Returns:
        Label image assigning every pixel to the nearest peak, with ties going
        to the lowest label like MATLAB ``min`` over columns.
    """
    axis0_grid, axis1_grid = np.indices(shape)
    distances = np.stack(
        tuple(
            (axis0_grid - peak0) ** 2 + (axis1_grid - peak1) ** 2
            for peak0, peak1 in peak_coordinates
        ),
        axis=-1,
    )
    return np.argmin(distances, axis=-1).astype(np.int64) + 1


def _validated_image(image: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a finite two-dimensional image as float64.

    Args:
        image: Candidate psycho5 ROI image.

    Returns:
        Validated image values.
    """
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2:
        msg = f"psycho5 ROI image must be two-dimensional. Got {values.shape}"
        raise ValueError(msg)
    if values.size == 0:
        msg = "psycho5 ROI image cannot be empty"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "psycho5 ROI image must contain only finite values"
        raise ValueError(msg)
    return values


def _validated_preseg(
    preseg: npt.ArrayLike,
    image_shape: tuple[int, int],
) -> npt.NDArray[np.int64]:
    """Return a MATLAB watershed preseg label image as integers.

    Args:
        preseg: Candidate MATLAB watershed labels.
        image_shape: Required image shape.

    Returns:
        Integer label image.
    """
    values = np.asarray(preseg, dtype=np.float64)
    if values.shape != image_shape:
        msg = f"psycho5 watershed preseg shape {values.shape} must match {image_shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "psycho5 watershed preseg must contain only finite values"
        raise ValueError(msg)
    rounded = np.rint(values)
    if not np.allclose(values, rounded):
        msg = "psycho5 watershed preseg labels must be integer-like"
        raise ValueError(msg)
    labels = rounded.astype(np.int64)
    if np.any(labels < 0):
        msg = "psycho5 watershed preseg labels cannot be negative"
        raise ValueError(msg)
    return labels


def _validated_spatial_shape(spatial_shape: tuple[int, int]) -> tuple[int, int]:
    """Validate a two-dimensional spatial shape.

    Args:
        spatial_shape: Candidate movie spatial shape.

    Returns:
        Validated shape tuple.
    """
    if len(spatial_shape) != 2 or spatial_shape[0] <= 0 or spatial_shape[1] <= 0:
        msg = f"psycho5 ROI spatial shape must be positive 2-D. Got {spatial_shape}"
        raise ValueError(msg)
    return spatial_shape
