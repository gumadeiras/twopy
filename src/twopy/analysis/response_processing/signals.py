"""Reusable numeric kernels for response post-processing.

Inputs: frame-by-channel response arrays and explicit signal-processing
parameters.
Outputs: arrays with the same shape and NaN positions handled audibly.

The functions in this module know nothing about ROIs, epochs, napari, HDF5, or
workflow state. They are pure signal transforms used by higher-level response
processing code.
"""

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, savgol_filter, sosfiltfilt

__all__ = [
    "butterworth_low_pass",
    "nan_aware_moving_average",
    "nan_aware_savgol_filter",
]


def nan_aware_moving_average(
    values: npt.NDArray[np.float64],
    *,
    window_frames: int,
) -> npt.NDArray[np.float64]:
    """Smooth frame-by-channel values with a NaN-aware moving average.

    Args:
        values: One-dimensional or ``(frames, channels)`` response values.
        window_frames: Number of frames in the moving-average window.

    Returns:
        Smoothed values with the same shape as ``values``.

    Raises:
        ValueError: If ``window_frames`` is less than one or the input rank is
            not one or two.

    NaN samples do not contribute to neighboring averages. Output positions
    whose whole window is NaN remain NaN.
    """
    if window_frames < 1:
        msg = f"window_frames must be at least 1; got {window_frames}"
        raise ValueError(msg)
    matrix, was_1d = _as_frame_channel_matrix(values)
    if window_frames == 1:
        return _restore_shape(matrix.copy(), was_1d=was_1d)

    kernel = np.ones(window_frames, dtype=np.float64)
    finite = np.isfinite(matrix)
    weighted_values = np.where(finite, matrix, 0.0)
    counts = _convolve_columns(finite.astype(np.float64), kernel)
    sums = _convolve_columns(weighted_values, kernel)
    smoothed = np.full(matrix.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=smoothed, where=counts > 0)
    return _restore_shape(smoothed, was_1d=was_1d)


def butterworth_low_pass(
    values: npt.NDArray[np.float64],
    *,
    sample_rate_hz: float,
    cutoff_hz: float,
    order: int = 2,
) -> npt.NDArray[np.float64]:
    """Apply a zero-phase Butterworth low-pass filter to response values.

    Args:
        values: One-dimensional or ``(frames, channels)`` response values.
        sample_rate_hz: Sampling rate in hertz.
        cutoff_hz: Low-pass cutoff in hertz. Must be below Nyquist.
        order: Butterworth filter order.

    Returns:
        Filtered values with the same shape as ``values``.

    Raises:
        ValueError: If the filter parameters or input rank are invalid.

    Each finite contiguous run is filtered independently so motion-masked NaNs
    remain NaN and do not leak into nearby valid frames. Runs too short for
    stable zero-phase filtering are copied unchanged.
    """
    _validate_low_pass_parameters(
        sample_rate_hz=sample_rate_hz,
        cutoff_hz=cutoff_hz,
        order=order,
    )
    matrix, was_1d = _as_frame_channel_matrix(values)
    filtered = np.full(matrix.shape, np.nan, dtype=np.float64)
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sample_rate_hz, output="sos")
    for column_index in range(matrix.shape[1]):
        column = matrix[:, column_index]
        for start, stop in _finite_runs(np.isfinite(column)):
            segment = column[start:stop]
            filtered[start:stop, column_index] = _filter_segment_or_copy(segment, sos)
    return _restore_shape(filtered, was_1d=was_1d)


def nan_aware_savgol_filter(
    values: npt.NDArray[np.float64],
    *,
    window_frames: int = 7,
    polynomial_order: int = 2,
) -> npt.NDArray[np.float64]:
    """Smooth values with a NaN-aware Savitzky-Golay filter.

    Args:
        values: One-dimensional or ``(frames, channels)`` response values.
        window_frames: Odd number of frames in the local regression window.
        polynomial_order: Polynomial order fit inside each local window.

    Returns:
        Smoothed values with the same shape as ``values``.

    Raises:
        ValueError: If parameters or input rank are invalid.

    Each finite contiguous run is smoothed independently so masked samples stay
    NaN. Runs shorter than the requested window are copied unchanged because a
    smaller implicit window would silently change the selected smoothing scale.
    """
    _validate_savgol_parameters(
        window_frames=window_frames,
        polynomial_order=polynomial_order,
    )
    matrix, was_1d = _as_frame_channel_matrix(values)
    smoothed = np.full(matrix.shape, np.nan, dtype=np.float64)
    for column_index in range(matrix.shape[1]):
        column = matrix[:, column_index]
        for start, stop in _finite_runs(np.isfinite(column)):
            segment = column[start:stop]
            if segment.size < window_frames:
                smoothed[start:stop, column_index] = segment
                continue
            smoothed[start:stop, column_index] = savgol_filter(
                segment,
                window_length=window_frames,
                polyorder=polynomial_order,
                mode="interp",
            )
    return _restore_shape(smoothed, was_1d=was_1d)


def _as_frame_channel_matrix(
    values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], bool]:
    """Return values as a ``(frames, channels)`` float64 matrix."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(-1, 1), True
    if array.ndim == 2:
        return array, False
    msg = f"response values must be 1D or 2D; got shape {array.shape}"
    raise ValueError(msg)


def _restore_shape(
    matrix: npt.NDArray[np.float64],
    *,
    was_1d: bool,
) -> npt.NDArray[np.float64]:
    """Return a filtered matrix with the original vector rank."""
    if was_1d:
        return matrix[:, 0]
    return matrix


def _convolve_columns(
    values: npt.NDArray[np.float64],
    kernel: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convolve each column with the same one-dimensional kernel."""
    convolved = np.empty(values.shape, dtype=np.float64)
    for column_index in range(values.shape[1]):
        column = np.asarray(
            np.convolve(
                values[:, column_index],
                kernel,
                mode="same",
            ),
            dtype=np.float64,
        )
        convolved[:, column_index] = _center_crop(column, values.shape[0])
    return convolved


def _center_crop(
    values: npt.NDArray[np.float64],
    length: int,
) -> npt.NDArray[np.float64]:
    """Return the centered part of a convolution result."""
    if values.shape[0] == length:
        return values
    start = (values.shape[0] - length) // 2
    return values[start : start + length]


def _validate_low_pass_parameters(
    *,
    sample_rate_hz: float,
    cutoff_hz: float,
    order: int,
) -> None:
    """Validate Butterworth filter parameters."""
    if sample_rate_hz <= 0:
        msg = f"sample_rate_hz must be positive; got {sample_rate_hz}"
        raise ValueError(msg)
    if cutoff_hz <= 0:
        msg = f"cutoff_hz must be positive; got {cutoff_hz}"
        raise ValueError(msg)
    nyquist_hz = sample_rate_hz / 2.0
    if cutoff_hz >= nyquist_hz:
        msg = (
            "cutoff_hz must be below Nyquist; "
            f"got {cutoff_hz} Hz for {sample_rate_hz} Hz data"
        )
        raise ValueError(msg)
    if order < 1:
        msg = f"order must be at least 1; got {order}"
        raise ValueError(msg)


def _validate_savgol_parameters(
    *,
    window_frames: int,
    polynomial_order: int,
) -> None:
    """Validate Savitzky-Golay smoothing parameters."""
    if window_frames < 1:
        msg = f"window_frames must be at least 1; got {window_frames}"
        raise ValueError(msg)
    if window_frames % 2 == 0:
        msg = f"window_frames must be odd for Savitzky-Golay; got {window_frames}"
        raise ValueError(msg)
    if polynomial_order < 0:
        msg = f"polynomial_order must be non-negative; got {polynomial_order}"
        raise ValueError(msg)
    if polynomial_order >= window_frames:
        msg = (
            "polynomial_order must be below window_frames; "
            f"got {polynomial_order} and {window_frames}"
        )
        raise ValueError(msg)


def _finite_runs(mask: npt.NDArray[np.bool_]) -> Iterator[tuple[int, int]]:
    """Yield half-open contiguous ``True`` runs from one Boolean mask."""
    start: int | None = None
    for index, value in enumerate(mask):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, int(mask.size)


def _filter_segment_or_copy(
    segment: npt.NDArray[np.float64],
    sos: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Filter one finite segment, copying it when it is too short."""
    try:
        return sosfiltfilt(sos, segment)
    except ValueError:
        return segment.copy()
