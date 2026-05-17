"""Compute stimulus response heatmaps from converted two-photon movies.

Inputs: converted recordings, photodiode-aligned epoch windows, and response-map
options.
Outputs: one mean movie-coordinate response map per stimulus epoch.

This module is GUI-independent. Napari widgets may expose the options and draw
the maps, but the response-map data contract and validation live here.

The calculation is intentionally image-based rather than ROI-based. For each
photodiode-aligned epoch window, twopy averages a short pre-epoch baseline image
and the response frames inside that epoch, then computes signed dF/F as
``(response - baseline) / max(abs(baseline), denominator_floor)``. The
denominator floor is the foreground threshold from the mean image, so dim
background pixels cannot dominate through division by near-zero baseline values.
Motion-artifact frames and below-threshold background pixels are masked before
trial averaging.

Two spatial methods are available. Pixel mode computes dF/F at each foreground
pixel and can apply NaN-aware Gaussian smoothing. Window mode computes one dF/F
value per square window, steps windows by the configured stride, paints each
window value back over the covered pixels, and averages overlapping windows.
Window starts always include the trailing edge so valid pixels are covered even
when the stride does not divide the crop size.

Persisted response values are normalized once after epoch averaging by the
largest absolute finite response across all epochs. ``ResponseMapData`` stores
that divisor in ``response_scale`` so users can audit or recover the original
dF/F units. Display color limits are handled separately by napari plotting code.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from twopy.analysis.epoch_mapping import interpolate_stimulus_epochs_to_frame_windows
from twopy.analysis.trials import EpochFrameWindow
from twopy.converted import RecordingData, recording_frame_rate_hz
from twopy.spatial import SpatialCrop

__all__ = [
    "EpochResponseMap",
    "ResponseMapData",
    "ResponseMapMode",
    "ResponseMapOptions",
    "compute_recording_response_maps",
    "load_response_map_data",
    "save_response_map_data",
]

RESPONSE_MAP_FILE_FORMAT = "twopy-response-heatmaps"
ResponseMapMode = Literal["pixel", "window"]


@dataclass(frozen=True)
class ResponseMapOptions:
    """Options for building epoch response heatmaps.

    Args:
        mode: ``"pixel"`` computes one dF/F value per pixel. ``"window"``
            computes one dF/F value per square moving window and paints it back
            onto covered pixels.
        pixel_smoothing_sigma: Gaussian sigma in pixels for pixel mode.
        window_size_pixels: Square window width for window mode.
        window_stride_pixels: Step between window starts for window mode.
        baseline_sample_seconds: Seconds before each epoch onset used as the
            local baseline image.
        foreground_percentile: Mean-image percentile below which pixels are
            excluded from response-map scoring.

    The foreground threshold also becomes the dF/F denominator floor. This keeps
    dim background pixels from dominating heatmaps through division by near-zero
    baseline values.

    Response values are normalized after epoch averaging so the largest absolute
    finite response across all epochs has magnitude 1. The original dF/F divisor
    is stored on ``ResponseMapData``.

    Pixel mode is the default because it preserves spatial detail; its default
    two-pixel Gaussian smoothing reduces isolated noisy pixels without changing
    the saved analysis contract. Window mode is useful when users want explicit
    block averaging before dF/F. Window size and stride are constrained by the
    displayed crop, and stride may not exceed size.
    """

    mode: ResponseMapMode = "pixel"
    pixel_smoothing_sigma: float = 2.0
    window_size_pixels: int = 3
    window_stride_pixels: int = 1
    baseline_sample_seconds: float = 1.0
    foreground_percentile: float = 25.0


@dataclass(frozen=True)
class EpochResponseMap:
    """Mean response map for one stimulus epoch.

    Args:
        epoch_name: Human-readable stimulus epoch name.
        epoch_number: One-based stimulus epoch number.
        response_values: Movie-coordinate signed response map for the displayed
            crop. Values are normalized by ``ResponseMapData.response_scale``
            after epoch averaging; multiply by that scale to recover the
            original dF/F magnitude.
        trial_count: Number of trials averaged into the map.
    """

    epoch_name: str
    epoch_number: int
    response_values: npt.NDArray[np.float64]
    trial_count: int


@dataclass(frozen=True)
class ResponseMapData:
    """Response heatmap data for one recording and option set.

    Args:
        mean_image: Movie-coordinate mean image for the same crop as all maps.
        epochs: One mean response map per stimulus epoch.
        options: Options used to compute the maps.
        spatial_crop: Movie-coordinate crop represented by the maps.
        response_scale: Original absolute dF/F value that corresponds to a
            persisted heatmap value of 1.
    """

    mean_image: npt.NDArray[np.float64]
    epochs: tuple[EpochResponseMap, ...]
    options: ResponseMapOptions
    spatial_crop: SpatialCrop
    response_scale: float


def compute_recording_response_maps(
    recording: RecordingData,
    *,
    epoch_windows: Sequence[EpochFrameWindow] | None = None,
    options: ResponseMapOptions | None = None,
) -> ResponseMapData:
    """Compute response heatmaps for one converted recording.

    Args:
        recording: Loaded converted recording.
        epoch_windows: Optional stimulus windows. When omitted, twopy derives
            photodiode-aligned windows from converted metadata.
        options: Optional response-map settings.

    Returns:
        Mean image plus one normalized response map per epoch, all in the
        alignment-valid crop represented by ``recording.alignment_valid_crop``.

    Raises:
        ValueError: If timing, foreground, or spatial-window settings are
            invalid.

    This reads the converted aligned movie for the valid crop, not microscope
    source MAT/TIFF files. Callers can pass explicit epoch windows for tests or
    audits; normal workflows let twopy derive photodiode-aligned windows from
    converted synchronization metadata.
    """
    resolved_options = options or ResponseMapOptions()
    resolved_windows = (
        tuple(epoch_windows)
        if epoch_windows is not None
        else interpolate_stimulus_epochs_to_frame_windows(recording).windows
    )
    _validate_options(resolved_options, recording.alignment_valid_crop.shape)
    if len(resolved_windows) == 0:
        msg = "Response maps require at least one epoch window."
        raise ValueError(msg)

    crop = recording.alignment_valid_crop
    mean_image = crop.crop_image(recording.mean_image).astype(np.float64, copy=False)
    threshold = _foreground_threshold(mean_image, resolved_options)
    foreground_mask = mean_image >= threshold
    if not np.any(foreground_mask):
        msg = "Response-map foreground mask is empty."
        raise ValueError(msg)

    frame_rate_hz = recording_frame_rate_hz(recording)
    baseline_frames = max(
        1,
        int(round(resolved_options.baseline_sample_seconds * frame_rate_hz)),
    )
    _validate_epoch_window_frames(
        resolved_windows,
        frame_count=recording.movie.shape[0],
        baseline_frames=baseline_frames,
    )

    trial_maps_by_epoch: dict[tuple[int, str], list[npt.NDArray[np.float64]]] = {}
    epoch_order: list[tuple[int, str]] = []
    for epoch_window in resolved_windows:
        key = (epoch_window.epoch_number, epoch_window.epoch_name)
        if key not in trial_maps_by_epoch:
            epoch_order.append(key)
            trial_maps_by_epoch[key] = []
        trial_maps_by_epoch[key].append(
            _trial_response_map(
                recording,
                crop,
                foreground_mask,
                epoch_window,
                baseline_frames=baseline_frames,
                denominator_floor=threshold,
                options=resolved_options,
            ),
        )

    raw_epochs = tuple(
        _epoch_response_map(
            epoch_number=epoch_number,
            epoch_name=epoch_name,
            trial_maps=tuple(trial_maps_by_epoch[(epoch_number, epoch_name)]),
        )
        for epoch_number, epoch_name in sorted(epoch_order)
    )
    epochs, response_scale = _normalize_epoch_response_maps(raw_epochs)
    return ResponseMapData(
        mean_image=mean_image,
        epochs=epochs,
        options=resolved_options,
        spatial_crop=crop,
        response_scale=response_scale,
    )


def save_response_map_data(path: Path, map_data: ResponseMapData) -> None:
    """Save response heatmap data to a recording-level HDF5 file.

    Args:
        path: Destination ``response_heatmaps.h5`` path.
        map_data: Computed response heatmaps to persist.

    Returns:
        None.

    The file is independent from ROI analysis outputs because response maps are
    recording-level movie products. Values are stored as normalized signed maps
    with the original dF/F scale in ``response_scale`` for audit.
    """
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = RESPONSE_MAP_FILE_FORMAT
        h5_file.attrs["response_scale"] = map_data.response_scale
        _write_response_map_options(h5_file.create_group("options"), map_data.options)
        _write_spatial_crop(h5_file.create_group("spatial_crop"), map_data.spatial_crop)
        h5_file.create_dataset(
            "mean_image",
            data=map_data.mean_image,
            compression="gzip",
        )
        epochs_group = h5_file.create_group("epochs")
        for index, epoch in enumerate(map_data.epochs):
            epoch_group = epochs_group.create_group(f"{index:04d}")
            epoch_group.attrs["epoch_name"] = epoch.epoch_name
            epoch_group.attrs["epoch_number"] = epoch.epoch_number
            epoch_group.attrs["trial_count"] = epoch.trial_count
            epoch_group.create_dataset(
                "response_values",
                data=epoch.response_values,
                compression="gzip",
            )


def load_response_map_data(path: Path) -> ResponseMapData:
    """Load response heatmap data from a recording-level HDF5 file.

    Args:
        path: HDF5 path written by ``save_response_map_data``.

    Returns:
        Persisted response heatmap data.

    Raises:
        ValueError: If the file is not a twopy response heatmap file.
    """
    input_path = path.expanduser()
    with h5py.File(input_path, "r") as h5_file:
        _require_response_map_format(h5_file, input_path)
        options = _read_response_map_options(h5_file["options"])
        spatial_crop = _read_spatial_crop(h5_file["spatial_crop"])
        mean_image = np.asarray(h5_file["mean_image"], dtype=np.float64)
        epochs_group = h5_file["epochs"]
        epochs = tuple(
            _read_epoch_response_map(epochs_group[name])
            for name in sorted(epochs_group.keys())
        )
        return ResponseMapData(
            mean_image=mean_image,
            epochs=epochs,
            options=options,
            spatial_crop=spatial_crop,
            response_scale=float(h5_file.attrs["response_scale"]),
        )


def _write_response_map_options(
    group: h5py.Group,
    options: ResponseMapOptions,
) -> None:
    """Write response-map options as plain HDF5 attributes."""
    group.attrs["mode"] = options.mode
    group.attrs["pixel_smoothing_sigma"] = options.pixel_smoothing_sigma
    group.attrs["window_size_pixels"] = options.window_size_pixels
    group.attrs["window_stride_pixels"] = options.window_stride_pixels
    group.attrs["baseline_sample_seconds"] = options.baseline_sample_seconds
    group.attrs["foreground_percentile"] = options.foreground_percentile


def _read_response_map_options(group: h5py.Group) -> ResponseMapOptions:
    """Read response-map options from plain HDF5 attributes."""
    mode = _response_map_mode(_string_attr(group, "mode"))
    return ResponseMapOptions(
        mode=mode,
        pixel_smoothing_sigma=float(group.attrs["pixel_smoothing_sigma"]),
        window_size_pixels=int(group.attrs["window_size_pixels"]),
        window_stride_pixels=int(group.attrs["window_stride_pixels"]),
        baseline_sample_seconds=float(group.attrs["baseline_sample_seconds"]),
        foreground_percentile=float(group.attrs["foreground_percentile"]),
    )


def _write_spatial_crop(group: h5py.Group, crop: SpatialCrop) -> None:
    """Write the spatial crop represented by heatmap arrays."""
    group.attrs["axis0_start"] = crop.axis0_start
    group.attrs["axis0_stop"] = crop.axis0_stop
    group.attrs["axis1_start"] = crop.axis1_start
    group.attrs["axis1_stop"] = crop.axis1_stop
    group.attrs["original_shape"] = crop.original_shape
    group.attrs["source"] = crop.source


def _read_spatial_crop(group: h5py.Group) -> SpatialCrop:
    """Read the spatial crop represented by heatmap arrays."""
    original_shape = tuple(int(value) for value in group.attrs["original_shape"])
    if len(original_shape) != 2:
        msg = f"Response heatmap crop original_shape must have 2 axes: {original_shape}"
        raise ValueError(msg)
    two_axis_shape = (original_shape[0], original_shape[1])
    return SpatialCrop(
        axis0_start=int(group.attrs["axis0_start"]),
        axis0_stop=int(group.attrs["axis0_stop"]),
        axis1_start=int(group.attrs["axis1_start"]),
        axis1_stop=int(group.attrs["axis1_stop"]),
        original_shape=two_axis_shape,
        source=_string_attr(group, "source"),
    )


def _read_epoch_response_map(group: h5py.Group) -> EpochResponseMap:
    """Read one persisted epoch response map."""
    return EpochResponseMap(
        epoch_name=_string_attr(group, "epoch_name"),
        epoch_number=int(group.attrs["epoch_number"]),
        response_values=np.asarray(group["response_values"], dtype=np.float64),
        trial_count=int(group.attrs["trial_count"]),
    )


def _require_response_map_format(h5_file: h5py.File, path: Path) -> None:
    """Raise when an HDF5 file is not a twopy response heatmap file."""
    actual = _string_attr(h5_file, "twopy_format")
    if actual == RESPONSE_MAP_FILE_FORMAT:
        return
    msg = f"{path} is not a twopy response heatmap file."
    raise ValueError(msg)


def _response_map_mode(value: str) -> ResponseMapMode:
    """Return a validated response-map mode literal."""
    if value == "pixel":
        return "pixel"
    if value == "window":
        return "window"
    msg = f"Unknown response map mode: {value!r}"
    raise ValueError(msg)


def _string_attr(group: h5py.Group | h5py.File, name: str) -> str:
    """Read one HDF5 string attribute as text."""
    value = group.attrs[name]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _validate_options(options: ResponseMapOptions, shape: tuple[int, int]) -> None:
    """Validate response-map options against one spatial shape."""
    if options.mode not in {"pixel", "window"}:
        msg = f"Unknown response map mode: {options.mode!r}"
        raise ValueError(msg)
    if (
        not np.isfinite(options.pixel_smoothing_sigma)
        or options.pixel_smoothing_sigma < 0
    ):
        msg = (
            "pixel_smoothing_sigma must be finite and nonnegative; "
            f"got {options.pixel_smoothing_sigma}"
        )
        raise ValueError(msg)
    if options.window_size_pixels < 1:
        msg = f"window_size_pixels must be positive; got {options.window_size_pixels}"
        raise ValueError(msg)
    if options.window_stride_pixels < 1:
        msg = (
            f"window_stride_pixels must be positive; got {options.window_stride_pixels}"
        )
        raise ValueError(msg)
    if options.window_stride_pixels > options.window_size_pixels:
        msg = (
            "window_stride_pixels cannot exceed window_size_pixels; "
            f"got {options.window_stride_pixels} > {options.window_size_pixels}"
        )
        raise ValueError(msg)
    if options.window_size_pixels > min(shape):
        msg = (
            f"window_size_pixels {options.window_size_pixels} exceeds response-map "
            f"crop shape {shape}"
        )
        raise ValueError(msg)
    if not np.isfinite(options.baseline_sample_seconds) or (
        options.baseline_sample_seconds <= 0
    ):
        msg = (
            "baseline_sample_seconds must be finite and positive; "
            f"got {options.baseline_sample_seconds}"
        )
        raise ValueError(msg)
    if not 0.0 <= options.foreground_percentile <= 100.0:
        msg = (
            "foreground_percentile must be between 0 and 100; "
            f"got {options.foreground_percentile}"
        )
        raise ValueError(msg)


def _foreground_threshold(
    mean_image: npt.NDArray[np.float64],
    options: ResponseMapOptions,
) -> float:
    """Return the foreground threshold and dF/F denominator floor."""
    finite = mean_image[np.isfinite(mean_image)]
    if finite.size == 0:
        msg = "Response-map mean image has no finite pixels."
        raise ValueError(msg)
    return max(
        float(np.percentile(finite, options.foreground_percentile)),
        float(np.finfo(np.float64).eps),
    )


def _validate_epoch_window_frames(
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    frame_count: int,
    baseline_frames: int,
) -> None:
    """Validate response-map frame windows before streaming movie data."""
    for epoch_window in epoch_windows:
        start = epoch_window.window.start_frame
        stop = epoch_window.window.stop_frame
        baseline_start = start - baseline_frames
        if baseline_start < 0:
            msg = (
                f"Epoch window {epoch_window.window.index} starts at frame {start}, "
                f"leaving fewer than {baseline_frames} baseline frames."
            )
            raise ValueError(msg)
        if stop <= start:
            msg = f"Epoch window {epoch_window.window.index} has no response frames."
            raise ValueError(msg)
        if stop > frame_count:
            msg = (
                f"Epoch window {epoch_window.window.index} stops at frame {stop}, "
                f"beyond movie frame count {frame_count}."
            )
            raise ValueError(msg)


def _trial_response_map(
    recording: RecordingData,
    crop: SpatialCrop,
    foreground_mask: npt.NDArray[np.bool_],
    epoch_window: EpochFrameWindow,
    *,
    baseline_frames: int,
    denominator_floor: float,
    options: ResponseMapOptions,
) -> npt.NDArray[np.float64]:
    """Return one trial response map from a local baseline and epoch response.

    Pixel mode returns a pixelwise dF/F image, optionally smoothed with
    NaN-aware Gaussian weights. Window mode averages baseline and response
    intensities inside square windows before dF/F, then paints window responses
    back over the covered pixels so the output shape stays identical to the
    crop.
    """
    start = epoch_window.window.start_frame
    stop = epoch_window.window.stop_frame
    baseline = _mean_masked_frames(
        recording,
        crop,
        start=start - baseline_frames,
        stop=start,
        foreground_mask=foreground_mask,
    )
    response = _mean_masked_frames(
        recording,
        crop,
        start=start,
        stop=stop,
        foreground_mask=foreground_mask,
    )
    if options.mode == "pixel":
        values = _delta_f_over_f(response, baseline, denominator_floor)
        if options.pixel_smoothing_sigma > 0:
            return _smooth_nan_image(values, options.pixel_smoothing_sigma)
        return values
    return _window_delta_f_over_f(
        response,
        baseline,
        denominator_floor=denominator_floor,
        window_size=options.window_size_pixels,
        stride=options.window_stride_pixels,
    )


def _mean_masked_frames(
    recording: RecordingData,
    crop: SpatialCrop,
    *,
    start: int,
    stop: int,
    foreground_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Return a frame mean image while streaming bounded movie chunks.

    Motion-artifact frames do not contribute. Background pixels outside the
    foreground mask remain NaN so downstream dF/F and smoothing treat them as
    missing data rather than zero response.
    """
    sums = np.zeros(crop.shape, dtype=np.float64)
    counts = np.zeros(crop.shape, dtype=np.int64)
    for chunk_start, chunk_stop, frames in recording.movie.iter_frame_batches(
        start=start,
        stop=stop,
        spatial_crop=crop,
    ):
        motion_mask = recording.motion_artifact_mask[chunk_start:chunk_stop]
        if motion_mask.shape[0] != frames.shape[0]:
            msg = (
                "Motion-artifact mask length does not match movie frame count "
                f"for frames [{chunk_start}, {chunk_stop})."
            )
            raise ValueError(msg)
        valid_frames = frames[~motion_mask]
        if valid_frames.size == 0:
            continue
        finite = np.isfinite(valid_frames) & foreground_mask[None, :, :]
        sums += np.sum(np.where(finite, valid_frames, 0.0), axis=0)
        counts += np.sum(finite, axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(crop.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _delta_f_over_f(
    response: npt.NDArray[np.float64],
    baseline: npt.NDArray[np.float64],
    denominator_floor: float,
) -> npt.NDArray[np.float64]:
    """Return pixelwise dF/F using a denominator floor."""
    denominator = np.maximum(np.abs(baseline), denominator_floor)
    values = (response - baseline) / denominator
    values[~np.isfinite(values)] = np.nan
    return values.astype(np.float64, copy=False)


def _window_delta_f_over_f(
    response: npt.NDArray[np.float64],
    baseline: npt.NDArray[np.float64],
    *,
    denominator_floor: float,
    window_size: int,
    stride: int,
) -> npt.NDArray[np.float64]:
    """Return a full-size map from square-window dF/F values.

    Each window contributes one scalar response to every pixel it covers.
    Overlapping windows are averaged by ``counts``. Pixels covered only by
    windows whose baseline or response is entirely masked remain NaN.
    """
    output = np.zeros(response.shape, dtype=np.float64)
    counts = np.zeros(response.shape, dtype=np.float64)
    for row_start in _window_starts(response.shape[0], window_size, stride):
        row_stop = row_start + window_size
        for col_start in _window_starts(response.shape[1], window_size, stride):
            col_stop = col_start + window_size
            baseline_window = baseline[row_start:row_stop, col_start:col_stop]
            response_window = response[row_start:row_stop, col_start:col_stop]
            if not np.any(np.isfinite(baseline_window)) or not np.any(
                np.isfinite(response_window),
            ):
                continue
            baseline_value = float(np.nanmean(baseline_window))
            response_value = float(np.nanmean(response_window))
            denominator = max(abs(baseline_value), denominator_floor)
            output[row_start:row_stop, col_start:col_stop] += (
                response_value - baseline_value
            ) / denominator
            counts[row_start:row_stop, col_start:col_stop] += 1.0
    return np.divide(
        output,
        counts,
        out=np.full_like(output, np.nan),
        where=counts > 0,
    )


def _window_starts(length: int, window_size: int, stride: int) -> tuple[int, ...]:
    """Return starts that cover the full axis, including the trailing edge."""
    final_start = length - window_size
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


def _smooth_nan_image(
    image: npt.NDArray[np.float64],
    sigma: float,
) -> npt.NDArray[np.float64]:
    """Gaussian smooth an image while preserving NaN-masked areas.

    Values and finite-pixel weights are filtered separately, then divided. This
    prevents masked background pixels from being treated as zero responses while
    still allowing nearby foreground pixels to share information. Locations with
    too little finite support stay NaN.
    """
    finite = np.isfinite(image)
    values = gaussian_filter(np.where(finite, image, 0.0), sigma=sigma)
    weights = gaussian_filter(finite.astype(np.float64), sigma=sigma)
    return np.divide(
        values,
        weights,
        out=np.full_like(image, np.nan),
        where=weights > 0.25,
    )


def _epoch_response_map(
    *,
    epoch_number: int,
    epoch_name: str,
    trial_maps: tuple[npt.NDArray[np.float64], ...],
) -> EpochResponseMap:
    """Average trial maps for one epoch."""
    sums = np.zeros(trial_maps[0].shape, dtype=np.float64)
    counts = np.zeros(trial_maps[0].shape, dtype=np.int64)
    for trial_map in trial_maps:
        finite = np.isfinite(trial_map)
        sums += np.where(finite, trial_map, 0.0)
        counts += finite
    response = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )
    return EpochResponseMap(
        epoch_name=epoch_name,
        epoch_number=epoch_number,
        response_values=response.astype(np.float64, copy=False),
        trial_count=len(trial_maps),
    )


def _normalize_epoch_response_maps(
    epochs: tuple[EpochResponseMap, ...],
) -> tuple[tuple[EpochResponseMap, ...], float]:
    """Return epochs normalized by the shared absolute response maximum.

    This is analysis persistence normalization, not display scaling. The saved
    maps stay in a fixed ``[-1, 1]`` audit space, while ``response_scale`` keeps
    the original dF/F magnitude that produced a persisted value of 1.
    """
    response_scale = 0.0
    for epoch in epochs:
        values = np.abs(epoch.response_values[np.isfinite(epoch.response_values)])
        if values.size > 0:
            response_scale = max(response_scale, float(np.max(values)))
    if response_scale == 0.0:
        return epochs, 1.0
    response_scale = max(response_scale, 1e-6)
    return (
        tuple(
            EpochResponseMap(
                epoch_name=epoch.epoch_name,
                epoch_number=epoch.epoch_number,
                response_values=epoch.response_values / response_scale,
                trial_count=epoch.trial_count,
            )
            for epoch in epochs
        ),
        response_scale,
    )
