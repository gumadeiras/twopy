"""Represent ROIs and extract fluorescence traces from converted movies.

Inputs: ROI masks plus a loaded twopy ``RecordingData`` object.
Outputs: saved ROI HDF5 files and ROI-by-frame fluorescence traces.

This module is GUI-independent. napari can create or edit masks, but analysis
code should use these typed objects and functions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import h5py
import numpy as np
import numpy.typing as npt

from twopy.converted import RecordingData

__all__ = [
    "BackgroundCorrectedRoiTraces",
    "BackgroundCorrectionMethod",
    "ROI_FILE_FORMAT",
    "RoiSet",
    "RoiTraces",
    "extract_background_corrected_roi_traces",
    "extract_roi_traces",
    "load_roi_set",
    "make_roi_set",
    "save_roi_set",
]

ROI_FILE_FORMAT = "twopy-roi-set"
TraceStatistic = Literal["mean"]
BackgroundCorrectionMethod = Literal[
    "none",
    "movie_global_percentile",
    "roi_local_percentile_y",
]
BackgroundMetadataValue = str | int | float | bool


@dataclass(frozen=True)
class RoiSet:
    """A named set of spatial ROI masks.

    Inputs: boolean masks with shape ``(rois, x, y)`` and one label per ROI.
    Outputs: validated ROI masks that can be saved or used for trace extraction.

    Masks are stored in movie coordinates so the same object works for scripts,
    tests, and napari-derived ROIs.
    """

    masks: npt.NDArray[np.bool_]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class RoiTraces:
    """Fluorescence traces extracted from a converted movie.

    Inputs: one converted recording and one ``RoiSet``.
    Outputs: ``values`` with shape ``(frames, rois)`` plus labels and frame
    bounds.

    The trace values are computed from converted aligned movie frames, not from
    source MATLAB or raw TIFF files.
    """

    values: npt.NDArray[np.float64]
    labels: tuple[str, ...]
    start_frame: int
    stop_frame: int
    statistic: TraceStatistic


@dataclass(frozen=True)
class BackgroundCorrectedRoiTraces:
    """ROI traces with explicit background correction outputs.

    Inputs: one converted recording, one ROI set, and one background correction
    method.
    Outputs: raw fluorescence, estimated background, corrected fluorescence,
    labels, frame bounds, and method metadata.

    Keeping all three arrays makes the correction auditable: downstream dF/F
    can use ``corrected_values`` while plots and tests can still inspect the
    raw signal and the subtracted background.
    """

    raw_values: npt.NDArray[np.float64]
    background_values: npt.NDArray[np.float64]
    corrected_values: npt.NDArray[np.float64]
    labels: tuple[str, ...]
    start_frame: int
    stop_frame: int
    statistic: TraceStatistic
    method: BackgroundCorrectionMethod
    metadata: dict[str, BackgroundMetadataValue]


def make_roi_set(
    masks: npt.ArrayLike,
    *,
    labels: tuple[str, ...] | None = None,
) -> RoiSet:
    """Create a validated ROI set from mask data.

    Args:
        masks: Array shaped ``(rois, x, y)``. Nonzero values are treated as
            inside the ROI.
        labels: Optional ROI labels. When omitted, labels are ``roi_0001``,
            ``roi_0002``, and so on.

    Returns:
        ``RoiSet`` with boolean masks and labels.

    Raises:
        ValueError: If masks are not three-dimensional, labels do not match, or
            any ROI has no pixels.
    """
    bool_masks = np.asarray(masks, dtype=np.bool_)
    if bool_masks.ndim != 3:
        msg = f"ROI masks must have shape (rois, x, y); got {bool_masks.shape}"
        raise ValueError(msg)
    if bool_masks.shape[0] == 0:
        msg = "ROI masks must contain at least one ROI"
        raise ValueError(msg)

    pixel_counts = bool_masks.reshape(bool_masks.shape[0], -1).sum(axis=1)
    if np.any(pixel_counts == 0):
        empty_indices = np.where(pixel_counts == 0)[0].tolist()
        msg = f"ROI masks cannot be empty; empty ROI indices: {empty_indices}"
        raise ValueError(msg)

    resolved_labels = labels
    if resolved_labels is None:
        resolved_labels = tuple(
            f"roi_{roi_index + 1:04d}" for roi_index in range(bool_masks.shape[0])
        )
    if len(resolved_labels) != bool_masks.shape[0]:
        msg = (
            f"ROI labels length {len(resolved_labels)} does not match "
            f"{bool_masks.shape[0]} masks"
        )
        raise ValueError(msg)

    return RoiSet(masks=bool_masks, labels=resolved_labels)


def save_roi_set(roi_set: RoiSet, path: Path) -> None:
    """Save ROI masks and labels to an inspectable HDF5 file.

    Args:
        roi_set: Validated ROI masks and labels.
        path: Destination HDF5 file.

    Returns:
        None.

    The file uses gzip compression because ROI masks are usually sparse and
    compress well.
    """
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = ROI_FILE_FORMAT
        h5_file.create_dataset("masks", data=roi_set.masks, compression="gzip")
        h5_file.create_dataset(
            "labels",
            data=np.asarray(roi_set.labels, dtype=h5py.string_dtype("utf-8")),
        )


def load_roi_set(path: Path) -> RoiSet:
    """Load a saved ROI set from HDF5.

    Args:
        path: ROI HDF5 file path.

    Returns:
        Validated ``RoiSet``.

    Raises:
        ValueError: If the file is not a twopy ROI file.
    """
    input_path = path.expanduser()
    with h5py.File(input_path, "r") as h5_file:
        actual_format = str(h5_file.attrs.get("twopy_format", ""))
        if actual_format != ROI_FILE_FORMAT:
            msg = f"Expected {ROI_FILE_FORMAT!r} file at {input_path}"
            raise ValueError(msg)
        masks = cast(npt.NDArray[np.bool_], h5_file["masks"][()])
        raw_labels = h5_file["labels"][()]

    labels = tuple(_decode_label(label) for label in raw_labels)
    return make_roi_set(masks, labels=labels)


def extract_roi_traces(
    recording: RecordingData,
    roi_set: RoiSet,
    *,
    start_frame: int | None = None,
    stop_frame: int | None = None,
    chunk_frames: int = 128,
    statistic: TraceStatistic = "mean",
) -> RoiTraces:
    """Extract ROI fluorescence traces from the converted aligned movie.

    Args:
        recording: Loaded converted recording.
        roi_set: ROI masks in aligned movie spatial coordinates.
        start_frame: Optional first frame to extract.
        stop_frame: Optional exclusive stop frame.
        chunk_frames: Number of movie frames to read per chunk.
        statistic: Pixel statistic inside each ROI. Currently only ``mean`` is
            supported.

    Returns:
        ``RoiTraces`` with shape ``(frames, rois)``.

    Raises:
        ValueError: If ROI shape differs from movie spatial shape.

    The function streams movie chunks from HDF5. Only one chunk and the output
    traces are held in memory.
    """
    if roi_set.masks.shape[1:] != recording.movie.shape[1:]:
        _raise_roi_movie_shape_error(roi_set, recording)

    frame_start, frame_stop = _normalize_trace_frame_range(
        frame_count=recording.movie.shape[0],
        start_frame=start_frame,
        stop_frame=stop_frame,
    )
    roi_pixel_indices = _roi_pixel_indices(roi_set)
    traces = _extract_mean_traces_from_indices(
        recording=recording,
        pixel_indices_by_trace=roi_pixel_indices,
        frame_start=frame_start,
        frame_stop=frame_stop,
        chunk_frames=chunk_frames,
    )

    return RoiTraces(
        values=traces,
        labels=roi_set.labels,
        start_frame=frame_start,
        stop_frame=frame_stop,
        statistic=statistic,
    )


def extract_background_corrected_roi_traces(
    recording: RecordingData,
    roi_set: RoiSet,
    *,
    method: BackgroundCorrectionMethod = "none",
    start_frame: int | None = None,
    stop_frame: int | None = None,
    chunk_frames: int = 128,
    statistic: TraceStatistic = "mean",
    global_percentile: float = 10.0,
    local_y_radius: int | None = None,
    local_percentile: float = 20.0,
) -> BackgroundCorrectedRoiTraces:
    """Extract ROI traces plus an explicit background correction.

    Args:
        recording: Loaded converted recording with lazy aligned movie access.
        roi_set: ROI masks in aligned movie spatial coordinates.
        method: Background correction method. ``"none"`` keeps raw traces,
            ``"movie_global_percentile"`` matches the audited MATLAB
            ``backgroundSubtractMovie = 1`` path, and
            ``"roi_local_percentile_y"`` subtracts a local y-neighborhood
            background trace from each ROI.
        start_frame: Optional first frame to extract.
        stop_frame: Optional exclusive stop frame.
        chunk_frames: Number of movie frames to read per chunk.
        statistic: Pixel statistic inside each ROI. Currently only ``mean`` is
            supported.
        global_percentile: Percentile of the recording mean image used to pick
            global background pixels.
        local_y_radius: Half-width in rows for local-y background pixels. When
            omitted, twopy uses roughly five percent of the image height.
        local_percentile: Percentile inside each local row window used to keep
            dim background pixels and reject bright non-ROI structures.

    Returns:
        ``BackgroundCorrectedRoiTraces`` with raw, background, and corrected
        frame-by-ROI arrays.

    Raises:
        ValueError: If masks do not match the movie, a background mask is empty,
            or an unsupported method is requested.

    The function streams movie chunks from HDF5 and never loads the full movie
    into memory. Each method records enough metadata to audit how the background
    values were produced.
    """
    if roi_set.masks.shape[1:] != recording.movie.shape[1:]:
        _raise_roi_movie_shape_error(roi_set, recording)

    frame_start, frame_stop = _normalize_trace_frame_range(
        frame_count=recording.movie.shape[0],
        start_frame=start_frame,
        stop_frame=stop_frame,
    )
    roi_pixel_indices = _roi_pixel_indices(roi_set)

    if method == "none":
        raw_values = _extract_mean_traces_from_indices(
            recording=recording,
            pixel_indices_by_trace=roi_pixel_indices,
            frame_start=frame_start,
            frame_stop=frame_stop,
            chunk_frames=chunk_frames,
        )
        background_values = np.zeros_like(raw_values)
        corrected_values = raw_values.copy()
        metadata: dict[str, BackgroundMetadataValue] = {"method": method}
    elif method == "movie_global_percentile":
        raw_values, background_values, corrected_values, metadata = (
            _extract_movie_global_percentile_corrected_traces(
                recording=recording,
                roi_pixel_indices=roi_pixel_indices,
                frame_start=frame_start,
                frame_stop=frame_stop,
                chunk_frames=chunk_frames,
                percentile=global_percentile,
            )
        )
    elif method == "roi_local_percentile_y":
        raw_values, background_values, corrected_values, metadata = (
            _extract_roi_local_y_corrected_traces(
                recording=recording,
                roi_set=roi_set,
                roi_pixel_indices=roi_pixel_indices,
                frame_start=frame_start,
                frame_stop=frame_stop,
                chunk_frames=chunk_frames,
                local_y_radius=local_y_radius,
                local_percentile=local_percentile,
            )
        )

    return BackgroundCorrectedRoiTraces(
        raw_values=raw_values,
        background_values=background_values,
        corrected_values=corrected_values,
        labels=roi_set.labels,
        start_frame=frame_start,
        stop_frame=frame_stop,
        statistic=statistic,
        method=method,
        metadata=metadata,
    )


def _roi_pixel_indices(roi_set: RoiSet) -> tuple[npt.NDArray[np.int64], ...]:
    """Flatten ROI masks into pixel-index arrays.

    Args:
        roi_set: Validated ROI set.

    Returns:
        Tuple with one integer index array per ROI.

    Precomputing indices keeps each movie chunk extraction simple and avoids
    copying whole masks repeatedly.
    """
    flat_masks = roi_set.masks.reshape(roi_set.masks.shape[0], -1)
    return tuple(np.flatnonzero(mask).astype(np.int64) for mask in flat_masks)


def _raise_roi_movie_shape_error(roi_set: RoiSet, recording: RecordingData) -> None:
    """Raise a consistent ROI/movie shape error.

    Args:
        roi_set: ROI masks being used for trace extraction.
        recording: Converted recording with movie shape metadata.

    Returns:
        None.
    """
    msg = (
        f"ROI mask shape {roi_set.masks.shape[1:]} does not match movie "
        f"shape {recording.movie.shape[1:]}"
    )
    raise ValueError(msg)


def _extract_mean_traces_from_indices(
    *,
    recording: RecordingData,
    pixel_indices_by_trace: tuple[npt.NDArray[np.int64], ...],
    frame_start: int,
    frame_stop: int,
    chunk_frames: int,
) -> npt.NDArray[np.float64]:
    """Extract mean traces for arbitrary flattened pixel-index groups.

    Args:
        recording: Converted recording with lazy movie batches.
        pixel_indices_by_trace: One flattened pixel-index array per output
            trace.
        frame_start: First frame to read.
        frame_stop: Exclusive stop frame.
        chunk_frames: Maximum number of frames per HDF5 read.

    Returns:
        Array shaped ``(frames, traces)``.

    This shared helper is used for ROI traces and background traces. It streams
    one movie chunk at a time so memory scales with ``chunk_frames`` rather than
    total recording length.
    """
    traces = np.empty((frame_stop - frame_start, len(pixel_indices_by_trace)))
    for chunk_start, chunk_stop, frames in recording.movie.iter_frame_batches(
        chunk_frames=chunk_frames,
        start=frame_start,
        stop=frame_stop,
    ):
        chunk_offset = chunk_start - frame_start
        chunk_slice = slice(chunk_offset, chunk_offset + (chunk_stop - chunk_start))
        chunk_flat = frames.reshape(frames.shape[0], -1)
        for trace_index, pixel_indices in enumerate(pixel_indices_by_trace):
            traces[chunk_slice, trace_index] = chunk_flat[:, pixel_indices].mean(
                axis=1,
            )

    return traces


def _extract_movie_global_percentile_corrected_traces(
    *,
    recording: RecordingData,
    roi_pixel_indices: tuple[npt.NDArray[np.int64], ...],
    frame_start: int,
    frame_stop: int,
    chunk_frames: int,
    percentile: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    dict[str, BackgroundMetadataValue],
]:
    """Extract MATLAB-compatible global movie-background-corrected traces.

    Args:
        recording: Converted recording with mean image and aligned movie.
        roi_pixel_indices: Flattened pixel indices for each ROI.
        frame_start: First frame to read.
        frame_stop: Exclusive stop frame.
        chunk_frames: Maximum number of frames per HDF5 read.
        percentile: Mean-image percentile used to define background pixels.

    Returns:
        Raw ROI traces, repeated background traces, corrected ROI traces, and
        audit metadata.

    The corrected trace matches the audited MATLAB movie-level correction:
    subtract one framewise background value from each pixel, clamp negatives to
    zero, and then average corrected pixels inside each ROI.
    """
    _validate_percentile(percentile)
    background_indices, threshold = _global_background_pixel_indices(
        recording.mean_image,
        percentile=percentile,
    )
    frame_count = frame_stop - frame_start
    roi_count = len(roi_pixel_indices)
    raw_values = np.empty((frame_count, roi_count), dtype=np.float64)
    background_values = np.empty((frame_count, roi_count), dtype=np.float64)
    corrected_values = np.empty((frame_count, roi_count), dtype=np.float64)

    for chunk_start, chunk_stop, frames in recording.movie.iter_frame_batches(
        chunk_frames=chunk_frames,
        start=frame_start,
        stop=frame_stop,
    ):
        chunk_offset = chunk_start - frame_start
        chunk_slice = slice(chunk_offset, chunk_offset + (chunk_stop - chunk_start))
        chunk_flat = frames.reshape(frames.shape[0], -1)
        background_trace = chunk_flat[:, background_indices].mean(axis=1)
        for roi_index, pixel_indices in enumerate(roi_pixel_indices):
            roi_pixels = chunk_flat[:, pixel_indices]
            raw_values[chunk_slice, roi_index] = roi_pixels.mean(axis=1)
            background_values[chunk_slice, roi_index] = background_trace
            # MATLAB clamps the movie after subtraction. Corrected ROI traces
            # therefore need pixel-level subtraction before averaging.
            corrected_pixels = np.maximum(roi_pixels - background_trace[:, None], 0.0)
            corrected_values[chunk_slice, roi_index] = corrected_pixels.mean(axis=1)

    metadata: dict[str, BackgroundMetadataValue] = {
        "method": "movie_global_percentile",
        "global_percentile": float(percentile),
        "background_threshold": threshold,
        "background_pixel_count": int(background_indices.size),
        "negative_values_clamped": True,
    }
    return raw_values, background_values, corrected_values, metadata


def _extract_roi_local_y_corrected_traces(
    *,
    recording: RecordingData,
    roi_set: RoiSet,
    roi_pixel_indices: tuple[npt.NDArray[np.int64], ...],
    frame_start: int,
    frame_stop: int,
    chunk_frames: int,
    local_y_radius: int | None,
    local_percentile: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    dict[str, BackgroundMetadataValue],
]:
    """Extract ROI traces corrected by local y-neighborhood background.

    Args:
        recording: Converted recording with lazy movie batches.
        roi_set: ROI masks used to define excluded foreground pixels.
        roi_pixel_indices: Flattened pixel indices for each ROI.
        frame_start: First frame to read.
        frame_stop: Exclusive stop frame.
        chunk_frames: Maximum number of frames per HDF5 read.
        local_y_radius: Optional row half-width for local background pixels.
        local_percentile: Percentile used to keep dim local background pixels.

    Returns:
        Raw ROI traces, local background traces, corrected traces, and metadata.

    The method estimates a background trace per ROI from dim pixels that are
    outside every ROI and close to that ROI along the scanline axis. It is
    designed for stimulus bleedthrough that varies across rows without
    requiring relaxation labeling or connected-component heuristics.
    """
    radius = _resolve_local_y_radius(
        row_count=roi_set.masks.shape[1],
        local_y_radius=local_y_radius,
    )
    _validate_percentile(local_percentile)
    background_indices_by_roi = _local_y_background_pixel_indices(
        roi_set=roi_set,
        mean_image=recording.mean_image,
        local_y_radius=radius,
        local_percentile=local_percentile,
    )
    raw_values = _extract_mean_traces_from_indices(
        recording=recording,
        pixel_indices_by_trace=roi_pixel_indices,
        frame_start=frame_start,
        frame_stop=frame_stop,
        chunk_frames=chunk_frames,
    )
    background_values = _extract_mean_traces_from_indices(
        recording=recording,
        pixel_indices_by_trace=background_indices_by_roi,
        frame_start=frame_start,
        frame_stop=frame_stop,
        chunk_frames=chunk_frames,
    )
    corrected_values = raw_values - background_values
    background_counts = tuple(
        int(pixel_indices.size) for pixel_indices in background_indices_by_roi
    )
    metadata: dict[str, BackgroundMetadataValue] = {
        "method": "roi_local_percentile_y",
        "local_y_radius": radius,
        "local_percentile": float(local_percentile),
        "background_pixel_count_min": min(background_counts),
        "background_pixel_count_max": max(background_counts),
        "negative_values_clamped": False,
    }
    return raw_values, background_values, corrected_values, metadata


def _validate_percentile(percentile: float) -> None:
    """Validate a percentile used to define background pixels.

    Args:
        percentile: Percentile in the open interval ``(0, 100)``.

    Returns:
        None.

    Strict bounds avoid empty masks from the active ``mean_image < threshold``
    rule at the extremes.
    """
    if not 0.0 < percentile < 100.0:
        msg = f"Background percentile must be between 0 and 100; got {percentile}"
        raise ValueError(msg)


def _global_background_pixel_indices(
    mean_image: npt.NDArray[np.float64],
    *,
    percentile: float,
) -> tuple[npt.NDArray[np.int64], float]:
    """Find global background pixels from a mean image.

    Args:
        mean_image: Mean image from the converted recording.
        percentile: Percentile threshold used to select dim pixels.

    Returns:
        Flattened background pixel indices and the numeric threshold.

    Raises:
        ValueError: If no pixels are below the percentile threshold.
    """
    threshold = float(np.percentile(mean_image, percentile))
    background_mask = mean_image < threshold
    background_indices = np.flatnonzero(background_mask.reshape(-1)).astype(np.int64)
    if background_indices.size == 0:
        msg = (
            "Global background mask is empty. Use a higher percentile or inspect "
            "the recording mean image."
        )
        raise ValueError(msg)
    return background_indices, threshold


def _resolve_local_y_radius(
    *,
    row_count: int,
    local_y_radius: int | None,
) -> int:
    """Resolve the row radius for local-y background correction.

    Args:
        row_count: Number of image rows.
        local_y_radius: Optional caller-provided radius.

    Returns:
        Nonnegative row radius.
    """
    radius = max(1, round(row_count / 20)) if local_y_radius is None else local_y_radius
    if radius < 0:
        msg = f"local_y_radius must be nonnegative; got {radius}"
        raise ValueError(msg)
    return radius


def _local_y_background_pixel_indices(
    *,
    roi_set: RoiSet,
    mean_image: npt.NDArray[np.float64],
    local_y_radius: int,
    local_percentile: float,
) -> tuple[npt.NDArray[np.int64], ...]:
    """Find local y-neighborhood background pixels for each ROI.

    Args:
        roi_set: ROI masks used to exclude foreground pixels.
        mean_image: Recording mean image used to choose dim local pixels.
        local_y_radius: Row half-width around each ROI center.
        local_percentile: Percentile used inside the local candidate pixels.

    Returns:
        Tuple of flattened background pixel-index arrays, one per ROI.

    Raises:
        ValueError: If any ROI has no background pixels in its local row window.

    Candidate background pixels are outside every ROI. The local window is
    centered on each ROI's row center, and the percentile threshold keeps dim
    pixels inside that window so bright unlabeled structures do not dominate the
    background trace.
    """
    occupied_mask = np.any(roi_set.masks, axis=0)
    background_mask = ~occupied_mask
    row_grid = np.arange(background_mask.shape[0])[:, None]
    background_indices: list[npt.NDArray[np.int64]] = []

    for roi_index, roi_mask in enumerate(roi_set.masks):
        roi_rows = np.flatnonzero(np.any(roi_mask, axis=1))
        center_row = int(round(float(np.mean(roi_rows))))
        local_rows = np.abs(row_grid - center_row) <= local_y_radius
        local_candidate_mask = background_mask & local_rows
        local_candidate_values = mean_image[local_candidate_mask]
        if local_candidate_values.size == 0:
            msg = (
                f"ROI {roi_index} has no local background candidates within "
                f"{local_y_radius} rows"
            )
            raise ValueError(msg)
        local_threshold = float(np.percentile(local_candidate_values, local_percentile))
        local_background_mask = local_candidate_mask & (mean_image <= local_threshold)
        local_indices = np.flatnonzero(local_background_mask.reshape(-1)).astype(
            np.int64,
        )
        if local_indices.size == 0:
            msg = (
                f"ROI {roi_index} has no local background pixels within "
                f"{local_y_radius} rows"
            )
            raise ValueError(msg)
        background_indices.append(local_indices)

    return tuple(background_indices)


def _decode_label(value: object) -> str:
    """Decode one HDF5 label value into text.

    Args:
        value: Raw scalar read from the labels dataset.

    Returns:
        ROI label as a string.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_trace_frame_range(
    *,
    frame_count: int,
    start_frame: int | None,
    stop_frame: int | None,
) -> tuple[int, int]:
    """Normalize and validate the frame range for trace extraction.

    Args:
        frame_count: Number of frames in the movie.
        start_frame: Optional first frame index.
        stop_frame: Optional exclusive stop frame.

    Returns:
        ``(start, stop)`` frame bounds.
    """
    start = 0 if start_frame is None else start_frame
    stop = frame_count if stop_frame is None else stop_frame
    if start < 0 or stop > frame_count or start >= stop:
        msg = f"Invalid trace frame range [{start}, {stop}) for {frame_count} frames"
        raise ValueError(msg)
    return start, stop
