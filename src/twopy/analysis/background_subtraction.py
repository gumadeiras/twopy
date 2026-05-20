"""Background correction for ROI fluorescence traces.

Inputs: converted aligned movie frames, ROI masks, and correction parameters.
Outputs: raw ROI traces, estimated background traces, corrected traces, and
method metadata.

This module owns analysis-time background correction. ROI storage and raw ROI
trace extraction stay in ``twopy.roi``; dF/F and trial analysis can consume the
auditable outputs produced here.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import sparse

from twopy.converted import RecordingData
from twopy.frame_ranges import normalize_frame_range
from twopy.roi import RoiSet, TraceStatistic, validate_trace_statistic
from twopy.spatial import SpatialCrop, SpatialDomain, full_frame_crop

__all__ = [
    "BackgroundCorrectedRoiTraces",
    "BackgroundCorrectionMethod",
    "extract_background_corrected_roi_traces",
]

BackgroundCorrectionMethod = Literal[
    "none",
    "movie_global_percentile",
    "movie_y_stripe_percentile",
    "roi_y_stripe_percentile",
]
BackgroundMetadataValue = str | int | float | bool


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
    stripe_y_height: int | None = None,
    stripe_percentile: float = 20.0,
    local_y_radius: int | None = None,
    local_percentile: float = 20.0,
    spatial_domain: SpatialDomain = "alignment_valid_crop",
    check_cancelled: Callable[[], None] | None = None,
) -> BackgroundCorrectedRoiTraces:
    """Extract ROI traces plus an explicit background correction.

    Args:
        recording: Loaded converted recording with lazy aligned movie access.
        roi_set: ROI masks in aligned movie spatial coordinates.
        method: Background correction method. ``"none"`` keeps raw traces inside
            the selected spatial domain, ``"movie_global_percentile"`` subtracts
            a framewise background estimated from dim global pixels in the mean
            image, ``"movie_y_stripe_percentile"`` subtracts a framewise
            low-percentile y-stripe background field, and
            ``"roi_y_stripe_percentile"`` subtracts a per-ROI y-stripe
            background trace.
        start_frame: Optional first frame to extract.
        stop_frame: Optional exclusive stop frame.
        chunk_frames: Number of movie frames to read per chunk.
        statistic: Pixel statistic inside each ROI. Currently only ``mean`` is
            supported.
        global_percentile: Percentile of the recording mean image used to pick
            global background pixels.
        stripe_y_height: Height in rows for framewise y-stripe background
            correction. When omitted, twopy uses roughly five percent of the
            image height.
        stripe_percentile: Percentile inside each frame stripe used to estimate
            the framewise y-stripe background field.
        local_y_radius: Half-width in rows for the per-ROI y-stripe. When
            omitted, twopy uses roughly five percent of the image height.
        local_percentile: Percentile inside each ROI y-stripe used to keep dim
            background pixels and reject bright non-ROI structures.
        spatial_domain: Spatial region used for analysis extraction. The default
            uses the alignment-valid crop saved during conversion, matching the
            crop-domain analysis contract while keeping ROI masks stored in
            full-frame coordinates.
        check_cancelled: Optional callback that raises ``CancelledError`` when
            an interactive caller has superseded this computation.

    Returns:
        ``BackgroundCorrectedRoiTraces`` with raw, background, and corrected
        frame-by-ROI arrays.

    Raises:
        ValueError: If masks do not match the movie or a background mask is
            empty. ROI y-stripe background correction also raises when the
            selected spatial domain has no unlabeled pixels outside all ROIs,
            which is expected for dense grid ROIs that cover the whole analysis
            crop.

    The function streams movie chunks from HDF5 and never loads the full movie
    into memory. Each method records enough metadata to audit how the background
    values were produced.
    """
    _validate_background_method(method)
    validate_trace_statistic(statistic)
    _validate_roi_movie_shape(roi_set, recording)
    _check_cancelled(check_cancelled)

    frame_start, frame_stop = normalize_frame_range(
        frame_count=recording.movie.shape[0],
        start_frame=start_frame,
        stop_frame=stop_frame,
        context="trace frame range",
    )

    crop = _resolve_spatial_domain(recording, spatial_domain)
    cropped_roi_masks = _crop_roi_masks_for_domain(roi_set, crop)
    roi_pixel_indices = _roi_pixel_indices_from_masks(cropped_roi_masks)

    if method == "none":
        raw_values = _extract_mean_traces_from_indices(
            recording=recording,
            pixel_indices_by_trace=roi_pixel_indices,
            frame_start=frame_start,
            frame_stop=frame_stop,
            chunk_frames=chunk_frames,
            spatial_crop=crop,
            check_cancelled=check_cancelled,
        )
        background_values = np.zeros_like(raw_values)
        corrected_values = raw_values.copy()
        metadata: dict[str, BackgroundMetadataValue] = {
            "method": method,
            **_spatial_crop_metadata(spatial_domain, crop),
        }
    elif method == "movie_global_percentile":
        raw_values, background_values, corrected_values, metadata = (
            _extract_movie_global_percentile_corrected_traces(
                recording=recording,
                roi_pixel_indices=roi_pixel_indices,
                frame_start=frame_start,
                frame_stop=frame_stop,
                chunk_frames=chunk_frames,
                percentile=global_percentile,
                spatial_crop=crop,
                spatial_domain=spatial_domain,
                check_cancelled=check_cancelled,
            )
        )
    elif method == "movie_y_stripe_percentile":
        raw_values, background_values, corrected_values, metadata = (
            _extract_movie_y_stripe_percentile_corrected_traces(
                recording=recording,
                roi_pixel_indices=roi_pixel_indices,
                frame_start=frame_start,
                frame_stop=frame_stop,
                chunk_frames=chunk_frames,
                stripe_y_height=stripe_y_height,
                percentile=stripe_percentile,
                spatial_crop=crop,
                spatial_domain=spatial_domain,
                check_cancelled=check_cancelled,
            )
        )
    elif method == "roi_y_stripe_percentile":
        raw_values, background_values, corrected_values, metadata = (
            _extract_roi_y_stripe_percentile_corrected_traces(
                recording=recording,
                roi_masks=cropped_roi_masks,
                roi_pixel_indices=roi_pixel_indices,
                frame_start=frame_start,
                frame_stop=frame_stop,
                chunk_frames=chunk_frames,
                local_y_radius=local_y_radius,
                local_percentile=local_percentile,
                spatial_crop=crop,
                spatial_domain=spatial_domain,
                check_cancelled=check_cancelled,
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


def _validate_background_method(method: BackgroundCorrectionMethod) -> None:
    """Validate the requested background correction method.

    Args:
        method: Background correction method requested by the caller.

    Returns:
        None.

    Raises:
        ValueError: If ``method`` names behavior twopy does not implement.

    Runtime validation keeps untyped scripts from falling through the branch
    table and getting a misleading low-level error.
    """
    allowed_methods = {
        "none",
        "movie_global_percentile",
        "movie_y_stripe_percentile",
        "roi_y_stripe_percentile",
    }
    if method not in allowed_methods:
        msg = (
            f"Unknown background correction method {method!r}; "
            f"supported methods: {', '.join(sorted(allowed_methods))}"
        )
        raise ValueError(msg)


def _extract_mean_traces_from_indices(
    *,
    recording: RecordingData,
    pixel_indices_by_trace: tuple[npt.NDArray[np.int64], ...],
    frame_start: int,
    frame_stop: int,
    chunk_frames: int,
    spatial_crop: SpatialCrop,
    check_cancelled: Callable[[], None] | None = None,
) -> npt.NDArray[np.float64]:
    """Extract mean traces for arbitrary flattened pixel-index groups.

    Args:
        recording: Converted recording with lazy movie batches.
        pixel_indices_by_trace: One flattened pixel-index array per output
            trace.
        frame_start: First frame to read.
        frame_stop: Exclusive stop frame.
        chunk_frames: Maximum number of frames per HDF5 read.
        spatial_crop: Spatial domain read from each movie frame.
        check_cancelled: Optional callback that raises when work is obsolete.

    Returns:
        Array shaped ``(frames, traces)``.

    This shared helper is used for raw ROI traces and background traces. It
    streams one movie chunk at a time so memory scales with ``chunk_frames``
    rather than total recording length.
    """
    traces = np.empty((frame_stop - frame_start, len(pixel_indices_by_trace)))
    projection = _mean_projection(pixel_indices_by_trace)
    _check_cancelled(check_cancelled)
    for chunk_start, chunk_stop, frames in recording.movie.iter_frame_batches(
        chunk_frames=chunk_frames,
        start=frame_start,
        stop=frame_stop,
        spatial_crop=spatial_crop,
    ):
        _check_cancelled(check_cancelled)
        chunk_offset = chunk_start - frame_start
        chunk_slice = slice(chunk_offset, chunk_offset + (chunk_stop - chunk_start))
        chunk_flat = frames.reshape(frames.shape[0], -1)
        traces[chunk_slice, :] = _project_trace_pixels(chunk_flat, projection)
        _check_cancelled(check_cancelled)

    return traces


def _extract_movie_global_percentile_corrected_traces(
    *,
    recording: RecordingData,
    roi_pixel_indices: tuple[npt.NDArray[np.int64], ...],
    frame_start: int,
    frame_stop: int,
    chunk_frames: int,
    percentile: float,
    spatial_crop: SpatialCrop,
    spatial_domain: SpatialDomain,
    check_cancelled: Callable[[], None] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    dict[str, BackgroundMetadataValue],
]:
    """Extract global movie-background-corrected traces.

    Args:
        recording: Converted recording with mean image and aligned movie.
        roi_pixel_indices: Flattened pixel indices for each ROI.
        frame_start: First frame to read.
        frame_stop: Exclusive stop frame.
        chunk_frames: Maximum number of frames per HDF5 read.
        percentile: Mean-image percentile used to define background pixels.
        spatial_crop: Spatial domain used for background and ROI pixels.
        spatial_domain: User-facing name of ``spatial_crop``.
        check_cancelled: Optional callback that raises when work is obsolete.

    Returns:
        Raw ROI traces, repeated background traces, corrected ROI traces, and
        audit metadata.

    This subtracts one framewise background value from each ROI pixel, clamps
    negatives to zero, and then averages corrected pixels inside each ROI.
    """
    _validate_percentile(percentile)
    _check_cancelled(check_cancelled)
    mean_image = spatial_crop.crop_image(recording.mean_image)
    background_indices, threshold = _global_background_pixel_indices(
        mean_image,
        percentile=percentile,
    )
    frame_count = frame_stop - frame_start
    roi_count = len(roi_pixel_indices)
    projection = _mean_projection(roi_pixel_indices)
    raw_values = np.empty((frame_count, roi_count), dtype=np.float64)
    background_values = np.empty((frame_count, roi_count), dtype=np.float64)
    corrected_values = np.empty((frame_count, roi_count), dtype=np.float64)

    for chunk_start, chunk_stop, frames in recording.movie.iter_frame_batches(
        chunk_frames=chunk_frames,
        start=frame_start,
        stop=frame_stop,
        spatial_crop=spatial_crop,
    ):
        _check_cancelled(check_cancelled)
        chunk_offset = chunk_start - frame_start
        chunk_slice = slice(chunk_offset, chunk_offset + (chunk_stop - chunk_start))
        chunk_flat = frames.reshape(frames.shape[0], -1)
        background_trace = chunk_flat[:, background_indices].mean(axis=1)
        roi_pixels = chunk_flat[:, projection.pixel_indices]
        raw_values[chunk_slice, :] = _project_selected_pixels(roi_pixels, projection)
        background_values[chunk_slice, :] = background_trace[:, np.newaxis]
        # Clamp pixels before averaging so dim pixels cannot pull the
        # corrected ROI trace below zero.
        corrected_pixels = np.maximum(
            roi_pixels - background_trace[:, np.newaxis],
            0.0,
        )
        corrected_values[chunk_slice, :] = _project_selected_pixels(
            corrected_pixels,
            projection,
        )
        _check_cancelled(check_cancelled)

    metadata: dict[str, BackgroundMetadataValue] = {
        "method": "movie_global_percentile",
        "global_percentile": float(percentile),
        "background_threshold": threshold,
        "background_pixel_count": int(background_indices.size),
        "negative_values_clamped": True,
        **_spatial_crop_metadata(spatial_domain, spatial_crop),
    }
    return raw_values, background_values, corrected_values, metadata


def _extract_movie_y_stripe_percentile_corrected_traces(
    *,
    recording: RecordingData,
    roi_pixel_indices: tuple[npt.NDArray[np.int64], ...],
    frame_start: int,
    frame_stop: int,
    chunk_frames: int,
    stripe_y_height: int | None,
    percentile: float,
    spatial_crop: SpatialCrop,
    spatial_domain: SpatialDomain,
    check_cancelled: Callable[[], None] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    dict[str, BackgroundMetadataValue],
]:
    """Extract traces corrected by a framewise y-stripe background field.

    Args:
        recording: Converted recording with lazy movie batches.
        roi_pixel_indices: Flattened pixel indices for each ROI.
        frame_start: First frame to read.
        frame_stop: Exclusive stop frame.
        chunk_frames: Maximum number of frames per HDF5 read.
        stripe_y_height: Optional stripe height in image rows. When omitted,
            twopy uses roughly five percent of the image height.
        percentile: Percentile used inside each frame stripe.
        spatial_crop: Spatial domain used for background and ROI pixels.
        spatial_domain: User-facing name of ``spatial_crop``.
        check_cancelled: Optional callback that raises when work is obsolete.

    Returns:
        Raw ROI traces, stripe-field background traces, corrected traces, and
        audit metadata.

    Plain algorithm:
        divide each frame into y-stripes
        take the dim percentile inside each stripe for that frame
        assign each ROI pixel the background from its stripe
        average those background pixels over each ROI
        subtract the stripe background from each ROI pixel

    This models broad, row-dependent background without assuming that pixels
    near one ROI are safe neuropil reference pixels.
    """
    _validate_percentile(percentile)
    _check_cancelled(check_cancelled)
    crop_shape = spatial_crop.crop_image(recording.mean_image).shape
    stripe_height = _resolve_stripe_y_height(
        row_count=crop_shape[0],
        stripe_y_height=stripe_y_height,
    )
    row_slices = _stripe_row_slices(
        row_count=crop_shape[0],
        stripe_y_height=stripe_height,
    )
    frame_count = frame_stop - frame_start
    roi_count = len(roi_pixel_indices)
    projection = _mean_projection(roi_pixel_indices)
    raw_values = np.empty((frame_count, roi_count), dtype=np.float64)
    background_values = np.empty((frame_count, roi_count), dtype=np.float64)
    corrected_values = np.empty((frame_count, roi_count), dtype=np.float64)

    for chunk_start, chunk_stop, frames in recording.movie.iter_frame_batches(
        chunk_frames=chunk_frames,
        start=frame_start,
        stop=frame_stop,
        spatial_crop=spatial_crop,
    ):
        _check_cancelled(check_cancelled)
        chunk_offset = chunk_start - frame_start
        chunk_slice = slice(chunk_offset, chunk_offset + (chunk_stop - chunk_start))
        background_field = _framewise_y_stripe_background_field(
            frames,
            row_slices=row_slices,
            percentile=percentile,
        )
        chunk_flat = frames.reshape(frames.shape[0], -1)
        background_flat = background_field.reshape(background_field.shape[0], -1)
        roi_pixels = chunk_flat[:, projection.pixel_indices]
        roi_background_pixels = background_flat[:, projection.pixel_indices]
        raw_values[chunk_slice, :] = _project_selected_pixels(roi_pixels, projection)
        background_values[chunk_slice, :] = _project_selected_pixels(
            roi_background_pixels,
            projection,
        )
        corrected_pixels = np.maximum(
            roi_pixels - roi_background_pixels,
            0.0,
        )
        corrected_values[chunk_slice, :] = _project_selected_pixels(
            corrected_pixels,
            projection,
        )
        _check_cancelled(check_cancelled)

    metadata: dict[str, BackgroundMetadataValue] = {
        "method": "movie_y_stripe_percentile",
        "stripe_y_height": stripe_height,
        "stripe_percentile": float(percentile),
        "stripe_count": len(row_slices),
        "negative_values_clamped": True,
        **_spatial_crop_metadata(spatial_domain, spatial_crop),
    }
    return raw_values, background_values, corrected_values, metadata


def _extract_roi_y_stripe_percentile_corrected_traces(
    *,
    recording: RecordingData,
    roi_masks: npt.NDArray[np.bool_],
    roi_pixel_indices: tuple[npt.NDArray[np.int64], ...],
    frame_start: int,
    frame_stop: int,
    chunk_frames: int,
    local_y_radius: int | None,
    local_percentile: float,
    spatial_crop: SpatialCrop,
    spatial_domain: SpatialDomain,
    check_cancelled: Callable[[], None] | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    dict[str, BackgroundMetadataValue],
]:
    """Extract ROI traces corrected by per-ROI y-stripe background.

    Args:
        recording: Converted recording with lazy movie batches.
        roi_masks: ROI masks in the selected spatial domain.
        roi_pixel_indices: Flattened pixel indices for each ROI.
        frame_start: First frame to read.
        frame_stop: Exclusive stop frame.
        chunk_frames: Maximum number of frames per HDF5 read.
        local_y_radius: Optional row half-width for the ROI y-stripe.
        local_percentile: Percentile used to keep dim pixels in each ROI
            y-stripe.
        spatial_crop: Spatial domain used for background and ROI pixels.
        spatial_domain: User-facing name of ``spatial_crop``.
        check_cancelled: Optional callback that raises when work is obsolete.

    Returns:
        Raw ROI traces, local background traces, corrected traces, and metadata.

    Plain algorithm:
        take rows near that ROI center
        exclude all ROI pixels
        keep dim pixels by percentile
        average those pixels over time
        subtract that trace from that ROI only

    The method estimates a background trace per ROI from dim pixels that are
    outside every ROI and inside that ROI's y-stripe. It is designed for
    stimulus bleedthrough that varies across rows without requiring relaxation
    labeling or connected-component heuristics.
    """
    radius = _resolve_roi_y_stripe_radius(
        row_count=roi_masks.shape[1],
        local_y_radius=local_y_radius,
    )
    _validate_percentile(local_percentile)
    _check_cancelled(check_cancelled)
    background_indices_by_roi = _roi_y_stripe_background_pixel_indices(
        roi_masks=roi_masks,
        mean_image=spatial_crop.crop_image(recording.mean_image),
        local_y_radius=radius,
        local_percentile=local_percentile,
    )
    raw_values = _extract_mean_traces_from_indices(
        recording=recording,
        pixel_indices_by_trace=roi_pixel_indices,
        frame_start=frame_start,
        frame_stop=frame_stop,
        chunk_frames=chunk_frames,
        spatial_crop=spatial_crop,
        check_cancelled=check_cancelled,
    )
    background_values = _extract_mean_traces_from_indices(
        recording=recording,
        pixel_indices_by_trace=background_indices_by_roi,
        frame_start=frame_start,
        frame_stop=frame_stop,
        chunk_frames=chunk_frames,
        spatial_crop=spatial_crop,
        check_cancelled=check_cancelled,
    )
    corrected_values = raw_values - background_values
    background_counts = tuple(
        int(pixel_indices.size) for pixel_indices in background_indices_by_roi
    )
    metadata: dict[str, BackgroundMetadataValue] = {
        "method": "roi_y_stripe_percentile",
        "local_y_radius": radius,
        "local_percentile": float(local_percentile),
        "background_pixel_count_min": min(background_counts),
        "background_pixel_count_max": max(background_counts),
        "negative_values_clamped": False,
        **_spatial_crop_metadata(spatial_domain, spatial_crop),
    }
    return raw_values, background_values, corrected_values, metadata


def _roi_pixel_indices_from_masks(
    masks: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.int64], ...]:
    """Flatten ROI masks from one spatial domain into pixel-index arrays.

    Args:
        masks: Boolean ROI masks shaped ``(rois, axis0, axis1)``.

    Returns:
        Tuple with one integer index array per ROI.
    """
    flat_masks = masks.reshape(masks.shape[0], -1)
    return tuple(np.flatnonzero(mask).astype(np.int64) for mask in flat_masks)


@dataclass(frozen=True)
class _MeanProjection:
    """Sparse averaging matrix plus selected flattened pixel indices."""

    pixel_indices: npt.NDArray[np.int64]
    matrix: sparse.csc_matrix


def _mean_projection(
    pixel_indices_by_trace: tuple[npt.NDArray[np.int64], ...],
) -> _MeanProjection:
    """Return selected pixels and a sparse pixel-to-trace averaging matrix.

    Args:
        pixel_indices_by_trace: One flattened pixel-index array per trace.

    Returns:
        Selected flattened pixel indices plus a sparse matrix shaped
        ``(selected_pixels, traces)`` whose columns average each trace.

    Pixel indices are concatenated per trace instead of deduplicated. That
    preserves the existing behavior for overlapping ROIs, where the same movie
    pixel can contribute independently to more than one ROI mean.
    """
    selected_pixel_indices: list[npt.NDArray[np.int64]] = []
    rows: list[npt.NDArray[np.int64]] = []
    columns: list[npt.NDArray[np.int64]] = []
    values: list[npt.NDArray[np.float64]] = []
    row_start = 0
    for trace_index, pixel_indices in enumerate(pixel_indices_by_trace):
        if pixel_indices.size == 0:
            continue
        selected_pixel_indices.append(pixel_indices)
        rows.append(
            np.arange(
                row_start,
                row_start + pixel_indices.size,
                dtype=np.int64,
            ),
        )
        row_start += pixel_indices.size
        columns.append(
            np.full(pixel_indices.shape, trace_index, dtype=np.int64),
        )
        values.append(
            np.full(
                pixel_indices.shape,
                1.0 / float(pixel_indices.size),
                dtype=np.float64,
            ),
        )
    if len(selected_pixel_indices) == 0:
        return _MeanProjection(
            pixel_indices=np.array([], dtype=np.int64),
            matrix=sparse.csc_matrix((0, len(pixel_indices_by_trace))),
        )
    return _MeanProjection(
        pixel_indices=np.concatenate(selected_pixel_indices),
        matrix=sparse.csc_matrix(
            (
                np.concatenate(values),
                (np.concatenate(rows), np.concatenate(columns)),
            ),
            shape=(row_start, len(pixel_indices_by_trace)),
        ),
    )


def _project_trace_pixels(
    chunk_flat: npt.NDArray[np.float64],
    projection: _MeanProjection,
) -> npt.NDArray[np.float64]:
    """Return trace means from a flattened movie chunk and projection."""
    return _project_selected_pixels(chunk_flat[:, projection.pixel_indices], projection)


def _project_selected_pixels(
    selected_pixels: npt.NDArray[np.float64],
    projection: _MeanProjection,
) -> npt.NDArray[np.float64]:
    """Return trace means from selected pixel columns."""
    return np.asarray(
        selected_pixels @ projection.matrix,
        dtype=np.float64,
    )


def _check_cancelled(check_cancelled: Callable[[], None] | None) -> None:
    """Raise ``CancelledError`` when an interactive caller superseded this work."""
    if check_cancelled is None:
        return
    check_cancelled()


def _crop_roi_masks_for_domain(
    roi_set: RoiSet,
    spatial_crop: SpatialCrop,
) -> npt.NDArray[np.bool_]:
    """Crop ROI masks and require every ROI pixel to remain inside the domain.

    Args:
        roi_set: Full-frame ROI masks.
        spatial_crop: Spatial domain selected for analysis.

    Returns:
        ROI masks restricted to ``spatial_crop``.

    The check prevents a quiet coordinate error: if an ROI crosses the invalid
    alignment border, crop-domain analysis would otherwise drop pixels silently.
    """
    cropped_masks = spatial_crop.crop_masks(roi_set.masks)
    full_counts = roi_set.masks.reshape(roi_set.masks.shape[0], -1).sum(axis=1)
    cropped_counts = cropped_masks.reshape(cropped_masks.shape[0], -1).sum(axis=1)
    outside = np.flatnonzero(full_counts != cropped_counts)
    if outside.size:
        msg = (
            "ROI pixels fall outside the selected spatial domain "
            f"{spatial_crop.source}; affected ROI indices: {outside.tolist()}"
        )
        raise ValueError(msg)
    return cropped_masks


def _validate_roi_movie_shape(roi_set: RoiSet, recording: RecordingData) -> None:
    """Confirm ROI masks match the converted movie shape.

    Args:
        roi_set: ROI masks being used for trace extraction.
        recording: Converted recording with movie shape metadata.

    Returns:
        None.
    """
    if roi_set.masks.shape[1:] != recording.movie.shape[1:]:
        msg = (
            f"ROI mask shape {roi_set.masks.shape[1:]} does not match movie "
            f"shape {recording.movie.shape[1:]}"
        )
        raise ValueError(msg)


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


def _resolve_roi_y_stripe_radius(
    *,
    row_count: int,
    local_y_radius: int | None,
) -> int:
    """Resolve the row radius for ROI y-stripe background correction.

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


def _resolve_stripe_y_height(
    *,
    row_count: int,
    stripe_y_height: int | None,
) -> int:
    """Resolve y-stripe height for framewise background correction.

    Args:
        row_count: Number of image rows.
        stripe_y_height: Optional caller-provided stripe height.

    Returns:
        Positive stripe height in rows.
    """
    height = (
        max(1, round(row_count / 20)) if stripe_y_height is None else stripe_y_height
    )
    if height <= 0:
        msg = f"stripe_y_height must be positive; got {height}"
        raise ValueError(msg)
    return height


def _stripe_row_slices(
    *,
    row_count: int,
    stripe_y_height: int,
) -> tuple[slice, ...]:
    """Split image rows into contiguous y-stripe slices.

    Args:
        row_count: Number of image rows.
        stripe_y_height: Stripe height in rows.

    Returns:
        Tuple of row slices covering the image.
    """
    return tuple(
        slice(start, min(start + stripe_y_height, row_count))
        for start in range(0, row_count, stripe_y_height)
    )


def _framewise_y_stripe_background_field(
    frames: npt.NDArray[np.float64],
    *,
    row_slices: tuple[slice, ...],
    percentile: float,
) -> npt.NDArray[np.float64]:
    """Build one low-percentile y-stripe background image per frame.

    Args:
        frames: Movie chunk shaped ``(frames, rows, columns)``.
        row_slices: Row stripes used for the background field.
        percentile: Percentile used inside each frame stripe.

    Returns:
        Background field shaped like ``frames``.
    """
    background = np.empty_like(frames, dtype=np.float64)
    for row_slice in row_slices:
        stripe_values = frames[:, row_slice, :].reshape(frames.shape[0], -1)
        stripe_background = np.percentile(stripe_values, percentile, axis=1)
        background[:, row_slice, :] = stripe_background[:, None, None]
    return background


def _roi_y_stripe_background_pixel_indices(
    *,
    roi_masks: npt.NDArray[np.bool_],
    mean_image: npt.NDArray[np.float64],
    local_y_radius: int,
    local_percentile: float,
) -> tuple[npt.NDArray[np.int64], ...]:
    """Find ROI y-stripe background pixels for each ROI.

    Args:
        roi_masks: ROI masks used to exclude foreground pixels.
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
    occupied_mask = np.any(roi_masks, axis=0)
    background_mask = ~occupied_mask
    if not np.any(background_mask):
        msg = (
            "ROI y-stripe P% background subtraction needs unlabeled background "
            "pixels outside all ROIs in the selected spatial domain. The current "
            "ROI masks cover every pixel, which is common for dense grid ROIs; "
            "use shared y-stripe P% or leave unlabeled background pixels."
        )
        raise ValueError(msg)

    row_grid = np.arange(background_mask.shape[0])[:, None]
    background_indices: list[npt.NDArray[np.int64]] = []

    for roi_index, roi_mask in enumerate(roi_masks):
        roi_rows = np.flatnonzero(np.any(roi_mask, axis=1))
        center_row = int(round(float(np.mean(roi_rows))))
        local_rows = np.abs(row_grid - center_row) <= local_y_radius
        local_candidate_mask = background_mask & local_rows
        local_candidate_values = mean_image[local_candidate_mask]
        if local_candidate_values.size == 0:
            msg = (
                f"ROI {roi_index} has no local background candidates within "
                f"{local_y_radius} rows. ROI y-stripe P% only uses unlabeled "
                "pixels outside all ROIs; increase the local y radius, use "
                "shared y-stripe P%, or leave unlabeled background pixels near "
                "this ROI."
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


def _resolve_spatial_domain(
    recording: RecordingData,
    spatial_domain: SpatialDomain,
) -> SpatialCrop:
    """Resolve a user-facing spatial domain name into crop bounds.

    Args:
        recording: Loaded converted recording.
        spatial_domain: Requested spatial domain.

    Returns:
        ``SpatialCrop`` used for image and movie reads.
    """
    if spatial_domain == "alignment_valid_crop":
        return recording.alignment_valid_crop
    if spatial_domain == "full_frame":
        return full_frame_crop(recording.movie.shape[1:])
    msg = f"Unknown spatial_domain {spatial_domain!r}"
    raise ValueError(msg)


def _spatial_crop_metadata(
    spatial_domain: SpatialDomain,
    spatial_crop: SpatialCrop,
) -> dict[str, BackgroundMetadataValue]:
    """Return scalar metadata that records the spatial analysis contract.

    Args:
        spatial_domain: User-facing domain name.
        spatial_crop: Crop bounds used by analysis.

    Returns:
        Metadata dictionary safe to store in trace outputs.
    """
    return {
        "spatial_domain": spatial_domain,
        "spatial_axis0_start": spatial_crop.axis0_start,
        "spatial_axis0_stop": spatial_crop.axis0_stop,
        "spatial_axis1_start": spatial_crop.axis1_start,
        "spatial_axis1_stop": spatial_crop.axis1_stop,
        "spatial_original_axis0": spatial_crop.original_shape[0],
        "spatial_original_axis1": spatial_crop.original_shape[1],
    }
