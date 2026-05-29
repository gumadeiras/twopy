"""Compute alignment-valid spatial crops during source conversion.

Inputs: alignment text offsets, acquisition metadata, movie shape, and
photodiode signals.
Outputs: a typed crop audit object saved into converted HDF5 metadata.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from twopy.conversion.orientation import twopy_spatial_shape_from_source_spatial_shape
from twopy.conversion.types import (
    AcquisitionMetadata,
    AlignedMovieSource,
    AlignmentCropAudit,
    PhotodiodeSignals,
)
from twopy.photodiode import stimulus_frame_bounds_from_photodiode
from twopy.spatial import SpatialCrop

__all__ = ["load_alignment_valid_crop"]

ONE_X_PIXELS_PER_MICRON_AT_256_X = 0.4
DEFAULT_MOTION_THRESHOLD_PERCENT = 150.0
DEFAULT_PLANE_MOTION_LIMIT_MICRONS = 5.0
DEFAULT_LINESCAN_MOTION_LIMIT_MICRONS = 15.0


def load_alignment_valid_crop(
    *,
    alignment_path: Path,
    aligned_movie: AlignedMovieSource,
    acquisition: AcquisitionMetadata,
    photodiode: PhotodiodeSignals,
) -> AlignmentCropAudit:
    """Compute the alignment-valid crop used by crop-domain analysis.

    Args:
        alignment_path: Text file with per-frame alignment offsets.
        aligned_movie: Aligned movie source metadata.
        acquisition: Acquisition metadata with zoom and frame-rate fields.
        photodiode: Photodiode vectors used to find stimulus-bounded frames.

    Returns:
        Crop bounds plus audit values.

    The source analysis path estimates this crop from alignment rows inside the
    stimulus presentation window, then applies background correction inside that
    crop. twopy saves the same contract while keeping the converted movie
    full-frame.
    """
    alignment = np.asarray(np.loadtxt(alignment_path), dtype=np.float64)
    if alignment.ndim != 2 or alignment.shape[1] < 2:
        msg = f"Alignment text must be a 2D table with x/y offsets: {alignment_path}"
        raise ValueError(msg)

    frame_start, frame_stop = _stimulus_alignment_frame_range(
        aligned_movie=aligned_movie,
        acquisition=acquisition,
        photodiode=photodiode,
    )
    if frame_stop > alignment.shape[0]:
        msg = (
            "Stimulus-bounded alignment range exceeds alignment rows: "
            f"[{frame_start}, {frame_stop}) for {alignment.shape[0]} rows"
        )
        raise ValueError(msg)

    alignment_window = alignment[frame_start:frame_stop, :]
    line_scan = _is_line_scan(acquisition)
    source_spatial_shape = _source_movie_spatial_shape(aligned_movie)
    twopy_spatial_shape = twopy_spatial_shape_from_source_spatial_shape(
        source_spatial_shape,
    )
    motion_threshold_pixels = _motion_threshold_pixels(
        acquisition=acquisition,
        source_x_pixel_count=source_spatial_shape[0],
        line_scan=line_scan,
    )
    alignment_shift_pixels = _alignment_shift_pixels(
        alignment[: aligned_movie.shape[0], :],
        line_scan=line_scan,
    )
    motion_artifact_mask = alignment_shift_pixels > motion_threshold_pixels
    kept_alignment, over_moved_count = _remove_over_moved_alignment_rows(
        alignment_window,
        line_scan=line_scan,
        motion_threshold_pixels=motion_threshold_pixels,
    )
    crop, x_cutoff, y_cutoff = _alignment_crop_from_offsets(
        kept_alignment,
        spatial_shape=twopy_spatial_shape,
        line_scan=line_scan,
    )

    return AlignmentCropAudit(
        crop=crop,
        alignment_frame_start=frame_start,
        alignment_frame_stop=frame_stop,
        x_cutoff_pixels=x_cutoff,
        y_cutoff_pixels=y_cutoff,
        over_moved_frame_count=over_moved_count,
        motion_threshold_pixels=motion_threshold_pixels,
        alignment_shift_pixels=alignment_shift_pixels,
        motion_artifact_mask=motion_artifact_mask,
    )


def _stimulus_alignment_frame_range(
    *,
    aligned_movie: AlignedMovieSource,
    acquisition: AcquisitionMetadata,
    photodiode: PhotodiodeSignals,
) -> tuple[int, int]:
    """Find the alignment rows that cover stimulus presentation.

    Args:
        aligned_movie: Movie metadata that gives frame count.
        acquisition: Acquisition metadata with frame rate and line-scan flag.
        photodiode: High-resolution photodiode signal.

    Returns:
        Half-open frame range used for the alignment-valid crop.
    """
    if _is_line_scan(acquisition):
        return (0, aligned_movie.shape[0])

    return stimulus_frame_bounds_from_photodiode(
        photodiode.high_res_pd,
        movie_frame_count=aligned_movie.shape[0],
        frame_rate_hz=_required_float_field(acquisition, "acq.frameRate"),
    )


def _motion_threshold_pixels(
    *,
    acquisition: AcquisitionMetadata,
    source_x_pixel_count: int,
    line_scan: bool,
) -> float:
    """Convert the motion rejection threshold from microns to pixels.

    Args:
        acquisition: Acquisition metadata with zoom.
        source_x_pixel_count: Number of source pixels along the physical x axis.
        line_scan: Whether this recording is a line scan.

    Returns:
        Maximum allowed alignment movement in pixels.
    """
    micron_limit = (
        DEFAULT_LINESCAN_MOTION_LIMIT_MICRONS
        if line_scan
        else DEFAULT_PLANE_MOTION_LIMIT_MICRONS
    )
    zoom = _required_float_field(acquisition, "acq.zoomFactor")
    x_axis_conversion = 256.0 / float(source_x_pixel_count)
    return micron_limit * ONE_X_PIXELS_PER_MICRON_AT_256_X * zoom * x_axis_conversion


def _remove_over_moved_alignment_rows(
    alignment: npt.NDArray[np.float64],
    *,
    line_scan: bool,
    motion_threshold_pixels: float,
) -> tuple[npt.NDArray[np.float64], int]:
    """Drop alignment rows with motion beyond the allowed threshold.

    Args:
        alignment: Stimulus-bounded alignment table.
        line_scan: Whether this recording is a line scan.
        motion_threshold_pixels: Maximum allowed movement in pixels.

    Returns:
        Filtered alignment rows and the number of dropped rows.
    """
    full_shift = _alignment_shift_pixels(alignment, line_scan=line_scan)
    over_moved = full_shift > motion_threshold_pixels
    over_moved_percent = (
        100.0 * float(np.count_nonzero(over_moved)) / alignment.shape[0]
    )
    if over_moved_percent > DEFAULT_MOTION_THRESHOLD_PERCENT:
        return alignment, 0

    kept = alignment[~over_moved, :]
    if kept.size == 0:
        msg = (
            "Alignment-valid crop cannot be computed; all alignment rows moved too far"
        )
        raise ValueError(msg)
    return kept, int(np.count_nonzero(over_moved))


def _alignment_shift_pixels(
    alignment: npt.NDArray[np.float64],
    *,
    line_scan: bool,
) -> npt.NDArray[np.float64]:
    """Return per-frame alignment shift magnitudes in pixels.

    Args:
        alignment: Alignment table with x/y offset columns.
        line_scan: Whether this recording is a line scan.

    Returns:
        One shift magnitude per alignment row.
    """
    if line_scan:
        return np.abs(alignment[:, 0])
    return np.sqrt(alignment[:, 0] ** 2 + alignment[:, 1] ** 2)


def _alignment_crop_from_offsets(
    alignment: npt.NDArray[np.float64],
    *,
    spatial_shape: tuple[int, int],
    line_scan: bool,
) -> tuple[SpatialCrop, int, int]:
    """Convert alignment offsets into twopy movie-axis crop bounds.

    Args:
        alignment: Filtered alignment table.
        spatial_shape: Full movie spatial shape in twopy array axes.
        line_scan: Whether this recording is a line scan.

    Returns:
        Spatial crop plus x/y cutoff values.

    Source alignment columns are x/y motion offsets. Converted twopy arrays use
    Python image axis order matching MATLAB display of the source data: spatial
    axis 0 is displayed y/rows, and spatial axis 1 is displayed x/columns.
    """
    x_cutoff = _ceil_abs_offset(alignment[:, 0])
    y_cutoff = 1 if line_scan else _ceil_abs_offset(alignment[:, 1])
    axis0_start, axis0_stop = _crop_bounds_from_cutoff(
        length=spatial_shape[0],
        cutoff=y_cutoff,
    )
    axis1_start, axis1_stop = _crop_bounds_from_cutoff(
        length=spatial_shape[1],
        cutoff=x_cutoff,
    )
    return (
        SpatialCrop(
            axis0_start=axis0_start,
            axis0_stop=axis0_stop,
            axis1_start=axis1_start,
            axis1_stop=axis1_stop,
            original_shape=spatial_shape,
            source="alignment_valid_crop",
        ),
        x_cutoff,
        y_cutoff,
    )


def _ceil_abs_offset(values: npt.NDArray[np.float64]) -> int:
    """Return the ceiling of the largest absolute alignment offset.

    Args:
        values: One alignment offset column.

    Returns:
        Nonnegative integer cutoff.
    """
    return int(np.ceil(float(np.nanmax(np.abs(values)))))


def _crop_bounds_from_cutoff(*, length: int, cutoff: int) -> tuple[int, int]:
    """Convert a source cutoff value into Python half-open bounds.

    Args:
        length: Axis length.
        cutoff: One-based cutoff convention from the source crop code.

    Returns:
        ``(start, stop)`` bounds.
    """
    if cutoff <= 1:
        return (0, length)
    start = cutoff - 1
    stop = length - cutoff + 1
    if start >= stop:
        msg = f"Alignment cutoff {cutoff} leaves no valid pixels for length {length}"
        raise ValueError(msg)
    return start, stop


def _source_movie_spatial_shape(aligned_movie: AlignedMovieSource) -> tuple[int, int]:
    """Return the source spatial dimensions of the aligned source movie.

    Args:
        aligned_movie: Source movie metadata.

    Returns:
        ``(source_x, source_y)`` spatial shape.
    """
    if len(aligned_movie.shape) != 3:
        msg = f"Aligned movie must be 3D; got {aligned_movie.shape}"
        raise ValueError(msg)
    return (aligned_movie.shape[1], aligned_movie.shape[2])


def _required_float_field(acquisition: AcquisitionMetadata, key: str) -> float:
    """Read one required numeric acquisition metadata field.

    Args:
        acquisition: Loaded acquisition metadata.
        key: Field name to read.

    Returns:
        Field value as a float.
    """
    value = acquisition.fields[key]
    if isinstance(value, str | bytes | int | float | np.generic):
        return float(value)

    value_type = type(value).__name__
    msg = f"Acquisition field {key!r} must be numeric, got {value_type}"
    raise ValueError(msg)


def _is_line_scan(acquisition: AcquisitionMetadata) -> bool:
    """Return whether acquisition metadata describes a line scan.

    Args:
        acquisition: Loaded acquisition metadata.

    Returns:
        ``True`` for line-scan recordings.
    """
    return _required_float_field(acquisition, "acq.scanAngleMultiplierSlow") == 0.0
