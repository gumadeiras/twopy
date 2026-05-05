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
    "ROI_FILE_FORMAT",
    "RoiSet",
    "RoiTraces",
    "extract_roi_traces",
    "load_roi_set",
    "make_roi_set",
    "save_roi_set",
]

ROI_FILE_FORMAT = "twopy-roi-set"
TraceStatistic = Literal["mean"]


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
        msg = (
            f"ROI mask shape {roi_set.masks.shape[1:]} does not match movie "
            f"shape {recording.movie.shape[1:]}"
        )
        raise ValueError(msg)

    frame_start, frame_stop = _normalize_trace_frame_range(
        frame_count=recording.movie.shape[0],
        start_frame=start_frame,
        stop_frame=stop_frame,
    )
    traces = np.empty(
        (frame_stop - frame_start, roi_set.masks.shape[0]),
        dtype=np.float64,
    )
    roi_pixel_indices = _roi_pixel_indices(roi_set)

    for chunk_start, chunk_stop, frames in recording.movie.iter_frame_batches(
        chunk_frames=chunk_frames,
        start=frame_start,
        stop=frame_stop,
    ):
        chunk_offset = chunk_start - frame_start
        chunk_flat = frames.reshape(frames.shape[0], -1)
        for roi_index, pixel_indices in enumerate(roi_pixel_indices):
            traces[
                chunk_offset : chunk_offset + (chunk_stop - chunk_start),
                roi_index,
            ] = chunk_flat[:, pixel_indices].mean(axis=1)

    return RoiTraces(
        values=traces,
        labels=roi_set.labels,
        start_frame=frame_start,
        stop_frame=frame_stop,
        statistic=statistic,
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
