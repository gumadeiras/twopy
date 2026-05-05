"""Split ROI traces into frame windows for response analysis.

Inputs: ROI traces and frame boundaries from photodiode alignment.
Outputs: windowed ROI response arrays.

This module keeps trial slicing independent from the GUI and from source
MATLAB/TIFF files. Stimulus-specific labels can be added when the stimulus
metadata has been decoded.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twopy.roi import RoiTraces
from twopy.synchronization import PhotodiodeAlignment

__all__ = [
    "FrameWindow",
    "WindowedRoiResponse",
    "frame_windows_from_photodiode_alignment",
    "make_frame_windows",
    "split_traces_by_frame_windows",
]


@dataclass(frozen=True)
class FrameWindow:
    """One half-open frame interval.

    Inputs: start and stop frame indices in aligned movie coordinates.
    Outputs: a labeled interval that can slice ROI traces.

    ``start_frame`` is inclusive and ``stop_frame`` is exclusive, matching
    Python slicing and HDF5 frame reads.
    """

    index: int
    start_frame: int
    stop_frame: int
    label: str


@dataclass(frozen=True)
class WindowedRoiResponse:
    """ROI traces for one frame window.

    Inputs: one ``FrameWindow`` and the matching slice of ``RoiTraces``.
    Outputs: frame-by-ROI values plus ROI labels.

    The values remain in frame order so callers can compute baselines,
    peak responses, averages, or plots without losing timing.
    """

    window: FrameWindow
    values: npt.NDArray[np.float64]
    roi_labels: tuple[str, ...]


def make_frame_windows(
    boundary_frames: Sequence[int],
    *,
    frame_count: int,
    labels: Sequence[str] | None = None,
) -> tuple[FrameWindow, ...]:
    """Create frame windows from explicit frame boundaries.

    Args:
        boundary_frames: Ordered frame boundaries. Each neighboring pair forms
            one window.
        frame_count: Total number of aligned movie frames.
        labels: Optional label for each output window.

    Returns:
        Tuple of ``FrameWindow`` objects.

    Raises:
        ValueError: If boundaries are out of range, not increasing, or too few.

    The caller owns what the boundaries mean: trials, epochs, baseline periods,
    or any other stimulus-derived interval.
    """
    boundaries = tuple(int(frame) for frame in boundary_frames)
    if len(boundaries) < 2:
        msg = "At least two frame boundaries are required to make windows"
        raise ValueError(msg)
    if any(frame < 0 or frame > frame_count for frame in boundaries):
        msg = f"Frame boundaries must be within [0, {frame_count}]: {boundaries}"
        raise ValueError(msg)
    if any(
        stop <= start for start, stop in zip(boundaries, boundaries[1:], strict=False)
    ):
        msg = f"Frame boundaries must be strictly increasing: {boundaries}"
        raise ValueError(msg)

    window_count = len(boundaries) - 1
    resolved_labels = _resolve_window_labels(window_count, labels)
    return tuple(
        FrameWindow(
            index=index,
            start_frame=boundaries[index],
            stop_frame=boundaries[index + 1],
            label=resolved_labels[index],
        )
        for index in range(window_count)
    )


def frame_windows_from_photodiode_alignment(
    alignment: PhotodiodeAlignment,
    *,
    frame_count: int,
    labels: Sequence[str] | None = None,
) -> tuple[FrameWindow, ...]:
    """Create frame windows from paired photodiode event frames.

    Args:
        alignment: Photodiode events paired to imaging frames.
        frame_count: Total number of aligned movie frames.
        labels: Optional label for each neighboring event interval.

    Returns:
        Frame windows between consecutive paired photodiode events.

    This turns decoded photodiode boundaries into frame intervals without
    assuming what each event means in a stimulus-specific protocol.
    """
    return make_frame_windows(
        tuple(event.imaging_frame for event in alignment.paired_events),
        frame_count=frame_count,
        labels=labels,
    )


def split_traces_by_frame_windows(
    traces: RoiTraces,
    windows: Sequence[FrameWindow],
) -> tuple[WindowedRoiResponse, ...]:
    """Split ROI traces into explicit frame windows.

    Args:
        traces: ROI traces with known start and stop frame indices.
        windows: Frame windows to slice.

    Returns:
        One ``WindowedRoiResponse`` per window.

    Raises:
        ValueError: If a window falls outside the trace frame range.

    This function only slices; it does not compute response metrics. Keeping
    those steps separate makes later baseline and trial logic auditable.
    """
    responses: list[WindowedRoiResponse] = []
    for window in windows:
        _validate_window_inside_traces(window, traces)
        start = window.start_frame - traces.start_frame
        stop = window.stop_frame - traces.start_frame
        responses.append(
            WindowedRoiResponse(
                window=window,
                values=traces.values[start:stop, :],
                roi_labels=traces.labels,
            ),
        )

    return tuple(responses)


def _resolve_window_labels(
    window_count: int,
    labels: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve optional frame-window labels.

    Args:
        window_count: Number of windows being created.
        labels: Optional caller labels.

    Returns:
        One label per window.
    """
    if labels is None:
        return tuple(f"window_{index + 1:04d}" for index in range(window_count))
    if len(labels) != window_count:
        msg = f"Expected {window_count} labels; got {len(labels)}"
        raise ValueError(msg)
    return tuple(labels)


def _validate_window_inside_traces(window: FrameWindow, traces: RoiTraces) -> None:
    """Confirm one frame window can slice a trace matrix.

    Args:
        window: Frame window to check.
        traces: ROI traces with available frame bounds.

    Returns:
        None.
    """
    if window.start_frame < traces.start_frame or window.stop_frame > traces.stop_frame:
        msg = (
            f"Window [{window.start_frame}, {window.stop_frame}) is outside "
            f"trace range [{traces.start_frame}, {traces.stop_frame})"
        )
        raise ValueError(msg)
