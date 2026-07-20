"""Split ROI traces into frame windows for response analysis.

Inputs: ROI traces and explicit stimulus frame windows.
Outputs: frame windows, stimulus-epoch labels, and windowed ROI response
arrays.

This module keeps trial slicing independent from the GUI and from source files.
Stimulus-specific labels can be added when the stimulus metadata has been
decoded.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twopy.roi import RoiTraces

__all__ = [
    "EpochFrameWindow",
    "FrameWindow",
    "WindowedRoiResponse",
    "default_baseline_epoch_number",
    "is_baseline_epoch_name",
    "make_frame_windows",
    "no_baseline_epoch_frame_windows",
    "resolve_baseline_frame_windows",
    "select_baseline_frame_windows",
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
class EpochFrameWindow:
    """One photodiode-aligned frame window with stimulus epoch metadata.

    Inputs: one frame window and the matching stimulus epoch run.
    Outputs: frame bounds plus the epoch number and epoch name.

    This object is the bridge between stimulus-computer rows and imaging-frame
    movie slices. It is intentionally small so later response code can inspect
    exactly which epoch each frame window came from.
    """

    window: FrameWindow
    epoch_number: int
    epoch_name: str


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


def is_baseline_epoch_name(epoch_name: str) -> bool:
    """Return whether an epoch name looks like a baseline epoch.

    Args:
        epoch_name: Stimulus epoch name.

    Returns:
        ``True`` for ``gray`` or ``grey`` spelling, or for names containing
        ``interleave``.
    """
    lowered = epoch_name.lower()
    return "gray" in lowered or "grey" in lowered or "interleave" in lowered


def default_baseline_epoch_number(epoch_names: Mapping[int, str]) -> int:
    """Return the default baseline epoch from readable epoch names.

    Args:
        epoch_names: Mapping from stimulus epoch number to epoch name.

    Returns:
        First epoch whose name looks like gray/grey/interleave, otherwise
        epoch 1.
    """
    for epoch_number, epoch_name in sorted(epoch_names.items()):
        if is_baseline_epoch_name(epoch_name):
            return epoch_number
    return 1


def resolve_baseline_frame_windows(
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    baseline_windows: Sequence[FrameWindow] | None = None,
    epoch_name: str | None = None,
    epoch_number: int | None = 1,
    epoch_numbers: Sequence[int] | None = None,
) -> tuple[FrameWindow, ...]:
    """Resolve explicit or epoch-selected baseline frame windows.

    Args:
        epoch_windows: Stimulus-labeled frame windows.
        baseline_windows: Optional explicit baseline windows. When present,
            these are returned directly.
        epoch_name: Optional baseline epoch name selector.
        epoch_number: Optional single baseline epoch number selector.
        epoch_numbers: Optional multiple baseline epoch number selector.

    Returns:
        Baseline frame windows.
    """
    if baseline_windows is not None:
        return tuple(baseline_windows)
    return select_baseline_frame_windows(
        epoch_windows,
        epoch_name=epoch_name,
        epoch_number=epoch_number,
        epoch_numbers=epoch_numbers,
    )


def select_baseline_frame_windows(
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    epoch_name: str | None = None,
    epoch_number: int | None = None,
    epoch_numbers: Sequence[int] | None = None,
) -> tuple[FrameWindow, ...]:
    """Select baseline frame windows from stimulus-labeled epoch windows.

    Args:
        epoch_windows: Stimulus-labeled frame windows.
        epoch_name: Optional epoch name to match exactly.
        epoch_number: Optional single epoch number to match exactly.
        epoch_numbers: Optional multiple epoch numbers to match exactly.

    Returns:
        Matching ``FrameWindow`` objects.

    Raises:
        ValueError: If no selector is supplied or both number selector forms are
            supplied.

    This is the native twopy baseline selector. Callers that need psycho5's
    historical default rules should compute epoch numbers in ``twopy.parity``
    and pass them here.
    """
    if epoch_number is not None and epoch_numbers is not None:
        msg = "Select by epoch_number or epoch_numbers, not both"
        raise ValueError(msg)
    selected_numbers = frozenset(
        epoch_numbers
        if epoch_numbers is not None
        else (epoch_number,)
        if epoch_number is not None
        else ()
    )
    if epoch_name is None and len(selected_numbers) == 0:
        msg = "Select by epoch_name, epoch_number, or epoch_numbers"
        raise ValueError(msg)

    return tuple(
        epoch_window.window
        for epoch_window in epoch_windows
        if _matches_baseline_epoch_window(
            epoch_window,
            epoch_name=epoch_name,
            epoch_numbers=selected_numbers,
        )
    )


def no_baseline_epoch_frame_windows(
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    start_epoch_number: int,
) -> tuple[FrameWindow, ...]:
    """Return one continuous baseline span from a starting epoch onward.

    Args:
        epoch_windows: Stimulus-labeled frame windows.
        start_epoch_number: First one-based epoch number included in the span.

    Returns:
        A single ``FrameWindow`` covering all frames from the earliest selected
        epoch occurrence through the end of the latest selected occurrence.

    This mode is for stimuli without a distinct baseline epoch. The baseline
    fit uses a continuous span rather than repeated baseline windows.
    """
    selected = tuple(
        epoch_window
        for epoch_window in epoch_windows
        if epoch_window.epoch_number >= start_epoch_number
    )
    if len(selected) == 0:
        return ()

    start_frame = min(epoch_window.window.start_frame for epoch_window in selected)
    stop_frame = max(epoch_window.window.stop_frame for epoch_window in selected)
    return (
        FrameWindow(
            index=0,
            start_frame=start_frame,
            stop_frame=stop_frame,
            label=f"no_baseline_epoch_from_epoch_{start_epoch_number:04d}",
        ),
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

    This function only slices. It does not compute response metrics. Keeping
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


def _matches_baseline_epoch_window(
    epoch_window: EpochFrameWindow,
    *,
    epoch_name: str | None,
    epoch_numbers: frozenset[int],
) -> bool:
    """Check whether one epoch window matches optional selectors.

    Args:
        epoch_window: Candidate stimulus-labeled window.
        epoch_name: Optional epoch name selector.
        epoch_numbers: Optional epoch number selectors.

    Returns:
        ``True`` when all supplied selectors match.
    """
    if epoch_name is not None and epoch_window.epoch_name != epoch_name:
        return False
    return len(epoch_numbers) == 0 or epoch_window.epoch_number in epoch_numbers


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
        msg = f"Expected {window_count} labels. Got {len(labels)}"
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
