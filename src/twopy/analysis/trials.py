"""Split ROI traces into frame windows for response analysis.

Inputs: converted recordings, ROI traces, frame boundaries, and photodiode
alignment.
Outputs: frame windows, stimulus-epoch labels, and windowed ROI response
arrays.

This module keeps trial slicing independent from the GUI and from source
MATLAB/TIFF files. Stimulus-specific labels can be added when the stimulus
metadata has been decoded.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twopy.converted import RecordingData
from twopy.roi import RoiTraces
from twopy.synchronization import PhotodiodeAlignment

__all__ = [
    "EpochFrameWindow",
    "FrameWindow",
    "WindowedRoiResponse",
    "frame_windows_from_photodiode_alignment",
    "make_frame_windows",
    "map_stimulus_epochs_to_frame_windows",
    "select_epoch_frame_windows",
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


def map_stimulus_epochs_to_frame_windows(
    recording: RecordingData,
    alignment: PhotodiodeAlignment,
) -> tuple[EpochFrameWindow, ...]:
    """Map stimulus epoch runs onto photodiode-aligned imaging-frame windows.

    Args:
        recording: Loaded converted recording with ``stimulus/data`` and epoch
            parameter metadata.
        alignment: Photodiode events paired to imaging frames.

    Returns:
        One ``EpochFrameWindow`` per nonzero stimulus epoch run.

    Raises:
        ValueError: If the stimulus epoch runs and photodiode windows do not
            have the same count.

    The stimulus computer writes one row per stimulus frame while the imaging
    computer writes one movie frame per microscope frame. The photodiode is the
    shared timing evidence. This helper only pairs by order after requiring the
    counts to match, which keeps the assumption visible and auditable.
    """
    epoch_runs = _stimulus_epoch_runs(recording)
    windows = frame_windows_from_photodiode_alignment(
        alignment,
        frame_count=recording.movie.shape[0],
    )
    if len(epoch_runs) != len(windows):
        msg = (
            "Cannot map stimulus epochs to frame windows by order because counts "
            f"differ: stimulus_epoch_runs={len(epoch_runs)}, "
            f"photodiode_windows={len(windows)}"
        )
        raise ValueError(msg)

    epoch_names = _epoch_names_by_number(recording)
    return tuple(
        EpochFrameWindow(
            window=FrameWindow(
                index=window.index,
                start_frame=window.start_frame,
                stop_frame=window.stop_frame,
                label=_epoch_window_label(epoch_number, epoch_names[epoch_number]),
            ),
            epoch_number=epoch_number,
            epoch_name=epoch_names[epoch_number],
        )
        for window, epoch_number in zip(windows, epoch_runs, strict=True)
    )


def select_epoch_frame_windows(
    epoch_windows: tuple[EpochFrameWindow, ...],
    *,
    epoch_name: str | None = None,
    epoch_number: int | None = None,
) -> tuple[FrameWindow, ...]:
    """Select frame windows for one stimulus epoch.

    Args:
        epoch_windows: Stimulus-labeled frame windows.
        epoch_name: Optional epoch name to match exactly.
        epoch_number: Optional epoch number to match exactly.

    Returns:
        Matching ``FrameWindow`` objects.

    Raises:
        ValueError: If neither selector is supplied.

    This is the small script-facing helper used for baseline windows such as
    ``Gray Interleave`` without making dF/F code know stimulus-table details.
    """
    if epoch_name is None and epoch_number is None:
        msg = "Select by epoch_name, epoch_number, or both"
        raise ValueError(msg)

    return tuple(
        epoch_window.window
        for epoch_window in epoch_windows
        if _matches_epoch_window(
            epoch_window,
            epoch_name=epoch_name,
            epoch_number=epoch_number,
        )
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


def _stimulus_epoch_runs(recording: RecordingData) -> tuple[int, ...]:
    """Return nonzero contiguous epoch numbers from ``stimulus/data``.

    Args:
        recording: Loaded converted recording.

    Returns:
        Epoch number for each contiguous nonzero epoch run.
    """
    try:
        epoch_column = recording.stimulus_data_column_names.index("epoch_number")
    except ValueError as error:
        msg = "stimulus/data is missing required column 'epoch_number'"
        raise ValueError(msg) from error

    epoch_numbers = recording.stimulus_data[:, epoch_column].astype(np.int64)
    epoch_numbers = epoch_numbers[epoch_numbers > 0]
    if epoch_numbers.size == 0:
        msg = "stimulus/data does not contain any positive epoch numbers"
        raise ValueError(msg)

    run_start_indices = np.r_[0, np.flatnonzero(np.diff(epoch_numbers) != 0) + 1]
    return tuple(int(epoch_numbers[index]) for index in run_start_indices)


def _epoch_names_by_number(recording: RecordingData) -> dict[int, str]:
    """Create a one-indexed epoch-number to epoch-name map.

    Args:
        recording: Loaded converted recording.

    Returns:
        Dictionary keyed by stimulus epoch number.
    """
    names: dict[int, str] = {}
    for index, parameters in enumerate(recording.stimulus_parameters, start=1):
        raw_name = parameters.get("epochName", f"epoch_{index}")
        names[index] = str(raw_name)
    return names


def _epoch_window_label(epoch_number: int, epoch_name: str) -> str:
    """Create a readable label for a stimulus epoch frame window.

    Args:
        epoch_number: One-indexed stimulus epoch number.
        epoch_name: Human-readable epoch name from ``stimParams.mat``.

    Returns:
        Label suitable for plots and logs.
    """
    safe_name = epoch_name.strip().replace(" ", "_") or "unnamed"
    return f"epoch_{epoch_number:04d}_{safe_name}"


def _matches_epoch_window(
    epoch_window: EpochFrameWindow,
    *,
    epoch_name: str | None,
    epoch_number: int | None,
) -> bool:
    """Check whether one epoch window matches optional selectors.

    Args:
        epoch_window: Candidate stimulus-labeled window.
        epoch_name: Optional epoch name selector.
        epoch_number: Optional epoch number selector.

    Returns:
        ``True`` when all supplied selectors match.
    """
    if epoch_name is not None and epoch_window.epoch_name != epoch_name:
        return False
    return epoch_number is None or epoch_window.epoch_number == epoch_number


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
