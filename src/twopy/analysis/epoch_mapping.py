"""Map stimulus epochs onto imaging frames by interpolation.

Inputs: converted stimulus data, high-rate photodiode bounds, and acquisition
frame rate.
Outputs: per-frame epoch numbers and stimulus-labeled frame windows.

This module reproduces the lab analysis timing contract for frame movies: use
the photodiode to find the stimulus-bounded imaging frames, then interpolate the
stimulus-computer epoch column onto those imaging frames. It does not extract
ROIs or compute responses.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twopy._segments import constant_segments
from twopy.analysis.trials import EpochFrameWindow, FrameWindow
from twopy.converted import RecordingData, recording_frame_rate_hz
from twopy.photodiode import stimulus_frame_bounds_from_photodiode
from twopy.stimulus import stimulus_column_index, stimulus_epoch_names_by_number

__all__ = [
    "InterpolatedEpochMapping",
    "interpolate_stimulus_epochs_to_frame_windows",
]


@dataclass(frozen=True)
class InterpolatedEpochMapping:
    """Stimulus epochs interpolated onto imaging frames.

    Inputs: converted stimulus rows and photodiode-derived frame bounds.
    Outputs: the absolute frame range, one epoch value per frame, and contiguous
    epoch windows with names.

    Keeping the per-frame list beside the windows makes timing easy to audit
    against saved analysis outputs.
    """

    start_frame: int
    stop_frame: int
    epoch_numbers: npt.NDArray[np.int64]
    windows: tuple[EpochFrameWindow, ...]


def interpolate_stimulus_epochs_to_frame_windows(
    recording: RecordingData,
) -> InterpolatedEpochMapping:
    """Interpolate stimulus epoch values onto stimulus-bounded movie frames.

    Args:
        recording: Loaded converted recording with stimulus data, acquisition
            metadata, and high-rate photodiode signal.

    Returns:
        ``InterpolatedEpochMapping`` with one epoch value per imaging frame and
        one window per contiguous positive epoch run.

    Raises:
        ValueError: If required columns or frame-rate metadata are missing.

    The stimulus computer owns epoch times. The imaging computer owns movie
    frames. The photodiode defines the stimulus-bounded movie frame range, and
    nearest-neighbor interpolation assigns the stimulus epoch number to each
    frame inside that range.
    """
    frame_rate_hz = recording_frame_rate_hz(recording)
    start_frame, stop_frame = stimulus_frame_bounds_from_photodiode(
        recording.high_res_pd,
        movie_frame_count=recording.movie.shape[0],
        frame_rate_hz=frame_rate_hz,
    )
    time_values = recording.stimulus_data[
        :,
        stimulus_column_index(recording, "time_seconds"),
    ]
    epoch_values = recording.stimulus_data[
        :,
        stimulus_column_index(recording, "epoch_number"),
    ]
    epoch_numbers = _interpolate_epoch_numbers(
        time_values=time_values,
        epoch_values=epoch_values,
        frame_count=stop_frame - start_frame,
        stimulus_duration_seconds=(stop_frame - start_frame) / frame_rate_hz,
    )
    windows = _epoch_windows_from_frame_values(
        epoch_numbers=epoch_numbers,
        start_frame=start_frame,
        epoch_names=stimulus_epoch_names_by_number(recording),
    )
    return InterpolatedEpochMapping(
        start_frame=start_frame,
        stop_frame=stop_frame,
        epoch_numbers=epoch_numbers,
        windows=windows,
    )


def _interpolate_epoch_numbers(
    *,
    time_values: npt.NDArray[np.float64],
    epoch_values: npt.NDArray[np.float64],
    frame_count: int,
    stimulus_duration_seconds: float,
) -> npt.NDArray[np.int64]:
    """Interpolate positive stimulus epochs onto imaging-frame slots.

    Args:
        time_values: Stimulus-computer times in seconds.
        epoch_values: Stimulus epoch numbers from the same rows.
        frame_count: Number of imaging frames to label.
        stimulus_duration_seconds: Stimulus-bounded movie duration.

    Returns:
        Integer epoch number per imaging frame.
    """
    if frame_count < 1:
        msg = f"frame_count must be positive; got {frame_count}"
        raise ValueError(msg)

    positive = epoch_values != 0
    times = np.asarray(time_values[positive], dtype=np.float64)
    epochs = np.asarray(epoch_values[positive], dtype=np.float64)
    keep = times <= stimulus_duration_seconds
    times = times[keep]
    epochs = epochs[keep]
    if times.size == 0:
        msg = "stimulus/data has no positive epoch rows inside the stimulus window"
        raise ValueError(msg)
    if np.any(np.diff(times) < 0):
        msg = "stimulus/data time_seconds must be sorted for epoch interpolation"
        raise ValueError(msg)

    target_times = np.linspace(times[0], times[-1], frame_count)
    nearest_indices = _nearest_time_indices(times, target_times)
    return np.rint(epochs[nearest_indices]).astype(np.int64)


def _nearest_time_indices(
    source_times: npt.NDArray[np.float64],
    target_times: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """Find nearest source row for each target time.

    Args:
        source_times: Sorted source times.
        target_times: Times that need source-row labels.

    Returns:
        Integer source index per target time.
    """
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, source_times.size - 1)
    left = np.clip(right - 1, 0, source_times.size - 1)
    choose_left = np.abs(target_times - source_times[left]) <= np.abs(
        source_times[right] - target_times
    )
    return np.where(choose_left, left, right).astype(np.int64)


def _epoch_windows_from_frame_values(
    *,
    epoch_numbers: npt.NDArray[np.int64],
    start_frame: int,
    epoch_names: dict[int, str],
) -> tuple[EpochFrameWindow, ...]:
    """Build contiguous epoch windows from per-frame epoch labels.

    Args:
        epoch_numbers: Epoch value per stimulus-bounded imaging frame.
        start_frame: Absolute movie frame for ``epoch_numbers[0]``.
        epoch_names: Epoch-number to name mapping.

    Returns:
        One window per positive contiguous epoch run.
    """
    windows: list[EpochFrameWindow] = []
    for start, stop in constant_segments(epoch_numbers):
        epoch_number = int(epoch_numbers[start])
        if epoch_number <= 0:
            continue
        epoch_name = epoch_names.get(epoch_number, f"epoch_{epoch_number}")
        absolute_start = start_frame + start
        absolute_stop = start_frame + stop
        windows.append(
            EpochFrameWindow(
                window=FrameWindow(
                    index=len(windows),
                    start_frame=absolute_start,
                    stop_frame=absolute_stop,
                    label=f"epoch_{epoch_number}:{epoch_name}",
                ),
                epoch_number=epoch_number,
                epoch_name=epoch_name,
            ),
        )
    return tuple(windows)
