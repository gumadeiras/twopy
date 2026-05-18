"""Resolve native stimulus timing for converted recordings.

Inputs: converted ``RecordingData`` stimulus and photodiode evidence.
Outputs: one audited set of epoch frame windows for native analysis callers.

This module owns the native timing decision. Parity code intentionally keeps
its own historical timing path and should not be routed through this resolver.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from twopy._segments import constant_segments
from twopy.analysis.trials import EpochFrameWindow, FrameWindow
from twopy.converted import RecordingData, recording_frame_rate_hz
from twopy.photodiode import stimulus_frame_bounds_from_photodiode
from twopy.photodiode_classification import (
    ClassifiedStimulusWindow,
    classify_recording_photodiode_events,
)
from twopy.stimulus import stimulus_column_index, stimulus_epoch_names_by_number
from twopy.synchronization import detect_recording_photodiode_events

__all__ = [
    "RecordingTiming",
    "TimingMetadataValue",
    "TimingSource",
    "resolve_recording_timing",
]

TimingSource = Literal["classified_photodiode", "interpolated_epochs"]
TimingMetadataValue = str | int | float | bool


@dataclass(frozen=True)
class RecordingTiming:
    """Resolved native timing for one converted recording.

    Args:
        frame_rate_hz: Imaging frame rate used for frame/time conversion.
        epoch_windows: Stimulus epoch windows in aligned-movie frame
            coordinates.
        source: Timing path that produced the windows.
        metadata: Small audit dictionary with path-specific counts and bounds.
    """

    frame_rate_hz: float
    epoch_windows: tuple[EpochFrameWindow, ...]
    source: TimingSource
    metadata: dict[str, TimingMetadataValue]


def resolve_recording_timing(recording: RecordingData) -> RecordingTiming:
    """Resolve the native stimulus timing for one converted recording.

    Args:
        recording: Loaded converted recording.

    Returns:
        Frame rate, epoch windows, source label, and audit metadata.

    Raises:
        ValueError: If the recording has applicable classified photodiode
            evidence but that evidence is internally inconsistent, or if the
            fallback interpolation path cannot produce windows.

    Native twopy prefers classified boundary-flash timing when the converted
    stimulus table contains active ``photodiode_flash`` evidence. Recordings
    without that contract use the interpolation timing path.
    """
    frame_rate_hz = recording_frame_rate_hz(recording)
    if _has_active_photodiode_flash_column(recording):
        return _classified_recording_timing(recording, frame_rate_hz=frame_rate_hz)
    return _interpolated_recording_timing(recording, frame_rate_hz=frame_rate_hz)


def _classified_recording_timing(
    recording: RecordingData,
    *,
    frame_rate_hz: float,
) -> RecordingTiming:
    """Resolve timing from paired and classified photodiode boundary events."""
    try:
        alignment = detect_recording_photodiode_events(recording)
        classified = classify_recording_photodiode_events(recording, alignment)
    except ValueError as error:
        msg = f"Could not resolve classified photodiode timing: {error}"
        raise ValueError(msg) from error

    windows = tuple(
        _epoch_window_from_classified(window) for window in classified.windows
    )
    if len(windows) == 0:
        msg = "Classified photodiode timing did not produce any epoch windows"
        raise ValueError(msg)

    metadata: dict[str, TimingMetadataValue] = {
        "event_count": len(classified.events),
        "window_count": len(windows),
        "high_res_event_count": len(alignment.high_res.events),
        "imaging_res_event_count": len(alignment.imaging_res.events),
    }
    metadata.update(classified.metadata)
    return RecordingTiming(
        frame_rate_hz=frame_rate_hz,
        epoch_windows=windows,
        source="classified_photodiode",
        metadata=metadata,
    )


def _interpolated_recording_timing(
    recording: RecordingData,
    *,
    frame_rate_hz: float,
) -> RecordingTiming:
    """Resolve timing by interpolating stimulus epochs onto movie frames."""
    start_frame, stop_frame = stimulus_frame_bounds_from_photodiode(
        recording.high_res_pd,
        movie_frame_count=recording.movie.shape[0],
        frame_rate_hz=frame_rate_hz,
    )
    frame_count = stop_frame - start_frame
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
        frame_count=frame_count,
        stimulus_duration_seconds=frame_count / frame_rate_hz,
    )
    windows = _epoch_windows_from_frame_values(
        epoch_numbers=epoch_numbers,
        start_frame=start_frame,
        epoch_names=stimulus_epoch_names_by_number(recording),
    )
    if len(windows) == 0:
        msg = "Interpolated timing did not produce any epoch windows"
        raise ValueError(msg)
    return RecordingTiming(
        frame_rate_hz=frame_rate_hz,
        epoch_windows=windows,
        source="interpolated_epochs",
        metadata={
            "start_frame": start_frame,
            "stop_frame": stop_frame,
            "frame_count": frame_count,
            "window_count": len(windows),
        },
    )


def _has_active_photodiode_flash_column(recording: RecordingData) -> bool:
    """Return whether the recording has active boundary-flash stimulus evidence."""
    try:
        flash_column = stimulus_column_index(recording, "photodiode_flash")
    except ValueError:
        return False
    flash_values = recording.stimulus_data[:, flash_column]
    return bool(np.any(flash_values > 0.5))


def _epoch_window_from_classified(
    window: ClassifiedStimulusWindow,
) -> EpochFrameWindow:
    """Convert a classified stimulus window into the common epoch-window shape."""
    return EpochFrameWindow(
        window=FrameWindow(
            index=window.index,
            start_frame=window.start_frame,
            stop_frame=window.stop_frame,
            label=_epoch_window_label(window.epoch_number, window.epoch_name),
        ),
        epoch_number=window.epoch_number,
        epoch_name=window.epoch_name,
    )


def _interpolate_epoch_numbers(
    *,
    time_values: npt.NDArray[np.float64],
    epoch_values: npt.NDArray[np.float64],
    frame_count: int,
    stimulus_duration_seconds: float,
) -> npt.NDArray[np.int64]:
    """Interpolate positive stimulus epochs onto imaging-frame slots."""
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
    """Find nearest source row for each target time."""
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
    """Build contiguous epoch windows from per-frame epoch labels."""
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
                    label=_epoch_window_label(epoch_number, epoch_name),
                ),
                epoch_number=epoch_number,
                epoch_name=epoch_name,
            ),
        )
    return tuple(windows)


def _epoch_window_label(epoch_number: int, epoch_name: str) -> str:
    """Return the standard readable epoch-window label."""
    return f"epoch_{epoch_number}:{epoch_name}"
