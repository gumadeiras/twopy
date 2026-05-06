"""Classify photodiode events into stimulus timing objects.

Inputs: converted stimulus data plus paired high-rate and imaging-frame
photodiode events.
Outputs: classified start, trial-transition, and end events plus imaging-frame
windows with stimulus epoch metadata.

This module is the semantic layer above raw photodiode event detection. It
cross-checks detected photodiode events against the converted
``stimulus/data`` ``photodiode_flash`` column before producing trial windows,
so response analysis does not rely on event order alone.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from twopy._segments import constant_segments, true_segments
from twopy.converted import RecordingData
from twopy.stimulus import stimulus_column_index, stimulus_epoch_names_by_number
from twopy.synchronization import AlignedPhotodiodeEvent, PhotodiodeAlignment

__all__ = [
    "ClassifiedPhotodiodeEvent",
    "ClassifiedStimulusTiming",
    "ClassifiedStimulusWindow",
    "PhotodiodeDurationClass",
    "PhotodiodeEventType",
    "classify_recording_photodiode_events",
]

PhotodiodeEventType = Literal[
    "stimulus_start",
    "trial_transition",
    "stimulus_end",
]
PhotodiodeDurationClass = Literal["short", "long"]
PhotodiodeClassificationMetadataValue = str | int | float | bool


@dataclass(frozen=True)
class ClassifiedPhotodiodeEvent:
    """One photodiode event with a stimulus-timing role.

    Inputs: one paired photodiode event and its matching
    ``stimulus/data`` flash segment.
    Outputs: event type, duration class, imaging frame, and stimulus row audit
    fields.

    ``stimulus_flash_start_row`` and ``stimulus_flash_stop_row`` point back to
    the stimulus-computer table rows that requested the flash. The aligned
    event carries the measured high-rate and frame-resolution photodiode
    evidence.
    """

    index: int
    event_type: PhotodiodeEventType
    duration_class: PhotodiodeDurationClass
    aligned_event: AlignedPhotodiodeEvent
    imaging_frame: int
    stimulus_flash_start_row: int
    stimulus_flash_stop_row: int
    stimulus_flash_duration_rows: int


@dataclass(frozen=True)
class ClassifiedStimulusWindow:
    """One stimulus epoch bounded by classified photodiode events.

    Inputs: neighboring classified photodiode events and one contiguous
    positive epoch run from ``stimulus/data``.
    Outputs: imaging frame bounds plus the stimulus epoch number/name and row
    bounds.

    This is the object later response code should use for trial or epoch
    slicing. It carries both imaging-frame coordinates and stimulus-table
    evidence.
    """

    index: int
    start_event: ClassifiedPhotodiodeEvent
    stop_event: ClassifiedPhotodiodeEvent
    start_frame: int
    stop_frame: int
    epoch_number: int
    epoch_name: str
    stimulus_start_row: int
    stimulus_stop_row: int


@dataclass(frozen=True)
class ClassifiedStimulusTiming:
    """Classified photodiode timing for one converted recording.

    Inputs: one converted recording and one photodiode alignment.
    Outputs: all classified events, all stimulus windows, and audit metadata.

    ``events`` has one item per detected flash. ``windows`` has one fewer item:
    each window spans the interval between neighboring events.
    """

    events: tuple[ClassifiedPhotodiodeEvent, ...]
    windows: tuple[ClassifiedStimulusWindow, ...]
    metadata: dict[str, PhotodiodeClassificationMetadataValue]


@dataclass(frozen=True)
class _StimulusFlashSegment:
    """One contiguous positive run in the photodiode flash column.

    Inputs: ``stimulus/data`` flash column.
    Outputs: half-open row bounds for one requested flash.
    """

    start_row: int
    stop_row: int


@dataclass(frozen=True)
class _StimulusEpochRun:
    """One contiguous positive stimulus epoch run.

    Inputs: ``stimulus/data`` epoch column.
    Outputs: epoch number and half-open row bounds.
    """

    epoch_number: int
    start_row: int
    stop_row: int


def classify_recording_photodiode_events(
    recording: RecordingData,
    alignment: PhotodiodeAlignment,
    *,
    flash_threshold: float = 0.5,
    long_flash_factor: float = 1.25,
) -> ClassifiedStimulusTiming:
    """Classify paired photodiode events into stimulus timing windows.

    Args:
        recording: Loaded converted recording with ``stimulus/data``.
        alignment: Paired high-rate and imaging-frame photodiode events.
        flash_threshold: Values above this threshold in ``photodiode_flash``
            count as stimulus-computer flash requests.
        long_flash_factor: High-rate photodiode events at least this multiple
            of the typical transition duration are marked ``"long"``.

    Returns:
        ``ClassifiedStimulusTiming`` with typed events and epoch windows.

    Raises:
        ValueError: If the photodiode, flash-column, and epoch-run contracts do
            not agree.

    The current classifier handles boundary-flash protocols: one flash marks
    stimulus start, each following flash marks an epoch/trial transition, and
    the final flash marks stimulus end. Extra within-epoch flashes should fail
    here until we add protocol-specific classification rules for them.
    """
    if long_flash_factor <= 1.0:
        msg = f"long_flash_factor must be greater than 1; got {long_flash_factor}"
        raise ValueError(msg)

    flash_segments = _stimulus_flash_segments(
        recording=recording,
        flash_threshold=flash_threshold,
    )
    _validate_event_and_flash_counts(
        paired_events=alignment.paired_events,
        flash_segments=flash_segments,
    )
    classified_events = _classify_events(
        paired_events=alignment.paired_events,
        flash_segments=flash_segments,
        long_flash_factor=long_flash_factor,
    )
    epoch_runs = _stimulus_epoch_runs(recording)
    _validate_event_and_epoch_counts(
        event_count=len(classified_events),
        epoch_runs=epoch_runs,
    )
    windows = _classified_windows(
        events=classified_events,
        epoch_runs=epoch_runs,
        epoch_names=stimulus_epoch_names_by_number(recording),
    )

    return ClassifiedStimulusTiming(
        events=classified_events,
        windows=windows,
        metadata={
            "event_count": len(classified_events),
            "window_count": len(windows),
            "stimulus_flash_segment_count": len(flash_segments),
            "stimulus_epoch_run_count": len(epoch_runs),
            "flash_threshold": float(flash_threshold),
            "long_flash_factor": float(long_flash_factor),
            "typical_transition_high_res_duration_samples": (
                _typical_transition_duration(alignment.paired_events)
            ),
        },
    )


def _stimulus_flash_segments(
    *,
    recording: RecordingData,
    flash_threshold: float,
) -> tuple[_StimulusFlashSegment, ...]:
    """Segment the converted stimulus photodiode-flash column.

    Args:
        recording: Loaded converted recording.
        flash_threshold: Values above this threshold are active flash rows.

    Returns:
        One segment per contiguous active flash run.
    """
    flash_column = stimulus_column_index(recording, "photodiode_flash")
    flash_values = recording.stimulus_data[:, flash_column]
    active = np.asarray(flash_values > flash_threshold, dtype=np.bool_)
    segments = true_segments(active)
    if len(segments) == 0:
        msg = "stimulus/data photodiode_flash column has no active flash segments"
        raise ValueError(msg)
    return tuple(
        _StimulusFlashSegment(start_row=start, stop_row=stop)
        for start, stop in segments
    )


def _stimulus_epoch_runs(recording: RecordingData) -> tuple[_StimulusEpochRun, ...]:
    """Find contiguous positive stimulus epoch runs.

    Args:
        recording: Loaded converted recording.

    Returns:
        One run per contiguous positive epoch number.
    """
    epoch_column = stimulus_column_index(recording, "epoch_number")
    epoch_numbers = recording.stimulus_data[:, epoch_column].astype(np.int64)
    all_segments = constant_segments(epoch_numbers)
    epoch_runs = tuple(
        _StimulusEpochRun(
            epoch_number=int(epoch_numbers[start]),
            start_row=start,
            stop_row=stop,
        )
        for start, stop in all_segments
        if epoch_numbers[start] > 0
    )
    if len(epoch_runs) == 0:
        msg = "stimulus/data does not contain any positive epoch runs"
        raise ValueError(msg)
    return epoch_runs


def _validate_event_and_flash_counts(
    *,
    paired_events: tuple[AlignedPhotodiodeEvent, ...],
    flash_segments: tuple[_StimulusFlashSegment, ...],
) -> None:
    """Check that measured flashes match stimulus-computer flash requests.

    Args:
        paired_events: Paired photodiode events detected from the recording.
        flash_segments: Flash request segments from ``stimulus/data``.

    Returns:
        None.
    """
    if len(paired_events) < 2:
        msg = f"At least two photodiode events are required; got {len(paired_events)}"
        raise ValueError(msg)
    if len(paired_events) != len(flash_segments):
        msg = (
            "Detected photodiode events do not match stimulus flash requests: "
            f"paired_events={len(paired_events)}, "
            f"stimulus_flash_segments={len(flash_segments)}"
        )
        raise ValueError(msg)


def _validate_event_and_epoch_counts(
    *,
    event_count: int,
    epoch_runs: tuple[_StimulusEpochRun, ...],
) -> None:
    """Check that neighboring events can bound every stimulus epoch run.

    Args:
        event_count: Number of classified photodiode events.
        epoch_runs: Contiguous positive epoch runs.

    Returns:
        None.
    """
    expected_windows = event_count - 1
    if len(epoch_runs) != expected_windows:
        msg = (
            "Classified photodiode windows do not match stimulus epoch runs: "
            f"windows={expected_windows}, stimulus_epoch_runs={len(epoch_runs)}"
        )
        raise ValueError(msg)


def _classify_events(
    *,
    paired_events: tuple[AlignedPhotodiodeEvent, ...],
    flash_segments: tuple[_StimulusFlashSegment, ...],
    long_flash_factor: float,
) -> tuple[ClassifiedPhotodiodeEvent, ...]:
    """Assign stimulus roles and duration classes to paired events.

    Args:
        paired_events: Ordered high-rate-to-frame photodiode events.
        flash_segments: Ordered stimulus-computer flash segments.
        long_flash_factor: Multiple used to call a high-rate event long.

    Returns:
        Classified events in recording order.
    """
    typical_duration = _typical_transition_duration(paired_events)
    return tuple(
        ClassifiedPhotodiodeEvent(
            index=index,
            event_type=_event_type_for_index(index, len(paired_events)),
            duration_class=_duration_class(
                event.high_res_event.duration_samples,
                typical_duration=typical_duration,
                long_flash_factor=long_flash_factor,
            ),
            aligned_event=event,
            imaging_frame=event.imaging_frame,
            stimulus_flash_start_row=segment.start_row,
            stimulus_flash_stop_row=segment.stop_row,
            stimulus_flash_duration_rows=segment.stop_row - segment.start_row,
        )
        for index, (event, segment) in enumerate(
            zip(paired_events, flash_segments, strict=True),
        )
    )


def _classified_windows(
    *,
    events: tuple[ClassifiedPhotodiodeEvent, ...],
    epoch_runs: tuple[_StimulusEpochRun, ...],
    epoch_names: dict[int, str],
) -> tuple[ClassifiedStimulusWindow, ...]:
    """Build stimulus windows from neighboring classified events.

    Args:
        events: Classified photodiode events.
        epoch_runs: Contiguous positive epoch runs.
        epoch_names: Epoch number to name mapping.

    Returns:
        One window per epoch run.
    """
    windows: list[ClassifiedStimulusWindow] = []
    for index, epoch_run in enumerate(epoch_runs):
        start_event = events[index]
        stop_event = events[index + 1]
        _validate_window_contract(
            start_event=start_event,
            stop_event=stop_event,
            epoch_run=epoch_run,
        )
        windows.append(
            ClassifiedStimulusWindow(
                index=index,
                start_event=start_event,
                stop_event=stop_event,
                start_frame=start_event.imaging_frame,
                stop_frame=stop_event.imaging_frame,
                epoch_number=epoch_run.epoch_number,
                epoch_name=epoch_names.get(
                    epoch_run.epoch_number,
                    f"epoch_{epoch_run.epoch_number}",
                ),
                stimulus_start_row=epoch_run.start_row,
                stimulus_stop_row=epoch_run.stop_row,
            ),
        )
    return tuple(windows)


def _validate_window_contract(
    *,
    start_event: ClassifiedPhotodiodeEvent,
    stop_event: ClassifiedPhotodiodeEvent,
    epoch_run: _StimulusEpochRun,
) -> None:
    """Validate one stimulus epoch run against its bounding flashes.

    Args:
        start_event: Classified event at the epoch start.
        stop_event: Classified event at the next boundary.
        epoch_run: Stimulus data epoch run.

    Returns:
        None.
    """
    if stop_event.imaging_frame <= start_event.imaging_frame:
        msg = (
            "Classified event frames must be increasing; "
            f"got {start_event.imaging_frame} then {stop_event.imaging_frame}"
        )
        raise ValueError(msg)
    if epoch_run.start_row != start_event.stimulus_flash_start_row:
        msg = (
            "Stimulus epoch run does not start at its photodiode flash: "
            f"epoch_start_row={epoch_run.start_row}, "
            f"flash_start_row={start_event.stimulus_flash_start_row}"
        )
        raise ValueError(msg)
    if epoch_run.stop_row != stop_event.stimulus_flash_start_row:
        msg = (
            "Stimulus epoch run does not stop at the next photodiode flash: "
            f"epoch_stop_row={epoch_run.stop_row}, "
            f"next_flash_start_row={stop_event.stimulus_flash_start_row}"
        )
        raise ValueError(msg)


def _event_type_for_index(index: int, event_count: int) -> PhotodiodeEventType:
    """Return the stimulus timing role for one ordered event.

    Args:
        index: Zero-based event index.
        event_count: Total event count.

    Returns:
        Event type label.
    """
    if index == 0:
        return "stimulus_start"
    if index == event_count - 1:
        return "stimulus_end"
    return "trial_transition"


def _duration_class(
    duration_samples: int,
    *,
    typical_duration: float,
    long_flash_factor: float,
) -> PhotodiodeDurationClass:
    """Classify one high-rate photodiode duration as short or long.

    Args:
        duration_samples: Measured high-rate event duration.
        typical_duration: Typical transition duration.
        long_flash_factor: Multiple used to call a duration long.

    Returns:
        Duration class label.
    """
    if duration_samples >= typical_duration * long_flash_factor:
        return "long"
    return "short"


def _typical_transition_duration(
    paired_events: tuple[AlignedPhotodiodeEvent, ...],
) -> float:
    """Estimate the typical high-rate transition-flash duration.

    Args:
        paired_events: Ordered high-rate-to-frame photodiode events.

    Returns:
        Median duration in high-rate samples.
    """
    transition_events = paired_events[1:-1] if len(paired_events) > 2 else paired_events
    durations = np.asarray(
        [event.high_res_event.duration_samples for event in transition_events],
        dtype=np.float64,
    )
    return float(np.median(durations))
