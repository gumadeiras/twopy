"""Detect photodiode events and map them to imaging frames.

Inputs: converted photodiode signals from ``RecordingData``.
Outputs: photodiode event segments and high-resolution events paired with
frame-resolution imaging events.

The stimulus and imaging computers have separate clocks. This module uses the
photodiode signals as the shared evidence for alignment instead of relying on
nominal frame rates.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from twopy.converted import RecordingData

__all__ = [
    "AlignedPhotodiodeEvent",
    "PhotodiodeAlignment",
    "PhotodiodeEvent",
    "PhotodiodeEventSet",
    "detect_photodiode_events",
    "detect_recording_photodiode_events",
    "pair_photodiode_events_to_imaging_frames",
]

PhotodiodePolarity = Literal["above", "below"]
PhotodiodeSignalName = Literal["high_res_pd", "imaging_res_pd"]


@dataclass(frozen=True)
class PhotodiodeEvent:
    """One contiguous photodiode flash segment.

    Inputs: one thresholded photodiode signal.
    Outputs: start, stop, center, duration, and peak for one detected event.

    ``start_sample`` is inclusive and ``stop_sample`` is exclusive. For
    ``imaging_res_pd``, sample indices are imaging-frame indices.
    """

    start_sample: int
    stop_sample: int
    center_sample: float
    duration_samples: int
    peak_value: float


@dataclass(frozen=True)
class PhotodiodeEventSet:
    """Detected events for one photodiode signal.

    Inputs: signal name, thresholding options, and detected event segments.
    Outputs: auditable event list plus the threshold that produced it.

    Keeping the threshold with the events makes later trial alignment easier to
    inspect and reproduce.
    """

    signal_name: PhotodiodeSignalName
    events: tuple[PhotodiodeEvent, ...]
    threshold: float
    polarity: PhotodiodePolarity


@dataclass(frozen=True)
class AlignedPhotodiodeEvent:
    """One high-resolution event paired with one imaging-frame event.

    Inputs: detected high-resolution and imaging-resolution photodiode events.
    Outputs: the imaging frame assigned to the high-resolution event.

    The imaging frame is the first frame-resolution sample where the paired
    photodiode event is detected. This is the conservative frame boundary used
    before stimulus-specific event classification exists.
    """

    high_res_event: PhotodiodeEvent
    imaging_res_event: PhotodiodeEvent
    imaging_frame: int


@dataclass(frozen=True)
class PhotodiodeAlignment:
    """Detected and paired photodiode events for one recording.

    Inputs: converted high-resolution and imaging-resolution photodiode signals.
    Outputs: event sets plus ordered high-res-to-frame pairings.

    This is the first timing object response analysis should consume. It still
    does not classify stimulus-specific flash patterns.
    """

    high_res: PhotodiodeEventSet
    imaging_res: PhotodiodeEventSet
    paired_events: tuple[AlignedPhotodiodeEvent, ...]


def detect_photodiode_events(
    signal: npt.ArrayLike,
    *,
    signal_name: PhotodiodeSignalName = "high_res_pd",
    threshold: float | None = None,
    polarity: PhotodiodePolarity = "above",
    minimum_width_samples: int = 1,
    merge_gap_samples: int = 0,
) -> PhotodiodeEventSet:
    """Detect contiguous photodiode events in one signal.

    Args:
        signal: One-dimensional photodiode samples.
        signal_name: Name of the converted photodiode signal being decoded.
        threshold: Optional threshold. When omitted, the midpoint between the
            signal minimum and maximum is used.
        polarity: ``above`` detects samples greater than threshold; ``below``
            detects samples less than threshold.
        minimum_width_samples: Drop events shorter than this width.
        merge_gap_samples: Merge neighboring events separated by this many
            samples or fewer.

    Returns:
        Event set with the requested signal name.

    Raises:
        ValueError: If the signal is not one-dimensional or widths are invalid.

    This function segments flashes. It does not classify stimulus-specific flash
    patterns because that requires stimulus protocol rules.
    """
    events, resolved_threshold = _detect_events(
        signal=signal,
        threshold=threshold,
        polarity=polarity,
        minimum_width_samples=minimum_width_samples,
        merge_gap_samples=merge_gap_samples,
    )
    return PhotodiodeEventSet(
        signal_name=signal_name,
        events=events,
        threshold=resolved_threshold,
        polarity=polarity,
    )


def detect_recording_photodiode_events(
    recording: RecordingData,
    *,
    high_res_threshold: float | None = None,
    imaging_res_threshold: float | None = None,
    polarity: PhotodiodePolarity = "above",
    minimum_width_samples: int = 1,
    merge_gap_samples: int = 0,
) -> PhotodiodeAlignment:
    """Detect and pair photodiode events for one converted recording.

    Args:
        recording: Loaded converted recording.
        high_res_threshold: Optional threshold for ``high_res_pd``.
        imaging_res_threshold: Optional threshold for ``imaging_res_pd``.
        polarity: Which side of threshold marks a photodiode flash.
        minimum_width_samples: Drop events shorter than this width in each
            signal.
        merge_gap_samples: Merge neighboring events separated by this many
            samples or fewer in each signal.

    Returns:
        ``PhotodiodeAlignment`` with high-resolution events paired to imaging
        frames by event order.

    Pairing by order uses the photodiode evidence in both signals. It avoids
    assuming the two computers share a frame rate or clock.
    """
    high_res_events, high_res_resolved_threshold = _detect_events(
        signal=recording.high_res_pd,
        threshold=high_res_threshold,
        polarity=polarity,
        minimum_width_samples=minimum_width_samples,
        merge_gap_samples=merge_gap_samples,
    )
    imaging_res_events, imaging_res_resolved_threshold = _detect_events(
        signal=recording.imaging_res_pd,
        threshold=imaging_res_threshold,
        polarity=polarity,
        minimum_width_samples=minimum_width_samples,
        merge_gap_samples=merge_gap_samples,
    )
    high_res = PhotodiodeEventSet(
        signal_name="high_res_pd",
        events=high_res_events,
        threshold=high_res_resolved_threshold,
        polarity=polarity,
    )
    imaging_res = PhotodiodeEventSet(
        signal_name="imaging_res_pd",
        events=imaging_res_events,
        threshold=imaging_res_resolved_threshold,
        polarity=polarity,
    )
    return PhotodiodeAlignment(
        high_res=high_res,
        imaging_res=imaging_res,
        paired_events=pair_photodiode_events_to_imaging_frames(
            high_res.events,
            imaging_res.events,
        ),
    )


def pair_photodiode_events_to_imaging_frames(
    high_res_events: tuple[PhotodiodeEvent, ...],
    imaging_res_events: tuple[PhotodiodeEvent, ...],
) -> tuple[AlignedPhotodiodeEvent, ...]:
    """Pair high-resolution photodiode events to imaging-frame events.

    Args:
        high_res_events: Events detected from ``high_res_pd``.
        imaging_res_events: Events detected from ``imaging_res_pd``.

    Returns:
        Ordered event pairs with assigned imaging-frame indices.

    Raises:
        ValueError: If event counts differ.

    Matching counts are required for this first implementation because silently
    dropping or inventing event pairings would corrupt trial alignment.
    """
    if len(high_res_events) != len(imaging_res_events):
        msg = (
            "Cannot pair photodiode events with different counts: "
            f"high_res={len(high_res_events)}, imaging_res={len(imaging_res_events)}"
        )
        raise ValueError(msg)

    return tuple(
        AlignedPhotodiodeEvent(
            high_res_event=high_res_event,
            imaging_res_event=imaging_res_event,
            imaging_frame=imaging_res_event.start_sample,
        )
        for high_res_event, imaging_res_event in zip(
            high_res_events,
            imaging_res_events,
            strict=True,
        )
    )


def _detect_events(
    *,
    signal: npt.ArrayLike,
    threshold: float | None,
    polarity: PhotodiodePolarity,
    minimum_width_samples: int,
    merge_gap_samples: int,
) -> tuple[tuple[PhotodiodeEvent, ...], float]:
    """Detect event segments and return the threshold used.

    Args:
        signal: One-dimensional photodiode samples.
        threshold: Optional threshold.
        polarity: Which side of threshold marks an event.
        minimum_width_samples: Drop events shorter than this width.
        merge_gap_samples: Merge neighboring events separated by this width.

    Returns:
        ``(events, threshold)``.
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        msg = f"Photodiode signal must be one-dimensional; got {values.shape}"
        raise ValueError(msg)
    if minimum_width_samples < 1:
        msg = "minimum_width_samples must be at least 1"
        raise ValueError(msg)
    if merge_gap_samples < 0:
        msg = "merge_gap_samples cannot be negative"
        raise ValueError(msg)

    resolved_threshold = (
        _default_threshold(values) if threshold is None else float(threshold)
    )
    if values.size == 0:
        return (), resolved_threshold

    active = _threshold_signal(
        values=values,
        threshold=resolved_threshold,
        polarity=polarity,
    )
    segments = _active_segments(active)
    merged_segments = _merge_close_segments(segments, merge_gap_samples)
    events = tuple(
        _event_from_segment(values, start, stop)
        for start, stop in merged_segments
        if stop - start >= minimum_width_samples
    )
    return events, resolved_threshold


def _default_threshold(values: npt.NDArray[np.float64]) -> float:
    """Choose a simple midpoint threshold for one signal.

    Args:
        values: One-dimensional signal values.

    Returns:
        Midpoint between minimum and maximum, or zero for an empty signal.
    """
    if values.size == 0:
        return 0.0
    return float((np.nanmin(values) + np.nanmax(values)) / 2.0)


def _threshold_signal(
    *,
    values: npt.NDArray[np.float64],
    threshold: float,
    polarity: PhotodiodePolarity,
) -> npt.NDArray[np.bool_]:
    """Convert signal values into an active-event mask.

    Args:
        values: One-dimensional signal values.
        threshold: Event threshold.
        polarity: Which side of threshold is active.

    Returns:
        Boolean mask with one value per input sample.
    """
    if polarity == "above":
        return values > threshold
    return values < threshold


def _active_segments(active: npt.NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    """Find contiguous true runs in an active-event mask.

    Args:
        active: Boolean threshold mask.

    Returns:
        Tuple of inclusive-start, exclusive-stop index pairs.
    """
    if active.size == 0:
        return ()

    padded = np.concatenate(([False], active, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return tuple(
        (int(changes[index]), int(changes[index + 1]))
        for index in range(0, len(changes), 2)
    )


def _merge_close_segments(
    segments: tuple[tuple[int, int], ...],
    merge_gap_samples: int,
) -> tuple[tuple[int, int], ...]:
    """Merge event segments separated by short inactive gaps.

    Args:
        segments: Event start/stop pairs.
        merge_gap_samples: Maximum inactive gap to merge.

    Returns:
        Merged start/stop pairs.
    """
    if len(segments) <= 1 or merge_gap_samples == 0:
        return segments

    merged: list[tuple[int, int]] = [segments[0]]
    for start, stop in segments[1:]:
        last_start, last_stop = merged[-1]
        if start - last_stop <= merge_gap_samples:
            merged[-1] = (last_start, stop)
        else:
            merged.append((start, stop))

    return tuple(merged)


def _event_from_segment(
    values: npt.NDArray[np.float64],
    start: int,
    stop: int,
) -> PhotodiodeEvent:
    """Summarize one active event segment.

    Args:
        values: Full one-dimensional photodiode signal.
        start: Inclusive event start sample.
        stop: Exclusive event stop sample.

    Returns:
        ``PhotodiodeEvent`` for the segment.
    """
    event_values = values[start:stop]
    return PhotodiodeEvent(
        start_sample=start,
        stop_sample=stop,
        center_sample=float(start + (stop - start - 1) / 2.0),
        duration_samples=stop - start,
        peak_value=float(np.nanmax(event_values)),
    )
