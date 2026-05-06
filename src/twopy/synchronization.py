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

import numpy.typing as npt

from twopy.converted import RecordingData
from twopy.photodiode import (
    DEFAULT_PHOTODIODE_THRESHOLD_FRACTION,
    PhotodiodeEvent,
    PhotodiodePolarity,
    detect_photodiode_event_segments,
)

__all__ = [
    "AlignedPhotodiodeEvent",
    "PhotodiodeAlignment",
    "PhotodiodeEvent",
    "PhotodiodeEventSet",
    "PhotodiodePolarity",
    "detect_photodiode_events",
    "detect_recording_photodiode_events",
    "pair_photodiode_events_to_imaging_frames",
]

PhotodiodeSignalName = Literal["high_res_pd", "imaging_res_pd"]


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
    threshold_fraction: float = DEFAULT_PHOTODIODE_THRESHOLD_FRACTION,
    polarity: PhotodiodePolarity = "above",
    minimum_width_samples: int = 1,
    merge_gap_samples: int = 0,
) -> PhotodiodeEventSet:
    """Detect contiguous photodiode events in one signal.

    Args:
        signal: One-dimensional photodiode samples.
        signal_name: Name of the converted photodiode signal being decoded.
        threshold: Optional threshold. When omitted, twopy uses
            ``min + threshold_fraction * range``.
        threshold_fraction: Signal-range fraction used when ``threshold`` is
            omitted. Defaults to the lab analysis convention, 0.3.
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
    events, resolved_threshold = detect_photodiode_event_segments(
        signal=signal,
        threshold=threshold,
        threshold_fraction=threshold_fraction,
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
    threshold_fraction: float = DEFAULT_PHOTODIODE_THRESHOLD_FRACTION,
    polarity: PhotodiodePolarity = "above",
    minimum_width_samples: int = 1,
    merge_gap_samples: int = 0,
) -> PhotodiodeAlignment:
    """Detect and pair photodiode events for one converted recording.

    Args:
        recording: Loaded converted recording.
        high_res_threshold: Optional threshold for ``high_res_pd``.
        imaging_res_threshold: Optional threshold for ``imaging_res_pd``.
        threshold_fraction: Signal-range fraction used for either signal when
            its explicit threshold is omitted.
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
    high_res = detect_photodiode_events(
        recording.high_res_pd,
        signal_name="high_res_pd",
        threshold=high_res_threshold,
        threshold_fraction=threshold_fraction,
        polarity=polarity,
        minimum_width_samples=minimum_width_samples,
        merge_gap_samples=merge_gap_samples,
    )
    imaging_res = detect_photodiode_events(
        recording.imaging_res_pd,
        signal_name="imaging_res_pd",
        threshold=imaging_res_threshold,
        threshold_fraction=threshold_fraction,
        polarity=polarity,
        minimum_width_samples=minimum_width_samples,
        merge_gap_samples=merge_gap_samples,
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
