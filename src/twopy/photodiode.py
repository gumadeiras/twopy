"""Shared photodiode timing helpers.

Inputs: converted or source photodiode vectors plus imaging-frame metadata.
Outputs: thresholded flash events and stimulus-bounded imaging-frame ranges.

The photodiode is the common timing evidence between the stimulus computer and
the imaging computer. This module keeps that low-level contract in one place so
conversion and response analysis do not drift into different threshold or
end-flash rules.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from twopy._segments import true_segments

__all__ = [
    "DEFAULT_END_FLASH_PROJECTOR_FRAME_THRESHOLD",
    "DEFAULT_PHOTODIODE_THRESHOLD_FRACTION",
    "DEFAULT_PROJECTOR_HZ",
    "PhotodiodeEvent",
    "PhotodiodePolarity",
    "detect_photodiode_event_segments",
    "photodiode_threshold",
    "stimulus_frame_bounds_from_photodiode",
]

PhotodiodePolarity = Literal["above", "below"]

DEFAULT_PHOTODIODE_THRESHOLD_FRACTION = 0.3
DEFAULT_PROJECTOR_HZ = 60.0
DEFAULT_END_FLASH_PROJECTOR_FRAME_THRESHOLD = 19.0


@dataclass(frozen=True)
class PhotodiodeEvent:
    """One contiguous photodiode flash segment.

    Inputs: one thresholded photodiode signal.
    Outputs: start, stop, center, duration, and peak for one detected event.

    ``start_sample`` is inclusive and ``stop_sample`` is exclusive. When the
    source signal is ``imaging_res_pd``, sample indices are imaging-frame
    indices. When the source signal is ``high_res_pd``, sample indices are
    high-rate photodiode samples.
    """

    start_sample: int
    stop_sample: int
    center_sample: float
    duration_samples: int
    peak_value: float


def photodiode_threshold(
    signal: npt.ArrayLike,
    *,
    fraction: float = DEFAULT_PHOTODIODE_THRESHOLD_FRACTION,
) -> float:
    """Compute the default active-flash threshold for a photodiode signal.

    Args:
        signal: One-dimensional photodiode samples.
        fraction: Fraction of the signal range above the minimum used as the
            threshold.

    Returns:
        ``min(signal) + fraction * (max(signal) - min(signal))``.

    Raises:
        ValueError: If the signal is not one-dimensional or ``fraction`` is
            outside ``[0, 1]``.

    The default fraction is 0.3 to match the lab analysis convention. This
    lower-than-midpoint threshold detects flash boundaries closer to the first
    rising edge while still using only local signal range information.
    """
    values = _photodiode_values(signal)
    if not 0.0 <= fraction <= 1.0:
        msg = f"Photodiode threshold fraction must be in [0, 1]. Got {fraction}"
        raise ValueError(msg)
    if values.size == 0:
        return 0.0

    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    return minimum + (fraction * (maximum - minimum))


def detect_photodiode_event_segments(
    signal: npt.ArrayLike,
    *,
    threshold: float | None = None,
    threshold_fraction: float = DEFAULT_PHOTODIODE_THRESHOLD_FRACTION,
    polarity: PhotodiodePolarity = "above",
    minimum_width_samples: int = 1,
    merge_gap_samples: int = 0,
) -> tuple[tuple[PhotodiodeEvent, ...], float]:
    """Detect event segments and return the threshold used.

    Args:
        signal: One-dimensional photodiode samples.
        threshold: Optional explicit event threshold. When omitted, the
            range-fraction threshold is used.
        threshold_fraction: Fraction used when ``threshold`` is omitted.
        polarity: Which side of threshold marks an event.
        minimum_width_samples: Drop events shorter than this width.
        merge_gap_samples: Merge neighboring events separated by this width.

    Returns:
        ``(events, threshold)``.

    Raises:
        ValueError: If signal shape or width arguments are invalid.
    """
    values = _photodiode_values(signal)
    if minimum_width_samples < 1:
        msg = "minimum_width_samples must be at least 1"
        raise ValueError(msg)
    if merge_gap_samples < 0:
        msg = "merge_gap_samples cannot be negative"
        raise ValueError(msg)

    resolved_threshold = (
        photodiode_threshold(values, fraction=threshold_fraction)
        if threshold is None
        else float(threshold)
    )
    if values.size == 0:
        return (), resolved_threshold

    active = _threshold_signal(
        values=values,
        threshold=resolved_threshold,
        polarity=polarity,
    )
    segments = true_segments(active)
    merged_segments = _merge_close_segments(segments, merge_gap_samples)
    events = tuple(
        _event_from_segment(values, start, stop)
        for start, stop in merged_segments
        if stop - start >= minimum_width_samples
    )
    return events, resolved_threshold


def stimulus_frame_bounds_from_photodiode(
    high_res_pd: npt.ArrayLike,
    *,
    movie_frame_count: int,
    frame_rate_hz: float,
    threshold: float | None = None,
    threshold_fraction: float = DEFAULT_PHOTODIODE_THRESHOLD_FRACTION,
    projector_hz: float = DEFAULT_PROJECTOR_HZ,
    end_flash_projector_frame_threshold: float = (
        DEFAULT_END_FLASH_PROJECTOR_FRAME_THRESHOLD
    ),
) -> tuple[int, int]:
    """Return imaging-frame bounds for the stimulus presentation window.

    Args:
        high_res_pd: High-rate photodiode signal.
        movie_frame_count: Number of aligned movie frames.
        frame_rate_hz: Imaging frame rate.
        threshold: Optional explicit event threshold.
        threshold_fraction: Range fraction used when ``threshold`` is omitted.
        projector_hz: Stimulus projector refresh rate.
        end_flash_projector_frame_threshold: Long-flash duration threshold in
            projector frames.

    Returns:
        Half-open imaging-frame range ``(start, stop)``.

    Raises:
        ValueError: If no start/end flash can be found or if long end flashes
            are ambiguous.

    The first detected flash marks stimulus start. The selected long flash marks
    stimulus end, and the end frame is the frame immediately before that flash.
    """
    events, _ = detect_photodiode_event_segments(
        high_res_pd,
        threshold=threshold,
        threshold_fraction=threshold_fraction,
    )
    if len(events) == 0:
        msg = "Cannot find stimulus frame bounds without photodiode flashes"
        raise ValueError(msg)

    samples_per_frame = _high_res_samples_per_frame(
        high_res_sample_count=_photodiode_values(high_res_pd).size,
        movie_frame_count=movie_frame_count,
    )
    end_index = _end_flash_event_index(
        events=events,
        high_res_samples_per_frame=samples_per_frame,
        frame_rate_hz=frame_rate_hz,
        projector_hz=projector_hz,
        end_flash_projector_frame_threshold=end_flash_projector_frame_threshold,
    )
    epoch_begin = events[0].start_sample / samples_per_frame
    epoch_end = (events[end_index].start_sample - 1) / samples_per_frame
    start = int(np.floor(epoch_begin))
    stop = int(np.floor(epoch_end)) + 1
    if start < 0 or stop > movie_frame_count or start >= stop:
        msg = (
            f"Invalid stimulus frame bounds [{start}, {stop}) for "
            f"{movie_frame_count} frames"
        )
        raise ValueError(msg)
    return start, stop


def _photodiode_values(signal: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a validated one-dimensional photodiode vector.

    Args:
        signal: Candidate photodiode samples.

    Returns:
        Float64 one-dimensional values.
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        msg = f"Photodiode signal must be one-dimensional. Got {values.shape}"
        raise ValueError(msg)
    return values


def _high_res_samples_per_frame(
    *,
    high_res_sample_count: int,
    movie_frame_count: int,
) -> int:
    """Return high-rate photodiode samples per imaging frame.

    Args:
        high_res_sample_count: Number of high-rate photodiode samples.
        movie_frame_count: Number of aligned movie frames.

    Returns:
        Integer high-rate samples per imaging frame.
    """
    if movie_frame_count <= 0:
        msg = f"movie_frame_count must be positive. Got {movie_frame_count}"
        raise ValueError(msg)
    if high_res_sample_count % movie_frame_count != 0:
        msg = (
            "high-rate photodiode sample count must be an integer multiple of "
            f"movie frames: {high_res_sample_count} samples for "
            f"{movie_frame_count} frames"
        )
        raise ValueError(msg)
    return high_res_sample_count // movie_frame_count


def _end_flash_event_index(
    *,
    events: tuple[PhotodiodeEvent, ...],
    high_res_samples_per_frame: int,
    frame_rate_hz: float,
    projector_hz: float,
    end_flash_projector_frame_threshold: float,
) -> int:
    """Identify the long photodiode event that marks stimulus end.

    Args:
        events: High-rate photodiode events.
        high_res_samples_per_frame: High-rate samples per imaging frame.
        frame_rate_hz: Imaging frame rate.
        projector_hz: Stimulus projector refresh rate.
        end_flash_projector_frame_threshold: Minimum projector-frame duration
            for an end flash.

    Returns:
        Index of the flash whose start bounds the stimulus end.
    """
    durations = np.asarray(
        [event.duration_samples for event in events],
        dtype=np.float64,
    )
    projector_frames = (
        durations / float(high_res_samples_per_frame) / frame_rate_hz * projector_hz
    )
    end_candidates = np.flatnonzero(
        projector_frames > end_flash_projector_frame_threshold,
    )
    if end_candidates.size == 0:
        msg = "Cannot find stimulus end because no long end flash was detected"
        raise ValueError(msg)
    if end_candidates.size == 2 and end_candidates[1] - end_candidates[0] > 3:
        return int(end_candidates[1])
    if end_candidates.size > 2:
        msg = (
            "Ambiguous stimulus end: more than two long photodiode flashes were "
            f"detected at event indices {end_candidates.tolist()}"
        )
        raise ValueError(msg)
    return int(end_candidates[0])


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
