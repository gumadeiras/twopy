"""Validate half-open frame ranges used by movie and trace helpers.

Inputs: frame counts and optional start/stop frame numbers.
Outputs: explicit valid ``(start, stop)`` frame ranges.
"""

__all__ = ["normalize_frame_range"]


def normalize_frame_range(
    *,
    frame_count: int,
    start_frame: int | None,
    stop_frame: int | None,
    context: str = "frame range",
) -> tuple[int, int]:
    """Normalize and validate an optional movie frame range.

    Args:
        frame_count: Total number of movie frames.
        start_frame: Optional first frame.
        stop_frame: Optional exclusive stop frame.
        context: Human-readable range name used in validation errors.

    Returns:
        Validated ``(start, stop)`` frame range.

    Raises:
        ValueError: If the requested range is empty or outside the movie.

    Half-open frame ranges are used in conversion, converted movie reads, ROI
    extraction, and background correction. Keeping the validation here makes
    every caller handle ``None`` defaults and bounds the same way.
    """
    start = 0 if start_frame is None else start_frame
    stop = frame_count if stop_frame is None else stop_frame
    if start < 0 or stop > frame_count or start >= stop:
        msg = f"Invalid {context} [{start}, {stop}) for {frame_count} frames"
        raise ValueError(msg)
    return start, stop
