"""Validate frame ranges used by conversion helpers.

Inputs: frame counts and optional start/stop frame numbers.
Outputs: explicit valid ``(start, stop)`` frame ranges.
"""


def normalize_frame_range(
    *,
    frame_count: int,
    start_frame: int | None,
    stop_frame: int | None,
) -> tuple[int, int]:
    """Normalize and validate an optional movie frame range.

    Args:
        frame_count: Total number of movie frames.
        start_frame: Optional first frame.
        stop_frame: Optional exclusive stop frame.

    Returns:
        Validated ``(start, stop)`` frame range.

    Raises:
        ValueError: If the requested range is empty or outside the movie.
    """
    start = 0 if start_frame is None else start_frame
    stop = frame_count if stop_frame is None else stop_frame
    if start < 0 or stop > frame_count or start >= stop:
        msg = (
            "Invalid frame range for mean image: "
            f"start={start}, stop={stop}, frame_count={frame_count}"
        )
        raise ValueError(msg)
    return start, stop
