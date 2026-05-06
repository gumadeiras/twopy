"""Movie frame-range helpers for the twopy napari adapter.

Inputs: GUI or CLI movie frame numbers.
Outputs: validated inclusive frame ranges and HDF5 reader stop indices.

Napari controls show inclusive frame numbers because that is the intuitive UI
contract. The converted movie reader uses Python's exclusive stop convention,
so the conversion happens in one small helper.
"""

__all__ = ["exclusive_stop", "resolve_movie_frame_range"]


def resolve_movie_frame_range(
    *,
    start_frame: int,
    end_frame: int | None,
    frame_count: int,
    zero_end_means_last: bool = False,
) -> tuple[int, int]:
    """Resolve and validate an inclusive movie frame range.

    Args:
        start_frame: First movie frame.
        end_frame: Last movie frame, or ``None`` for the final frame.
        frame_count: Number of available movie frames.
        zero_end_means_last: Whether ``start=0, end=0`` should mean the final
            frame. The empty-launch widget uses this before it knows the movie
            length.

    Returns:
        Inclusive ``(start, end)`` movie frame range.
    """
    if start_frame < 0:
        msg = f"movie_start_frame must be at least 0; got {start_frame}"
        raise ValueError(msg)
    if frame_count < 1:
        msg = f"movie must have at least 1 frame; got {frame_count}"
        raise ValueError(msg)

    last_frame = frame_count - 1
    if end_frame is None or (
        zero_end_means_last and start_frame == 0 and end_frame == 0
    ):
        resolved_end = last_frame
    else:
        resolved_end = end_frame

    if resolved_end < start_frame:
        msg = (
            "movie_end_frame must be greater than or equal to "
            f"movie_start_frame; got {resolved_end} < {start_frame}"
        )
        raise ValueError(msg)
    if resolved_end > last_frame:
        msg = f"movie_end_frame must be at most {last_frame}; got {resolved_end}"
        raise ValueError(msg)
    return (start_frame, resolved_end)


def exclusive_stop(end_frame: int | None) -> int | None:
    """Convert an inclusive end frame to a reader stop frame.

    Args:
        end_frame: Inclusive final frame, or ``None`` for the final movie
            frame.

    Returns:
        Exclusive stop index for ``ConvertedMovie.read_frames``.
    """
    return None if end_frame is None else end_frame + 1
