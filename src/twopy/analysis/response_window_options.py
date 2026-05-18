"""Typed options for response-trial time windows.

Inputs: user-selected automatic/manual window settings and recording-specific
window limits.
Outputs: resolved pre- and post-stimulus seconds for response grouping.

This module is GUI-independent so napari preview and script-facing analysis
save paths can use the same pre- and post-stimulus seconds.
"""

from dataclasses import dataclass

__all__ = [
    "DEFAULT_RESPONSE_POST_WINDOW_SECONDS",
    "DEFAULT_RESPONSE_PRE_WINDOW_SECONDS",
    "ResponseWindowOptions",
    "resolve_response_window_seconds",
]

DEFAULT_RESPONSE_PRE_WINDOW_SECONDS = 2.0
DEFAULT_RESPONSE_POST_WINDOW_SECONDS = 2.0


@dataclass(frozen=True)
class ResponseWindowOptions:
    """User-facing response-window settings.

    Inputs: whether the Plot tab should choose the window automatically, plus
    manual pre- and post-stimulus durations.
    Outputs: a small value object that can be passed from GUI controls into
    preview and persistence workflows.

    ``auto`` keeps the current twopy behavior: two seconds before stimulus
    onset, and post-stimulus context only when an interleave/gray epoch is
    available. Manual values are used only when ``auto`` is false.
    """

    auto: bool = True
    pre_window_seconds: float = DEFAULT_RESPONSE_PRE_WINDOW_SECONDS
    post_window_seconds: float = DEFAULT_RESPONSE_POST_WINDOW_SECONDS


def resolve_response_window_seconds(
    options: ResponseWindowOptions,
    *,
    automatic_pre_window_seconds: float,
    automatic_post_window_seconds: float,
    max_window_seconds: float | None = None,
) -> tuple[float, float]:
    """Return concrete pre/post seconds for response grouping.

    Args:
        options: User-selected response-window options.
        automatic_pre_window_seconds: Default pre-stimulus duration.
        automatic_post_window_seconds: Default post-stimulus duration, usually
            based on whether a gray/interleave epoch is present.
        max_window_seconds: Optional recording-derived gray/interleave duration
            that caps both pre and post windows.

    Returns:
        ``(pre_window_seconds, post_window_seconds)`` after validation and the
        optional recording-derived cap.

    Raises:
        ValueError: If any supplied window duration is negative.
    """
    if max_window_seconds is not None and max_window_seconds < 0.0:
        msg = f"max_window_seconds must be non-negative; got {max_window_seconds}"
        raise ValueError(msg)
    if options.auto:
        pre = _validate_non_negative(
            automatic_pre_window_seconds,
            name="automatic_pre_window_seconds",
        )
        post = _validate_non_negative(
            automatic_post_window_seconds,
            name="automatic_post_window_seconds",
        )
    else:
        pre = _validate_non_negative(
            options.pre_window_seconds,
            name="pre_window_seconds",
        )
        post = _validate_non_negative(
            options.post_window_seconds,
            name="post_window_seconds",
        )
    if max_window_seconds is None:
        return pre, post
    return min(pre, max_window_seconds), min(post, max_window_seconds)


def _validate_non_negative(value: float, *, name: str) -> float:
    """Return ``value`` as a float after enforcing the response-window domain."""
    resolved = float(value)
    if resolved < 0.0:
        msg = f"{name} must be non-negative; got {resolved}"
        raise ValueError(msg)
    return resolved
