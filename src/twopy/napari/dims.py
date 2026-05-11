"""Read and update napari dimension state for movie-frame aware tools.

Inputs: napari viewer objects with optional ``dims`` state.
Outputs: bounded frame indices and best-effort frame-seek updates.

The napari adapter has several features that need the currently displayed movie
frame. Keeping that small boundary here avoids each feature learning napari's
``current_step`` and ``point`` variants independently.
"""

from typing import Protocol, cast

__all__ = ["current_step_index", "set_current_step_index"]


class _ViewerWithDims(Protocol):
    """Small protocol for napari viewers that expose dimension state."""

    dims: object


def current_step_index(
    viewer: object | None,
    *,
    axis: int = 0,
    step_count: int | None = None,
) -> int:
    """Return the current integer step on one napari dimension axis.

    Args:
        viewer: Napari viewer, or ``None`` when no viewer is available.
        axis: Dimension axis to inspect. Movie frames use axis zero.
        step_count: Optional number of valid steps. When supplied, the returned
            value is clamped into ``[0, step_count - 1]``.

    Returns:
        Bounded current step. Missing or malformed napari state falls back to
        zero so callers can still export or render a sensible first frame.
    """
    if viewer is None or not hasattr(viewer, "dims"):
        return 0
    dims = cast(_ViewerWithDims, viewer).dims
    value = _step_value_from_dims(dims, axis=axis)
    if step_count is None:
        return max(0, value)
    if step_count < 1:
        return 0
    return max(0, min(value, step_count - 1))


def set_current_step_index(viewer: object | None, *, axis: int, step: int) -> None:
    """Move one napari dimension axis to an integer step when possible.

    Args:
        viewer: Napari viewer, or ``None`` when no viewer is available.
        axis: Dimension axis to update.
        step: New non-negative integer step.

    Returns:
        None.

    Real napari exposes ``dims.set_current_step``. Tests and older object
    shapes may only expose mutable ``current_step``/``point`` tuples, so this
    helper uses the method when present and otherwise updates tuple-like state
    best-effort.
    """
    if viewer is None or not hasattr(viewer, "dims"):
        return
    dims = cast(_ViewerWithDims, viewer).dims
    bounded_step = max(0, int(step))
    setter = getattr(dims, "set_current_step", None)
    if callable(setter):
        setter(axis, bounded_step)
        return
    _replace_step_attribute(dims, "current_step", axis=axis, step=bounded_step)
    _replace_step_attribute(dims, "point", axis=axis, step=bounded_step)


def _step_value_from_dims(dims: object, *, axis: int) -> int:
    """Read one step value from napari dims-like state."""
    for attribute in ("current_step", "point"):
        value = getattr(dims, attribute, None)
        if value is None:
            continue
        try:
            return int(value[axis])
        except (IndexError, TypeError, ValueError):
            continue
    return 0


def _replace_step_attribute(
    dims: object,
    attribute: str,
    *,
    axis: int,
    step: int,
) -> None:
    """Replace one tuple-like dims attribute when the object allows it."""
    value = getattr(dims, attribute, None)
    if value is None:
        return
    try:
        values = list(value)
    except TypeError:
        return
    if axis >= len(values):
        return
    values[axis] = step
    try:
        setattr(dims, attribute, tuple(values))
    except AttributeError:
        return
