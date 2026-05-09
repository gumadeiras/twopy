"""Visibility-state helpers for response plotting docks.

Inputs: plot data rows and checkbox callback keys from ROI or epoch visibility
controls.
Outputs: normalized visibility dictionaries and selected row indices.

The helpers keep callback value validation and row-index logic shared between
ROI and epoch controls.
"""

from twopy.analysis.trials import is_baseline_epoch_name
from twopy.napari.plotting.data import EpochResponsePlotData, ResponsePlotData
from twopy.roi import roi_label_values_from_labels


def row_index_or_none(value: object, row_count: int) -> int | None:
    """Return a valid widget row index from an external callback value.

    Args:
        value: Callback key from a visibility checkbox.
        row_count: Number of rows currently displayed by that widget.

    Returns:
        Zero-based row index, or ``None`` when ``value`` is invalid.
    """
    if type(value) is not int or value < 0 or value >= row_count:
        return None
    return value


def row_visibility(
    existing_visibility: dict[int, bool],
    row_count: int,
) -> dict[int, bool]:
    """Return visibility state for all rows, defaulting missing rows visible.

    Args:
        existing_visibility: Prior row visibility state.
        row_count: Number of rows currently displayed.

    Returns:
        Visibility dictionary with one entry per displayed row.
    """
    return {index: existing_visibility.get(index, True) for index in range(row_count)}


def epoch_visibility(
    existing_visibility: dict[int, bool],
    epochs: tuple[EpochResponsePlotData, ...],
) -> dict[int, bool]:
    """Return epoch visibility, hiding interleave rows by default.

    Args:
        existing_visibility: Prior epoch visibility state.
        epochs: Plot rows for each stimulus epoch.

    Returns:
        Visibility dictionary with one entry per displayed epoch row.
    """
    return {
        index: existing_visibility.get(
            index,
            not is_baseline_epoch_name(epoch.epoch_name),
        )
        for index, epoch in enumerate(epochs)
    }


def set_row_visibility(
    visibility: dict[int, bool],
    index: object,
    visible: bool,
    *,
    row_count: int,
) -> None:
    """Set one row visibility flag when the callback key is valid.

    Args:
        visibility: Mutable visibility dictionary to update.
        index: Callback key from a visibility checkbox.
        visible: Whether the row should be visible.
        row_count: Number of rows currently displayed.

    Returns:
        None.
    """
    row_index = row_index_or_none(index, row_count)
    if row_index is not None:
        visibility[row_index] = visible


def set_row_visibility_batch(
    visibility: dict[int, bool],
    updates: dict[object, bool],
    *,
    row_count: int,
) -> None:
    """Set several row visibility flags after validating callback keys.

    Args:
        visibility: Mutable visibility dictionary to update.
        updates: Callback keys and requested visibility values.
        row_count: Number of rows currently displayed.

    Returns:
        None.
    """
    for index, visible in updates.items():
        set_row_visibility(
            visibility,
            index,
            visible,
            row_count=row_count,
        )


def visible_indices(visibility: dict[int, bool], row_count: int) -> tuple[int, ...]:
    """Return row indices currently visible in plots.

    Args:
        visibility: Row visibility dictionary.
        row_count: Number of rows currently displayed.

    Returns:
        Tuple of visible row indices.
    """
    return tuple(index for index in range(row_count) if visibility.get(index, True))


def roi_labels_from_plot_data(plot_data: ResponsePlotData | None) -> tuple[str, ...]:
    """Return ROI labels from the current plot data.

    Args:
        plot_data: Current plot data, or None before data is loaded.

    Returns:
        ROI labels, or an empty tuple before data is loaded.
    """
    if plot_data is None or len(plot_data.epochs) == 0:
        return ()
    return plot_data.epochs[0].roi_labels


def roi_label_values(plot_data: ResponsePlotData | None) -> tuple[int, ...]:
    """Return integer Labels values in plot ROI order.

    Args:
        plot_data: Current plot data, or None before data is loaded.

    Returns:
        Integer napari Labels values parsed from ROI labels.
    """
    return roi_label_values_from_labels(roi_labels_from_plot_data(plot_data))
