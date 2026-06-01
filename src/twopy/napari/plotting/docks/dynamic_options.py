"""Dynamic Plot, ROI, and Epoch option rendering for response docks.

Inputs: currently loaded plot data, visibility state, axis bounds, and widget
callbacks.
Outputs: rebuilt Plot, ROI, and Epoch option controls.

The static tab shell lives in ``options_panel``. This module owns the parts that
must be rebuilt when plot data, axis defaults, or visibility rows change.
"""

from collections.abc import Callable

from qtpy.QtGui import QColor
from qtpy.QtWidgets import QCheckBox, QPushButton, QSpinBox, QVBoxLayout

from twopy.napari.plotting.data import ResponsePlotData
from twopy.napari.plotting.options import (
    plot_display_options_group,
    visibility_options_widget,
)
from twopy.napari.plotting.widgets import (
    clear_layout,
    global_time_bounds,
    global_value_bounds,
)


def clear_dynamic_option_tabs(
    *,
    plot_display_options_layout: QVBoxLayout,
    roi_options_layout: QVBoxLayout,
    epoch_options_layout: QVBoxLayout,
) -> None:
    """Clear dynamic widgets from Plot, ROI, and Epoch option tabs.

    Args:
        plot_display_options_layout: Layout containing the axis display group.
        roi_options_layout: Layout containing ROI visibility controls.
        epoch_options_layout: Layout containing epoch visibility controls.

    Returns:
        None.
    """
    clear_layout(plot_display_options_layout)
    clear_layout(roi_options_layout)
    clear_layout(epoch_options_layout)


def render_dynamic_options(
    *,
    plot_data: ResponsePlotData | None,
    plot_display_options_layout: QVBoxLayout,
    roi_options_layout: QVBoxLayout,
    epoch_options_layout: QVBoxLayout,
    show_sem: bool,
    plot_size: int,
    manual_x_min: float | None,
    manual_x_max: float | None,
    manual_y_min: float | None,
    manual_y_max: float | None,
    visible_roi_indices: tuple[int, ...],
    visible_epoch_indices: tuple[int, ...],
    roi_labels: tuple[str, ...],
    roi_visibility: dict[int, bool],
    roi_colors: tuple[QColor, ...],
    roi_area_pixel_details: tuple[str, ...] | None,
    epoch_visibility: dict[int, bool],
    on_show_sem_change: Callable[[int], None],
    on_plot_size_change: Callable[[int], None],
    on_axis_change: Callable[..., None],
    on_roi_visibility_change: Callable[[object, bool], None],
    on_roi_visibility_batch: Callable[[dict[object, bool]], None],
    on_merge_selected_rois: Callable[[], None],
    on_remove_selected_rois: Callable[[], None],
    on_epoch_visibility_change: Callable[[object, bool], None],
    on_epoch_visibility_batch: Callable[[dict[object, bool]], None],
) -> None:
    """Render current axis, ROI, and epoch visibility controls.

    Args:
        plot_data: Current response plot data, or None before data is loaded.
        plot_display_options_layout: Layout containing the axis display group.
        roi_options_layout: Layout containing ROI visibility controls.
        epoch_options_layout: Layout containing epoch visibility controls.
        show_sem: Whether the SEM checkbox should be checked.
        plot_size: Current plot width in pixels.
        manual_x_min: Manual x-axis minimum, if set.
        manual_x_max: Manual x-axis maximum, if set.
        manual_y_min: Manual y-axis minimum, if set.
        manual_y_max: Manual y-axis maximum, if set.
        visible_roi_indices: ROI rows currently visible in plots.
        visible_epoch_indices: Epoch rows currently visible in plots.
        roi_labels: ROI labels in plot order.
        roi_visibility: Current ROI visibility state.
        roi_colors: ROI colors in plot order.
        roi_area_pixel_details: Area text shown beside each ROI row, or
            ``None`` when no editable Labels layer is available.
        epoch_visibility: Current epoch visibility state.
        on_show_sem_change: Callback for the SEM checkbox.
        on_plot_size_change: Callback for the plot-size spinbox.
        on_axis_change: Callback for manual axis-bound edits.
        on_roi_visibility_change: Callback for one ROI checkbox.
        on_roi_visibility_batch: Callback for ROI batch buttons.
        on_merge_selected_rois: Callback that combines checked ROI labels in
            the active Labels layer.
        on_remove_selected_rois: Callback that deletes checked ROI labels from
            the active Labels layer.
        on_epoch_visibility_change: Callback for one epoch checkbox.
        on_epoch_visibility_batch: Callback for epoch batch buttons.

    Returns:
        None.
    """
    clear_dynamic_option_tabs(
        plot_display_options_layout=plot_display_options_layout,
        roi_options_layout=roi_options_layout,
        epoch_options_layout=epoch_options_layout,
    )
    if plot_data is None or len(plot_data.epochs) == 0:
        return

    auto_x_min, auto_x_max = global_time_bounds(plot_data)
    auto_y_min, auto_y_max = global_value_bounds(
        plot_data,
        visible_roi_indices,
        visible_epoch_indices,
    )
    add_plot_display_options_group(
        plot_display_options_layout=plot_display_options_layout,
        show_sem=show_sem,
        plot_size=plot_size,
        x_min=manual_x_min if manual_x_min is not None else auto_x_min,
        x_max=manual_x_max if manual_x_max is not None else auto_x_max,
        y_min=manual_y_min if manual_y_min is not None else auto_y_min,
        y_max=manual_y_max if manual_y_max is not None else auto_y_max,
        on_show_sem_change=on_show_sem_change,
        on_plot_size_change=on_plot_size_change,
        on_axis_change=on_axis_change,
    )
    render_roi_options(
        roi_options_layout=roi_options_layout,
        roi_labels=roi_labels,
        roi_visibility=roi_visibility,
        roi_colors=roi_colors,
        roi_area_pixel_details=roi_area_pixel_details,
        on_roi_visibility_change=on_roi_visibility_change,
        on_roi_visibility_batch=on_roi_visibility_batch,
        on_merge_selected_rois=on_merge_selected_rois,
        on_remove_selected_rois=on_remove_selected_rois,
    )
    epoch_options_layout.addWidget(
        visibility_options_widget(
            title="Epochs",
            labels=tuple(
                f"{epoch.epoch_number}: {epoch.epoch_name}"
                for epoch in plot_data.epochs
            ),
            visibility=epoch_visibility,
            on_change=on_epoch_visibility_change,
            on_change_batch=on_epoch_visibility_batch,
            keys=tuple(range(len(plot_data.epochs))),
        ),
    )
    epoch_options_layout.addStretch(1)


def render_roi_options(
    *,
    roi_options_layout: QVBoxLayout,
    roi_labels: tuple[str, ...],
    roi_visibility: dict[int, bool],
    roi_colors: tuple[QColor, ...],
    on_roi_visibility_change: Callable[[object, bool], None],
    on_roi_visibility_batch: Callable[[dict[object, bool]], None],
    on_merge_selected_rois: Callable[[], None],
    on_remove_selected_rois: Callable[[], None],
    roi_area_pixel_details: tuple[str, ...] | None = None,
) -> None:
    """Render the ROIs tab controls from current visibility state.

    Args:
        roi_options_layout: Layout containing ROI visibility controls.
        roi_labels: ROI labels in plot order.
        roi_visibility: Current ROI visibility state.
        roi_colors: ROI colors in plot order.
        on_roi_visibility_change: Callback for one ROI checkbox.
        on_roi_visibility_batch: Callback for ROI batch buttons.
        on_merge_selected_rois: Callback that combines checked ROI labels in
            the active Labels layer.
        on_remove_selected_rois: Callback that deletes checked ROI labels from
            the active Labels layer.
        roi_area_pixel_details: Area text shown beside each ROI row, or
            ``None`` when no editable Labels layer is available.

    Returns:
        None.
    """
    clear_layout(roi_options_layout)
    roi_widget = visibility_options_widget(
        title="ROIs",
        labels=roi_labels,
        visibility=roi_visibility,
        on_change=on_roi_visibility_change,
        on_change_batch=on_roi_visibility_batch,
        keys=tuple(range(len(roi_labels))),
        colors=roi_colors,
        details=roi_area_pixel_details,
        detail_header="area (px)" if roi_area_pixel_details is not None else None,
    )
    roi_options_layout.addWidget(roi_widget)
    merge_button = QPushButton("Merge Selected")
    merge_button.clicked.connect(on_merge_selected_rois)
    roi_options_layout.addWidget(merge_button)
    remove_button = QPushButton("Remove Selected")
    remove_button.clicked.connect(on_remove_selected_rois)
    roi_options_layout.addWidget(remove_button)
    roi_options_layout.addStretch(1)


def add_plot_display_options_group(
    *,
    plot_display_options_layout: QVBoxLayout,
    show_sem: bool,
    plot_size: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    on_show_sem_change: Callable[[int], None],
    on_plot_size_change: Callable[[int], None],
    on_axis_change: Callable[..., None],
) -> None:
    """Add the Plot group with the same form layout as processing groups.

    Args:
        plot_display_options_layout: Layout receiving the display option group.
        show_sem: Whether the SEM checkbox should be checked.
        plot_size: Current plot width in pixels.
        x_min: Current x-axis minimum.
        x_max: Current x-axis maximum.
        y_min: Current y-axis minimum.
        y_max: Current y-axis maximum.
        on_show_sem_change: Callback for SEM checkbox edits.
        on_plot_size_change: Callback for plot-size spinbox edits.
        on_axis_change: Callback for manual axis-bound edits.

    Returns:
        None.
    """
    show_sem_checkbox = QCheckBox("show SEM")
    show_sem_checkbox.setChecked(show_sem)
    show_sem_checkbox.stateChanged.connect(on_show_sem_change)
    plot_size_spin = QSpinBox()
    plot_size_spin.setRange(240, 900)
    plot_size_spin.setSingleStep(20)
    plot_size_spin.setSuffix(" px")
    plot_size_spin.setValue(plot_size)
    plot_size_spin.valueChanged.connect(on_plot_size_change)
    plot_display_options_layout.addWidget(
        plot_display_options_group(
            show_sem_checkbox=show_sem_checkbox,
            plot_size_spin=plot_size_spin,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            on_change=on_axis_change,
        ),
    )
