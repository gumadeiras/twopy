"""Response plotting widgets for the twopy napari adapter.

Inputs: loaded twopy analysis outputs or the current napari Labels ROI layer.
Outputs: Qt widgets showing one response plot per stimulus epoch and a
separate options panel.

The plotting code only owns GUI state and drawing. When the user asks to update
responses from the current Labels layer, this module converts labels to a
``RoiSet`` and calls the normal analysis workflow.
"""

from pathlib import Path

import numpy as np
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.responses import is_gray_epoch_name
from twopy.analysis.workflow import analyze_recording_responses
from twopy.converted import RecordingData
from twopy.napari.plotting.data import (
    ResponsePlotData,
    default_analysis_output_path,
    load_response_plot_data,
    response_plot_data_from_grouped,
)
from twopy.napari.plotting.label_visibility import (
    apply_roi_visibility_to_labels_layer,
    roi_label_value_from_label,
)
from twopy.napari.plotting.options import (
    axis_options_widget,
    visibility_options_widget,
)
from twopy.napari.plotting.widgets import (
    EpochPlotWidget,
    clear_layout,
    epoch_key_label,
    global_time_bounds,
    global_value_bounds,
    ordered_bounds,
    roi_colors_from_label_values,
)
from twopy.napari.protocols import NapariViewer
from twopy.napari.roi import roi_label_image_from_layer
from twopy.roi import make_roi_set_from_label_image

__all__ = [
    "add_twopy_response_options_widget",
    "add_twopy_response_plot_widget",
    "refresh_response_plot_widget",
]


def add_twopy_response_plot_widget(
    viewer: NapariViewer,
    *,
    recording: RecordingData | None,
    roi_labels_layer: object | None = None,
    dock_name: str = "twopy responses",
    dock_area: str = "top",
) -> tuple[object, object]:
    """Add the response plotting dock widget to a napari viewer.

    Args:
        viewer: Napari viewer that should receive the dock widget.
        recording: Optional loaded converted recording.
        roi_labels_layer: Optional napari Labels layer used for analysis
            previews from current ROIs.
        dock_name: Dock widget title.
        dock_area: Napari dock area.

    Returns:
        ``(widget, dock_widget)`` created by Qt and napari.
    """
    widget = create_response_plot_widget(
        recording,
        roi_labels_layer=roi_labels_layer,
    )
    dock_widget = viewer.window.add_dock_widget(
        widget,
        name=dock_name,
        area=dock_area,
    )
    return widget, dock_widget


def add_twopy_response_options_widget(
    viewer: NapariViewer,
    response_plot_widget: object,
    *,
    dock_name: str = "twopy response options",
    dock_area: str = "right",
) -> tuple[object, object] | tuple[None, None]:
    """Add the response plot options as a separate napari dock.

    Args:
        viewer: Napari viewer that should receive the dock widget.
        response_plot_widget: Widget returned by ``create_response_plot_widget``.
        dock_name: Dock widget title.
        dock_area: Napari dock area.

    Returns:
        ``(widget, dock_widget)`` when the plot widget owns options, otherwise
        ``(None, None)``.
    """
    if not isinstance(response_plot_widget, _ResponsePlotWidget):
        return None, None
    widget = response_plot_widget.options_widget()
    dock_widget = viewer.window.add_dock_widget(
        widget,
        name=dock_name,
        area=dock_area,
    )
    return widget, dock_widget


def create_response_plot_widget(
    recording: RecordingData | None,
    *,
    roi_labels_layer: object | None = None,
) -> object:
    """Create a response plotting widget.

    Args:
        recording: Optional loaded converted recording.
        roi_labels_layer: Optional napari Labels layer used when the user asks
            to compute responses from the current ROIs.

    Returns:
        Qt widget as a plain object for napari docking.
    """
    _ensure_qapplication()
    widget = _ResponsePlotWidget()
    refresh_response_plot_widget(
        widget,
        recording=recording,
        roi_labels_layer=roi_labels_layer,
    )
    return widget


def refresh_response_plot_widget(
    widget: object | None,
    *,
    recording: RecordingData | None,
    roi_labels_layer: object | None = None,
) -> None:
    """Refresh a response plotting widget after a recording is loaded.

    Args:
        widget: Optional widget returned by ``create_response_plot_widget``.
        recording: Optional loaded converted recording.
        roi_labels_layer: Optional current ROI Labels layer.

    Returns:
        None.
    """
    if not isinstance(widget, _ResponsePlotWidget):
        return
    widget.set_roi_labels_layer(roi_labels_layer)
    if recording is None:
        widget.clear_recording()
        return
    widget.load_recording(recording)


def _ensure_qapplication() -> None:
    """Create a Qt application when tests instantiate widgets without napari.

    Args:
        None.

    Returns:
        None.
    """
    from qtpy.QtWidgets import QApplication

    if QApplication.instance() is None:
        QApplication([])


class _ResponsePlotWidget(QWidget):
    """Qt widget for response plots plus a separately docked options panel."""

    def __init__(self) -> None:
        """Create empty response plots and an options panel."""
        super().__init__()
        self._recording: RecordingData | None = None
        self._roi_labels_layer: object | None = None
        self._analysis_path: Path | None = None
        self._plot_data: ResponsePlotData | None = None
        self._show_sem = True
        self._roi_visibility: dict[str, bool] = {}
        self._roi_colors: tuple[QColor, ...] = ()
        self._epoch_visibility: dict[tuple[int, str], bool] = {}
        self._manual_x_min: float | None = None
        self._manual_x_max: float | None = None
        self._manual_y_min: float | None = None
        self._manual_y_max: float | None = None
        self._status_label = QLabel("No recording loaded.")
        self._plot_layout = QHBoxLayout()
        self._plot_layout.addWidget(self._status_label)
        plot_content = QWidget()
        plot_content.setLayout(self._plot_layout)
        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setWidget(plot_content)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(plot_scroll)
        self.setLayout(layout)

        self._options_tabs = QTabWidget()
        self._show_sem_checkbox = QCheckBox("Show SEM")
        self._show_sem_checkbox.setChecked(True)
        self._show_sem_checkbox.stateChanged.connect(self._set_show_sem)
        refresh_button = QPushButton("Refresh responses")
        refresh_button.clicked.connect(self.reload)
        update_from_rois_button = QPushButton("Update from current ROIs")
        update_from_rois_button.clicked.connect(self.update_from_current_rois)
        self._analysis_path_label = QLabel("Analysis file: default")
        self._analysis_path_label.setWordWrap(True)
        self._plot_options_layout = QVBoxLayout()
        self._plot_axis_layout = QVBoxLayout()
        plot_axis_widget = QWidget()
        plot_axis_widget.setLayout(self._plot_axis_layout)
        self._plot_options_layout.addWidget(self._show_sem_checkbox)
        self._plot_options_layout.addWidget(plot_axis_widget)
        self._plot_options_layout.addStretch(1)
        self._roi_options_layout = QVBoxLayout()
        self._epoch_options_layout = QVBoxLayout()
        self._options_tabs.addTab(
            _response_update_tab(
                refresh_button=refresh_button,
                update_from_rois_button=update_from_rois_button,
                analysis_path_label=self._analysis_path_label,
            ),
            "Update",
        )
        self._options_tabs.addTab(
            _scrolling_tab(self._plot_options_layout),
            "Plot",
        )
        self._options_tabs.addTab(
            _scrolling_tab(self._roi_options_layout),
            "ROIs",
        )
        self._options_tabs.addTab(
            _scrolling_tab(self._epoch_options_layout),
            "Epochs",
        )

    def options_widget(self) -> object:
        """Return the separately docked response-options widget.

        Args:
            None.

        Returns:
            Tabbed Qt widget with response plot controls.
        """
        return self._options_tabs

    def set_roi_labels_layer(self, roi_labels_layer: object | None) -> None:
        """Store the current editable ROI Labels layer.

        Args:
            roi_labels_layer: Optional napari Labels layer.

        Returns:
            None.
        """
        self._roi_labels_layer = roi_labels_layer
        self._render_plots()

    def clear_recording(self) -> None:
        """Reset the widget to the empty-launch state.

        Args:
            None.

        Returns:
            None.
        """
        self._recording = None
        self._analysis_path = None
        self._plot_data = None
        self._reset_plot_state()
        self._analysis_path_label.setText("Analysis file: default")
        self._clear_dynamic_option_tabs()
        self._set_status("No recording loaded.")

    def load_recording(self, recording: RecordingData) -> None:
        """Load response plots for one recording.

        Args:
            recording: Loaded converted recording.

        Returns:
            None.
        """
        self._recording = recording
        self._analysis_path = default_analysis_output_path(recording)
        self._analysis_path_label.setText(f"Analysis file: {self._analysis_path}")
        self.reload()

    def update_from_current_rois(self) -> None:
        """Compute and plot responses from the current ROI Labels layer.

        Args:
            None.

        Returns:
            None.

        The analysis workflow streams the converted movie in chunks, so this
        action avoids loading a large full movie only to update plots.
        """
        if self._recording is None:
            self._set_status("No recording loaded.")
            return
        if self._roi_labels_layer is None:
            self._set_status("No ROI Labels layer is available.")
            return
        label_image = roi_label_image_from_layer(self._roi_labels_layer)
        if not np.any(label_image > 0):
            self._set_status("No ROI labels to analyze.")
            return

        try:
            roi_set = make_roi_set_from_label_image(label_image)
            run = analyze_recording_responses(
                self._recording.path,
                roi_set,
                output_path=self._analysis_path,
                write_summary_csv=False,
            )
        except ValueError as error:
            self._plot_data = None
            self._set_status(str(error))
            return

        plot_data = load_response_plot_data(run.output_path)
        if isinstance(plot_data, str):
            self._plot_data = response_plot_data_from_grouped(
                run.grouped_responses,
                source_path=run.output_path,
            )
        else:
            self._plot_data = plot_data
        self._sync_plot_state(reset_axes=True)
        self._render_plots()

    def reload(self) -> None:
        """Reload response plots from the current analysis output file."""
        if self._analysis_path is None:
            self._set_status("No recording loaded.")
            return
        result = load_response_plot_data(self._analysis_path)
        if isinstance(result, str):
            self._plot_data = None
            self._reset_plot_state()
            self._clear_dynamic_option_tabs()
            self._set_status(result)
            return
        self._plot_data = result
        self._sync_plot_state(reset_axes=True)
        self._render_plots()

    def _set_show_sem(self, state: int) -> None:
        """Update whether SEM bands are drawn.

        Args:
            state: Qt checkbox state.

        Returns:
            None.
        """
        del state
        self._show_sem = self._show_sem_checkbox.isChecked()
        self._render_plots()

    def _sync_plot_state(self, *, reset_axes: bool) -> None:
        """Synchronize option state with currently loaded plot data.

        Args:
            reset_axes: Whether axis spin boxes should return to data-derived
                defaults.

        Returns:
            None.
        """
        if self._plot_data is None:
            self._reset_plot_state()
            return

        roi_labels = self._plot_data.epochs[0].roi_labels
        self._roi_visibility = {
            roi_label: self._roi_visibility.get(roi_label, True)
            for roi_label in roi_labels
        }
        self._roi_colors = _opaque_colors(
            roi_colors_from_label_values(
                self._roi_labels_layer,
                _label_values_from_roi_labels(roi_labels),
                fallback_count=len(roi_labels),
            )
        )
        self._epoch_visibility = {
            (epoch.epoch_number, epoch.epoch_name): self._epoch_visibility.get(
                (epoch.epoch_number, epoch.epoch_name),
                not is_gray_epoch_name(epoch.epoch_name),
            )
            for epoch in self._plot_data.epochs
        }
        if reset_axes:
            self._manual_x_min = None
            self._manual_x_max = None
            self._manual_y_min = None
            self._manual_y_max = None
        self._render_options()

    def _reset_plot_state(self) -> None:
        """Clear plot-specific option state.

        Args:
            None.

        Returns:
            None.
        """
        self._roi_visibility = {}
        self._roi_colors = ()
        self._epoch_visibility = {}
        self._manual_x_min = None
        self._manual_x_max = None
        self._manual_y_min = None
        self._manual_y_max = None

    def _render_options(self) -> None:
        """Render axis, ROI, and epoch visibility controls."""
        self._clear_dynamic_option_tabs()
        if self._plot_data is None or len(self._plot_data.epochs) == 0:
            return

        auto_x_min, auto_x_max = global_time_bounds(self._plot_data)
        auto_y_min, auto_y_max = global_value_bounds(
            self._plot_data,
            self._visible_roi_indices(),
            self._visible_epoch_keys(),
        )
        axis_widget = axis_options_widget(
            x_min=self._manual_x_min if self._manual_x_min is not None else auto_x_min,
            x_max=self._manual_x_max if self._manual_x_max is not None else auto_x_max,
            y_min=self._manual_y_min if self._manual_y_min is not None else auto_y_min,
            y_max=self._manual_y_max if self._manual_y_max is not None else auto_y_max,
            on_change=self._set_manual_axis_bounds,
        )
        self._plot_axis_layout.addWidget(axis_widget)
        self._roi_options_layout.addWidget(
            visibility_options_widget(
                title="ROIs",
                labels=self._roi_labels(),
                visibility=self._roi_visibility,
                on_change=self._set_roi_visibility,
                colors=self._roi_colors,
            ),
        )
        self._roi_options_layout.addStretch(1)
        self._epoch_options_layout.addWidget(
            visibility_options_widget(
                title="Epochs",
                labels=tuple(
                    epoch_key_label(epoch.epoch_number, epoch.epoch_name)
                    for epoch in self._plot_data.epochs
                ),
                visibility={
                    epoch_key_label(epoch_number, epoch_name): visible
                    for (
                        epoch_number,
                        epoch_name,
                    ), visible in self._epoch_visibility.items()
                },
                on_change=self._set_epoch_visibility_by_label,
            ),
        )
        self._epoch_options_layout.addStretch(1)

    def _clear_dynamic_option_tabs(self) -> None:
        """Clear dynamic option widgets from plot, ROI, and epoch tabs.

        Args:
            None.

        Returns:
            None.
        """
        clear_layout(self._plot_axis_layout)
        clear_layout(self._roi_options_layout)
        clear_layout(self._epoch_options_layout)

    def _set_manual_axis_bounds(
        self,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        """Store user-selected plot bounds.

        Args:
            x_min: Minimum x-axis value in seconds.
            x_max: Maximum x-axis value in seconds.
            y_min: Minimum dF/F value.
            y_max: Maximum dF/F value.

        Returns:
            None.
        """
        self._manual_x_min = x_min
        self._manual_x_max = x_max
        self._manual_y_min = y_min
        self._manual_y_max = y_max
        self._render_plots()

    def _set_roi_visibility(self, label: str, visible: bool) -> None:
        """Update one ROI visibility flag.

        Args:
            label: ROI label.
            visible: Whether the ROI should be plotted.

        Returns:
            None.
        """
        self._roi_visibility[label] = visible
        self._render_plots()

    def _set_epoch_visibility_by_label(self, label: str, visible: bool) -> None:
        """Update one epoch visibility flag from an options label.

        Args:
            label: Display label for the epoch checkbox.
            visible: Whether the epoch should be plotted.

        Returns:
            None.
        """
        for key in self._epoch_visibility:
            if epoch_key_label(key[0], key[1]) == label:
                self._epoch_visibility[key] = visible
                break
        self._render_plots()

    def _visible_roi_indices(self) -> tuple[int, ...]:
        """Return zero-based ROI indices currently visible in plots.

        Args:
            None.

        Returns:
            Tuple of ROI indices.
        """
        return tuple(
            index
            for index, roi_label in enumerate(self._roi_labels())
            if self._roi_visibility.get(roi_label, True)
        )

    def _visible_epoch_keys(self) -> tuple[tuple[int, str], ...]:
        """Return epoch keys currently visible in plots.

        Args:
            None.

        Returns:
            Tuple of ``(epoch_number, epoch_name)`` keys.
        """
        return tuple(key for key, visible in self._epoch_visibility.items() if visible)

    def _roi_labels(self) -> tuple[str, ...]:
        """Return ROI labels from the current plot data.

        Args:
            None.

        Returns:
            ROI labels, or an empty tuple before data is loaded.
        """
        if self._plot_data is None or len(self._plot_data.epochs) == 0:
            return ()
        return self._plot_data.epochs[0].roi_labels

    def _set_status(self, text: str) -> None:
        """Replace plots with one status message.

        Args:
            text: User-facing status message.

        Returns:
            None.
        """
        clear_layout(self._plot_layout)
        self._status_label = QLabel(text)
        self._status_label.setWordWrap(True)
        self._plot_layout.addWidget(self._status_label)
        self._plot_layout.addStretch(1)

    def _render_plots(self) -> None:
        """Render one plot widget per epoch type."""
        clear_layout(self._plot_layout)
        if self._plot_data is None or len(self._plot_data.epochs) == 0:
            self._set_status("No responses available.")
            return
        self._apply_roi_layer_visibility()
        roi_indices = self._visible_roi_indices()
        epoch_keys = self._visible_epoch_keys()
        if len(roi_indices) == 0 or len(epoch_keys) == 0:
            self._set_status("No responses selected.")
            return
        auto_x_min, auto_x_max = global_time_bounds(self._plot_data, epoch_keys)
        auto_y_min, auto_y_max = global_value_bounds(
            self._plot_data,
            roi_indices,
            epoch_keys,
        )
        time_min = self._manual_x_min if self._manual_x_min is not None else auto_x_min
        time_max = self._manual_x_max if self._manual_x_max is not None else auto_x_max
        value_min = self._manual_y_min if self._manual_y_min is not None else auto_y_min
        value_max = self._manual_y_max if self._manual_y_max is not None else auto_y_max
        time_min, time_max = ordered_bounds(time_min, time_max)
        value_min, value_max = ordered_bounds(value_min, value_max)
        colors = self._roi_colors
        for epoch in self._plot_data.epochs:
            if not self._epoch_visibility.get(
                (epoch.epoch_number, epoch.epoch_name),
                True,
            ):
                continue
            title = f"Epoch {epoch.epoch_number}: {epoch.epoch_name}"
            self._plot_layout.addWidget(
                _epoch_plot_panel(
                    title=title,
                    plot=EpochPlotWidget(
                        epoch,
                        show_sem=self._show_sem,
                        roi_indices=roi_indices,
                        roi_colors=colors,
                        time_min=time_min,
                        time_max=time_max,
                        value_min=value_min,
                        value_max=value_max,
                    ),
                )
            )
        self._plot_layout.addStretch(1)

    def _apply_roi_layer_visibility(self) -> None:
        """Mirror plot ROI visibility onto the napari Labels overlay."""
        apply_roi_visibility_to_labels_layer(
            self._roi_labels_layer,
            roi_labels=self._roi_labels(),
            visibility=self._roi_visibility,
            colors=self._roi_colors,
        )


def _epoch_plot_panel(*, title: str, plot: EpochPlotWidget) -> QWidget:
    """Create one titled plot panel for the horizontal response strip.

    Args:
        title: Plot title.
        plot: Epoch response plot widget.

    Returns:
        Qt widget containing the title and plot.
    """
    panel = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(QLabel(title))
    layout.addWidget(plot)
    panel.setLayout(layout)
    return panel


def _response_update_tab(
    *,
    refresh_button: QPushButton,
    update_from_rois_button: QPushButton,
    analysis_path_label: QLabel,
) -> QWidget:
    """Create the tab that owns response recompute/reload actions.

    Args:
        refresh_button: Button that reloads persisted response outputs.
        update_from_rois_button: Button that computes responses from current
            Labels ROIs.
        analysis_path_label: Label showing the analysis output path.

    Returns:
        Qt widget for the Update tab.
    """
    tab = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(refresh_button)
    layout.addWidget(update_from_rois_button)
    layout.addWidget(analysis_path_label)
    layout.addStretch(1)
    tab.setLayout(layout)
    return tab


def _scrolling_tab(layout: QVBoxLayout) -> QScrollArea:
    """Create a scrollable options tab around one layout.

    Args:
        layout: Layout that callers will populate with controls.

    Returns:
        Scroll area suitable for a tab body.
    """
    content = QWidget()
    content.setLayout(layout)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    return scroll


def _opaque_colors(colors: tuple[QColor, ...]) -> tuple[QColor, ...]:
    """Return ROI colors with full alpha for plotting and visibility restore.

    Args:
        colors: Colors read from the Labels layer.

    Returns:
        Same RGB colors with alpha set to fully opaque.
    """
    opaque: list[QColor] = []
    for color in colors:
        updated = QColor(color)
        updated.setAlpha(255)
        opaque.append(updated)
    return tuple(opaque)


def _label_values_from_roi_labels(roi_labels: tuple[str, ...]) -> tuple[int, ...]:
    """Return napari label values for plot ROI labels.

    Args:
        roi_labels: ROI labels in plot order.

    Returns:
        Integer Labels-layer values in the same order.
    """
    return tuple(
        roi_label_value_from_label(roi_label, fallback=index + 1)
        for index, roi_label in enumerate(roi_labels)
    )
