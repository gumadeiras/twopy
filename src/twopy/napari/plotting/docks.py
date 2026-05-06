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
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.responses import is_gray_epoch_name
from twopy.analysis.workflow import analyze_recording_responses
from twopy.converted import RecordingData
from twopy.napari.display_paths import (
    format_output_folder,
    format_twopy_h5_output,
    recording_display_summary,
)
from twopy.napari.interactive import LiveResponseController
from twopy.napari.plotting.data import (
    ResponsePlotData,
    default_analysis_output_path,
    load_response_plot_data,
)
from twopy.napari.plotting.export_controls import (
    ResponseExportState,
    create_response_export_tab,
)
from twopy.napari.plotting.label_visibility import (
    apply_roi_visibility_to_labels_layer,
    roi_label_values_from_labels,
)
from twopy.napari.plotting.options import (
    axis_options_widget,
    visibility_options_widget,
)
from twopy.napari.plotting.panels import (
    epoch_plot_panel,
    plot_size_widget,
    response_update_tab,
    scrolling_tab,
)
from twopy.napari.plotting.widgets import (
    EpochPlotWidget,
    clear_layout,
    global_time_bounds,
    global_value_bounds,
    opaque_colors,
    resolved_time_bounds,
    resolved_value_bounds,
    roi_colors_from_label_values,
)
from twopy.napari.protocols import NapariViewer
from twopy.napari.roi import (
    roi_label_image_from_layer_for_recording,
    save_napari_label_rois,
)

__all__ = [
    "add_twopy_response_options_widget",
    "add_twopy_response_plot_widget",
    "refresh_response_plot_widget",
]

_QT_APPLICATION: object | None = None


def add_twopy_response_plot_widget(
    viewer: NapariViewer,
    *,
    recording: RecordingData | None,
    roi_labels_layer: object | None = None,
    roi_save_file: Path | None = None,
    dock_name: str = "twopy responses",
    dock_area: str = "top",
) -> tuple[object, object]:
    """Add the response plotting dock widget to a napari viewer.

    Args:
        viewer: Napari viewer that should receive the dock widget.
        recording: Optional loaded converted recording.
        roi_labels_layer: Optional napari Labels layer used for analysis
            previews from current ROIs.
        roi_save_file: Optional ROI HDF5 path used by the Save Analysis action.
        dock_name: Dock widget title.
        dock_area: Napari dock area.

    Returns:
        ``(widget, dock_widget)`` created by Qt and napari.
    """
    widget = create_response_plot_widget(
        recording,
        viewer=viewer,
        roi_labels_layer=roi_labels_layer,
        roi_save_file=roi_save_file,
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
    viewer: object | None = None,
    roi_labels_layer: object | None = None,
    roi_save_file: Path | None = None,
) -> object:
    """Create a response plotting widget.

    Args:
        recording: Optional loaded converted recording.
        viewer: Optional napari viewer used for figure export.
        roi_labels_layer: Optional napari Labels layer used when the user asks
            to compute responses from the current ROIs.
        roi_save_file: Optional ROI HDF5 path used by the Save Analysis action.

    Returns:
        Qt widget as a plain object for napari docking.
    """
    _ensure_qapplication()
    widget = _ResponsePlotWidget(viewer=viewer)
    refresh_response_plot_widget(
        widget,
        recording=recording,
        roi_labels_layer=roi_labels_layer,
        roi_save_file=roi_save_file,
    )
    return widget


def refresh_response_plot_widget(
    widget: object | None,
    *,
    recording: RecordingData | None,
    roi_labels_layer: object | None = None,
    roi_save_file: Path | None = None,
) -> None:
    """Refresh a response plotting widget after a recording is loaded.

    Args:
        widget: Optional widget returned by ``create_response_plot_widget``.
        recording: Optional loaded converted recording.
        roi_labels_layer: Optional current ROI Labels layer.
        roi_save_file: Optional ROI HDF5 path used by the Save Analysis action.

    Returns:
        None.
    """
    if not isinstance(widget, _ResponsePlotWidget):
        return
    widget.set_roi_labels_layer(roi_labels_layer)
    widget.set_roi_save_file(roi_save_file)
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

    global _QT_APPLICATION
    if QApplication.instance() is None:
        _QT_APPLICATION = QApplication([])


class _ResponsePlotWidget(QWidget):
    """Qt widget for response plots plus a separately docked options panel."""

    def __init__(self, *, viewer: object | None = None) -> None:
        """Create empty response plots and an options panel.

        Args:
            viewer: Optional napari viewer used by export buttons to read the
                currently displayed movie frame.
        """
        super().__init__()
        self._viewer = viewer
        self._recording: RecordingData | None = None
        self._roi_labels_layer: object | None = None
        self._analysis_path: Path | None = None
        self._roi_save_file: Path | None = None
        self._plot_data: ResponsePlotData | None = None
        self._show_sem = True
        self._plot_size = 360
        self._epoch_plot_widgets: dict[int, EpochPlotWidget] = {}
        self._epoch_plot_panels: dict[int, QWidget] = {}
        self._roi_visibility: dict[int, bool] = {}
        self._roi_colors: tuple[QColor, ...] = ()
        self._epoch_visibility: dict[int, bool] = {}
        self._manual_x_min: float | None = None
        self._manual_x_max: float | None = None
        self._manual_y_min: float | None = None
        self._manual_y_max: float | None = None
        self._live_controller = LiveResponseController(self)
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
        reload_saved_button = QPushButton("Reload saved analysis")
        reload_saved_button.clicked.connect(self.reload)
        recompute_preview_button = QPushButton("Recompute preview now")
        recompute_preview_button.clicked.connect(self.update_from_current_rois)
        save_analysis_button = QPushButton("Save analysis + ROIs")
        save_analysis_button.clicked.connect(self.save_analysis_and_rois)
        self._recording_summary_label = QLabel("No recording loaded.")
        self._recording_summary_label.setWordWrap(True)
        self._analysis_path_label = QLabel("Analysis output: default")
        self._analysis_path_label.setWordWrap(True)
        self._roi_save_path_label = QLabel("ROI output: default")
        self._roi_save_path_label.setWordWrap(True)
        self._update_status_label = QLabel("")
        self._update_status_label.setWordWrap(True)
        self._plot_options_layout = QVBoxLayout()
        self._plot_axis_layout = QVBoxLayout()
        plot_axis_widget = QWidget()
        plot_axis_widget.setLayout(self._plot_axis_layout)
        self._plot_size_spin = QSpinBox()
        self._plot_size_spin.setRange(240, 900)
        self._plot_size_spin.setSingleStep(20)
        self._plot_size_spin.setSuffix(" px")
        self._plot_size_spin.setValue(self._plot_size)
        self._plot_size_spin.valueChanged.connect(self._set_plot_size)
        self._plot_options_layout.addWidget(self._show_sem_checkbox)
        self._plot_options_layout.addWidget(plot_size_widget(self._plot_size_spin))
        self._plot_options_layout.addWidget(plot_axis_widget)
        self._plot_options_layout.addStretch(1)
        self._roi_options_layout = QVBoxLayout()
        self._epoch_options_layout = QVBoxLayout()
        self._options_tabs.addTab(
            response_update_tab(
                reload_saved_button=reload_saved_button,
                recompute_preview_button=recompute_preview_button,
                save_analysis_button=save_analysis_button,
                recording_summary_label=self._recording_summary_label,
                analysis_output_label=self._analysis_path_label,
                roi_output_label=self._roi_save_path_label,
                status_label=self._update_status_label,
            ),
            "Update",
        )
        self._options_tabs.addTab(
            scrolling_tab(self._plot_options_layout),
            "Plot",
        )
        self._options_tabs.addTab(
            scrolling_tab(self._roi_options_layout),
            "ROIs",
        )
        self._options_tabs.addTab(
            scrolling_tab(self._epoch_options_layout),
            "Epochs",
        )
        self._options_tabs.addTab(
            create_response_export_tab(self._export_state),
            "Export",
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
        self._live_controller.set_context(self._recording, self._roi_labels_layer)
        self._render_plots()

    def set_roi_save_file(self, roi_save_file: Path | None) -> None:
        """Store the ROI HDF5 path used by Save Analysis.

        Args:
            roi_save_file: Optional ROI output path for the selected recording.

        Returns:
            None.
        """
        self._roi_save_file = roi_save_file
        self._refresh_update_path_labels()

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
        self._live_controller.set_context(None, None)
        self._reset_plot_state()
        self._recording_summary_label.setText("No recording loaded.")
        self._analysis_path_label.setText("Analysis output: default")
        self._roi_save_path_label.setText("ROI output: default")
        self._update_status_label.setText("")
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
        self._refresh_update_path_labels()
        self.reload()
        self._live_controller.set_context(self._recording, self._roi_labels_layer)

    def update_from_current_rois(self) -> None:
        """Compute and plot responses from the current ROI Labels layer.

        Args:
            None.

        Returns:
            None.

        The live controller computes in memory and does not write analysis
        files. Explicit analysis saves remain separate from plot previews.
        """
        self._live_controller.update_now()

    def save_analysis_and_rois(self) -> None:
        """Save current ROI labels plus persisted analysis outputs.

        Args:
            None.

        Returns:
            None.

        Live response plots are previews. This method is the explicit
        persistence action: it saves the current Labels layer as ROI HDF5, then
        runs the normal script-facing analysis workflow to write analysis HDF5
        and response CSV outputs.
        """
        if self._recording is None:
            self._set_status("No recording loaded.")
            return
        if self._roi_labels_layer is None:
            self._set_status("No ROI Labels layer is available.")
            return
        try:
            label_image = roi_label_image_from_layer_for_recording(
                self._roi_labels_layer,
                self._recording,
            )
        except ValueError as error:
            self._set_status(str(error))
            return
        if not np.any(label_image > 0):
            self._set_status("No ROI labels to save or analyze.")
            return

        roi_output_path = self._resolved_roi_save_file()
        analysis_output_path = self._resolved_analysis_path()
        try:
            roi_set = save_napari_label_rois(label_image, roi_output_path)
            run = analyze_recording_responses(
                self._recording.path,
                roi_set,
                output_path=analysis_output_path,
            )
        except ValueError as error:
            self._set_status(str(error))
            return

        self._roi_save_file = roi_output_path
        self._analysis_path = run.output_path
        self._refresh_update_path_labels()
        output_folder = format_output_folder(run.output_path, self._recording)
        self._update_status_label.setText(
            f"Saved {len(roi_set.labels)} ROI(s) to {output_folder}"
        )

    def set_response_plot_data(
        self,
        plot_data: ResponsePlotData,
        *,
        reset_axes: bool,
    ) -> None:
        """Show plot data computed from live or saved analysis outputs.

        Args:
            plot_data: Plot-ready response data.
            reset_axes: Whether manual axis bounds should return to automatic
                data-derived values.

        Returns:
            None.
        """
        self._clear_epoch_plot_cache()
        self._plot_data = plot_data
        self._sync_plot_state(reset_axes=reset_axes)
        self._render_plots()

    def show_response_status(self, text: str) -> None:
        """Show a user-facing response-analysis status message.

        Args:
            text: Status text.

        Returns:
            None.
        """
        self._plot_data = None
        self._set_status(text)

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
        self.set_response_plot_data(result, reset_axes=True)

    def _set_show_sem(self, state: int) -> None:
        """Update whether SEM bands are drawn."""
        del state
        self._show_sem = self._show_sem_checkbox.isChecked()
        self._update_visible_plot_widgets(apply_roi_layer_visibility=False)

    def _set_plot_size(self, value: int) -> None:
        """Update the square pixel size used for live response plots."""
        self._plot_size = int(value)
        self._update_visible_plot_widgets(apply_roi_layer_visibility=False)

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
            index: self._roi_visibility.get(index, True)
            for index in range(len(roi_labels))
        }
        self._roi_colors = opaque_colors(
            roi_colors_from_label_values(
                self._roi_labels_layer,
                roi_label_values_from_labels(roi_labels),
                fallback_count=len(roi_labels),
            )
        )
        self._epoch_visibility = {
            index: self._epoch_visibility.get(
                index,
                not is_gray_epoch_name(epoch.epoch_name),
            )
            for index, epoch in enumerate(self._plot_data.epochs)
        }
        if reset_axes:
            self._manual_x_min = None
            self._manual_x_max = None
            self._manual_y_min = None
            self._manual_y_max = None
        self._render_options()

    def _reset_plot_state(self) -> None:
        """Clear plot-specific option state."""
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
            self._visible_epoch_indices(),
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
                on_change_batch=self._set_roi_visibility_batch,
                keys=tuple(range(len(self._roi_labels()))),
                colors=self._roi_colors,
            ),
        )
        self._roi_options_layout.addStretch(1)
        self._epoch_options_layout.addWidget(
            visibility_options_widget(
                title="Epochs",
                labels=tuple(
                    f"{epoch.epoch_number}: {epoch.epoch_name}"
                    for epoch in self._plot_data.epochs
                ),
                visibility=self._epoch_visibility,
                on_change=self._set_epoch_visibility,
                on_change_batch=self._set_epoch_visibility_batch,
                keys=tuple(range(len(self._plot_data.epochs))),
            ),
        )
        self._epoch_options_layout.addStretch(1)

    def _clear_dynamic_option_tabs(self) -> None:
        """Clear dynamic option widgets from plot, ROI, and epoch tabs."""
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
        self._update_visible_plot_widgets(apply_roi_layer_visibility=False)

    def _export_state(self) -> ResponseExportState:
        """Return the current state needed by export buttons."""
        return ResponseExportState(
            viewer=self._viewer,
            recording=self._recording,
            roi_labels_layer=self._roi_labels_layer,
            plot_data=self._plot_data,
            output_dir=self._export_output_dir(),
            roi_label_values=self._roi_label_values(),
            roi_colors=self._roi_color_hex(),
            epoch_indices=self._visible_epoch_indices(),
            roi_indices=self._visible_roi_indices(),
            show_sem=self._show_sem,
            time_bounds=resolved_time_bounds(
                self._plot_data,
                self._visible_epoch_indices(),
                manual_min=self._manual_x_min,
                manual_max=self._manual_x_max,
            )
            if self._plot_data is not None
            else (-1.0, 1.0),
            value_bounds=resolved_value_bounds(
                self._plot_data,
                self._visible_roi_indices(),
                self._visible_epoch_indices(),
                manual_min=self._manual_y_min,
                manual_max=self._manual_y_max,
            )
            if self._plot_data is not None
            else (-1.0, 1.0),
        )

    def _export_output_dir(self) -> Path:
        """Return the default export folder beside the current recording."""
        if self._recording is None:
            return Path("exports")
        return self._recording.path.parent / "exports"

    def _resolved_analysis_path(self) -> Path:
        """Return the analysis HDF5 path for the selected recording."""
        if self._analysis_path is not None:
            return self._analysis_path
        if self._recording is not None:
            return default_analysis_output_path(self._recording)
        return Path("analysis_outputs.h5")

    def _resolved_roi_save_file(self) -> Path:
        """Return the ROI HDF5 path for the selected recording."""
        if self._roi_save_file is not None:
            return self._roi_save_file
        if self._recording is not None:
            return self._recording.path.parent / "rois.h5"
        return Path("rois.h5")

    def _refresh_update_path_labels(self) -> None:
        """Refresh compact recording and output labels in the Update tab.

        Args:
            None.

        Returns:
            None.
        """
        if self._recording is None:
            self._recording_summary_label.setText("No recording loaded.")
            self._analysis_path_label.setText("Analysis output: default")
            self._roi_save_path_label.setText("ROI output: default")
            return

        summary = recording_display_summary(self._recording)
        self._recording_summary_label.setText("\n\n".join(summary.lines()))
        self._analysis_path_label.setText(
            f"Analysis output: {format_twopy_h5_output(self._resolved_analysis_path())}"
        )
        self._roi_save_path_label.setText(
            f"ROI output: {format_twopy_h5_output(self._resolved_roi_save_file())}"
        )

    def _roi_color_hex(self) -> tuple[str, ...]:
        """Return ROI colors as hex strings in plot ROI order."""
        return tuple(color.name() for color in self._roi_colors)

    def _roi_label_values(self) -> tuple[int, ...]:
        """Return integer Labels values in plot ROI order."""
        return roi_label_values_from_labels(self._roi_labels())

    def _set_roi_visibility(self, index: object, visible: bool) -> None:
        """Update one ROI visibility flag from its displayed row index."""
        row_index = _row_index_or_none(index, len(self._roi_labels()))
        if row_index is not None:
            self._roi_visibility[row_index] = visible
        self._update_visible_plot_widgets()

    def _set_roi_visibility_batch(self, visibility: dict[object, bool]) -> None:
        """Update several ROI visibility flags with one plot refresh."""
        for index, visible in visibility.items():
            row_index = _row_index_or_none(index, len(self._roi_labels()))
            if row_index is not None:
                self._roi_visibility[row_index] = visible
        self._update_visible_plot_widgets()

    def _set_epoch_visibility(self, index: object, visible: bool) -> None:
        """Update one epoch visibility flag from its displayed row index."""
        row_index = _row_index_or_none(index, self._epoch_count())
        if row_index is not None:
            self._epoch_visibility[row_index] = visible
        self._render_plots(apply_roi_layer_visibility=False)

    def _set_epoch_visibility_batch(self, visibility: dict[object, bool]) -> None:
        """Update several epoch visibility flags with one plot refresh."""
        for index, visible in visibility.items():
            row_index = _row_index_or_none(index, self._epoch_count())
            if row_index is not None:
                self._epoch_visibility[row_index] = visible
        self._render_plots(apply_roi_layer_visibility=False)

    def _visible_roi_indices(self) -> tuple[int, ...]:
        """Return zero-based ROI indices currently visible in plots.

        Args:
            None.

        Returns:
            Tuple of ROI indices.
        """
        return tuple(
            index
            for index in range(len(self._roi_labels()))
            if self._roi_visibility.get(index, True)
        )

    def _visible_epoch_indices(self) -> tuple[int, ...]:
        """Return epoch row indices currently visible in plots.

        Args:
            None.

        Returns:
            Tuple of displayed epoch row indices.
        """
        return tuple(
            index
            for index in range(self._epoch_count())
            if self._epoch_visibility.get(index, True)
        )

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

    def _epoch_count(self) -> int:
        """Return the number of epoch rows in the current plot data."""
        if self._plot_data is None:
            return 0
        return len(self._plot_data.epochs)

    def _set_status(self, text: str) -> None:
        """Replace plots with one status message.

        Args:
            text: User-facing status message.

        Returns:
            None.
        """
        self._clear_epoch_plot_cache()
        clear_layout(self._plot_layout)
        self._status_label = QLabel(text)
        self._status_label.setWordWrap(True)
        self._plot_layout.addWidget(self._status_label)
        self._plot_layout.addStretch(1)

    def _render_plots(self, *, apply_roi_layer_visibility: bool = True) -> None:
        """Render one plot widget per epoch type."""
        self._clear_plot_layout_preserving_epoch_cache()
        if self._plot_data is None or len(self._plot_data.epochs) == 0:
            self._set_status("No responses available.")
            return
        if apply_roi_layer_visibility:
            self._apply_roi_layer_visibility()
        roi_indices = self._visible_roi_indices()
        epoch_indices = self._visible_epoch_indices()
        if len(roi_indices) == 0 or len(epoch_indices) == 0:
            self._set_status("No responses selected.")
            return
        time_min, time_max = resolved_time_bounds(
            self._plot_data,
            epoch_indices,
            manual_min=self._manual_x_min,
            manual_max=self._manual_x_max,
        )
        value_min, value_max = resolved_value_bounds(
            self._plot_data,
            roi_indices,
            epoch_indices,
            manual_min=self._manual_y_min,
            manual_max=self._manual_y_max,
        )
        colors = self._roi_colors
        self._ensure_epoch_plot_cache()
        for epoch_index, _epoch in enumerate(self._plot_data.epochs):
            if not self._epoch_visibility.get(epoch_index, True):
                continue
            self._plot_layout.addWidget(self._epoch_plot_panels[epoch_index])
            self._epoch_plot_panels[epoch_index].show()
        self._plot_layout.addStretch(1)
        self._update_epoch_plot_widgets(
            roi_indices=roi_indices,
            epoch_indices=epoch_indices,
            time_min=time_min,
            time_max=time_max,
            value_min=value_min,
            value_max=value_max,
            colors=colors,
        )

    def _update_visible_plot_widgets(
        self,
        *,
        apply_roi_layer_visibility: bool = True,
    ) -> None:
        """Repaint currently shown plot widgets without rebuilding the layout.

        Args:
            apply_roi_layer_visibility: Whether ROI visibility should also dim
                matching Labels-layer values in the napari viewer.

        Returns:
            None.
        """
        if self._plot_data is None or len(self._plot_data.epochs) == 0:
            self._render_plots(apply_roi_layer_visibility=apply_roi_layer_visibility)
            return
        roi_indices = self._visible_roi_indices()
        epoch_indices = self._visible_epoch_indices()
        if len(roi_indices) == 0 or len(epoch_indices) == 0:
            self._render_plots(apply_roi_layer_visibility=apply_roi_layer_visibility)
            return
        if any(index not in self._epoch_plot_widgets for index in epoch_indices):
            self._render_plots(apply_roi_layer_visibility=apply_roi_layer_visibility)
            return
        if apply_roi_layer_visibility:
            self._apply_roi_layer_visibility()
        time_min, time_max = resolved_time_bounds(
            self._plot_data,
            epoch_indices,
            manual_min=self._manual_x_min,
            manual_max=self._manual_x_max,
        )
        value_min, value_max = resolved_value_bounds(
            self._plot_data,
            roi_indices,
            epoch_indices,
            manual_min=self._manual_y_min,
            manual_max=self._manual_y_max,
        )
        self._update_epoch_plot_widgets(
            roi_indices=roi_indices,
            epoch_indices=epoch_indices,
            time_min=time_min,
            time_max=time_max,
            value_min=value_min,
            value_max=value_max,
            colors=self._roi_colors,
        )

    def _update_epoch_plot_widgets(
        self,
        *,
        roi_indices: tuple[int, ...],
        epoch_indices: tuple[int, ...],
        time_min: float,
        time_max: float,
        value_min: float,
        value_max: float,
        colors: tuple[QColor, ...],
    ) -> None:
        """Update cached epoch widgets from already-computed plot data.

        Args:
            roi_indices: ROI rows to draw.
            epoch_indices: Epoch plot rows to update.
            time_min: Shared x-axis minimum.
            time_max: Shared x-axis maximum.
            value_min: Shared y-axis minimum.
            value_max: Shared y-axis maximum.
            colors: ROI plot colors.

        Returns:
            None.
        """
        for epoch_index in epoch_indices:
            plot = self._epoch_plot_widgets[epoch_index]
            plot.update_display(
                show_sem=self._show_sem,
                roi_indices=roi_indices,
                roi_colors=colors,
                time_min=time_min,
                time_max=time_max,
                value_min=value_min,
                value_max=value_max,
                plot_size=self._plot_size,
            )

    def _ensure_epoch_plot_cache(self) -> None:
        """Create one persistent plot widget per loaded epoch."""
        if self._plot_data is None:
            self._clear_epoch_plot_cache()
            return
        expected_indices = set(range(len(self._plot_data.epochs)))
        for epoch_index in tuple(self._epoch_plot_widgets):
            if epoch_index not in expected_indices:
                self._epoch_plot_panels[epoch_index].deleteLater()
                del self._epoch_plot_panels[epoch_index]
                del self._epoch_plot_widgets[epoch_index]
        for epoch_index, epoch in enumerate(self._plot_data.epochs):
            if epoch_index in self._epoch_plot_widgets:
                continue
            plot = EpochPlotWidget(
                epoch,
                show_sem=self._show_sem,
                roi_indices=self._visible_roi_indices(),
                roi_colors=self._roi_colors,
                time_min=0.0,
                time_max=1.0,
                value_min=0.0,
                value_max=1.0,
                plot_size=self._plot_size,
            )
            self._epoch_plot_widgets[epoch_index] = plot
            panel = epoch_plot_panel(
                title=f"Epoch {epoch.epoch_number}: {epoch.epoch_name}",
                plot=plot,
            )
            panel.hide()
            self._epoch_plot_panels[epoch_index] = panel

    def _clear_epoch_plot_cache(self) -> None:
        """Delete cached epoch plot panels and widgets."""
        for panel in self._epoch_plot_panels.values():
            panel.deleteLater()
        self._epoch_plot_widgets.clear()
        self._epoch_plot_panels.clear()

    def _clear_plot_layout_preserving_epoch_cache(self) -> None:
        """Clear the plot strip while keeping cached epoch panels alive."""
        cached_panels = set(self._epoch_plot_panels.values())
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is None:
                continue
            if widget in cached_panels:
                # Qt keeps a widget visible after it is removed from a layout.
                # Hidden epoch plots must be hidden explicitly so stale panels
                # cannot remain on screen after checkbox changes.
                widget.hide()
            else:
                widget.deleteLater()

    def _apply_roi_layer_visibility(self) -> None:
        """Mirror plot ROI visibility onto the napari Labels overlay."""
        apply_roi_visibility_to_labels_layer(
            self._roi_labels_layer,
            roi_labels=self._roi_labels(),
            visibility=self._roi_visibility,
            colors=self._roi_colors,
            keys=tuple(range(len(self._roi_labels()))),
        )


def _row_index_or_none(value: object, row_count: int) -> int | None:
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
