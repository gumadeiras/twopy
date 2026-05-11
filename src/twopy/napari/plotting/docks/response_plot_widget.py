"""Response plotting widgets for the twopy napari adapter.

Inputs: loaded twopy analysis outputs or the current napari Labels ROI layer.
Outputs: Qt widgets showing one response plot per stimulus epoch and a
separate options panel.

The plotting code only owns GUI state and drawing. When the user asks to update
responses from the current Labels layer, this module converts labels to a
``RoiSet`` and calls the normal analysis workflow.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

from qtpy.QtCore import QTimer
from qtpy.QtGui import QCloseEvent, QColor
from qtpy.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.response_processing import (
    NormalizationOptions,
    ResponseProcessingOptions,
)
from twopy.analysis.response_window_options import ResponseWindowOptions
from twopy.analysis.trials import default_baseline_epoch_number
from twopy.analysis_cache import (
    AnalysisSyncPlan,
    AnalysisSyncResult,
    start_analysis_sync,
)
from twopy.converted import RecordingData
from twopy.napari.interactive import LiveResponseController
from twopy.napari.plotting.data import (
    ResponsePlotData,
    filter_response_plot_data_rois,
    load_response_plot_data,
    response_plot_baseline_window_limit_for_recording,
)
from twopy.napari.plotting.docks.dynamic_options import (
    add_plot_display_options_group,
    clear_dynamic_option_tabs,
    render_dynamic_options,
)
from twopy.napari.plotting.docks.options_panel import create_response_options_panel
from twopy.napari.plotting.docks.paths import (
    refresh_update_path_labels,
    resolved_analysis_path,
)
from twopy.napari.plotting.docks.plot_area import ResponsePlotArea
from twopy.napari.plotting.docks.save_actions import save_current_roi_analysis
from twopy.napari.plotting.docks.visibility_state import (
    epoch_visibility,
    roi_label_values,
    roi_labels_from_plot_data,
    row_visibility,
    set_row_visibility,
    set_row_visibility_batch,
    visible_indices,
)
from twopy.napari.plotting.export_controls import ResponseExportState
from twopy.napari.plotting.label_visibility import apply_roi_visibility_to_labels_layer
from twopy.napari.plotting.widgets import (
    clear_layout,
    opaque_colors,
    resolved_time_bounds,
    resolved_value_bounds,
    roi_colors_from_label_values,
)
from twopy.napari.roi import remove_roi_label_values_from_layer
from twopy.stimulus import stimulus_epoch_names_by_number


class _ResponsePlotWidget(QWidget):
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
        self._plot_size = 300
        self._roi_visibility: dict[int, bool] = {}
        self._roi_colors: tuple[QColor, ...] = ()
        self._epoch_visibility: dict[int, bool] = {}
        self._manual_x_min: float | None = None
        self._manual_x_max: float | None = None
        self._manual_y_min: float | None = None
        self._manual_y_max: float | None = None
        self._delta_f_over_f_options = DeltaFOverFOptions()
        self._response_window_options = ResponseWindowOptions()
        self._response_processing_options = ResponseProcessingOptions()
        self._is_shutdown = False
        self._sync_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="twopy-analysis-sync",
        )
        self._sync_future: Future[AnalysisSyncResult] | None = None
        self._sync_timer = QTimer()
        self._sync_timer.setInterval(200)
        self._sync_timer.timeout.connect(self._collect_finished_sync)
        self._live_controller = LiveResponseController(self)
        self._live_controller.set_delta_f_over_f_options(
            self._delta_f_over_f_options,
        )
        self._live_controller.set_response_window_options(
            self._response_window_options,
        )
        self._live_controller.set_response_processing_options(
            self._response_processing_options,
        )
        self._quit_application = QApplication.instance()
        if self._quit_application is not None:
            self._quit_application.aboutToQuit.connect(self.shutdown)
        self._plot_area = ResponsePlotArea("No recording loaded.")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_area.scroll_area)
        self.setLayout(layout)

        options_panel = create_response_options_panel(
            response_processing_options=self._response_processing_options,
            response_window_options=self._response_window_options,
            delta_f_over_f_options=self._delta_f_over_f_options,
            on_response_processing_change=self._set_response_processing_options,
            on_response_window_change=self._set_response_window_options,
            on_delta_f_over_f_change=self._set_delta_f_over_f_options,
            on_normalization_change=self._set_normalization_options,
            on_reload_saved=self.reload,
            on_recompute_preview=self.update_from_current_rois,
            on_save_analysis=self.save_analysis_and_rois,
            export_state=self._export_state,
        )
        self._options_tabs = options_panel.tabs
        self._recording_summary_label = options_panel.recording_summary_label
        self._analysis_path_label = options_panel.analysis_path_label
        self._roi_save_path_label = options_panel.roi_save_path_label
        self._update_status_label = options_panel.update_status_label
        self._plot_options_layout = options_panel.plot_options_layout
        self._plot_display_options_layout = options_panel.plot_display_options_layout
        self._roi_options_layout = options_panel.roi_options_layout
        self._epoch_options_layout = options_panel.epoch_options_layout
        self._processing_options_widget = options_panel.processing_options_widget
        self._response_window_options_widget = (
            options_panel.response_window_options_widget
        )
        self._delta_f_over_f_options_widget = (
            options_panel.delta_f_over_f_options_widget
        )
        self._normalization_options_widget = options_panel.normalization_options_widget
        self._add_default_plot_display_options()

    def options_widget(self) -> object:
        """Return the response-options widget owned by this plot widget.

        Args:
            None.

        Returns:
            Tabbed Qt widget with response plot controls.
        """
        return self._options_tabs

    def shutdown(self) -> None:
        """Release worker, layer, and plot state owned by this widget.

        Args:
            None.

        Returns:
            None.

        The live controller owns a worker thread plus napari layer event
        connections. Closing napari must stop those resources explicitly so they
        cannot keep the Python process or loaded imaging arrays alive.
        """
        if self._is_shutdown:
            return
        self._is_shutdown = True
        if self._quit_application is not None:
            with suppress(RuntimeError, TypeError):
                self._quit_application.aboutToQuit.disconnect(self.shutdown)
            self._quit_application = None
        self._sync_timer.stop()
        if self._sync_future is not None:
            self._sync_future.cancel()
            self._sync_future = None
        self._sync_executor.shutdown(wait=True, cancel_futures=True)
        self._live_controller.shutdown()
        self._recording = None
        self._roi_labels_layer = None
        self._plot_data = None
        self._viewer = None
        self._plot_area.clear_epoch_cache()
        clear_dynamic_option_tabs(
            plot_display_options_layout=self._plot_display_options_layout,
            roi_options_layout=self._roi_options_layout,
            epoch_options_layout=self._epoch_options_layout,
        )
        self._plot_area.clear()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Shut down background work before Qt closes the widget.

        Args:
            a0: Qt close event supplied by the application.

        Returns:
            None.
        """
        self.shutdown()
        if a0 is not None:
            super().closeEvent(a0)

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
        self._delta_f_over_f_options_widget.set_epoch_choices(
            {},
            selected_epoch_number=self._delta_f_over_f_options.baseline_epoch_number,
        )
        self._normalization_options_widget.set_epoch_choices(
            {},
            selected_epoch_number=(
                self._response_processing_options.normalization.epoch_number
            ),
        )
        self._response_window_options_widget.set_max_window_seconds(None)
        self._reset_plot_state()
        self._recording_summary_label.setText("No recording loaded.")
        self._analysis_path_label.setText("Analysis output: default")
        self._roi_save_path_label.setText("ROI output: default")
        self._update_status_label.setText("")
        self._reset_empty_option_tabs()
        self._set_status("No recording loaded.")

    def load_recording(self, recording: RecordingData) -> None:
        """Load response plots for one recording.

        Args:
            recording: Loaded converted recording.

        Returns:
            None.
        """
        self._recording = recording
        self._analysis_path = resolved_analysis_path(None, recording)
        epoch_names = stimulus_epoch_names_by_number(recording)
        self._delta_f_over_f_options_widget.set_epoch_choices(
            epoch_names,
            selected_epoch_number=default_baseline_epoch_number(epoch_names),
        )
        self._normalization_options_widget.set_epoch_choices(
            epoch_names,
            selected_epoch_number=(
                self._response_processing_options.normalization.epoch_number
            ),
        )
        self._response_window_options_widget.set_max_window_seconds(
            response_plot_baseline_window_limit_for_recording(recording),
        )
        self._delta_f_over_f_options = self._delta_f_over_f_options_widget.options()
        self._live_controller.set_delta_f_over_f_options(
            self._delta_f_over_f_options,
        )
        self._response_window_options = self._response_window_options_widget.options()
        self._live_controller.set_response_window_options(
            self._response_window_options,
        )
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
        try:
            result = save_current_roi_analysis(
                recording=self._recording,
                roi_labels_layer=self._roi_labels_layer,
                roi_save_file=self._roi_save_file,
                analysis_path=self._analysis_path,
                delta_f_over_f_options=self._delta_f_over_f_options,
                response_window_options=self._response_window_options,
                response_processing_options=self._response_processing_options,
            )
        except ValueError as error:
            self._set_status(str(error))
            return

        self._roi_save_file = result.roi_output_path
        self._analysis_path = result.analysis_output_path
        self._refresh_update_path_labels()
        self._update_status_label.setText(result.status_text)
        if result.sync_plan is not None:
            self._start_sync(result.sync_plan)

    def _start_sync(self, sync_plan: AnalysisSyncPlan) -> None:
        """Start background sync for just-saved analysis outputs.

        Args:
            sync_plan: Local files and publish root to copy in the background.

        Returns:
            None.
        """
        if self._is_shutdown:
            return
        if self._sync_future is not None and not self._sync_future.done():
            self._sync_future.cancel()
        self._sync_future = start_analysis_sync(
            sync_plan,
            executor=self._sync_executor,
        )
        self._sync_timer.start()

    def _collect_finished_sync(self) -> None:
        """Update the UI when a background analysis sync finishes."""
        future = self._sync_future
        if future is None or not future.done():
            return
        self._sync_future = None
        self._sync_timer.stop()
        try:
            result = future.result()
        except Exception as error:
            self._update_status_label.setText(f"Saved locally; sync failed: {error}")
            return
        if len(result.copied_paths) == 0:
            self._update_status_label.setText("Saved locally; sync already current")
            return
        copied_count = len(result.copied_paths)
        self._update_status_label.setText(
            f"Saved locally; synced {copied_count} files to {result.publish_root}",
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
        self._plot_area.clear_epoch_cache()
        self._plot_data = plot_data
        if plot_data.response_processing_options is not None:
            self._load_response_processing_options(
                plot_data.response_processing_options,
            )
        if plot_data.delta_f_over_f_options is not None:
            self._load_delta_f_over_f_options(plot_data.delta_f_over_f_options)
        if plot_data.response_window_options is not None:
            self._load_response_window_options(plot_data.response_window_options)
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
        self._reset_plot_state()
        self._reset_empty_option_tabs()
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
            self._reset_empty_option_tabs()
            self._set_status(result)
            return
        self.set_response_plot_data(result, reset_axes=True)

    def _set_show_sem(self, state: int) -> None:
        self._show_sem = state != 0
        self._update_visible_plot_widgets(apply_roi_layer_visibility=False)

    def _set_plot_size(self, value: int) -> None:
        self._plot_size = int(value)
        self._update_visible_plot_widgets(apply_roi_layer_visibility=False)

    def _set_response_processing_options(
        self,
        options: ResponseProcessingOptions,
    ) -> None:
        self._response_processing_options = options
        self._live_controller.set_response_processing_options(options)
        if self._recording is not None and self._roi_labels_layer is not None:
            self._live_controller.request_update()
            self._update_status_label.setText("Processing settings updated.")
            return
        self._update_status_label.setText("Processing settings updated for next run.")

    def _set_normalization_options(self, options: NormalizationOptions) -> None:
        self._processing_options_widget.set_normalization_options(options)
        self._set_response_processing_options(
            replace(self._response_processing_options, normalization=options),
        )
        if options.method == "epoch_peak":
            self._update_status_label.setText("Normalization settings updated.")
            return
        self._update_status_label.setText("Normalization disabled.")

    def _set_delta_f_over_f_options(self, options: DeltaFOverFOptions) -> None:
        self._delta_f_over_f_options = options
        self._live_controller.set_delta_f_over_f_options(options)
        if self._recording is not None and self._roi_labels_layer is not None:
            self._live_controller.request_update()
            self._update_status_label.setText("dF/F settings updated.")
            return
        self._update_status_label.setText("dF/F settings updated for next run.")

    def _set_response_window_options(self, options: ResponseWindowOptions) -> None:
        self._response_window_options = options
        self._live_controller.set_response_window_options(options)
        if self._recording is not None and self._roi_labels_layer is not None:
            self._live_controller.request_update()
            self._update_status_label.setText("Response window updated.")
            return
        self._update_status_label.setText("Response window updated for next run.")

    def _load_response_processing_options(
        self,
        options: ResponseProcessingOptions,
    ) -> None:
        self._response_processing_options = options
        self._processing_options_widget.set_options(options)
        self._normalization_options_widget.set_options(options.normalization)
        self._processing_options_widget.set_normalization_options(
            options.normalization,
        )
        self._live_controller.set_response_processing_options(options)

    def _load_delta_f_over_f_options(self, options: DeltaFOverFOptions) -> None:
        self._delta_f_over_f_options = options
        self._delta_f_over_f_options_widget.set_options(options)
        self._live_controller.set_delta_f_over_f_options(options)

    def _load_response_window_options(self, options: ResponseWindowOptions) -> None:
        self._response_window_options_widget.set_options(options)
        self._response_window_options = self._response_window_options_widget.options()
        self._live_controller.set_response_window_options(self._response_window_options)

    def _sync_plot_state(self, *, reset_axes: bool) -> None:
        if self._plot_data is None:
            self._reset_plot_state()
            return

        roi_labels = self._roi_labels()
        self._roi_visibility = row_visibility(self._roi_visibility, len(roi_labels))
        self._roi_colors = opaque_colors(
            roi_colors_from_label_values(
                self._roi_labels_layer,
                roi_label_values(self._plot_data),
                fallback_count=len(roi_labels),
            )
        )
        self._epoch_visibility = epoch_visibility(
            self._epoch_visibility,
            self._plot_data.epochs,
        )
        if reset_axes:
            self._manual_x_min = None
            self._manual_x_max = None
            self._manual_y_min = None
            self._manual_y_max = None
        self._render_options()

    def _reset_plot_state(self) -> None:
        self._roi_visibility = {}
        self._roi_colors = ()
        self._epoch_visibility = {}
        self._manual_x_min = None
        self._manual_x_max = None
        self._manual_y_min = None
        self._manual_y_max = None

    def _reset_empty_option_tabs(self) -> None:
        clear_layout(self._plot_display_options_layout)
        self._add_default_plot_display_options()
        clear_layout(self._roi_options_layout)
        clear_layout(self._epoch_options_layout)

    def _add_default_plot_display_options(self) -> None:
        add_plot_display_options_group(
            plot_display_options_layout=self._plot_display_options_layout,
            show_sem=self._show_sem,
            plot_size=self._plot_size,
            x_min=0.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            on_show_sem_change=self._set_show_sem,
            on_plot_size_change=self._set_plot_size,
            on_axis_change=self._set_manual_axis_bounds,
        )

    def _render_options(self) -> None:
        render_dynamic_options(
            plot_data=self._plot_data,
            plot_display_options_layout=self._plot_display_options_layout,
            roi_options_layout=self._roi_options_layout,
            epoch_options_layout=self._epoch_options_layout,
            show_sem=self._show_sem,
            plot_size=self._plot_size,
            manual_x_min=self._manual_x_min,
            manual_x_max=self._manual_x_max,
            manual_y_min=self._manual_y_min,
            manual_y_max=self._manual_y_max,
            visible_roi_indices=self._visible_roi_indices(),
            visible_epoch_indices=self._visible_epoch_indices(),
            roi_labels=self._roi_labels(),
            roi_visibility=self._roi_visibility,
            roi_colors=self._roi_colors,
            epoch_visibility=self._epoch_visibility,
            on_show_sem_change=self._set_show_sem,
            on_plot_size_change=self._set_plot_size,
            on_axis_change=self._set_manual_axis_bounds,
            on_roi_visibility_change=self._set_roi_visibility,
            on_roi_visibility_batch=self._set_roi_visibility_batch,
            on_remove_selected_rois=self.remove_selected_rois,
            on_epoch_visibility_change=self._set_epoch_visibility,
            on_epoch_visibility_batch=self._set_epoch_visibility_batch,
        )

    def remove_selected_rois(self) -> None:
        """Delete checked ROI labels from the active Labels layer.

        Args:
            None.

        Returns:
            None.

        The ROI checkboxes already define which ROIs are selected for plotting.
        This action reuses that state and edits only the Labels layer; response
        recomputation remains owned by the live controller.
        """
        label_values = self._roi_label_values()
        selected_label_values = tuple(
            label_values[index]
            for index in self._visible_roi_indices()
            if index < len(label_values)
        )
        removed_count = remove_roi_label_values_from_layer(
            self._roi_labels_layer,
            selected_label_values,
        )
        if removed_count == 0:
            self._update_status_label.setText("No selected ROIs to remove.")
            return
        self._remove_rois_from_current_plot_data(selected_label_values)
        noun = "ROI" if removed_count == 1 else "ROIs"
        self._update_status_label.setText(f"Removed {removed_count} selected {noun}.")
        self._live_controller.request_update()

    def _remove_rois_from_current_plot_data(
        self,
        removed_label_values: tuple[int, ...],
    ) -> None:
        """Remove deleted ROI rows from the displayed response state.

        Args:
            removed_label_values: Napari Labels values that were just erased.

        Returns:
            None.
        """
        if self._plot_data is None:
            return
        removed_values = set(removed_label_values)
        keep_indices = tuple(
            index
            for index, label_value in enumerate(self._roi_label_values())
            if label_value not in removed_values
        )
        self._plot_data = filter_response_plot_data_rois(
            self._plot_data,
            keep_indices,
        )
        self._sync_plot_state(reset_axes=False)
        self._render_plots()

    def _set_manual_axis_bounds(
        self,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        self._manual_x_min = x_min
        self._manual_x_max = x_max
        self._manual_y_min = y_min
        self._manual_y_max = y_max
        self._update_visible_plot_widgets(apply_roi_layer_visibility=False)

    def _export_state(self) -> ResponseExportState:
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
        if self._recording is None:
            return Path("exports")
        return self._recording.path.parent / "exports"

    def _refresh_update_path_labels(self) -> None:
        refresh_update_path_labels(
            recording=self._recording,
            analysis_path=self._analysis_path,
            roi_save_file=self._roi_save_file,
            recording_summary_label=self._recording_summary_label,
            analysis_path_label=self._analysis_path_label,
            roi_save_path_label=self._roi_save_path_label,
        )

    def _roi_color_hex(self) -> tuple[str, ...]:
        return tuple(color.name() for color in self._roi_colors)

    def _roi_label_values(self) -> tuple[int, ...]:
        return roi_label_values(self._plot_data)

    def _set_roi_visibility(self, index: object, visible: bool) -> None:
        set_row_visibility(
            self._roi_visibility,
            index,
            visible,
            row_count=len(self._roi_labels()),
        )
        self._update_visible_plot_widgets()

    def _set_roi_visibility_batch(self, visibility: dict[object, bool]) -> None:
        set_row_visibility_batch(
            self._roi_visibility,
            visibility,
            row_count=len(self._roi_labels()),
        )
        self._update_visible_plot_widgets()

    def _set_epoch_visibility(self, index: object, visible: bool) -> None:
        set_row_visibility(
            self._epoch_visibility,
            index,
            visible,
            row_count=self._epoch_count(),
        )
        self._render_plots(apply_roi_layer_visibility=False)

    def _set_epoch_visibility_batch(self, visibility: dict[object, bool]) -> None:
        set_row_visibility_batch(
            self._epoch_visibility,
            visibility,
            row_count=self._epoch_count(),
        )
        self._render_plots(apply_roi_layer_visibility=False)

    def _visible_roi_indices(self) -> tuple[int, ...]:
        return visible_indices(self._roi_visibility, len(self._roi_labels()))

    def _visible_epoch_indices(self) -> tuple[int, ...]:
        return visible_indices(self._epoch_visibility, self._epoch_count())

    def _roi_labels(self) -> tuple[str, ...]:
        return roi_labels_from_plot_data(self._plot_data)

    def _epoch_count(self) -> int:
        if self._plot_data is None:
            return 0
        return len(self._plot_data.epochs)

    def _set_status(self, text: str) -> None:
        self._plot_area.set_status(text)

    def _render_plots(self, *, apply_roi_layer_visibility: bool = True) -> None:
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
        self._plot_area.render(
            plot_data=self._plot_data,
            roi_indices=roi_indices,
            epoch_indices=epoch_indices,
            show_sem=self._show_sem,
            roi_colors=self._roi_colors,
            time_min=time_min,
            time_max=time_max,
            value_min=value_min,
            value_max=value_max,
            plot_size=self._plot_size,
        )

    def _update_visible_plot_widgets(
        self,
        *,
        apply_roi_layer_visibility: bool = True,
    ) -> None:
        if self._plot_data is None or len(self._plot_data.epochs) == 0:
            self._render_plots(apply_roi_layer_visibility=apply_roi_layer_visibility)
            return
        roi_indices = self._visible_roi_indices()
        epoch_indices = self._visible_epoch_indices()
        if len(roi_indices) == 0 or len(epoch_indices) == 0:
            self._render_plots(apply_roi_layer_visibility=apply_roi_layer_visibility)
            return
        if not self._plot_area.has_epoch_widgets(epoch_indices):
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
        self._plot_area.update_epoch_plot_widgets(
            roi_indices=roi_indices,
            epoch_indices=epoch_indices,
            show_sem=self._show_sem,
            roi_colors=self._roi_colors,
            time_min=time_min,
            time_max=time_max,
            value_min=value_min,
            value_max=value_max,
            plot_size=self._plot_size,
        )

    def _apply_roi_layer_visibility(self) -> None:
        apply_roi_visibility_to_labels_layer(
            self._roi_labels_layer,
            roi_labels=self._roi_labels(),
            visibility=self._roi_visibility,
            colors=self._roi_colors,
            keys=tuple(range(len(self._roi_labels()))),
        )
