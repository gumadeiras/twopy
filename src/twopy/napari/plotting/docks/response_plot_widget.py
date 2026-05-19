"""Response plotting widgets for the twopy napari adapter.

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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.response_context import default_recording_baseline_epoch_number
from twopy.analysis.response_maps import (
    ResponseMapData,
    ResponseMapOptions,
    compute_recording_response_maps,
)
from twopy.analysis.response_processing import (
    NormalizationOptions,
    ResponseProcessingOptions,
)
from twopy.analysis.response_window_options import ResponseWindowOptions
from twopy.analysis_cache import (
    AnalysisSyncPlan,
    AnalysisSyncResult,
    build_analysis_sync_plan,
    start_analysis_sync,
)
from twopy.config import load_config
from twopy.converted import RecordingData
from twopy.custom import (
    CustomParameterSpec,
    CustomResult,
    CustomRunContext,
    CustomWorkflow,
    custom_result_artifact_paths,
    native_custom_workflow_paths,
    parameter_specs,
    parameter_values,
    validate_custom_result,
    workflow_provenance,
    write_result_provenance,
)
from twopy.filenames import EXPORTS_DIRNAME
from twopy.napari.custom_tab import CustomWorkflowPanel
from twopy.napari.interactive import LiveResponseController
from twopy.napari.latest_worker import LatestWorker
from twopy.napari.paths import DEFAULT_PATH_TEXT
from twopy.napari.plotting.data import (
    ResponsePlotData,
    filter_response_plot_data_rois,
    load_response_plot_data,
    response_plot_baseline_window_limit_for_recording,
    response_plot_min_epoch_duration_for_recording,
    response_plot_window_seconds_for_recording,
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
    resolved_roi_save_file,
)
from twopy.napari.plotting.docks.plot_area import ResponsePlotArea
from twopy.napari.plotting.docks.response_map_area import ResponseMapArea
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
from twopy.napari.plotting.roi_generation import (
    RoiGenerationOptions,
    generate_roi_labels,
)
from twopy.napari.plotting.widgets import (
    clear_layout,
    opaque_colors,
    resolved_time_bounds,
    resolved_value_bounds,
    roi_colors_from_label_values,
)
from twopy.napari.responses import (
    ResponseAnalysisRequest,
    response_analysis_request_from_labels,
)
from twopy.napari.roi import (
    load_roi_file_on_layer,
    remove_roi_label_values_from_layer,
    set_roi_label_image_on_layer,
)
from twopy.pixel_calibration import (
    DEFAULT_PIXEL_CALIBRATION_PATH,
    PixelCalibrationRow,
    load_pixel_calibrations,
)
from twopy.pixel_calibration_profiles import (
    DEFAULT_PIXEL_CALIBRATION_PROFILE_PATH,
    PixelCalibrationProfileMapping,
    load_pixel_calibration_profile_mappings,
)
from twopy.roi import RoiSet, roi_set_to_label_image
from twopy.stimulus import stimulus_epoch_names_by_number

_CUSTOM_EPOCH_WINDOW_MIN_SECONDS = 0.0
_CUSTOM_EPOCH_WINDOW_STEP_SECONDS = 0.1
_CUSTOM_RESPONSE_WINDOW_STEP_SECONDS = 0.1
_CUSTOM_RESPONSE_METRIC_CHOICES = ("mean", "peak", "minimum")
_CUSTOM_ROI_SELECTOR_CHOICES = ("all_rois", "visible_rois")
_CUSTOM_EPOCH_WINDOW_ROLES = ("epoch_window_start", "epoch_window_stop")
_CUSTOM_RESPONSE_WINDOW_ROLES = ("response_window_start", "response_window_stop")


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
        self._response_map_data: ResponseMapData | None = None
        self._show_sem = True
        self._plot_size = 300
        self._roi_visibility: dict[int, bool] = {}
        self._correlation_roi_visibility: dict[int, bool] | None = None
        self._roi_colors: tuple[QColor, ...] = ()
        self._epoch_visibility: dict[int, bool] = {}
        self._manual_x_min: float | None = None
        self._manual_x_max: float | None = None
        self._manual_y_min: float | None = None
        self._manual_y_max: float | None = None
        self._delta_f_over_f_options = DeltaFOverFOptions()
        self._response_window_options = ResponseWindowOptions()
        self._response_map_options = ResponseMapOptions()
        self._response_map_shared_limits = True
        self._response_processing_options = ResponseProcessingOptions()
        self._is_shutdown = False
        self._sync_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="twopy-analysis-sync",
        )
        self._sync_future: Future[AnalysisSyncResult] | None = None
        self._response_map_worker = LatestWorker[ResponseMapData](
            thread_name_prefix="twopy-response-heatmaps",
            on_result=self._apply_response_map_data,
            on_value_error=self._show_response_map_value_error,
            on_exception=self._show_response_map_exception,
        )
        self._pixel_calibrations = _load_pixel_calibrations_for_ui()
        self._pixel_calibration_profile_mappings = (
            _load_pixel_calibration_profile_mappings_for_ui()
        )
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
        self._response_map_area = ResponseMapArea("No recording loaded.")
        self._plot_tabs = QTabWidget()
        self._plot_tabs.addTab(self._plot_area.scroll_area, "Responses")
        self._plot_tabs.addTab(self._response_map_area.scroll_area, "Heatmaps")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_tabs)
        self.setLayout(layout)

        options_panel = create_response_options_panel(
            response_processing_options=self._response_processing_options,
            response_window_options=self._response_window_options,
            response_map_options=self._response_map_options,
            delta_f_over_f_options=self._delta_f_over_f_options,
            on_response_processing_change=self._set_response_processing_options,
            on_response_window_change=self._set_response_window_options,
            on_response_map_change=self._set_response_map_options,
            on_response_map_shared_limits_change=(self._set_response_map_shared_limits),
            on_delta_f_over_f_change=self._set_delta_f_over_f_options,
            on_normalization_change=self._set_normalization_options,
            on_reload_saved=self.reload,
            on_save_analysis=self.save_analysis_and_rois,
            on_create_generated_rois=self.create_generated_rois,
            export_state=self._export_state,
            pixel_calibrations=self._pixel_calibrations,
            pixel_calibration_profile_mappings=self._pixel_calibration_profile_mappings,
        )
        self._options_tabs = options_panel.tabs
        self._recording_summary_label = options_panel.recording_summary_label
        self._microscope_summary_label = options_panel.microscope_summary_label
        self._analysis_path_label = options_panel.analysis_path_label
        self._roi_save_path_label = options_panel.roi_save_path_label
        self._update_status_label = options_panel.update_status_label
        self._reload_saved_button = options_panel.reload_saved_button
        self._plot_options_layout = options_panel.plot_options_layout
        self._plot_display_options_layout = options_panel.plot_display_options_layout
        self._roi_options_layout = options_panel.roi_options_layout
        self._roi_generation_widget = options_panel.roi_generation_widget
        self._epoch_options_layout = options_panel.epoch_options_layout
        self._processing_options_widget = options_panel.processing_options_widget
        self._response_window_options_widget = (
            options_panel.response_window_options_widget
        )
        self._response_map_options_widget = options_panel.response_map_options_widget
        self._delta_f_over_f_options_widget = (
            options_panel.delta_f_over_f_options_widget
        )
        self._normalization_options_widget = options_panel.normalization_options_widget
        self._custom_workflow_panel = CustomWorkflowPanel(
            workflow_paths=_load_custom_workflow_paths_for_ui(),
            on_run=self._run_custom_workflow,
            parameter_specs_for_workflow=self._custom_parameter_specs_for_workflow,
        )
        self._options_tabs.insertTab(
            max(0, self._options_tabs.count() - 1),
            self._custom_workflow_panel,
            "Custom",
        )
        self._add_default_plot_display_options()

    def options_widget(self) -> object:
        """Return the response-options widget owned by this plot widget.

        Args:
            None.

        Returns:
            Tabbed Qt widget with response plot controls.
        """
        return self._options_tabs

    def load_tab_button(self) -> object:
        """Return the response action button that belongs on the Load tab.

        Args:
            None.

        Returns:
            Qt button that reloads persisted analysis outputs.
        """
        return self._reload_saved_button

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
        self._response_map_worker.shutdown()
        self._live_controller.shutdown()
        self._recording = None
        self._roi_labels_layer = None
        self._plot_data = None
        self._response_map_data = None
        self._viewer = None
        self._plot_area.clear_epoch_cache()
        self._response_map_area.clear_epoch_cache()
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
        self._roi_generation_widget.set_recording(None)
        self._analysis_path = None
        self._plot_data = None
        self._response_map_data = None
        self._invalidate_response_map_jobs()
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
        self._response_map_options_widget.set_spatial_shape(None)
        self._processing_options_widget.set_correlation_window_stop_default(None)
        self._custom_workflow_panel.refresh_parameters()
        self._reset_plot_state()
        self._recording_summary_label.setText("No recording loaded.")
        self._microscope_summary_label.setText("No microscope metadata.")
        self._analysis_path_label.setText(f"Analysis output: {DEFAULT_PATH_TEXT}")
        self._roi_save_path_label.setText(f"ROI output: {DEFAULT_PATH_TEXT}")
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
        self._recording = None
        self._response_map_data = None
        self._invalidate_response_map_jobs()
        self._response_map_options_widget.set_spatial_shape(
            recording.alignment_valid_crop.shape,
        )
        self._recording = recording
        self._roi_generation_widget.set_recording(recording)
        self._analysis_path = resolved_analysis_path(None, recording)
        epoch_names = stimulus_epoch_names_by_number(recording)
        self._delta_f_over_f_options_widget.set_epoch_choices(
            epoch_names,
            selected_epoch_number=default_recording_baseline_epoch_number(recording),
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
        self._processing_options_widget.set_correlation_window_stop_default(
            response_plot_min_epoch_duration_for_recording(recording),
        )
        self._custom_workflow_panel.refresh_parameters()
        self._delta_f_over_f_options = self._delta_f_over_f_options_widget.options()
        self._live_controller.set_delta_f_over_f_options(
            self._delta_f_over_f_options,
        )
        self._response_window_options = self._response_window_options_widget.options()
        self._live_controller.set_response_window_options(
            self._response_window_options,
        )
        self._refresh_update_path_labels()
        self._refresh_response_maps()
        self._render_response_maps()
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

    def create_generated_rois(self, options: RoiGenerationOptions) -> None:
        """Replace the active Labels layer with generated ROIs.

        Args:
            options: ROI generation options from the ROIs tab.

        Returns:
            None.

        The generated labels remain editable napari ROI labels. Saving and live
        response recomputation use the same paths as manually drawn ROIs.
        """
        if self._recording is None:
            self._set_roi_generation_status("Load a recording before creating ROIs.")
            return
        if self._roi_labels_layer is None:
            self._set_roi_generation_status("No ROI Labels layer is available.")
            return

        try:
            generated = generate_roi_labels(
                self._recording,
                options,
                self._pixel_calibrations,
            )
            set_roi_label_image_on_layer(
                self._roi_labels_layer,
                self._recording,
                generated.label_image,
            )
        except ValueError as error:
            self._set_roi_generation_status(str(error))
            return

        self._set_roi_generation_status(generated.status_text)
        self._update_status_label.setText(generated.status_text)
        self._live_controller.request_update()

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
            request = self._response_analysis_request()
            result = save_current_roi_analysis(
                request,
                roi_save_file=self._roi_save_file,
                analysis_path=self._analysis_path,
                response_map_data=self._response_map_data,
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

    def _response_analysis_request(self) -> ResponseAnalysisRequest:
        """Return the current ROI response-analysis request."""
        return response_analysis_request_from_labels(
            self._recording,
            self._roi_labels_layer,
            delta_f_over_f_options=self._delta_f_over_f_options,
            response_window_options=self._response_window_options,
            response_processing_options=self._response_processing_options,
        )

    def _run_custom_workflow(
        self,
        workflow: CustomWorkflow,
        params: object | None,
    ) -> CustomResult:
        """Run a custom workflow on the current recording.

        Args:
            workflow: Workflow selected in the Custom tab.
            params: Parameter values from the Custom tab.

        Returns:
            Result after validation and output sync setup.
        """
        if self._recording is None:
            msg = "Load a recording before running a custom workflow."
            raise ValueError(msg)
        roi_set = self._current_roi_set_for_custom_workflow(workflow)
        pre_window_seconds, post_window_seconds = (
            response_plot_window_seconds_for_recording(
                self._recording,
                self._response_window_options,
            )
        )
        parameters = parameter_values(params)
        provenance = workflow_provenance(
            workflow,
            parameters=parameters,
            recording=self._recording,
        )
        output_dir = _custom_workflow_output_dir(self._recording, workflow)
        context = CustomRunContext(
            recording=self._recording,
            roi_set=roi_set,
            output_dir=output_dir,
            delta_f_over_f_options=self._delta_f_over_f_options,
            response_processing_options=self._response_processing_options,
            response_pre_window_seconds=pre_window_seconds,
            response_post_window_seconds=post_window_seconds,
            provenance=provenance,
            visible_roi_indices=self._custom_workflow_visible_roi_indices(roi_set),
        )
        if params is None:
            raw_result = workflow.function(context)
        else:
            raw_result = workflow.function(context, params)
        result = validate_custom_result(
            raw_result,
            output_dir=output_dir,
            expected_roi_shape=self._recording.movie.shape[1:],
        )
        provenance_paths = write_result_provenance(result, provenance)
        sync_plan = build_analysis_sync_plan(
            recording=self._recording,
            local_paths=(
                *custom_result_artifact_paths(result),
                *provenance_paths,
            ),
        )
        if sync_plan is not None:
            self._start_sync(sync_plan)
        self._apply_custom_workflow_result(result)
        return _custom_workflow_display_result(result, sync_plan)

    def _custom_parameter_specs_for_workflow(
        self,
        workflow: CustomWorkflow,
    ) -> tuple[CustomParameterSpec, ...]:
        """Return parameter controls adjusted for the loaded recording."""
        if workflow.params_type is None:
            return ()
        specs = parameter_specs(workflow.params_type)
        if self._recording is None:
            return specs
        epoch_choices = _custom_epoch_choices(self._recording)
        metric_stop_seconds = response_plot_min_epoch_duration_for_recording(
            self._recording,
        )
        pre_window_seconds, post_window_seconds = (
            response_plot_window_seconds_for_recording(
                self._recording,
                self._response_window_options,
            )
        )
        response_stop_seconds = None
        if metric_stop_seconds is not None:
            response_stop_seconds = metric_stop_seconds + post_window_seconds
        return tuple(
            _recording_parameter_spec(
                spec,
                recording=self._recording,
                epoch_choices=epoch_choices,
                metric_stop_seconds=metric_stop_seconds,
                response_start_seconds=-pre_window_seconds,
                response_stop_seconds=response_stop_seconds,
            )
            for spec in specs
        )

    def _custom_workflow_visible_roi_indices(
        self,
        roi_set: RoiSet | None,
    ) -> tuple[int, ...]:
        """Return ROI visibility for custom workflows."""
        if roi_set is None:
            return ()
        if self._plot_data is None:
            return tuple(range(len(roi_set.labels)))
        return self._visible_roi_indices()

    def _current_roi_set_for_custom_workflow(
        self,
        workflow: CustomWorkflow,
    ) -> RoiSet | None:
        """Return current ROIs, or raise only when the workflow needs them."""
        try:
            return self._response_analysis_request().roi_set
        except ValueError:
            if workflow.requires_rois:
                raise
            return None

    def _apply_custom_workflow_result(self, result: CustomResult) -> None:
        """Apply returned ROIs or responses to napari."""
        if result.roi_set is not None:
            if self._roi_labels_layer is None or self._recording is None:
                msg = "Custom workflow returned ROIs, but no Labels layer is available."
                raise ValueError(msg)
            set_roi_label_image_on_layer(
                self._roi_labels_layer,
                self._recording,
                roi_set_to_label_image(result.roi_set),
            )
            self._live_controller.request_update()
        if result.response_plot_data is not None:
            self.set_response_plot_data(result.response_plot_data, reset_axes=True)

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
        self._plot_data = plot_data
        if plot_data.response_processing_options is not None:
            self._load_response_processing_options(
                plot_data.response_processing_options,
            )
        if plot_data.delta_f_over_f_options is not None:
            self._load_delta_f_over_f_options(plot_data.delta_f_over_f_options)
        if plot_data.response_window_options is not None:
            self._load_response_window_options(plot_data.response_window_options)
        self._processing_options_widget.set_correlation_window_stop_default(
            plot_data.correlation_window_stop_default_seconds,
        )
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
        self._plot_area.set_status(text)
        self._render_response_maps()

    def reload(self) -> None:
        """Reload response plots and saved ROIs from disk."""
        if self._analysis_path is None:
            self._set_status("No recording loaded.")
            return
        result = load_response_plot_data(self._analysis_path)
        if isinstance(result, str):
            self._plot_data = None
            self._reset_plot_state()
            self._reset_empty_option_tabs()
            self._plot_area.set_status(result)
            self._render_response_maps()
            return
        roi_error = self._reload_saved_rois()
        self.set_response_plot_data(result, reset_axes=True)
        if roi_error is not None:
            self._update_status_label.setText(f"Reloaded saved analysis; {roi_error}")

    def _reload_saved_rois(self) -> str | None:
        """Replace the active Labels layer with the saved ROI HDF5 file."""
        if self._recording is None or self._roi_labels_layer is None:
            return "no ROI Labels layer is available."
        roi_path = resolved_roi_save_file(self._roi_save_file, self._recording)
        try:
            self._live_controller.set_context(self._recording, None)
            load_roi_file_on_layer(roi_path, self._roi_labels_layer, self._recording)
        except (FileNotFoundError, OSError, ValueError) as error:
            return f"ROI reload failed: {error}"
        finally:
            self._live_controller.set_context(self._recording, self._roi_labels_layer)
        self._roi_save_file = roi_path
        self._refresh_update_path_labels()
        return None

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

    def _set_response_map_options(self, options: ResponseMapOptions) -> None:
        self._response_map_options = options
        self._schedule_response_map_refresh()
        self._update_status_label.setText("Heatmap settings updated; recomputing.")

    def _set_response_map_shared_limits(self, shared_limits: bool) -> None:
        self._response_map_shared_limits = shared_limits
        self._render_response_maps()
        self._update_status_label.setText("Heatmap display limits updated.")

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
        self._sync_roi_visibility_from_plot_data(roi_labels)
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
        self._correlation_roi_visibility = None
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

    def _set_roi_generation_status(self, text: str) -> None:
        """Show ROI-generation status in the ROIs tab.

        Args:
            text: User-facing status.

        Returns:
            None.
        """
        self._roi_generation_widget.set_status(text)

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
            response_map_data=self._response_map_data,
            response_map_shared_limits=self._response_map_shared_limits,
            output_dir=self._export_output_dir(),
            roi_label_values=self._roi_label_values(),
            roi_colors=self._roi_color_hex(),
            epoch_indices=self._visible_epoch_indices(),
            response_map_epoch_indices=self._visible_response_map_epoch_indices(),
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
            return Path(EXPORTS_DIRNAME)
        return self._recording.path.parent / EXPORTS_DIRNAME

    def _refresh_update_path_labels(self) -> None:
        refresh_update_path_labels(
            recording=self._recording,
            analysis_path=self._analysis_path,
            roi_save_file=self._roi_save_file,
            recording_summary_label=self._recording_summary_label,
            microscope_summary_label=self._microscope_summary_label,
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

    def _visible_response_map_epoch_indices(self) -> tuple[int, ...]:
        """Return heatmap row indices matching visible response-plot epochs."""
        if self._response_map_data is None:
            return ()
        if self._plot_data is None:
            return visible_indices(
                self._epoch_visibility,
                len(self._response_map_data.epochs),
            )
        visible_plot_epochs = tuple(
            self._plot_data.epochs[index]
            for index in self._visible_epoch_indices()
            if 0 <= index < len(self._plot_data.epochs)
        )
        map_indices_by_epoch = {
            (epoch.epoch_number, epoch.epoch_name): index
            for index, epoch in enumerate(self._response_map_data.epochs)
        }
        indices: list[int] = []
        seen_indices: set[int] = set()
        for epoch in visible_plot_epochs:
            index = map_indices_by_epoch.get((epoch.epoch_number, epoch.epoch_name))
            if index is None or index in seen_indices:
                continue
            indices.append(index)
            seen_indices.add(index)
        return tuple(indices)

    def _roi_labels(self) -> tuple[str, ...]:
        return roi_labels_from_plot_data(self._plot_data)

    def _sync_roi_visibility_from_plot_data(self, roi_labels: tuple[str, ...]) -> None:
        """Update ROI visibility from correlation QC or existing user choices."""
        if (
            self._plot_data is not None
            and self._plot_data.visible_roi_indices is not None
        ):
            visible_indices_set = set(self._plot_data.visible_roi_indices)
            self._roi_visibility = {
                index: index in visible_indices_set for index in range(len(roi_labels))
            }
            self._correlation_roi_visibility = None
            return
        correlation_visibility = _correlation_roi_visibility(
            self._plot_data,
            roi_labels,
        )
        if correlation_visibility is not None:
            self._roi_visibility = correlation_visibility
            self._correlation_roi_visibility = dict(correlation_visibility)
            return
        existing_visibility = row_visibility(self._roi_visibility, len(roi_labels))
        if _visibility_matches_previous_correlation_filter(
            existing_visibility,
            self._correlation_roi_visibility,
        ):
            self._roi_visibility = {index: True for index in range(len(roi_labels))}
        else:
            self._roi_visibility = existing_visibility
        self._correlation_roi_visibility = None

    def _epoch_count(self) -> int:
        if self._plot_data is not None:
            return len(self._plot_data.epochs)
        if self._response_map_data is not None:
            return len(self._response_map_data.epochs)
        return 0

    def _set_status(self, text: str) -> None:
        self._plot_area.set_status(text)
        self._response_map_area.set_status(text)

    def _render_plots(self, *, apply_roi_layer_visibility: bool = True) -> None:
        if self._plot_data is None or len(self._plot_data.epochs) == 0:
            self._plot_area.set_status("No responses available.")
            self._render_response_maps()
            return
        if apply_roi_layer_visibility:
            self._apply_roi_layer_visibility()
        roi_indices = self._visible_roi_indices()
        epoch_indices = self._visible_epoch_indices()
        if len(epoch_indices) == 0:
            self._set_status("No responses selected.")
            return
        if len(roi_indices) == 0:
            self._plot_area.set_status("No ROI responses selected.")
            self._render_response_maps()
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
        self._render_response_maps()

    def _refresh_response_maps(self) -> None:
        """Recompute heatmaps for the selected recording and map options."""
        if self._recording is None:
            self._response_map_data = None
            self._response_map_area.set_status("No heatmaps available.")
            return
        try:
            self._response_map_data = compute_recording_response_maps(
                self._recording,
                options=self._response_map_options,
            )
        except ValueError as error:
            self._response_map_data = None
            self._response_map_area.set_status(f"Heatmaps unavailable: {error}")

    def _schedule_response_map_refresh(self) -> None:
        """Debounce option-driven heatmap recomputation off the Qt thread."""
        self._response_map_data = None
        self._response_map_area.clear_epoch_cache()
        if self._recording is None:
            self._response_map_area.set_status("No heatmaps available.")
            return
        self._response_map_area.set_status("Recomputing heatmaps...")
        recording = self._recording
        options = self._response_map_options
        self._response_map_worker.schedule(
            lambda: compute_recording_response_maps(recording, options=options),
        )

    def _invalidate_response_map_jobs(self) -> None:
        """Discard queued or stale heatmap worker results."""
        self._response_map_worker.invalidate()

    def _apply_response_map_data(self, map_data: ResponseMapData) -> None:
        """Apply latest worker heatmaps to the display cache."""
        self._response_map_data = map_data
        self._render_response_maps()

    def _show_response_map_value_error(self, error: ValueError) -> None:
        """Show latest expected heatmap-computation failure."""
        self._response_map_data = None
        self._response_map_area.set_status(f"Heatmaps unavailable: {error}")

    def _show_response_map_exception(self, error: Exception) -> None:
        """Show latest unexpected heatmap-computation failure."""
        self._response_map_data = None
        self._response_map_area.set_status(f"Heatmap computation failed: {error}")

    def _render_response_maps(self) -> None:
        """Render heatmaps for the same visible epochs as response plots."""
        if self._response_map_data is None or len(self._response_map_data.epochs) == 0:
            self._response_map_area.set_status("No heatmaps available.")
            return
        epoch_indices = self._visible_response_map_epoch_indices()
        if len(epoch_indices) == 0:
            self._response_map_area.set_status("No epochs selected.")
            return
        self._response_map_area.render(
            map_data=self._response_map_data,
            epoch_indices=epoch_indices,
            map_size=self._plot_size,
            shared_limits=self._response_map_shared_limits,
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
        if self._response_map_data is not None:
            response_map_epoch_indices = self._visible_response_map_epoch_indices()
            if self._response_map_area.has_epoch_widgets(response_map_epoch_indices):
                self._response_map_area.update_epoch_map_widgets(
                    epoch_indices=response_map_epoch_indices,
                    map_size=self._plot_size,
                )
            else:
                self._render_response_maps()

    def _apply_roi_layer_visibility(self) -> None:
        apply_roi_visibility_to_labels_layer(
            self._roi_labels_layer,
            roi_labels=self._roi_labels(),
            visibility=self._roi_visibility,
            colors=self._roi_colors,
            keys=tuple(range(len(self._roi_labels()))),
        )


def _correlation_roi_visibility(
    plot_data: ResponsePlotData | None,
    roi_labels: tuple[str, ...],
) -> dict[int, bool] | None:
    """Return ROI row visibility from correlation QC scores when available."""
    if plot_data is None or plot_data.correlation_scores is None:
        return None
    scores = plot_data.correlation_scores
    if scores.roi_labels != roi_labels:
        return None
    if scores.included_mask.shape != (len(roi_labels),):
        return None
    return {
        index: bool(scores.included_mask[index]) for index in range(len(roi_labels))
    }


def _visibility_matches_previous_correlation_filter(
    visibility: dict[int, bool],
    previous_correlation_visibility: dict[int, bool] | None,
) -> bool:
    """Return whether current visibility still reflects the previous QC mask."""
    if previous_correlation_visibility is None:
        return False
    return visibility == previous_correlation_visibility


def _load_pixel_calibrations_for_ui() -> tuple[PixelCalibrationRow, ...]:
    """Load calibration rows for napari ROI generation controls.

    Returns:
        Calibration rows from ``config.yml`` when available, otherwise twopy's
        tracked default registry.

    The UI can open a converted recording without a machine-local config file,
    so the default registry remains available as the fallback. Invalid config
    files still fail loudly through ``load_config``.
    """
    try:
        calibration_path = load_config().pixel_calibration_path
    except FileNotFoundError:
        calibration_path = DEFAULT_PIXEL_CALIBRATION_PATH
    return load_pixel_calibrations(calibration_path)


def _load_custom_workflow_paths_for_ui() -> tuple[Path, ...]:
    """Load workflow paths for the Custom tab.

    Returns:
        Built-in workflow paths plus paths from ``config.yml``.
    """
    try:
        configured_paths = load_config().custom_workflow_paths
    except FileNotFoundError:
        configured_paths = ()
    return (*native_custom_workflow_paths(), *configured_paths)


def _custom_workflow_output_dir(
    recording: RecordingData,
    workflow: CustomWorkflow,
) -> Path:
    """Return the output folder for one workflow run.

    Args:
        recording: Active converted recording.
        workflow: Workflow being run.

    Returns:
        Folder for custom output files.
    """
    return (
        recording.path.parent
        / "custom_outputs"
        / workflow.output_prefix
        / workflow.version
    )


def _custom_workflow_display_result(
    result: CustomResult,
    sync_plan: AnalysisSyncPlan | None,
) -> CustomResult:
    """Show publish paths for tables when sync is active."""
    if sync_plan is None:
        return result
    return replace(
        result,
        tables=tuple(
            replace(
                table,
                display_path=_custom_workflow_publish_path(table.path, sync_plan),
            )
            for table in result.tables
        ),
    )


def _custom_workflow_publish_path(
    path: Path,
    sync_plan: AnalysisSyncPlan,
) -> Path | None:
    """Return the publish path for a local custom output."""
    try:
        relative_path = path.expanduser().relative_to(sync_plan.local_root)
    except ValueError:
        return None
    return sync_plan.publish_root / relative_path


def _custom_epoch_choices(recording: RecordingData) -> tuple[str, ...]:
    """Return epoch dropdown choices for custom workflows."""
    epoch_names = stimulus_epoch_names_by_number(recording)
    return tuple(
        _custom_epoch_choice_label(epoch_number, epoch_name)
        for epoch_number, epoch_name in sorted(epoch_names.items())
    )


def _recording_parameter_spec(
    spec: CustomParameterSpec,
    *,
    recording: RecordingData,
    epoch_choices: tuple[str, ...],
    metric_stop_seconds: float | None,
    response_start_seconds: float,
    response_stop_seconds: float | None,
) -> CustomParameterSpec:
    """Apply recording-specific defaults to one parameter control."""
    if spec.role in {"comparison_epoch", "epoch"} and len(epoch_choices) > 0:
        return replace(
            spec,
            kind="choice",
            choices=epoch_choices,
            default=_matched_epoch_choice(epoch_choices, str(spec.default)),
        )
    if spec.role == "baseline_epoch" and len(epoch_choices) > 0:
        return replace(
            spec,
            kind="choice",
            choices=epoch_choices,
            default=_baseline_epoch_choice(recording, epoch_choices, str(spec.default)),
        )
    if spec.role in _CUSTOM_EPOCH_WINDOW_ROLES:
        return replace(
            spec,
            minimum=_CUSTOM_EPOCH_WINDOW_MIN_SECONDS,
            maximum=metric_stop_seconds,
            step=_CUSTOM_EPOCH_WINDOW_STEP_SECONDS,
        )
    if spec.role == "response_metric":
        return _custom_choice_parameter_spec(spec, _CUSTOM_RESPONSE_METRIC_CHOICES)
    if spec.role in _CUSTOM_RESPONSE_WINDOW_ROLES:
        return replace(
            spec,
            minimum=response_start_seconds,
            maximum=response_stop_seconds,
            step=_CUSTOM_RESPONSE_WINDOW_STEP_SECONDS,
            decimals=spec.decimals if spec.decimals is not None else 3,
        )
    if spec.role == "roi_limit":
        return replace(
            spec,
            minimum=spec.minimum if spec.minimum is not None else 1.0,
            step=spec.step if spec.step is not None else 1.0,
        )
    if spec.role == "roi_selector":
        return _custom_choice_parameter_spec(spec, _CUSTOM_ROI_SELECTOR_CHOICES)
    if spec.role == "table_highlight_threshold":
        return replace(
            spec,
            minimum=spec.minimum if spec.minimum is not None else 0.0,
            step=spec.step if spec.step is not None else 0.05,
            decimals=spec.decimals if spec.decimals is not None else 3,
        )
    return spec


def _custom_choice_parameter_spec(
    spec: CustomParameterSpec,
    choices: tuple[str, ...],
) -> CustomParameterSpec:
    """Return a dropdown spec with a valid default choice."""
    default = str(spec.default)
    if default not in choices:
        default = choices[0]
    return replace(spec, kind="choice", choices=choices, default=default)


def _matched_epoch_choice(
    choices: tuple[str, ...],
    preferred_name: str,
) -> str:
    """Return the matching epoch choice or the first choice."""
    selector = preferred_name.strip()
    preferred = selector.casefold()
    for choice in choices:
        if (
            choice.casefold() == preferred
            or _custom_epoch_number_from_choice(choice) == selector
            or _custom_epoch_name_from_choice(choice).casefold() == preferred
        ):
            return choice
    return choices[0]


def _baseline_epoch_choice(
    recording: RecordingData,
    choices: tuple[str, ...],
    preferred_name: str,
) -> str:
    """Return the default baseline epoch choice for one recording."""
    if preferred_name != "":
        return _matched_epoch_choice(choices, preferred_name)
    baseline_epoch_number = default_recording_baseline_epoch_number(recording)
    prefix = f"{baseline_epoch_number}:"
    for choice in choices:
        if choice.startswith(prefix):
            return choice
    return choices[0]


def _custom_epoch_choice_label(epoch_number: int, epoch_name: str) -> str:
    """Return one readable custom epoch choice."""
    if epoch_name == "":
        return f"Epoch {epoch_number}"
    return f"{epoch_number}: {epoch_name}"


def _custom_epoch_name_from_choice(choice: str) -> str:
    """Return the epoch-name part of a custom epoch choice."""
    if ":" not in choice:
        return choice
    return choice.split(":", maxsplit=1)[1].strip()


def _custom_epoch_number_from_choice(choice: str) -> str:
    """Return the epoch-number part of a custom epoch choice."""
    return choice.split(":", maxsplit=1)[0].strip()


def _load_pixel_calibration_profile_mappings_for_ui() -> tuple[
    PixelCalibrationProfileMapping,
    ...,
]:
    """Load ScanImage config mappings for napari ROI generation controls.

    Returns:
        Tracked profile mappings shipped with twopy.

    The mapping file is package data, not machine-local configuration. Loading
    it here keeps file I/O at the dock construction boundary instead of inside
    the ROI option widget.
    """
    return load_pixel_calibration_profile_mappings(
        DEFAULT_PIXEL_CALIBRATION_PROFILE_PATH,
    )
