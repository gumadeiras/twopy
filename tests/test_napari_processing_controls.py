"""Napari response processing-control tests.

Inputs: shared fake napari state and tiny response data.
Outputs: assertions for plot-control behavior.
"""

from qtpy.QtCore import Qt
from tests.napari_support import (
    Any,
    DeltaFOverFOptions,
    DeltaFOverFOptionsWidget,
    EpochFrameWindow,
    FrameWindow,
    NapariAdapterTestCase,
    NormalizationOptions,
    Path,
    QApplication,
    ResponsePlotData,
    ResponseProcessingOptions,
    ResponseProcessingOptionsWidget,
    RoiDeltaFOverF,
    SmoothingOptions,
    _FakeViewer,
    _tiny_response_plot_data,
    _write_converted_recording,
    cast,
    create_response_plot_widget,
    group_delta_f_over_f_by_epoch,
    np,
    open_recording_in_napari,
    response_plot_data_from_grouped,
    response_plot_min_epoch_duration_seconds,
    temporary_directory,
    unittest,
)

from twopy.custom import (
    CustomLinePlot,
    CustomParameterSpec,
    discover_custom_workflows,
    native_custom_workflow_paths,
)
from twopy.napari.custom_tab import (
    _line_plot_widget,
    _line_plot_y_bounds,
    _parameter_widget,
)
from twopy.napari.plotting.docks.response_plot_widget import _recording_parameter_spec


class NapariProcessingControlsTest(NapariAdapterTestCase):
    """Napari response processing-control tests."""

    def test_correlation_window_stop_defaults_to_shortest_epoch_duration(
        self,
    ) -> None:
        """Confirm correlation stop starts at the shortest epoch duration.

        Inputs: plot data with three-second and two-second stimulus epochs.
        Outputs: enabling the correlation window stop uses two seconds instead
        of the invalid zero-second default.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((5, 1), dtype=np.float64),
            baseline=np.ones((5, 1), dtype=np.float64),
            values=np.arange(5, dtype=np.float64).reshape(5, 1),
            labels=("roi_1",),
            start_frame=0,
            stop_frame=5,
            tau=0.0,
            amplitudes=np.ones(1, dtype=np.float64),
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 1), dtype=np.float64),
            metadata={"method": "test"},
        )
        epoch_windows = (
            EpochFrameWindow(FrameWindow(0, 0, 3, "slow"), 1, "Slow"),
            EpochFrameWindow(FrameWindow(1, 3, 5, "fast"), 2, "Fast"),
        )
        plot_data = response_plot_data_from_grouped(
            group_delta_f_over_f_by_epoch(
                dff,
                epoch_windows,
                data_rate_hz=1.0,
            ),
            correlation_window_stop_default_seconds=(
                response_plot_min_epoch_duration_seconds(
                    epoch_windows,
                    data_rate_hz=1.0,
                )
            ),
        )

        response_widget.set_response_plot_data(plot_data, reset_axes=True)
        processing_widget = response_widget._processing_options_widget
        processing_widget._correlation_window_has_stop.setChecked(True)

        self.assertEqual(processing_widget._correlation_window_stop.value(), 2.0)
        self.assertEqual(
            processing_widget.options().correlation_filter.window_seconds,
            (0.0, 2.0),
        )

    def test_dff_baseline_mode_switches_epoch_label_and_option(self) -> None:
        """Confirm no-baseline mode uses native UI wording.

        Inputs: a Plot-tab dF/F widget with default settings.
        Outputs: the epoch selector is labeled as a baseline selector for
        normal dF/F and as the first epoch selector for no-baseline analysis.
        """
        _ = QApplication.instance() or QApplication([])
        dff_widget = cast(Any, DeltaFOverFOptionsWidget(DeltaFOverFOptions()))

        self.assertEqual(dff_widget._baseline_epoch_label.text(), "Baseline epoch")
        dff_widget._baseline_mode.setCurrentIndex(
            dff_widget._baseline_mode.findData("no_baseline_epoch"),
        )

        self.assertEqual(dff_widget._baseline_epoch_label.text(), "First epoch")
        self.assertEqual(dff_widget.options().baseline_mode, "no_baseline_epoch")

    def test_baseline_epoch_dropdown_uses_recording_epoch_names(self) -> None:
        """Confirm baseline selection shows actual recording epoch names.

        Inputs: converted recording with stimulus parameter epoch names.
        Outputs: the dF/F baseline dropdown lists those names and stores the
        selected epoch selector.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                stimulus_parameters_json=(
                    '[{"epochName": "Gray Interleave"}, {"epochName": "Odor A"}]'
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            dff_widget = response_widget._delta_f_over_f_options_widget

            labels = tuple(
                dff_widget._baseline_epoch.itemText(index)
                for index in range(dff_widget._baseline_epoch.count())
            )
            dff_widget._baseline_epoch.setCurrentIndex(1)

            self.assertEqual(labels, ("1: Gray Interleave", "2: Odor A"))
            self.assertEqual(
                dff_widget.options().baseline_epoch_name,
                "Odor A",
            )

    def test_native_dsi_epoch_parameters_use_recording_epoch_dropdowns(self) -> None:
        """Confirm native DSI uses recording-aware Custom tab controls."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                stimulus_parameters_json=(
                    '[{"epochName": "Left"}, {"epochName": "Right"}]'
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            workflows = discover_custom_workflows(native_custom_workflow_paths())
            workflow = next(
                item
                for item in workflows.workflows
                if item.id == "direction-selectivity"
            )

            specs = {
                spec.name: spec
                for spec in response_widget._custom_parameter_specs_for_workflow(
                    workflow,
                )
            }

            self.assertEqual(specs["preferred_epoch"].kind, "choice")
            self.assertEqual(specs["null_epoch"].kind, "choice")
            self.assertEqual(specs["preferred_epoch"].choices, ("1: Left", "2: Right"))
            self.assertEqual(specs["null_epoch"].choices, ("1: Left", "2: Right"))
            self.assertEqual(specs["metric"].kind, "choice")
            self.assertEqual(specs["metric"].choices, ("mean", "peak", "minimum"))
            self.assertEqual(specs["roi_selector"].kind, "choice")
            self.assertEqual(
                specs["roi_selector"].choices,
                ("all_rois", "visible_rois"),
            )
            self.assertEqual(specs["window_start_seconds"].minimum, 0.0)
            self.assertEqual(specs["window_start_seconds"].step, 0.1)
            self.assertEqual(specs["window_stop_seconds"].default, 3.3)
            self.assertEqual(specs["window_stop_seconds"].minimum, 0.0)
            self.assertEqual(specs["window_stop_seconds"].step, 0.1)
            self.assertEqual(specs["dsi_threshold"].minimum, 0.0)
            self.assertEqual(specs["dsi_threshold"].step, 0.05)
            self.assertEqual(specs["dsi_threshold"].decimals, 3)

    def test_roi_selector_dropdown_uses_readable_labels(self) -> None:
        """Confirm ROI selector values keep stable IDs but display readable text."""
        _ = QApplication.instance() or QApplication([])
        spec = CustomParameterSpec(
            name="roi_selector",
            label="ROIs",
            kind="choice",
            default="visible_rois",
            description="",
            role="roi_selector",
            choices=("all_rois", "visible_rois"),
        )

        widget = cast(Any, _parameter_widget(spec))

        self.assertEqual(
            [widget.itemText(index) for index in range(widget.count())],
            ["all ROIs", "visible ROIs"],
        )
        self.assertEqual(
            [widget.itemData(index) for index in range(widget.count())],
            ["all_rois", "visible_rois"],
        )
        self.assertEqual(widget.currentData(), "visible_rois")

    def test_custom_line_plot_uses_response_plot_style(self) -> None:
        """Confirm custom line plots share the response-widget visual palette."""
        from matplotlib.colors import to_hex

        _ = QApplication.instance() or QApplication([])
        plot = CustomLinePlot(
            "Kernel",
            np.array([-1.0, 0.0, 1.0], dtype=np.float64),
            np.array([[-1.0, 0.0, 1.0]], dtype=np.float64),
            ("roi_0001",),
            y_label="Weight",
        )

        widget = cast(Any, _line_plot_widget(plot, y_bounds=(-2.0, 2.0)))
        axes = widget.figure.axes[0]

        self.assertEqual(to_hex(widget.figure.get_facecolor()), "#20252d")
        self.assertEqual(to_hex(axes.get_facecolor()), "#20252d")
        self.assertFalse(axes.spines["top"].get_visible())
        self.assertFalse(axes.spines["right"].get_visible())
        self.assertEqual(to_hex(axes.lines[0].get_color()), "#4cc9f0")
        self.assertEqual(axes.get_ylabel(), "Weight")
        self.assertEqual(tuple(float(value) for value in axes.get_ylim()), (-2.0, 2.0))
        self.assertEqual(widget.focusPolicy(), Qt.FocusPolicy.NoFocus)
        self.assertGreaterEqual(len(axes.lines), 3)

    def test_custom_line_plot_y_bounds_span_all_result_plots(self) -> None:
        """Confirm a workflow result can use one y scale for every line plot."""
        plots = (
            CustomLinePlot(
                "A",
                np.array([0.0, 1.0], dtype=np.float64),
                np.array([[1.0, 2.0]], dtype=np.float64),
                ("a",),
            ),
            CustomLinePlot(
                "B",
                np.array([0.0, 1.0], dtype=np.float64),
                np.array([[-10.0, 5.0]], dtype=np.float64),
                ("b",),
            ),
        )

        self.assertEqual(_line_plot_y_bounds(plots), (-10.75, 5.75))

    def test_native_kernel_modality_parameter_uses_dropdown(self) -> None:
        """Confirm native kernel stimulus mode is a dropdown."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                stimulus_data=np.zeros((1, 4), dtype=np.float64),
                stimulus_data_column_names=(
                    "time_seconds",
                    "epoch_number",
                    "stimulus_frame_number",
                    "stimulus_specific_05",
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            workflows = discover_custom_workflows(native_custom_workflow_paths())
            workflow = next(
                item for item in workflows.workflows if item.id == "response-kernels"
            )

            specs = {
                spec.name: spec
                for spec in response_widget._custom_parameter_specs_for_workflow(
                    workflow,
                )
            }

            self.assertEqual(specs["stimulus_modality"].kind, "choice")
            self.assertEqual(
                specs["stimulus_modality"].choices,
                ("olfaction", "vision"),
            )
            self.assertEqual(specs["stimulus_modality"].default, "olfaction")
            hemisphere_widget = cast(Any, _parameter_widget(specs["hemisphere"]))

            self.assertEqual(
                [
                    hemisphere_widget.itemText(index)
                    for index in range(hemisphere_widget.count())
                ],
                ["auto", "right", "left"],
            )
            self.assertEqual(hemisphere_widget.currentData(), "recording_metadata")

    def test_custom_epoch_parameter_default_matches_supported_selectors(self) -> None:
        """Confirm epoch dropdown defaults accept number, label, and name."""
        with temporary_directory() as temp_dir:
            recording_path = _write_converted_recording(
                Path(temp_dir),
                stimulus_parameters_json=(
                    '[{"epochName": "Left"}, {"epochName": "Right"}]'
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            choices = ("1: Left", "2: Right")

            for default in ("2", "2: Right", "Right"):
                spec = CustomParameterSpec(
                    name="epoch",
                    label="Epoch",
                    kind="str",
                    default=default,
                    description="",
                    role="epoch",
                )
                adjusted = _recording_parameter_spec(
                    spec,
                    recording=opened.recording,
                    epoch_choices=choices,
                    stimulus_column_choices=(),
                    metric_stop_seconds=None,
                    response_start_seconds=-1.0,
                    response_stop_seconds=2.0,
                )

                self.assertEqual(adjusted.default, "2: Right")

            spec = CustomParameterSpec(
                name="epoch",
                label="Epoch",
                kind="str",
                default="2",
                description="",
                role="epoch",
            )
            adjusted = _recording_parameter_spec(
                spec,
                recording=opened.recording,
                epoch_choices=("Epoch 1", "Epoch 2"),
                stimulus_column_choices=(),
                metric_stop_seconds=None,
                response_start_seconds=-1.0,
                response_stop_seconds=2.0,
            )

            self.assertEqual(adjusted.default, "Epoch 2")

    def test_baseline_epoch_dropdown_defaults_to_gray_like_epoch(self) -> None:
        """Confirm the dF/F baseline defaults to the gray/interleave epoch.

        Inputs: converted recording where the gray-like epoch is not epoch one.
        Outputs: the baseline dropdown selects that named epoch by default.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                stimulus_parameters_json=(
                    '[{"epochName": "Odor A"}, {"epochName": "Grey screen"}]'
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            options = response_widget._delta_f_over_f_options_widget.options()

            self.assertEqual(options.baseline_epoch_number, 2)
            self.assertEqual(options.baseline_epoch_name, "Grey screen")

    def test_baseline_epoch_dropdown_accepts_interleave_without_gray(self) -> None:
        """Confirm interleave names are accepted even without gray spelling.

        Inputs: converted recording with an ``interleave`` baseline name.
        Outputs: the baseline dropdown selects that epoch by default.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                stimulus_parameters_json=(
                    '[{"epochName": "Odor A"}, {"epochName": "Baseline Interleave"}]'
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            options = response_widget._delta_f_over_f_options_widget.options()

            self.assertEqual(options.baseline_epoch_number, 2)
            self.assertEqual(options.baseline_epoch_name, "Baseline Interleave")

    def test_normalization_epoch_defaults_to_first_response_epoch(self) -> None:
        """Confirm normalization defaults to a non-baseline epoch when present.

        Inputs: converted recording with a gray baseline followed by odor.
        Outputs: normalization is disabled by default but points at the odor
        epoch when enabled.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                stimulus_parameters_json=(
                    '[{"epochName": "Gray Interleave"}, {"epochName": "Odor A"}]'
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            normalization_widget = response_widget._normalization_options_widget

            self.assertFalse(normalization_widget._normalize_to_epoch_peak.isChecked())
            self.assertFalse(normalization_widget._epoch.isEnabled())
            self.assertEqual(normalization_widget.options().method, "none")
            self.assertEqual(normalization_widget.options().epoch_number, 2)
            self.assertEqual(normalization_widget.options().epoch_name, "Odor A")

    def test_saved_processing_options_update_plot_tab_controls(self) -> None:
        """Confirm saved analysis settings restore Plot-tab controls.

        Inputs: plot data carrying persisted response-processing options.
        Outputs: response widget controls show the saved smoothing settings.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        options = ResponseProcessingOptions(
            smoothing=SmoothingOptions(
                method="savgol",
                window_frames=11,
                polynomial_order=3,
            ),
            normalization=NormalizationOptions(
                method="epoch_peak",
                epoch_number=2,
                epoch_name="Odor A",
            ),
        )
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=_tiny_response_plot_data().epochs,
            response_processing_options=options,
        )

        response_widget.set_response_plot_data(plot_data, reset_axes=True)

        self.assertEqual(
            response_widget._processing_options_widget.options().smoothing,
            options.smoothing,
        )
        self.assertEqual(
            response_widget._normalization_options_widget.options(),
            options.normalization,
        )
        self.assertEqual(response_widget._response_processing_options, options)

    def test_saved_delta_f_over_f_options_update_plot_tab_controls(self) -> None:
        """Confirm saved dF/F settings restore Plot-tab controls.

        Inputs: plot data carrying persisted dF/F options.
        Outputs: response widget controls show the saved dF/F settings.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        options = DeltaFOverFOptions(
            baseline_mode="no_baseline_epoch",
            baseline_epoch_number=3,
            baseline_epoch_name="Manual baseline",
            background_method="roi_y_stripe_percentile",
            baseline_sample_seconds=None,
            fit_mode="direct_bounded_tau_and_log_amplitude",
            apply_motion_mask=False,
        )
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=_tiny_response_plot_data().epochs,
            delta_f_over_f_options=options,
        )

        response_widget.set_response_plot_data(plot_data, reset_axes=True)

        self.assertEqual(
            response_widget._delta_f_over_f_options_widget.options(),
            options,
        )
        self.assertEqual(response_widget._delta_f_over_f_options, options)
        self.assertEqual(
            response_widget._delta_f_over_f_options_widget._baseline_epoch_label.text(),
            "First epoch",
        )

    def test_savgol_window_control_steps_through_valid_values(self) -> None:
        """Confirm Savitzky-Golay window clicks stay on valid values.

        Inputs: processing controls configured for second-order Savitzky-Golay.
        Outputs: the window spinbox uses odd values above the polynomial order.
        """
        _ = QApplication.instance() or QApplication([])
        widget = cast(
            Any,
            ResponseProcessingOptionsWidget(
                ResponseProcessingOptions(
                    smoothing=SmoothingOptions(
                        method="savgol",
                        window_frames=7,
                        polynomial_order=2,
                    ),
                ),
            ),
        )
        window_spin_box = widget._smoothing_window_frames

        self.assertEqual(window_spin_box.minimum(), 3)
        self.assertEqual(window_spin_box.singleStep(), 2)

        window_spin_box.stepUp()
        self.assertEqual(window_spin_box.value(), 9)

        window_spin_box.stepDown()
        self.assertEqual(window_spin_box.value(), 7)

        window_spin_box.setValue(8)
        self.assertEqual(window_spin_box.value(), 9)

    def test_savgol_window_minimum_tracks_polynomial_order(self) -> None:
        """Confirm Savitzky-Golay window minimum follows polynomial order.

        Inputs: processing controls with a small Savitzky-Golay window.
        Outputs: increasing polynomial order raises the window to the next
        valid odd length.
        """
        _ = QApplication.instance() or QApplication([])
        widget = cast(
            Any,
            ResponseProcessingOptionsWidget(
                ResponseProcessingOptions(
                    smoothing=SmoothingOptions(
                        method="savgol",
                        window_frames=3,
                        polynomial_order=2,
                    ),
                ),
            ),
        )

        widget._smoothing_polynomial_order.setValue(4)

        self.assertEqual(widget._smoothing_window_frames.minimum(), 5)
        self.assertEqual(widget._smoothing_window_frames.value(), 5)
        self.assertEqual(widget.options().smoothing.window_frames, 5)


if __name__ == "__main__":
    unittest.main()
