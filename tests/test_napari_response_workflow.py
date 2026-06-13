"""Napari response workflow tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from tests.napari_support import (
    Any,
    BackgroundCorrectedRoiTraces,
    DeltaFOverFOptions,
    EpochFrameWindow,
    FrameWindow,
    LiveResponseAnalysisCache,
    NapariAdapterTestCase,
    NormalizationOptions,
    Path,
    QCheckBox,
    QLabel,
    ResponseProcessingOptions,
    ResponseWindowOptions,
    RoiCorrelationScores,
    SimpleNamespace,
    SmoothingOptions,
    SpatialCrop,
    _FakeViewer,
    _tiny_grouped_responses,
    _tiny_response_map_data,
    _tiny_response_plot_data,
    _write_converted_recording,
    _write_source_recording_shape,
    cast,
    chdir,
    compute_response_preview,
    load_converted_recording,
    load_response_map_data,
    make_roi_set,
    np,
    open_recording_in_napari,
    patch,
    response_analysis_request_from_label_image,
    roi_label_image_from_layer,
    roi_set_to_label_image,
    save_analysis_outputs,
    save_roi_set,
    temporary_directory,
    unittest,
)
from twopy.napari.output_routing import NapariOutputRoute
from twopy.napari.plotting.docks.save_actions import save_current_roi_analysis
from twopy.napari.plotting.roi_generation import RoiGenerationOptions
from twopy.roi import load_roi_generation_metadata


class NapariResponseWorkflowTest(NapariAdapterTestCase):
    """Napari response workflow tests."""

    def test_startup_open_routes_output_to_resolved_publish_folder(self) -> None:
        """Confirm startup-loaded recordings keep launcher output routing.

        Inputs: converted recording opened through the startup viewer helper
            with an explicit local and publish route.
        Outputs: response metadata labels show the publish folder from the
            route instead of recomputing a cache-local fallback.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            local_dir = root / "cache"
            publish_dir = root / "source" / "twopy"
            recording_path = _write_converted_recording(local_dir)
            viewer = _FakeViewer()

            opened = open_recording_in_napari(
                recording_path,
                viewer=viewer,
                output_route=NapariOutputRoute(
                    local_root=local_dir,
                    publish_root=publish_dir,
                ),
            )

            response_widget = cast(Any, opened.response_plot_widget)
            self.assertIn(
                f"Output: {publish_dir / 'analysis_outputs.h5'}",
                response_widget._analysis_path_label.text(),
            )
            self.assertIn(
                f"Output: {publish_dir / 'rois.h5'}",
                response_widget._roi_save_path_label.text(),
            )
            response_widget.shutdown()

    def test_response_update_rejects_full_frame_labels_layer(self) -> None:
        """Confirm response updates use the same crop-native ROI contract.

        Inputs: cropped recording display plus a stale full-frame Labels image.
        Outputs: response widget status explains that the layer shape is
        invalid before analysis runs.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                movie_values=np.arange(27, dtype=np.float64).reshape(3, 3, 3),
                alignment_valid_crop=SpatialCrop(
                    axis0_start=1,
                    axis0_stop=3,
                    axis1_start=0,
                    axis1_stop=2,
                    original_shape=(3, 3),
                    source="alignment_valid_crop",
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            viewer.labels[0].data = np.ones((3, 3), dtype=np.int64)
            response_widget = cast(Any, opened.response_plot_widget)

            response_widget.update_from_current_rois()

            status_text = "\n".join(
                label.text() for label in response_widget.findChildren(QLabel)
            )
            self.assertIn(
                "ROI Labels layer must use the cropped recording view",
                status_text,
            )

    def test_response_update_from_rois_does_not_write_analysis_file(self) -> None:
        """Confirm napari plot previews compute in memory only.

        Inputs: edited Labels layer and a patched response calculation.
        Outputs: plot data shown in the widget without creating analysis HDF5.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            viewer.labels[0].data = np.array([[1, 0], [0, 0]], dtype=np.int64)
            response_widget = cast(Any, opened.response_plot_widget)

            with patch(
                "twopy.napari.interactive.compute_response_preview",
                return_value=_tiny_response_plot_data(),
            ):
                response_widget.update_from_current_rois()

            self.assertFalse((root / "analysis_outputs.h5").exists())
            self.assertFalse((root / "response_summary_trials.csv").exists())
            self.assertFalse((root / "response_summary_grouped.csv").exists())
            self.assertFalse((root / "exports" / "csvs").exists())
            self.assertIsNotNone(response_widget._plot_data)

    def test_live_response_plot_data_preserves_gray_post_window(self) -> None:
        """Confirm live Plot-tab recompute keeps post-stimulus plot context.

        Inputs: converted recording with a gray epoch and patched analysis
        computation.
        Outputs: the live plot path requests the same two-second post window
        used when saved analysis outputs are loaded for plotting.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(
                _write_converted_recording(
                    root,
                    stimulus_parameters_json=(
                        '[{"epochName": "Gray"}, {"epochName": "Odor A"}]'
                    ),
                ),
            )
            roi_set = make_roi_set(
                np.array([[[True, False], [False, False]]]),
                labels=("roi_1",),
            )
            computation = SimpleNamespace(
                grouped_responses=_tiny_grouped_responses(),
                epoch_windows=(
                    EpochFrameWindow(FrameWindow(0, 0, 2, "odor"), 1, "Odor"),
                ),
                response_processing_options=ResponseProcessingOptions(),
                correlation_scores=RoiCorrelationScores(
                    roi_labels=("roi_1",),
                    scores=np.array([1.0], dtype=np.float64),
                    included_mask=np.array([True], dtype=np.bool_),
                    minimum_correlation=0.5,
                    reference="epoch_mean",
                    window_seconds=(0.0, None),
                ),
            )

            with patch(
                "twopy.napari.responses.compute_recording_responses",
                return_value=computation,
            ) as compute:
                request = response_analysis_request_from_label_image(
                    recording,
                    roi_set_to_label_image(roi_set),
                )
                plot_data = compute_response_preview(request)

            self.assertIsNotNone(plot_data)
            self.assertIs(plot_data.correlation_scores, computation.correlation_scores)
            self.assertEqual(
                compute.call_args.kwargs["response_pre_window_seconds"],
                2.0,
            )
            self.assertEqual(
                compute.call_args.kwargs["response_post_window_seconds"],
                2.0,
            )

    def test_live_response_plot_data_omits_post_window_without_gray(self) -> None:
        """Confirm non-interleave recordings keep epoch-bounded live plots.

        Inputs: converted recording without gray/grey/interleave epoch names.
        Outputs: the live plot path requests no post-epoch plotting context.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(
                _write_converted_recording(
                    root,
                    stimulus_parameters_json=(
                        '[{"epochName": "Odor A"}, {"epochName": "Odor B"}]'
                    ),
                ),
            )
            roi_set = make_roi_set(
                np.array([[[True, False], [False, False]]]),
                labels=("roi_1",),
            )
            computation = SimpleNamespace(
                grouped_responses=_tiny_grouped_responses(),
                epoch_windows=(
                    EpochFrameWindow(FrameWindow(0, 0, 2, "odor"), 1, "Odor"),
                ),
                response_processing_options=ResponseProcessingOptions(),
                correlation_scores=None,
            )

            with patch(
                "twopy.napari.responses.compute_recording_responses",
                return_value=computation,
            ) as compute:
                request = response_analysis_request_from_label_image(
                    recording,
                    roi_set_to_label_image(roi_set),
                )
                plot_data = compute_response_preview(request)

            self.assertIsNotNone(plot_data)
            self.assertEqual(
                compute.call_args.kwargs["response_pre_window_seconds"],
                2.0,
            )
            self.assertEqual(
                compute.call_args.kwargs["response_post_window_seconds"],
                0.0,
            )

    def test_live_response_plot_data_uses_manual_response_window(self) -> None:
        """Confirm manual Plot-tab response windows reach live analysis.

        Inputs: converted recording, manual response-window options, and
        patched analysis computation.
        Outputs: live analysis receives the requested pre/post seconds.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(_write_converted_recording(root))
            roi_set = make_roi_set(
                np.array([[[True, False], [False, False]]]),
                labels=("roi_1",),
            )
            computation = SimpleNamespace(
                grouped_responses=_tiny_grouped_responses(),
                epoch_windows=(
                    EpochFrameWindow(FrameWindow(0, 0, 2, "odor"), 1, "Odor"),
                ),
                response_processing_options=ResponseProcessingOptions(),
                correlation_scores=None,
            )

            with patch(
                "twopy.napari.responses.compute_recording_responses",
                return_value=computation,
            ) as compute:
                request = response_analysis_request_from_label_image(
                    recording,
                    roi_set_to_label_image(roi_set),
                    response_window_options=ResponseWindowOptions(
                        auto=False,
                        pre_window_seconds=0.5,
                        post_window_seconds=1.5,
                    ),
                )
                plot_data = compute_response_preview(request)

            self.assertIsNotNone(plot_data)
            self.assertEqual(
                compute.call_args.kwargs["response_pre_window_seconds"],
                0.5,
            )
            self.assertEqual(
                compute.call_args.kwargs["response_post_window_seconds"],
                1.5,
            )

    def test_response_analysis_request_options_reach_preview_save_and_live_cache(
        self,
    ) -> None:
        """Confirm one request drives all napari ROI response paths.

        Inputs: one response-analysis request with non-default dF/F,
        response-window, and processing settings.
        Outputs: preview, Save Analysis, and live cached preview pass the same
        request-derived options to their underlying analysis calls.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(
                _write_converted_recording(
                    root,
                    stimulus_parameters_json=(
                        '[{"epochName": "Gray"}, {"epochName": "Odor"}]'
                    ),
                ),
            )
            roi_set = make_roi_set(
                np.array([[[True, False], [False, False]]], dtype=np.bool_),
                labels=("roi_0001",),
            )
            dff_options = DeltaFOverFOptions(
                baseline_epoch_number=2,
                baseline_epoch_name="Odor",
                baseline_mode="no_baseline_epoch",
                background_method="none",
                baseline_sample_seconds=None,
                fit_mode="log_linear",
                apply_motion_mask=False,
            )
            response_window_options = ResponseWindowOptions(
                auto=False,
                pre_window_seconds=0.25,
                post_window_seconds=0.75,
            )
            processing_options = ResponseProcessingOptions(
                smoothing=SmoothingOptions(
                    method="moving_average",
                    window_frames=3,
                ),
                normalization=NormalizationOptions(
                    method="epoch_peak",
                    epoch_number=2,
                    epoch_name="Odor",
                ),
            )
            request = response_analysis_request_from_label_image(
                recording,
                roi_set_to_label_image(roi_set),
                delta_f_over_f_options=dff_options,
                response_window_options=response_window_options,
                response_processing_options=processing_options,
            )
            computation = SimpleNamespace(
                grouped_responses=_tiny_grouped_responses(),
                epoch_windows=(
                    EpochFrameWindow(FrameWindow(0, 0, 2, "odor"), 2, "Odor"),
                ),
                response_processing_options=processing_options,
                correlation_scores=None,
            )

            with patch(
                "twopy.napari.responses.compute_recording_responses",
                return_value=computation,
            ) as preview_compute:
                compute_response_preview(request)

            preview_kwargs = preview_compute.call_args.kwargs
            self._assert_request_workflow_kwargs(
                preview_kwargs,
                processing_options=processing_options,
            )

            analysis_path = root / "analysis_outputs.h5"
            with patch(
                (
                    "twopy.napari.plotting.docks.save_actions."
                    "analyze_recording_responses"
                ),
                return_value=SimpleNamespace(
                    output_path=analysis_path,
                    response_summary_trials_csv_path=None,
                    response_summary_grouped_csv_path=None,
                ),
            ) as save_analyze:
                save_current_roi_analysis(
                    request,
                    roi_save_file=None,
                    analysis_path=analysis_path,
                    output_route=None,
                )

            save_kwargs = save_analyze.call_args.kwargs
            self._assert_request_workflow_kwargs(
                save_kwargs,
                processing_options=processing_options,
            )
            self.assertFalse(save_kwargs["response_window_auto"])

            live_cache = LiveResponseAnalysisCache()
            context = SimpleNamespace(trace_start=0, trace_stop=2)

            def extract(
                _recording: object,
                extracted_roi_set: object,
                **kwargs: object,
            ) -> BackgroundCorrectedRoiTraces:
                del _recording, kwargs
                roi_labels = cast(Any, extracted_roi_set).labels
                values = np.ones((2, len(roi_labels)), dtype=np.float64)
                return BackgroundCorrectedRoiTraces(
                    raw_values=values,
                    background_values=np.zeros_like(values),
                    corrected_values=values,
                    labels=roi_labels,
                    start_frame=0,
                    stop_frame=2,
                    statistic="mean",
                    method="none",
                    metadata={"method": "none"},
                )

            with (
                patch(
                    "twopy.napari.live_analysis.resolve_response_analysis_context",
                    return_value=context,
                ) as live_context,
                patch(
                    "twopy.napari.live_analysis."
                    "extract_background_corrected_roi_traces",
                    side_effect=extract,
                ) as live_extract,
                patch(
                    "twopy.napari.live_analysis."
                    "compute_recording_responses_from_traces",
                    return_value=computation,
                ) as live_compute,
                patch(
                    "twopy.napari.live_analysis.response_plot_data_from_computation",
                    return_value=_tiny_response_plot_data(),
                ),
            ):
                live_cache.compute_response_preview(request)

            live_context_kwargs = live_context.call_args.kwargs
            self.assertEqual(live_context_kwargs["baseline_mode"], "no_baseline_epoch")
            self.assertEqual(live_context_kwargs["baseline_epoch_number"], 2)
            self.assertEqual(live_context_kwargs["baseline_epoch_name"], "Odor")
            self.assertEqual(live_context_kwargs["response_pre_window_seconds"], 0.25)
            self.assertEqual(live_context_kwargs["response_post_window_seconds"], 0.75)
            self.assertIs(
                live_context_kwargs["response_processing_options"],
                processing_options,
            )
            live_extract_kwargs = live_extract.call_args.kwargs
            self.assertEqual(live_extract_kwargs["method"], "none")
            self.assertEqual(
                live_extract_kwargs["spatial_domain"],
                "alignment_valid_crop",
            )
            live_compute_kwargs = live_compute.call_args.kwargs
            self.assertIsNone(live_compute_kwargs["baseline_sample_seconds"])
            self.assertEqual(live_compute_kwargs["fit_mode"], "log_linear")
            self.assertFalse(live_compute_kwargs["apply_motion_mask"])

    def test_save_analysis_button_writes_roi_and_analysis_outputs(self) -> None:
        """Confirm the Export tab can persist current ROI analysis.

        Inputs: edited Labels layer and a patched analysis workflow.
        Outputs: ROI HDF5 file written beside the recording without replacing
        the current in-memory plot preview.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "source" / "twopy"
            analysis_path = root / "analysis_outputs.h5"
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            viewer.labels[0].data = np.array([[1, 0], [0, 0]], dtype=np.int64)
            response_widget = cast(Any, opened.response_plot_widget)
            preview_plot_data = _tiny_response_plot_data()
            preview_map_data = _tiny_response_map_data()
            response_widget._plot_data = preview_plot_data
            response_widget._response_map_data = preview_map_data
            response_widget._set_response_window_options(
                ResponseWindowOptions(
                    auto=False,
                    pre_window_seconds=0.5,
                    post_window_seconds=1.5,
                ),
            )

            with patch(
                (
                    "twopy.napari.plotting.docks.save_actions."
                    "analyze_recording_responses"
                ),
                return_value=SimpleNamespace(
                    output_path=analysis_path,
                    grouped_responses=_tiny_grouped_responses(),
                    response_summary_trials_csv_path=None,
                    response_summary_grouped_csv_path=None,
                ),
            ) as analyze:
                response_widget.save_analysis_and_rois()

            roi_path = root / "rois.h5"
            heatmap_path = root / "response_heatmaps.h5"
            self.assertTrue(roi_path.is_file())
            self.assertTrue(heatmap_path.is_file())
            loaded_heatmaps = load_response_map_data(heatmap_path)
            np.testing.assert_allclose(
                loaded_heatmaps.epochs[0].response_values,
                preview_map_data.epochs[0].response_values,
            )
            analyze.assert_called_once()
            _, roi_set = analyze.call_args.args
            self.assertEqual(roi_set.labels, ("roi_0001",))
            self.assertEqual(analyze.call_args.kwargs["output_path"], analysis_path)
            self.assertEqual(
                analyze.call_args.kwargs["background_method"],
                "movie_global_percentile",
            )
            self.assertEqual(
                analyze.call_args.kwargs["baseline_sample_seconds"],
                1.0,
            )
            self.assertIsNone(analyze.call_args.kwargs["baseline_epoch_name"])
            self.assertEqual(
                analyze.call_args.kwargs["fit_mode"],
                "direct_bounded_tau",
            )
            self.assertTrue(analyze.call_args.kwargs["apply_motion_mask"])
            self.assertEqual(
                analyze.call_args.kwargs["response_pre_window_seconds"],
                0.5,
            )
            self.assertEqual(
                analyze.call_args.kwargs["response_post_window_seconds"],
                1.5,
            )
            self.assertFalse(analyze.call_args.kwargs["response_window_auto"])
            self.assertIsInstance(
                analyze.call_args.kwargs["response_processing_options"],
                ResponseProcessingOptions,
            )
            self.assertIs(response_widget._plot_data, preview_plot_data)

            recording_summary_text = response_widget._recording_summary_label.text()
            self.assertEqual(len(recording_summary_text.split("\n\n")), 4)
            microscope_summary_text = response_widget._microscope_summary_label.text()
            self.assertIn("Rig: TestRig", microscope_summary_text)
            self.assertIn("Frame rate: 10 Hz", microscope_summary_text)
            labels_text = "\n".join(
                (
                    response_widget._analysis_path_label.text(),
                    response_widget._roi_save_path_label.text(),
                    response_widget._update_status_label.text(),
                ),
            )
            self.assertIn(f"Local: {root / 'analysis_outputs.h5'}", labels_text)
            self.assertIn(f"Output: {output_dir / 'analysis_outputs.h5'}", labels_text)
            self.assertIn(f"Local: {root / 'rois.h5'}", labels_text)
            self.assertIn(f"Output: {output_dir / 'rois.h5'}", labels_text)
            self.assertIn(f"Saved 1 ROI locally to {root}", labels_text)
            self.assertIn(
                f"syncing to {output_dir}",
                response_widget._export_status_label.text(),
            )
            response_widget.shutdown()

    def test_save_analysis_button_writes_roi_generation_metadata(self) -> None:
        """Confirm generated ROI settings are saved with ROI masks.

        Inputs: a Labels layer made by watershed generation and a patched
            analysis workflow.
        Outputs: ``rois.h5`` records the watershed mode and parameters that
            produced the current masks.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            analysis_path = root / "analysis_outputs.h5"
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            response_widget._live_controller.request_update = lambda: None
            options = RoiGenerationOptions(
                roi_mode="watershed",
                units="pixels",
                grid_size_pixels=16,
                micron_grid_size=10.0,
                rig="",
                calibration_mode=0,
                scanner="",
                zoom=1.0,
                allow_extrapolation=True,
                watershed_min_pixels=15,
                watershed_smoothing_sigma=0.75,
            )

            with patch(
                "twopy.napari.plotting.docks.response_plot_widget.generate_roi_labels",
                return_value=SimpleNamespace(
                    label_image=np.array([[1, 0], [0, 0]], dtype=np.int64),
                    status_text="Created watershed ROIs: 1 ROI.",
                ),
            ):
                response_widget.create_generated_rois(options)
            with patch(
                (
                    "twopy.napari.plotting.docks.save_actions."
                    "analyze_recording_responses"
                ),
                return_value=SimpleNamespace(
                    output_path=analysis_path,
                    grouped_responses=_tiny_grouped_responses(),
                    response_summary_trials_csv_path=None,
                    response_summary_grouped_csv_path=None,
                ),
            ):
                response_widget.save_analysis_and_rois()

            self.assertEqual(
                load_roi_generation_metadata(root / "rois.h5"),
                {
                    "schema_version": 1,
                    "roi_mode": "watershed",
                    "watershed_min_pixels": 15,
                    "watershed_smoothing_sigma": 0.75,
                },
            )
            response_widget.shutdown()

    def test_save_analysis_marks_generated_rois_edited_after_manual_change(
        self,
    ) -> None:
        """Confirm manual edits to generated ROIs are saved as provenance.

        Inputs: watershed-generated ROI masks marked as hand-edited before
        saving.
        Outputs: saved ROI metadata keeps the watershed settings and records
        the post-generation edit marker.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            analysis_path = root / "analysis_outputs.h5"
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            response_widget._live_controller.request_update = lambda: None
            options = RoiGenerationOptions(
                roi_mode="watershed",
                units="pixels",
                grid_size_pixels=16,
                micron_grid_size=10.0,
                rig="",
                calibration_mode=0,
                scanner="",
                zoom=1.0,
                allow_extrapolation=True,
                watershed_min_pixels=15,
                watershed_smoothing_sigma=0.75,
            )

            with patch(
                "twopy.napari.plotting.docks.response_plot_widget.generate_roi_labels",
                return_value=SimpleNamespace(
                    label_image=np.array([[1, 0], [0, 0]], dtype=np.int64),
                    status_text="Created watershed ROIs: 1 ROI.",
                ),
            ):
                response_widget.create_generated_rois(options)
            response_widget.mark_roi_labels_edited()
            with patch(
                (
                    "twopy.napari.plotting.docks.save_actions."
                    "analyze_recording_responses"
                ),
                return_value=SimpleNamespace(
                    output_path=analysis_path,
                    grouped_responses=_tiny_grouped_responses(),
                    response_summary_trials_csv_path=None,
                    response_summary_grouped_csv_path=None,
                ),
            ):
                response_widget.save_analysis_and_rois()

            self.assertEqual(
                load_roi_generation_metadata(root / "rois.h5"),
                {
                    "schema_version": 1,
                    "roi_mode": "watershed",
                    "watershed_min_pixels": 15,
                    "watershed_smoothing_sigma": 0.75,
                    "edited_after_generation": True,
                },
            )
            response_widget.shutdown()

    def test_save_analysis_button_syncs_cached_outputs_in_background(self) -> None:
        """Confirm cached saves publish changed outputs without blocking save.

        Inputs: local cached recording with a source publish destination.
        Outputs: background sync copies converted, ROI, and analysis files to
            publish path.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            local_dir = root / "cache" / "fly" / "stim" / "2023" / "10_17"
            publish_dir = source_dir / "twopy"
            local_dir.mkdir(parents=True)
            _write_source_recording_shape(source_dir)
            recording_path = _write_converted_recording(
                local_dir,
                source_session_dir=source_dir,
            )
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                "data_paths:\n"
                f"  - {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {root / 'cache'}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            self._activate_test_config(config_path)
            viewer = _FakeViewer()
            original_cwd = Path.cwd()
            response_widget: Any | None = None
            try:
                chdir(root)
                opened = open_recording_in_napari(recording_path, viewer=viewer)
                viewer.labels[0].data = np.array([[1, 0], [0, 0]], dtype=np.int64)
                response_widget = cast(Any, opened.response_plot_widget)
                response_widget._response_map_data = _tiny_response_map_data()
                analysis_path = local_dir / "analysis_outputs.h5"

                def fake_analyze(*_args: object, **_kwargs: object) -> object:
                    analysis_path.write_text("analysis", encoding="utf-8")
                    return SimpleNamespace(
                        output_path=analysis_path,
                        grouped_responses=_tiny_grouped_responses(),
                        response_summary_trials_csv_path=None,
                        response_summary_grouped_csv_path=None,
                    )

                with patch(
                    (
                        "twopy.napari.plotting.docks.save_actions."
                        "analyze_recording_responses"
                    ),
                    side_effect=fake_analyze,
                ):
                    response_widget.save_analysis_and_rois()

                futures = tuple(response_widget._sync_futures)
                self.assertEqual(len(futures), 1)
                future = futures[0]
                future.result(timeout=2)
                response_widget._collect_finished_sync()
            finally:
                chdir(original_cwd)
                if response_widget is not None:
                    response_widget.shutdown()

            self.assertTrue((publish_dir / "rois.h5").is_file())
            self.assertTrue((publish_dir / "recording_data.h5").is_file())
            self.assertTrue((publish_dir / "aligned_movie.h5").is_file())
            self.assertEqual(
                (publish_dir / "analysis_outputs.h5").read_text(encoding="utf-8"),
                "analysis",
            )
            self.assertTrue((publish_dir / "response_heatmaps.h5").is_file())

    def test_reload_saved_analysis_reloads_saved_rois(self) -> None:
        """Confirm Reload saved analysis replaces the editable ROI Labels layer.

        Inputs: saved analysis output, saved ROI HDF5, and edited in-memory
        Labels pixels.
        Outputs: reloading restores the saved ROI labels from disk.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            save_analysis_outputs(
                root / "analysis_outputs.h5",
                grouped_responses=_tiny_grouped_responses(),
            )
            save_roi_set(
                make_roi_set(
                    np.array([[[True, False], [False, False]]], dtype=np.bool_),
                    labels=("roi_0001",),
                ),
                root / "rois.h5",
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            viewer.labels[0].data = np.array([[0, 0], [0, 2]], dtype=np.int64)
            response_widget = cast(Any, opened.response_plot_widget)

            response_widget.reload()

            np.testing.assert_array_equal(
                roi_label_image_from_layer(viewer.labels[0]),
                np.array([[1, 0], [0, 0]], dtype=np.int64),
            )
            response_widget.shutdown()

    def test_reload_saved_analysis_restores_roi_generation_settings(self) -> None:
        """Confirm Reload saved analysis restores saved ROI-generation controls.

        Inputs: saved analysis output and saved watershed ROI metadata.
        Outputs: the ROIs tab shows watershed mode with the saved min-pixel and
        smoothing settings after reload.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            save_analysis_outputs(
                root / "analysis_outputs.h5",
                grouped_responses=_tiny_grouped_responses(),
            )
            save_roi_set(
                make_roi_set(
                    np.array([[[True, False], [False, False]]], dtype=np.bool_),
                    labels=("roi_0001",),
                ),
                root / "rois.h5",
                generation_metadata={
                    "roi_mode": "watershed",
                    "watershed_min_pixels": 15,
                    "watershed_smoothing_sigma": 0.75,
                    "edited_after_generation": True,
                },
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)

            response_widget.reload()
            options = response_widget._roi_generation_widget.options()

            self.assertEqual(options.roi_mode, "watershed")
            self.assertEqual(options.watershed_min_pixels, 15)
            self.assertEqual(options.watershed_smoothing_sigma, 0.75)
            self.assertTrue(response_widget._roi_generation_edited_after_generation)
            self.assertEqual(
                response_widget._roi_generation_widget._status.text(),
                "Saved ROI masks were edited after generation.",
            )
            response_widget.shutdown()

    def test_open_recording_restores_roi_generation_settings_without_analysis(
        self,
    ) -> None:
        """Confirm ROI metadata loads even when saved analysis is absent.

        Inputs: a recording with ``rois.h5`` generation metadata but no saved
        analysis output.
        Outputs: opening the ROI file restores ROIs-tab settings from the ROI
        file instead of requiring analysis outputs to exist first.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            save_roi_set(
                make_roi_set(
                    np.array([[[True, False], [False, False]]], dtype=np.bool_),
                    labels=("roi_0001",),
                ),
                root / "rois.h5",
                generation_metadata={
                    "roi_mode": "watershed",
                    "watershed_min_pixels": 15,
                    "watershed_smoothing_sigma": 0.75,
                },
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(
                recording_path,
                viewer=viewer,
                roi_set=root / "rois.h5",
            )
            response_widget = cast(Any, opened.response_plot_widget)
            options = response_widget._roi_generation_widget.options()

            self.assertEqual(options.roi_mode, "watershed")
            self.assertEqual(options.watershed_min_pixels, 15)
            self.assertEqual(options.watershed_smoothing_sigma, 0.75)
            self.assertEqual(
                {
                    checkbox.text()
                    for checkbox in response_widget.options_widget().findChildren(
                        QCheckBox
                    )
                    if checkbox.text().startswith("roi_")
                },
                {"roi_0001"},
            )
            response_widget.shutdown()

    def test_processing_option_changes_request_preview_update(self) -> None:
        """Confirm processing controls trigger debounced preview recompute.

        Inputs: loaded recording, active ROI Labels layer, and a processing
        settings change.
        Outputs: the live response controller receives an update request.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            requests: list[str] = []
            response_widget._live_controller.request_update = lambda: requests.append(
                "requested"
            )

            response_widget._set_response_processing_options(
                ResponseProcessingOptions(),
            )

            self.assertEqual(requests, ["requested"])

    def test_normalization_option_changes_request_preview_update(self) -> None:
        """Confirm normalization controls update the shared processing options.

        Inputs: loaded recording, active ROI Labels layer, and normalization
        settings.
        Outputs: the live response controller receives one update request and
        processing controls preserve the selected normalization.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            requests: list[str] = []
            response_widget._live_controller.request_update = lambda: requests.append(
                "requested"
            )
            options = NormalizationOptions(method="epoch_peak", epoch_number=1)

            response_widget._set_normalization_options(options)

            self.assertEqual(requests, ["requested"])
            self.assertEqual(
                response_widget._response_processing_options.normalization,
                options,
            )
            self.assertEqual(
                response_widget._processing_options_widget.options().normalization,
                options,
            )

    def test_delta_f_over_f_option_changes_request_preview_update(self) -> None:
        """Confirm dF/F controls trigger debounced preview recompute.

        Inputs: loaded recording, active ROI Labels layer, and a dF/F settings
        change.
        Outputs: the live response controller receives an update request.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            requests: list[str] = []
            response_widget._live_controller.request_update = lambda: requests.append(
                "requested"
            )

            response_widget._set_delta_f_over_f_options(
                DeltaFOverFOptions(
                    background_method="none",
                    baseline_sample_seconds=None,
                    fit_mode="direct_bounded_tau_and_log_amplitude",
                    apply_motion_mask=False,
                ),
            )

            self.assertEqual(requests, ["requested"])

    def _assert_request_workflow_kwargs(
        self,
        kwargs: Any,
        *,
        processing_options: ResponseProcessingOptions,
    ) -> None:
        """Assert request options passed to core response workflow kwargs."""
        self.assertEqual(kwargs["baseline_mode"], "no_baseline_epoch")
        self.assertEqual(kwargs["baseline_epoch_number"], 2)
        self.assertEqual(kwargs["baseline_epoch_name"], "Odor")
        self.assertEqual(kwargs["background_method"], "none")
        self.assertIsNone(kwargs["baseline_sample_seconds"])
        self.assertEqual(kwargs["fit_mode"], "log_linear")
        self.assertFalse(kwargs["apply_motion_mask"])
        self.assertEqual(kwargs["response_pre_window_seconds"], 0.25)
        self.assertEqual(kwargs["response_post_window_seconds"], 0.75)
        self.assertIs(kwargs["response_processing_options"], processing_options)


if __name__ == "__main__":
    unittest.main()
