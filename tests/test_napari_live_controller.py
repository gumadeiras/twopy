"""Napari live response controller tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from tests.napari_support import (
    Any,
    BackgroundCorrectedRoiTraces,
    Callable,
    CancelledError,
    DeltaFOverFOptions,
    Event,
    LiveResponseAnalysisCache,
    LiveResponseController,
    NapariAdapterTestCase,
    Path,
    QApplication,
    QCloseEvent,
    ResponsePlotData,
    ResponseProcessingOptions,
    SimpleNamespace,
    _FakeLabelEvents,
    _FakeLayer,
    _FakePlotReceiver,
    _ProcessingEchoPlotReceiver,
    _tiny_grouped_responses,
    _tiny_response_plot_data,
    _wait_for_live_response_job,
    _write_converted_recording,
    cast,
    create_response_plot_widget,
    format_output_folder,
    format_twopy_h5_output,
    load_converted_recording,
    np,
    patch,
    recording_display_summary,
    response_analysis_request_from_label_image,
    sleep,
    temporary_directory,
    unittest,
    write_converted_recording_files,
)

from twopy.napari.display_paths import microscope_display_lines


class NapariLiveControllerTest(NapariAdapterTestCase):
    """Napari live response controller tests."""

    def test_response_widget_close_shuts_down_live_controller(self) -> None:
        """Confirm closing the response widget releases live-update resources.

        Inputs: a response plot widget with its live controller patched.
        Outputs: Qt close handling calls the controller shutdown path.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        shutdowns: list[str] = []
        response_widget._live_controller.shutdown = lambda: shutdowns.append("shutdown")

        response_widget.closeEvent(QCloseEvent())

        self.assertEqual(shutdowns, ["shutdown"])
        self.assertTrue(response_widget._is_shutdown)

    def test_response_display_paths_use_recording_identity(self) -> None:
        """Confirm response widgets show compact recording and output paths.

        Inputs: converted recording whose source path follows the lab date
        layout.
        Outputs: parsed root, genotype, stimulus, recording time, and
        twopy-relative output paths.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source_dir = (
                root / "data" / "gh146" / "combo_stim" / "2025" / "12_21" / "17_42_22"
            )
            recording_path = _write_converted_recording(
                root,
                source_session_dir=source_dir,
            )
            recording = load_converted_recording(recording_path)

            summary = recording_display_summary(recording)

            self.assertEqual(summary.root, root / "data")
            self.assertEqual(summary.genotype, "gh146")
            self.assertEqual(summary.stimulus, "combo_stim")
            self.assertEqual(summary.recording, "2025-12-21 17:42:22")
            self.assertEqual(
                format_twopy_h5_output(recording.path.parent / "rois.h5"),
                "./twopy/rois.h5",
            )
            self.assertEqual(
                format_output_folder(
                    recording.path.parent / "exports" / "plots" / "plot.png",
                    recording,
                ),
                "./twopy/exports/plots",
            )

    def test_metadata_tab_shows_hemisphere(self) -> None:
        """Confirm Metadata-tab microscope details include fly hemisphere.

        Inputs: converted recording with run-level hemisphere metadata.
        Outputs: compact microscope summary lines include that hemisphere.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = write_converted_recording_files(
                root,
                acquisition_metadata={"acq.frameRate": 10.0},
                run_metadata={"rig_name": "TestRig", "hemisphere": "left"},
            )
            recording = load_converted_recording(recording_path)

            self.assertIn("Hemisphere: left", microscope_display_lines(recording))

    def test_live_response_controller_updates_after_paint_event(self) -> None:
        """Confirm committed Labels painting triggers a response plot refresh.

        Inputs: selected recording, fake Labels paint emitter, and patched
        response calculation.
        Outputs: the plot receiver gets new plot data after an event.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(_write_converted_recording(root))
            events = _FakeLabelEvents()
            layer = _FakeLayer(
                name="rois",
                data=np.array([[1, 0], [0, 0]], dtype=np.int64),
                options={},
                events=events,
            )
            receiver = _FakePlotReceiver()
            controller = LiveResponseController(
                receiver,
                debounce_ms=0,
                run_async=False,
            )
            controller.set_context(recording, layer)

            with patch(
                "twopy.napari.interactive.compute_response_preview",
                return_value=_tiny_response_plot_data(),
            ):
                events.paint.emit()

            self.assertIsNotNone(receiver.plot_data)
            self.assertIsNone(receiver.status)

    def test_live_response_controller_defaults_to_short_update_delay(self) -> None:
        """Confirm live ROI updates use a short drawing debounce.

        Inputs: default live controller.
        Outputs: debounce and poll intervals stay low enough for responsive
        post-drawing plot updates.
        """
        _ = QApplication.instance() or QApplication([])
        receiver = _FakePlotReceiver()
        controller = LiveResponseController(receiver)

        self.assertEqual(controller._debounce_ms, 200)
        self.assertEqual(controller._poll_timer.interval(), 30)
        controller.shutdown()

    def test_live_response_controller_does_not_loop_on_echoed_options(self) -> None:
        """Confirm result option hydration does not schedule another recompute.

        Inputs: async live controller whose receiver echoes result processing
        options, matching the real response plot widget.
        Outputs: applying the finished result does not mark the job stale and
        start another background computation.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(_write_converted_recording(root))
            layer = _FakeLayer(
                name="rois",
                data=np.array([[1, 0], [0, 0]], dtype=np.int64),
                options={},
                events=_FakeLabelEvents(),
            )
            receiver = _ProcessingEchoPlotReceiver()
            controller = LiveResponseController(receiver, debounce_ms=0)
            receiver.controller = controller
            controller.set_context(recording, layer)
            plot_data = ResponsePlotData(
                source_path=None,
                epochs=_tiny_response_plot_data().epochs,
                response_processing_options=ResponseProcessingOptions(),
            )

            with patch.object(
                controller._analysis_cache,
                "compute_response_preview",
                return_value=plot_data,
            ) as compute:
                controller.request_update()
                _wait_for_live_response_job(controller)
                controller._collect_finished_job()

            self.assertIs(receiver.plot_data, plot_data)
            self.assertIsNone(controller._future)
            compute.assert_called_once()
            controller.shutdown()

    def test_live_response_controller_cancels_stale_worker_job(self) -> None:
        """Confirm newer ROI edits cancel obsolete background analysis.

        Inputs: async live controller with a deliberately slow first job.
        Outputs: the first job observes cancellation and the next job publishes
        the latest plot data.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(_write_converted_recording(root))
            layer = _FakeLayer(
                name="rois",
                data=np.array([[1, 0], [0, 0]], dtype=np.int64),
                options={},
                events=_FakeLabelEvents(),
            )
            receiver = _FakePlotReceiver()
            controller = LiveResponseController(receiver, debounce_ms=0)
            controller.set_context(recording, layer)
            first_started = Event()
            first_cancelled = Event()
            calls = 0
            plot_data = _tiny_response_plot_data()

            def compute(*args: object, **kwargs: object) -> ResponsePlotData:
                del args
                nonlocal calls
                calls += 1
                check_cancelled = cast(Callable[[], None], kwargs["check_cancelled"])
                if calls == 1:
                    first_started.set()
                    while True:
                        try:
                            check_cancelled()
                        except CancelledError:
                            first_cancelled.set()
                            raise
                        sleep(0.01)
                return plot_data

            with patch.object(
                controller._analysis_cache,
                "compute_response_preview",
                side_effect=compute,
            ):
                controller.request_update()
                self.assertTrue(first_started.wait(timeout=2.0))
                controller.request_update()
                self.assertTrue(first_cancelled.wait(timeout=2.0))
                _wait_for_live_response_job(controller)
                controller._collect_finished_job()
                _wait_for_live_response_job(controller)
                controller._collect_finished_job()

            self.assertIs(receiver.plot_data, plot_data)
            self.assertEqual(calls, 2)
            controller.shutdown()

    def test_live_response_controller_replaces_cache_on_context_change(self) -> None:
        """Confirm context changes do not clear a cache still used by a worker.

        Inputs: live controller with an active worker using a blocking fake cache.
        Outputs: setting a new context swaps in a fresh cache without mutating
        the cache object captured by the active worker.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(_write_converted_recording(root))
            layer = _FakeLayer(
                name="rois",
                data=np.array([[1, 0], [0, 0]], dtype=np.int64),
                options={},
                events=_FakeLabelEvents(),
            )
            controller = LiveResponseController(_FakePlotReceiver(), debounce_ms=0)
            controller.set_context(recording, layer)
            started = Event()
            proceed = Event()
            testcase = self

            class BlockingCache:
                def __init__(self) -> None:
                    self.clear_calls = 0

                def clear(self) -> None:
                    self.clear_calls += 1

                def compute_response_preview(
                    self,
                    *args: object,
                    **kwargs: object,
                ) -> ResponsePlotData:
                    del args
                    started.set()
                    testcase.assertTrue(proceed.wait(timeout=2.0))
                    check_cancelled = cast(
                        Callable[[], None], kwargs["check_cancelled"]
                    )
                    check_cancelled()
                    return _tiny_response_plot_data()

            old_cache = BlockingCache()
            controller._analysis_cache = cast(Any, old_cache)

            controller.request_update()
            self.assertTrue(started.wait(timeout=2.0))
            controller.set_context(recording, layer)
            self.assertIsNot(controller._analysis_cache, old_cache)
            self.assertEqual(old_cache.clear_calls, 0)

            proceed.set()
            _wait_for_live_response_job(controller)
            controller._collect_finished_job()
            controller.shutdown()

    def test_live_response_cache_reuses_unchanged_roi_traces(self) -> None:
        """Confirm live cache extracts only changed ROI masks.

        Inputs: two ROI labels followed by an edit to only ROI 2.
        Outputs: first run extracts both traces, second run extracts none, and
        the edited run extracts only ROI 2.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(_write_converted_recording(root))
            cache = LiveResponseAnalysisCache()
            first_labels = np.array([[1, 0], [0, 2]], dtype=np.int64)
            edited_labels = np.array([[1, 0], [2, 2]], dtype=np.int64)
            extracted: list[tuple[str, ...]] = []

            def extract(
                _recording: object,
                roi_set: object,
                **kwargs: object,
            ) -> BackgroundCorrectedRoiTraces:
                roi_labels = cast(Any, roi_set).labels
                extracted.append(roi_labels)
                start_frame = cast(int, kwargs["start_frame"])
                stop_frame = cast(int, kwargs["stop_frame"])
                frame_count = stop_frame - start_frame
                roi_count = len(roi_labels)
                values = np.ones((frame_count, roi_count), dtype=np.float64)
                return BackgroundCorrectedRoiTraces(
                    raw_values=values,
                    background_values=np.zeros_like(values),
                    corrected_values=values,
                    labels=roi_labels,
                    start_frame=start_frame,
                    stop_frame=stop_frame,
                    statistic="mean",
                    method="movie_global_percentile",
                    metadata={"method": "movie_global_percentile"},
                )

            computation = SimpleNamespace(
                grouped_responses=_tiny_grouped_responses(),
                response_processing_options=ResponseProcessingOptions(),
                correlation_scores=None,
            )
            context = SimpleNamespace(trace_start=0, trace_stop=2)
            with (
                patch(
                    "twopy.napari.live_analysis.resolve_response_analysis_context",
                    return_value=context,
                ),
                patch(
                    "twopy.napari.live_analysis."
                    "extract_background_corrected_roi_traces",
                    side_effect=extract,
                ),
                patch(
                    "twopy.napari.live_analysis."
                    "compute_recording_responses_from_traces",
                    return_value=computation,
                ),
                patch(
                    "twopy.napari.live_analysis.response_plot_data_from_computation",
                    return_value=_tiny_response_plot_data(),
                ),
            ):
                first_request = response_analysis_request_from_label_image(
                    recording,
                    first_labels,
                )
                edited_request = response_analysis_request_from_label_image(
                    recording,
                    edited_labels,
                )
                cache.compute_response_preview(first_request)
                cache.compute_response_preview(first_request)
                cache.compute_response_preview(edited_request)

            self.assertEqual(
                extracted,
                [("roi_0001", "roi_0002"), ("roi_0002",)],
            )

    def test_live_response_cache_recomputes_roi_y_stripe_after_deletion(self) -> None:
        """Confirm ROI-y-stripe traces are invalidated when any ROI is deleted.

        Inputs: two ROI labels using per-ROI y-stripe background, then deletion
        of ROI 2.
        Outputs: ROI 1 is recomputed because its background excludes all ROIs.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(_write_converted_recording(root))
            cache = LiveResponseAnalysisCache()
            first_labels = np.array([[1, 0], [0, 2]], dtype=np.int64)
            deleted_labels = np.array([[1, 0], [0, 0]], dtype=np.int64)
            extracted: list[tuple[str, ...]] = []

            def extract(
                _recording: object,
                roi_set: object,
                **kwargs: object,
            ) -> BackgroundCorrectedRoiTraces:
                roi_labels = cast(Any, roi_set).labels
                extracted.append(roi_labels)
                start_frame = cast(int, kwargs["start_frame"])
                stop_frame = cast(int, kwargs["stop_frame"])
                frame_count = stop_frame - start_frame
                roi_count = len(roi_labels)
                values = np.ones((frame_count, roi_count), dtype=np.float64)
                return BackgroundCorrectedRoiTraces(
                    raw_values=values,
                    background_values=np.zeros_like(values),
                    corrected_values=values,
                    labels=roi_labels,
                    start_frame=start_frame,
                    stop_frame=stop_frame,
                    statistic="mean",
                    method="roi_y_stripe_percentile",
                    metadata={"method": "roi_y_stripe_percentile"},
                )

            computation = SimpleNamespace(
                grouped_responses=_tiny_grouped_responses(),
                response_processing_options=ResponseProcessingOptions(),
                correlation_scores=None,
            )
            context = SimpleNamespace(trace_start=0, trace_stop=2)
            options = DeltaFOverFOptions(
                background_method="roi_y_stripe_percentile",
            )
            with (
                patch(
                    "twopy.napari.live_analysis.resolve_response_analysis_context",
                    return_value=context,
                ),
                patch(
                    "twopy.napari.live_analysis."
                    "extract_background_corrected_roi_traces",
                    side_effect=extract,
                ),
                patch(
                    "twopy.napari.live_analysis."
                    "compute_recording_responses_from_traces",
                    return_value=computation,
                ),
                patch(
                    "twopy.napari.live_analysis.response_plot_data_from_computation",
                    return_value=_tiny_response_plot_data(),
                ),
            ):
                first_request = response_analysis_request_from_label_image(
                    recording,
                    first_labels,
                    delta_f_over_f_options=options,
                )
                deleted_request = response_analysis_request_from_label_image(
                    recording,
                    deleted_labels,
                    delta_f_over_f_options=options,
                )
                cache.compute_response_preview(first_request)
                cache.compute_response_preview(deleted_request)

            self.assertEqual(
                extracted,
                [("roi_0001", "roi_0002"), ("roi_0001",)],
            )

    def test_live_response_controller_shutdown_disconnects_events(self) -> None:
        """Confirm shutdown breaks event callbacks and ignores later updates.

        Inputs: selected recording, fake Labels events, and a synchronous
        controller.
        Outputs: event emitters no longer hold callbacks or run analysis after
        shutdown.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(_write_converted_recording(root))
            events = _FakeLabelEvents()
            layer = _FakeLayer(
                name="rois",
                data=np.array([[1, 0], [0, 0]], dtype=np.int64),
                options={},
                events=events,
            )
            receiver = _FakePlotReceiver()
            controller = LiveResponseController(
                receiver,
                debounce_ms=0,
                run_async=False,
            )
            controller.set_context(recording, layer)

            controller.shutdown()
            with patch(
                "twopy.napari.interactive.compute_response_preview",
            ) as compute:
                events.paint.emit()
                controller.request_update()

            compute.assert_not_called()
            self.assertEqual(events.paint._callbacks, [])
            self.assertEqual(events.data._callbacks, [])

    def test_live_response_controller_ignores_labels_update_event(self) -> None:
        """Confirm display refresh events do not recompute responses.

        Inputs: selected recording and fake Labels events.
        Outputs: ``labels_update`` emits no plot data because napari can emit it
        during mouse movement without a committed ROI edit.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(_write_converted_recording(root))
            events = _FakeLabelEvents()
            layer = _FakeLayer(
                name="rois",
                data=np.array([[1, 0], [0, 0]], dtype=np.int64),
                options={},
                events=events,
            )
            receiver = _FakePlotReceiver()
            controller = LiveResponseController(
                receiver,
                debounce_ms=0,
                run_async=False,
            )
            controller.set_context(recording, layer)

            with patch(
                "twopy.napari.interactive.compute_response_preview",
                return_value=_tiny_response_plot_data(),
            ) as compute:
                events.labels_update.emit()

            compute.assert_not_called()
            self.assertIsNone(receiver.plot_data)


if __name__ == "__main__":
    unittest.main()
