"""Napari response plot widget tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from qtpy.QtGui import QPalette
from tests.napari_support import (
    Any,
    EpochFrameWindow,
    EpochPlotWidget,
    EpochResponsePlotData,
    FrameWindow,
    Future,
    NapariAdapterTestCase,
    Path,
    QApplication,
    QColor,
    QLabel,
    Qt,
    QWidget,
    ResponseMapData,
    ResponseMapOptions,
    ResponsePlotData,
    RoiDeltaFOverF,
    SimpleNamespace,
    _FakeColorLayer,
    _FakeLayer,
    _tiny_response_map_data,
    _tiny_response_plot_data,
    _write_converted_recording,
    cast,
    create_response_plot_widget,
    epoch_plot_panel,
    global_time_bounds,
    global_value_bounds,
    group_delta_f_over_f_by_epoch,
    group_matching_roi,
    load_converted_recording,
    mean_image_roi_overlay_pixmap,
    np,
    patch,
    replace,
    response_plot_data_from_grouped,
    roi_colors_from_layer,
    temporary_directory,
    unittest,
)

from twopy.napari.group_matching_style import style_group_matching_panel


class NapariPlotWidgetTest(NapariAdapterTestCase):
    """Napari response plot widget tests."""

    def test_epoch_visibility_toggle_is_idempotent_by_row_index(self) -> None:
        """Confirm hiding and showing an epoch restores the same row.

        Inputs: two epoch plots and direct row-index toggles.
        Outputs: the selected epoch set returns to the original row indices.
        """
        _ = QApplication.instance() or QApplication([])
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((4, 1), dtype=np.float64),
            baseline=np.ones((4, 1), dtype=np.float64),
            values=np.arange(4, dtype=np.float64).reshape(4, 1),
            labels=("roi_1",),
            start_frame=0,
            stop_frame=4,
            tau=0.0,
            amplitudes=np.ones(1, dtype=np.float64),
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 1), dtype=np.float64),
            metadata={"method": "test"},
        )
        plot_data = response_plot_data_from_grouped(
            group_delta_f_over_f_by_epoch(
                dff,
                (
                    EpochFrameWindow(FrameWindow(0, 0, 2, "first"), 1, "First"),
                    EpochFrameWindow(FrameWindow(1, 2, 4, "second"), 2, "Second"),
                ),
                data_rate_hz=1.0,
            ),
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(plot_data, reset_axes=True)

        response_widget._set_epoch_visibility(0, False)
        response_widget._set_epoch_visibility(0, True)

        self.assertEqual(response_widget._visible_epoch_indices(), (0, 1))

    def test_duplicate_epoch_metadata_does_not_collide_in_visibility(self) -> None:
        """Confirm epoch rows do not share GUI state when metadata matches.

        Inputs: two plot rows with the same epoch number and name.
        Outputs: hiding the second row leaves only the first row visible.
        """
        _ = QApplication.instance() or QApplication([])
        time_seconds = np.array([0.0, 1.0], dtype=np.float64)
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Same",
                    epoch_number=1,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[0.0, 1.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
                EpochResponsePlotData(
                    epoch_name="Same",
                    epoch_number=1,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[2.0, 3.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
            ),
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(plot_data, reset_axes=True)

        response_widget._set_epoch_visibility(1, False)

        self.assertEqual(response_widget._visible_epoch_indices(), (0,))
        self.assertEqual(tuple(response_widget._plot_area.epoch_plot_widgets), (0, 1))

    def test_hidden_epoch_plot_panel_is_hidden_after_layout_refresh(self) -> None:
        """Confirm hidden epoch plots do not remain visible as stale Qt panels.

        Inputs: one gray epoch hidden by default plus two visible epochs.
        Outputs: hiding the last visible row explicitly hides that cached panel.
        """
        _ = QApplication.instance() or QApplication([])
        time_seconds = np.array([0.0, 1.0], dtype=np.float64)
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Gray Interleave",
                    epoch_number=1,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[0.0, 1.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
                EpochResponsePlotData(
                    epoch_name="Odor A",
                    epoch_number=2,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[2.0, 3.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
                EpochResponsePlotData(
                    epoch_name="Odor B",
                    epoch_number=3,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[4.0, 5.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
            ),
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(plot_data, reset_axes=True)

        response_widget._set_epoch_visibility(2, False)

        self.assertEqual(response_widget._visible_epoch_indices(), (1,))
        self.assertTrue(response_widget._plot_area.epoch_plot_panels[0].isHidden())
        self.assertFalse(response_widget._plot_area.epoch_plot_panels[1].isHidden())
        self.assertTrue(response_widget._plot_area.epoch_plot_panels[2].isHidden())

    def test_epoch_visibility_defaults_reset_when_epoch_identity_changes(self) -> None:
        """Confirm stale row visibility does not override baseline defaults.

        Inputs: first plot data with visible epoch row zero, then different plot
        data whose row zero is gray interleave.
        Outputs: row zero is hidden after the epoch identity changes.
        """
        _ = QApplication.instance() or QApplication([])
        time_seconds = np.array([0.0, 1.0], dtype=np.float64)
        first = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Odor",
                    epoch_number=1,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[0.0, 1.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
            ),
        )
        second = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Gray Interleave",
                    epoch_number=1,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[0.0, 1.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
                EpochResponsePlotData(
                    epoch_name="Odor",
                    epoch_number=2,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[2.0, 3.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
            ),
        )
        response_widget = cast(Any, create_response_plot_widget(None))

        response_widget.set_response_plot_data(first, reset_axes=True)
        response_widget.set_response_plot_data(second, reset_axes=True)

        self.assertEqual(response_widget._epoch_visibility, {0: False, 1: True})
        self.assertEqual(response_widget._visible_epoch_indices(), (1,))

    def test_epoch_visibility_preserves_user_choice_for_same_epochs(self) -> None:
        """Confirm live refreshes keep visibility when epoch identity is stable.

        Inputs: two plot-data objects with the same epoch rows and one hidden
        row selected by the user.
        Outputs: the hidden row stays hidden after the refreshed plot data loads.
        """
        _ = QApplication.instance() or QApplication([])
        time_seconds = np.array([0.0, 1.0], dtype=np.float64)
        first = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Odor A",
                    epoch_number=1,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[0.0, 1.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
                EpochResponsePlotData(
                    epoch_name="Odor B",
                    epoch_number=2,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[2.0, 3.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
            ),
        )
        second = replace(
            first,
            epochs=(
                replace(
                    first.epochs[0],
                    mean_values=np.array([[4.0, 5.0]], dtype=np.float64),
                ),
                replace(
                    first.epochs[1],
                    mean_values=np.array([[6.0, 7.0]], dtype=np.float64),
                ),
            ),
        )
        response_widget = cast(Any, create_response_plot_widget(None))

        response_widget.set_response_plot_data(first, reset_axes=True)
        response_widget._set_epoch_visibility(1, False)
        response_widget.set_response_plot_data(second, reset_axes=True)

        self.assertEqual(response_widget._visible_epoch_indices(), (0,))

    def test_epoch_visibility_reuses_cached_heatmap_widgets(self) -> None:
        """Confirm epoch toggles do not rebuild cached heatmap images.

        Inputs: two cached response plots and two cached heatmaps.
        Outputs: hiding one epoch reuses existing widgets instead of refreshing
        every heatmap from the computed map arrays.
        """
        _ = QApplication.instance() or QApplication([])
        time_seconds = np.array([0.0, 1.0], dtype=np.float64)
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Odor A",
                    epoch_number=1,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[0.0, 1.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
                EpochResponsePlotData(
                    epoch_name="Odor B",
                    epoch_number=2,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[2.0, 3.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
            ),
        )
        base_map_data = _tiny_response_map_data()
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(plot_data, reset_axes=True)
        response_widget._response_map_data = replace(
            base_map_data,
            epochs=(
                replace(base_map_data.epochs[0], epoch_name="Odor A", epoch_number=1),
                replace(base_map_data.epochs[0], epoch_name="Odor B", epoch_number=2),
            ),
        )
        response_widget._render_response_maps()
        refresh_calls: list[str] = []
        response_widget._response_map_area.ensure_epoch_map_cache = lambda **_kwargs: (
            refresh_calls.append("refresh")
        )

        response_widget._set_epoch_visibility(1, False)

        self.assertEqual(refresh_calls, [])
        self.assertEqual(response_widget._visible_epoch_indices(), (0,))
        self.assertFalse(
            response_widget._response_map_area.epoch_map_panels[0].isHidden()
        )
        self.assertTrue(
            response_widget._response_map_area.epoch_map_panels[1].isHidden()
        )

    def test_heatmap_epochs_match_visible_plot_epochs_by_identity(self) -> None:
        """Confirm omitted baseline heatmaps do not shift visible map rows.

        Inputs: plot data with a hidden gray epoch and two visible odor epochs,
        plus heatmap data containing only the odor epochs.
        Outputs: both odor heatmaps render even though their map indices differ
        from the response-plot row indices.
        """
        _ = QApplication.instance() or QApplication([])
        time_seconds = np.array([0.0, 1.0], dtype=np.float64)
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Gray Interleave",
                    epoch_number=1,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[0.0, 1.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
                EpochResponsePlotData(
                    epoch_name="Odor A",
                    epoch_number=2,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[2.0, 3.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
                EpochResponsePlotData(
                    epoch_name="Odor B",
                    epoch_number=3,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[4.0, 5.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
            ),
        )
        base_map_data = _tiny_response_map_data()
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(plot_data, reset_axes=True)
        response_widget._response_map_data = replace(
            base_map_data,
            epochs=(
                replace(base_map_data.epochs[0], epoch_name="Odor A", epoch_number=2),
                replace(base_map_data.epochs[0], epoch_name="Odor B", epoch_number=3),
            ),
        )

        response_widget._render_response_maps()

        self.assertEqual(response_widget._visible_epoch_indices(), (1, 2))
        self.assertEqual(response_widget._visible_response_map_epoch_indices(), (0, 1))
        self.assertEqual(
            tuple(response_widget._response_map_area.epoch_map_panels),
            (0, 1),
        )
        self.assertFalse(
            response_widget._response_map_area.epoch_map_panels[0].isHidden()
        )
        self.assertFalse(
            response_widget._response_map_area.epoch_map_panels[1].isHidden()
        )

    def test_roi_visibility_toggle_is_idempotent_by_row_index(self) -> None:
        """Confirm ROI visibility does not depend on unique display labels.

        Inputs: one plot data object with duplicate ROI names.
        Outputs: hiding and showing the second ROI restores both ROI indices.
        """
        _ = QApplication.instance() or QApplication([])
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((4, 2), dtype=np.float64),
            baseline=np.ones((4, 2), dtype=np.float64),
            values=np.arange(8, dtype=np.float64).reshape(4, 2),
            labels=("same", "same"),
            start_frame=0,
            stop_frame=4,
            tau=0.0,
            amplitudes=np.ones(2, dtype=np.float64),
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 2), dtype=np.float64),
            metadata={"method": "test"},
        )
        plot_data = response_plot_data_from_grouped(
            group_delta_f_over_f_by_epoch(
                dff,
                (EpochFrameWindow(FrameWindow(0, 0, 4, "first"), 1, "First"),),
                data_rate_hz=1.0,
            ),
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(plot_data, reset_axes=True)

        response_widget._set_roi_visibility(1, False)
        self.assertEqual(response_widget._visible_roi_indices(), (0,))

        response_widget._set_roi_visibility(1, True)
        self.assertEqual(response_widget._visible_roi_indices(), (0, 1))

    def test_roi_visibility_repaints_without_rebuilding_epoch_layout(self) -> None:
        """Confirm ROI toggles do not hide and reshow every epoch panel.

        Inputs: two ROI traces and one visible epoch plot.
        Outputs: hiding one ROI updates plot state without clearing the layout.
        """
        _ = QApplication.instance() or QApplication([])
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Odor",
                    epoch_number=1,
                    roi_labels=("roi_1", "roi_2"),
                    time_seconds=np.array([0.0, 1.0], dtype=np.float64),
                    mean_values=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
                    sem_values=np.zeros((2, 2), dtype=np.float64),
                ),
            ),
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(plot_data, reset_axes=True)
        clear_calls: list[str] = []
        response_widget._plot_area.clear_layout_preserving_epoch_cache = lambda: (
            clear_calls.append("clear")
        )

        response_widget._set_roi_visibility(1, False)

        self.assertEqual(clear_calls, [])
        self.assertEqual(response_widget._visible_roi_indices(), (0,))
        self.assertFalse(response_widget._plot_area.epoch_plot_panels[0].isHidden())

    def test_response_plot_bounds_use_selected_epochs_and_rois(self) -> None:
        """Confirm plot bounds follow selected ROI and epoch visibility.

        Inputs: two epoch plots and two ROIs with different ranges.
        Outputs: bounds for only the selected epoch and ROI.
        """
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((4, 2), dtype=np.float64),
            baseline=np.ones((4, 2), dtype=np.float64),
            values=np.array(
                [[0.0, 10.0], [1.0, 11.0], [2.0, 20.0], [3.0, 21.0]],
                dtype=np.float64,
            ),
            labels=("roi_1", "roi_2"),
            start_frame=0,
            stop_frame=4,
            tau=0.0,
            amplitudes=np.ones(2, dtype=np.float64),
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 2), dtype=np.float64),
            metadata={"method": "test"},
        )
        grouped = group_delta_f_over_f_by_epoch(
            dff,
            (
                EpochFrameWindow(FrameWindow(0, 0, 2, "gray"), 1, "Gray"),
                EpochFrameWindow(FrameWindow(1, 2, 4, "odor"), 2, "Odor"),
            ),
            data_rate_hz=1.0,
        )
        plot_data = response_plot_data_from_grouped(grouped)

        self.assertEqual(global_time_bounds(plot_data, (1,)), (0.0, 1.0))
        self.assertEqual(
            global_value_bounds(plot_data, (0,), (1,)),
            (1.6, 3.6),
        )

    def test_response_plot_colors_can_match_labels_layer(self) -> None:
        """Confirm plot colors can come from the napari Labels layer.

        Inputs: fake Labels layer with deterministic per-label RGBA values.
        Outputs: Qt colors matching those values.
        """
        colors = roi_colors_from_layer(_FakeColorLayer(), 2)

        self.assertEqual(colors[0].red(), 255)
        self.assertEqual(colors[0].blue(), 0)
        self.assertEqual(colors[1].red(), 0)
        self.assertEqual(colors[1].blue(), 255)

    def test_group_matching_overlay_plot_combines_selected_recording_rois(self) -> None:
        """Confirm group matching can overlay selected ROIs in one plot strip.

        Inputs: two selected ROI responses from different recording paths.
        Outputs: combined plot data has one trace row per recording.
        """
        first_data = _tiny_response_plot_data()
        second_data = _tiny_response_plot_data()
        combined = group_matching_roi._combined_response_plot_data(
            (
                group_matching_roi._SelectedRoiResponse(
                    recording_path=Path("/recordings/first"),
                    roi_label="roi_0001",
                    plot_data=first_data,
                    color=QColor("#1f77b4"),
                ),
                group_matching_roi._SelectedRoiResponse(
                    recording_path=Path("/recordings/second"),
                    roi_label="roi_0002",
                    plot_data=second_data,
                    color=QColor("#d95f02"),
                ),
            ),
        )

        assert combined is not None
        self.assertEqual(len(combined.epochs), len(first_data.epochs))
        self.assertEqual(combined.epochs[0].mean_values.shape[0], 2)
        self.assertEqual(
            combined.epochs[0].roi_labels,
            (
                "first roi_0001",
                "second roi_0002",
            ),
        )

    def test_group_matching_mean_plot_uses_sample_sem(self) -> None:
        """Confirm the ROI matching mean row uses twopy's shared SEM rule.

        Inputs: two overlaid recording traces.
        Outputs: mean trace and sample SEM across the active traces.
        """
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Odor",
                    epoch_number=1,
                    roi_labels=("first", "second"),
                    time_seconds=np.array([0.0, 1.0], dtype=np.float64),
                    mean_values=np.array(
                        [[1.0, 3.0], [3.0, 7.0]],
                        dtype=np.float64,
                    ),
                    sem_values=np.zeros((2, 2), dtype=np.float64),
                ),
            ),
        )

        mean_plot = group_matching_roi._mean_response_plot_data(plot_data)
        epoch = mean_plot.epochs[0]

        np.testing.assert_allclose(epoch.mean_values, np.array([[2.0, 5.0]]))
        np.testing.assert_allclose(epoch.sem_values, np.array([[1.0, 2.0]]))

    def test_group_matching_plot_size_redraws_without_recompute(self) -> None:
        """Confirm display-only plot sizing does not rerun ROI analysis.

        Inputs: an empty ROI assignment view and a patched recompute method.
        Outputs: size changes repaint cached data without asking for new
        selected response data.
        """
        _ = QApplication.instance() or QApplication([])
        view = group_matching_roi.RoiAssignmentView(
            state=SimpleNamespace(loaded_recordings=[]),
            fov_groups={},
            current_rois={},
            on_back=lambda: None,
        )

        with patch.object(
            view,
            "_selected_response_data",
            side_effect=AssertionError("plot size should not recompute responses"),
        ):
            view._set_plot_size(240)

        self.assertEqual(view._plot_size, 240)

    def test_group_matching_style_uses_active_qt_palette(self) -> None:
        """Confirm Group Matching colors follow the current Qt theme.

        Inputs: one widget with a dark palette.
        Outputs: the applied stylesheet uses palette colors instead of fixed
        light-card colors.
        """
        _ = QApplication.instance() or QApplication([])
        widget = QWidget()
        palette = widget.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#202020"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#303030"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#eeeeee"))
        widget.setPalette(palette)

        style_group_matching_panel(widget)

        stylesheet = widget.styleSheet()
        self.assertIn("#202020", stylesheet)
        self.assertIn("#303030", stylesheet)
        self.assertNotIn("#fffefa", stylesheet)

    def test_group_matching_roi_preview_omits_interleave_epochs(self) -> None:
        """Confirm ROI matching response plots omit interleave epochs.

        Inputs: one baseline-like interleave epoch and one odor epoch.
        Outputs: only the odor epoch index remains visible in ROI matching.
        """
        time_seconds = np.array([0.0, 1.0], dtype=np.float64)
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Gray Interleave",
                    epoch_number=1,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[0.0, 1.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
                EpochResponsePlotData(
                    epoch_name="Odor",
                    epoch_number=2,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[2.0, 3.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
            ),
        )

        self.assertEqual(
            group_matching_roi._visible_response_epoch_indices(plot_data),
            (1,),
        )

    def test_group_matching_roi_preview_uses_trace_color(self) -> None:
        """Confirm selected ROI masks match the recording trace color.

        Inputs: one selected ROI mask and one recording trace color.
        Outputs: the rendered ROI preview uses that color for selected pixels.
        """
        _ = QApplication.instance() or QApplication([])
        selected_color = QColor("#1f77b4")

        pixmap = mean_image_roi_overlay_pixmap(
            mean_image_layer=_FakeLayer(
                name="mean",
                data=np.zeros((12, 12), dtype=np.float64),
                options={},
            ),
            roi_labels_layer=_FakeLayer(
                name="rois",
                data=np.full((12, 12), 2, dtype=np.int64),
                options={},
            ),
            roi_label="roi_0002",
            selected_color=selected_color,
            contrast_percentile=100.0,
        )

        assert pixmap is not None
        pixel_color = pixmap.toImage().pixelColor(8, 8)
        self.assertLessEqual(abs(pixel_color.red() - selected_color.red()), 1)
        self.assertLessEqual(abs(pixel_color.green() - selected_color.green()), 1)
        self.assertLessEqual(abs(pixel_color.blue() - selected_color.blue()), 1)

    def test_epoch_plot_widget_uses_compact_height(self) -> None:
        """Confirm response plots trim unused height below the x-axis.

        Inputs: plot-ready data for one epoch.
        Outputs: widget height preserves the square data area without bottom
        slack.
        """
        _ = QApplication.instance() or QApplication([])
        plot_data = _tiny_response_plot_data()

        widget = EpochPlotWidget(
            plot_data.epochs[0],
            show_sem=True,
            roi_indices=(0,),
            roi_colors=(QColor("#ff0000"),),
            time_min=0.0,
            time_max=1.0,
            value_min=0.0,
            value_max=1.0,
            plot_size=480,
        )

        self.assertEqual(widget.sizeHint().width(), 480)
        self.assertEqual(widget.sizeHint().height(), 432)

        widget.update_display(
            show_sem=False,
            roi_indices=(0,),
            roi_colors=(QColor("#ff0000"),),
            time_min=-1.0,
            time_max=2.0,
            value_min=-0.5,
            value_max=1.5,
            plot_size=320,
        )

        self.assertEqual(widget.sizeHint().width(), 320)
        self.assertEqual(widget.sizeHint().height(), 272)

    def test_response_widget_reuses_epoch_plots_for_live_refreshes(self) -> None:
        """Confirm repeated live results update cached epoch widgets in place.

        Inputs: response widget receiving two plot-data objects with the same
        epoch identity.
        Outputs: the existing epoch plot widget is reused with new data.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        first = _tiny_response_plot_data()
        second = ResponsePlotData(
            source_path=None,
            epochs=(
                replace(
                    first.epochs[0],
                    mean_values=np.array([[2.0, 3.0]], dtype=np.float64),
                ),
            ),
        )

        response_widget.set_response_plot_data(first, reset_axes=True)
        cached_plot = response_widget._plot_area.epoch_plot_widgets[0]
        response_widget.set_response_plot_data(second, reset_axes=True)

        self.assertIs(response_widget._plot_area.epoch_plot_widgets[0], cached_plot)
        self.assertIs(cached_plot._data, second.epochs[0])

    def test_response_heatmaps_render_without_roi_plot_data(self) -> None:
        """Confirm heatmaps are available before any ROI responses exist.

        Inputs: a loaded recording with no saved ROI analysis and a mocked
        movie-derived response map.
        Outputs: the heatmap tab renders the map even though plot data is empty.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(Path(temp_dir)),
            )
            map_data = _tiny_response_map_data()
            with patch(
                "twopy.napari.plotting.docks.response_plot_widget."
                "compute_recording_response_maps",
                return_value=map_data,
            ) as compute_maps:
                response_widget = cast(Any, create_response_plot_widget(None))
                response_widget.load_recording(recording)

        compute_maps.assert_called_once()
        self.assertIs(response_widget._response_map_data, map_data)
        self.assertEqual(len(response_widget._response_map_area.epoch_map_widgets), 1)
        self.assertIsNone(response_widget._plot_data)

    def test_response_heatmaps_without_roi_plot_data_omit_interleave_epochs(
        self,
    ) -> None:
        """Confirm standalone heatmaps share the baseline epoch default.

        Inputs: heatmap data with one gray interleave epoch and one odor epoch,
        but no ROI plot data.
        Outputs: only the odor heatmap is shown.
        """
        _ = QApplication.instance() or QApplication([])
        base_map_data = _tiny_response_map_data()
        map_data = replace(
            base_map_data,
            epochs=(
                replace(
                    base_map_data.epochs[0],
                    epoch_name="Gray Interleave",
                    epoch_number=1,
                ),
                replace(
                    base_map_data.epochs[0],
                    epoch_name="Odor",
                    epoch_number=2,
                ),
            ),
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget._response_map_data = map_data

        response_widget._render_response_maps()

        self.assertIsNone(response_widget._plot_data)
        self.assertEqual(response_widget._visible_response_map_epoch_indices(), (1,))
        self.assertTrue(
            response_widget._response_map_area.epoch_map_panels[0].isHidden()
        )
        self.assertFalse(
            response_widget._response_map_area.epoch_map_panels[1].isHidden()
        )

    def test_response_heatmaps_do_not_recompute_for_roi_plot_updates(self) -> None:
        """Confirm ROI response refreshes do not recompute movie heatmaps.

        Inputs: a loaded recording with one cached heatmap, then ROI plot data.
        Outputs: heatmap computation runs only on recording load.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(Path(temp_dir)),
            )
            map_data = _tiny_response_map_data()
            with patch(
                "twopy.napari.plotting.docks.response_plot_widget."
                "compute_recording_response_maps",
                return_value=map_data,
            ) as compute_maps:
                response_widget = cast(Any, create_response_plot_widget(None))
                response_widget.load_recording(recording)
                compute_maps.reset_mock()
                response_widget.set_response_plot_data(
                    _tiny_response_plot_data(),
                    reset_axes=True,
                )

        compute_maps.assert_not_called()
        self.assertIs(response_widget._response_map_data, map_data)

    def test_response_map_shared_limits_updates_display_without_recompute(self) -> None:
        """Confirm shared heatmap limits are display-only.

        Inputs: a loaded recording with cached heatmaps, then the shared-limits
        checkbox toggled.
        Outputs: the cached heatmap image updates without recomputing maps.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(Path(temp_dir)),
            )
            map_data = _tiny_response_map_data()
            with patch(
                "twopy.napari.plotting.docks.response_plot_widget."
                "compute_recording_response_maps",
                return_value=map_data,
            ) as compute_maps:
                response_widget = cast(Any, create_response_plot_widget(None))
                response_widget.load_recording(recording)
                first_image = response_widget._response_map_area.epoch_map_widgets[
                    0
                ]._image
                compute_maps.reset_mock()
                response_widget._response_map_options_widget._shared_limits.setChecked(
                    False,
                )

        compute_maps.assert_not_called()
        self.assertFalse(response_widget._response_map_shared_limits)
        self.assertIsNot(
            response_widget._response_map_area.epoch_map_widgets[0]._image,
            first_image,
        )

    def test_response_map_option_edits_recompute_in_worker(self) -> None:
        """Confirm heatmap option edits do not compute on the Qt thread.

        Inputs: a loaded recording with cached heatmaps and a pending options
        edit.
        Outputs: recomputation is debounced into the heatmap worker and applied
        only after the worker future finishes.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(Path(temp_dir)),
            )
            first_map_data = _tiny_response_map_data()
            next_map_data = _tiny_response_map_data(epoch_name="next")
            with patch(
                "twopy.napari.plotting.docks.response_plot_widget."
                "compute_recording_response_maps",
                return_value=first_map_data,
            ):
                response_widget = cast(Any, create_response_plot_widget(None))
                response_widget.load_recording(recording)

            future: Future[ResponseMapData] = Future()
            with patch.object(
                response_widget._response_map_worker._executor,
                "submit",
                return_value=future,
            ) as submit:
                response_widget._set_response_map_options(
                    ResponseMapOptions(pixel_smoothing_sigma=1.0),
                )
                submit.assert_not_called()
                response_widget._response_map_worker._debounce_timer.stop()
                response_widget._response_map_worker._start_latest_job()
                submit.assert_called_once()
                self.assertIsNone(response_widget._response_map_data)

                future.set_result(next_map_data)
                response_widget._response_map_worker.collect_finished_job()

        self.assertIs(response_widget._response_map_data, next_map_data)
        self.assertEqual(
            response_widget._response_map_data.epochs[0].epoch_name, "next"
        )

    def test_epoch_plot_panel_centers_title(self) -> None:
        """Confirm epoch titles are centered above live response plots.

        Inputs: one titled epoch plot panel.
        Outputs: the title label uses horizontal center alignment.
        """
        _ = QApplication.instance() or QApplication([])
        plot_data = _tiny_response_plot_data()
        plot = EpochPlotWidget(
            plot_data.epochs[0],
            show_sem=True,
            roi_indices=(0,),
            roi_colors=(QColor("#ff0000"),),
            time_min=0.0,
            time_max=1.0,
            value_min=0.0,
            value_max=1.0,
            plot_size=320,
        )

        panel = epoch_plot_panel(title="Epoch 1: Odor", plot=plot)
        title_label = panel.findChild(QLabel)

        self.assertEqual(title_label.text(), "Epoch 1: Odor")
        self.assertTrue(title_label.alignment() & Qt.AlignmentFlag.AlignHCenter)


if __name__ == "__main__":
    unittest.main()
