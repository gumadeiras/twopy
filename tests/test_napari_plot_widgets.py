"""Napari response plot widget tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from qtpy.QtCore import QRectF
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QCheckBox, QGridLayout, QPushButton, QTableWidget

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
    SpatialCrop,
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
from twopy.analysis.group_matching import ManualRoiMatchRow, save_manual_roi_match_rows
from twopy.custom import CustomResult, CustomTable
from twopy.napari.group_matching import (
    response_preview as group_matching_response_preview,
)
from twopy.napari.group_matching.fov_assignment import FovAssignmentView
from twopy.napari.group_matching.fov_cards import FovRecordingCard
from twopy.napari.group_matching.roi_cards import RoiRecordingCard
from twopy.napari.group_matching.style import style_group_matching_panel
from twopy.napari.plotting import widgets as plotting_widgets
from twopy.napari.plotting.preview_strip import ResponsePreviewStrip
from twopy.napari.session import LoadedNapariRecording


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
        combined = group_matching_response_preview.combined_response_plot_data(
            (
                group_matching_response_preview.SelectedRoiResponse(
                    recording_path=Path("/recordings/first"),
                    roi_label="roi_0001",
                    plot_data=first_data,
                    color=QColor("#1f77b4"),
                ),
                group_matching_response_preview.SelectedRoiResponse(
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

    def test_group_matching_overlay_resamples_different_time_axes(self) -> None:
        """Confirm selected recordings stay visible when frame rates differ."""
        reference_data = _tiny_response_plot_data()
        dense_epoch = replace(
            reference_data.epochs[0],
            time_seconds=np.array([0.0, 0.5, 1.0], dtype=np.float64),
            mean_values=np.array([[0.0, 10.0, 20.0]], dtype=np.float64),
            sem_values=np.array([[0.0, 1.0, 2.0]], dtype=np.float64),
        )
        dense_data = ResponsePlotData(
            source_path=None,
            epochs=(dense_epoch,),
        )

        combined = group_matching_response_preview.combined_response_plot_data(
            (
                group_matching_response_preview.SelectedRoiResponse(
                    recording_path=Path("/recordings/reference"),
                    roi_label="roi_0001",
                    plot_data=reference_data,
                    color=QColor("#1f77b4"),
                ),
                group_matching_response_preview.SelectedRoiResponse(
                    recording_path=Path("/recordings/dense"),
                    roi_label="roi_0006",
                    plot_data=dense_data,
                    color=QColor("#17becf"),
                ),
            ),
        )

        assert combined is not None
        np.testing.assert_allclose(
            combined.epochs[0].mean_values[1],
            np.array([0.0, 20.0], dtype=np.float64),
        )
        np.testing.assert_allclose(
            combined.epochs[0].sem_values[1],
            np.array([0.0, 2.0], dtype=np.float64),
        )
        self.assertTrue(np.all(np.isfinite(combined.epochs[0].mean_values[1])))

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

        mean_plot = group_matching_response_preview.mean_response_plot_data(plot_data)
        epoch = mean_plot.epochs[0]

        np.testing.assert_allclose(epoch.mean_values, np.array([[2.0, 5.0]]))
        np.testing.assert_allclose(epoch.sem_values, np.array([[1.0, 2.0]]))

    def test_group_matching_selected_roi_buttons_fit_text_and_wrap(self) -> None:
        """Confirm selected-ROI chips fit their text and wrap by row width."""
        _ = QApplication.instance() or QApplication([])
        widget = QWidget()
        layout = QGridLayout()
        widget.setLayout(layout)
        first_path = Path("/recordings/gh146/stim/2025/12_21/17_42_22")
        second_path = Path("/recordings/gh146/stim/2026/01_03/08_09_10")

        selected_responses = (
            group_matching_response_preview.SelectedRoiResponse(
                recording_path=first_path,
                roi_label="roi_0001",
                plot_data=_tiny_response_plot_data(),
                color=QColor("#1f77b4"),
            ),
            group_matching_response_preview.SelectedRoiResponse(
                recording_path=second_path,
                roi_label="roi_0012",
                plot_data=_tiny_response_plot_data(),
                color=QColor("#d95f02"),
            ),
        )
        group_matching_response_preview.add_response_legend(
            layout,
            selected_responses,
            hidden_recordings=set(),
            set_visible=lambda _path, _visible: None,
            max_width=1000,
        )

        button_texts = [
            button.text()
            for button in widget.findChildren(QPushButton)
            if " - ROI " in button.text()
        ]
        self.assertEqual(
            button_texts,
            ["2025.12.21 17:42 - ROI 1", "2026.01.03 08:09 - ROI 12"],
        )
        for button in widget.findChildren(QPushButton):
            if " - ROI " not in button.text():
                continue
            self.assertEqual(button.minimumWidth(), button.sizeHint().width())
            self.assertEqual(button.maximumWidth(), button.sizeHint().width())
        row_item = layout.itemAtPosition(0, 0)
        self.assertIsNotNone(row_item)
        row_widget = row_item.widget() if row_item is not None else None
        self.assertIsNotNone(row_widget)
        row_layout = row_widget.layout() if row_widget is not None else None
        self.assertIsNotNone(row_layout)
        if row_layout is not None:
            self.assertEqual(row_layout.count(), 2)
            self.assertTrue(row_layout.alignment() & Qt.AlignmentFlag.AlignLeft)
            self.assertEqual(layout.itemAtPosition(0, 1), None)

        narrow_widget = QWidget()
        narrow_layout = QGridLayout()
        narrow_widget.setLayout(narrow_layout)
        group_matching_response_preview.add_response_legend(
            narrow_layout,
            selected_responses,
            hidden_recordings=set(),
            set_visible=lambda _path, _visible: None,
            max_width=320,
        )
        self.assertEqual(narrow_layout.count(), 2)
        for row_index in range(2):
            row_item = narrow_layout.itemAtPosition(row_index, 0)
            self.assertIsNotNone(row_item)
            row_widget = row_item.widget() if row_item is not None else None
            self.assertIsNotNone(row_widget)
            row_layout = row_widget.layout() if row_widget is not None else None
            self.assertIsNotNone(row_layout)
            if row_layout is not None:
                self.assertEqual(row_layout.count(), 1)

    def test_group_matching_card_overlays_use_recording_dates(self) -> None:
        """Confirm FOV and ROI card overlays show date plus recording minute.

        Inputs: one loaded recording whose source folder follows the lab
        ``YYYY/MM_DD/HH_MM_SS`` layout.
        Outputs: both assignment card overlays show ``YYYY.MM.DD HH:MM``.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source_dir = (
                root / "data" / "gh146" / "stimulus" / "2025" / "12_21" / "17_42_22"
            )
            recording = load_converted_recording(
                _write_converted_recording(root, source_session_dir=source_dir),
            )
            loaded = LoadedNapariRecording(
                recording=recording,
                roi_save_file=root / "rois.h5",
                mean_image_layer=_FakeLayer("mean", np.ones((2, 2)), {}),
                movie_layer=None,
                roi_labels_layer=_FakeLayer(
                    "rois",
                    np.array([[0, 1], [0, 0]], dtype=np.int64),
                    {},
                ),
            )

            fov_card = FovRecordingCard(
                recording=loaded,
                fov_group_id="fov_1",
                note="",
            )
            roi_card = RoiRecordingCard(
                recording=loaded,
                fov_group_id="fov_1",
                selected_roi="roi_0001",
                trace_color=QColor("#1f77b4"),
                on_selection_changed=lambda: None,
            )

            fov_overlay = fov_card.findChild(QLabel, "fov_card_overlay")
            roi_overlay = roi_card.findChild(QLabel, "fov_card_overlay")
            assert fov_overlay is not None
            assert roi_overlay is not None
            self.assertEqual(fov_overlay.text(), "2025.12.21 17:42 - FOV ID: 1")
            self.assertEqual(roi_overlay.text(), "2025.12.21 17:42 - FOV ID: 1")

    def test_group_matching_tables_use_recording_dates(self) -> None:
        """Confirm FOV and ROI tables summarize recordings by date and minute.

        Inputs: private table-summary methods with card paths in the lab
        ``YYYY/MM_DD/HH_MM_SS`` layout.
        Outputs: table display strings use ``YYYY.MM.DD HH:MM``.
        """
        _ = QApplication.instance() or QApplication([])
        first_path = Path("/recordings/gh146/stim/2025/12_21/17_42_22")
        second_path = Path("/recordings/gh146/stim/2026/01_03/08_09_10")
        fov_view = FovAssignmentView(
            state=SimpleNamespace(loaded_recordings=[]),
            fov_groups={},
            fov_notes={},
            output_path=Path("fov_groups.csv"),
            on_output_path_changed=lambda _path: None,
            on_finalize=lambda: None,
        )
        roi_view = group_matching_roi.RoiAssignmentView(
            state=SimpleNamespace(loaded_recordings=[]),
            fov_groups={},
            current_rois={},
            output_path=Path("roi_matches.csv"),
            on_output_path_changed=lambda _path: None,
            on_back=lambda: None,
        )
        cast(Any, fov_view)._cards = (
            SimpleNamespace(recording_path=first_path),
            SimpleNamespace(recording_path=second_path),
        )
        cast(Any, fov_view)._fov_groups = {
            first_path: "fov_1",
            second_path: "fov_1",
        }
        cast(Any, roi_view)._cards = (
            SimpleNamespace(recording_path=first_path),
            SimpleNamespace(recording_path=second_path),
        )

        self.assertEqual(
            fov_view._visible_fov_group_summary(),
            (("fov_1", ("2025.12.21 17:42", "2026.01.03 08:09")),),
        )
        self.assertEqual(
            roi_view._group_summary(
                (
                    ManualRoiMatchRow("fov_1", 1, first_path, "roi_0001", "matched"),
                    ManualRoiMatchRow("fov_1", 1, second_path, "roi_0012", "matched"),
                ),
            ),
            "2025.12.21 17:42: roi_0001 | 2026.01.03 08:09: roi_0012",
        )

    def test_group_matching_epoch_controls_filter_response_previews(self) -> None:
        """Confirm ROI matching has compact epoch visibility controls."""
        _ = QApplication.instance() or QApplication([])
        view = group_matching_roi.RoiAssignmentView(
            state=SimpleNamespace(loaded_recordings=[]),
            fov_groups={},
            current_rois={},
            output_path=Path("roi_matches.csv"),
            on_output_path_changed=lambda _path: None,
            on_back=lambda: None,
        )
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Gray Interleave",
                    epoch_number=1,
                    roi_labels=("roi_1",),
                    time_seconds=np.array([0.0, 1.0], dtype=np.float64),
                    mean_values=np.array([[0.0, 1.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
                EpochResponsePlotData(
                    epoch_name="Odor",
                    epoch_number=2,
                    roi_labels=("roi_1",),
                    time_seconds=np.array([0.0, 1.0], dtype=np.float64),
                    mean_values=np.array([[2.0, 3.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
            ),
        )

        view._refresh_epoch_controls(plot_data)

        checkboxes = view.findChildren(QCheckBox, "roi_epoch_visibility_checkbox")
        self.assertEqual(
            [checkbox.text() for checkbox in checkboxes],
            ["1: Gray Interleave", "2: Odor"],
        )
        self.assertFalse(checkboxes[0].isChecked())
        self.assertTrue(checkboxes[1].isChecked())
        self.assertEqual(view._visible_response_epoch_indices(plot_data), (1,))
        checkboxes[1].click()
        self.assertEqual(view._visible_response_epoch_indices(plot_data), ())

    def test_group_matching_processing_dropdowns_show_all_choices(self) -> None:
        """Confirm ROI matching plot-setting dropdowns open without scrolling.

        Inputs: an empty ROI assignment view and three normalization epochs.
        Outputs: smoothing and normalization dropdowns show all current items
        with the non-native Qt popup that honors visible-item counts.
        """
        _ = QApplication.instance() or QApplication([])
        view = group_matching_roi.RoiAssignmentView(
            state=SimpleNamespace(loaded_recordings=[]),
            fov_groups={},
            current_rois={},
            output_path=Path("roi_matches.csv"),
            on_output_path_changed=lambda _path: None,
            on_back=lambda: None,
        )
        view._normalization_widget.set_epoch_choices(
            {1: "Gray Interleave", 2: "Odor", 3: "Clean Air"},
        )

        dropdowns = (
            view._smoothing_widget._smoothing_method,
            view._normalization_widget._epoch,
        )

        for dropdown in dropdowns:
            self.assertEqual(dropdown.maxVisibleItems(), dropdown.count())
            self.assertIn("combobox-popup: 0", dropdown.styleSheet())
            dropdown_view = dropdown.view()
            self.assertIsNotNone(dropdown_view)
            if dropdown_view is None:
                continue
            row_heights = [
                dropdown_view.sizeHintForRow(index) for index in range(dropdown.count())
            ]
            expected_height = (
                sum(
                    row_height if row_height > 0 else dropdown.sizeHint().height()
                    for row_height in row_heights
                )
                + 2 * dropdown_view.frameWidth()
            )
            self.assertEqual(dropdown_view.height(), expected_height)
            try:
                dropdown.showPopup()
                QApplication.processEvents()
                scrollbar = dropdown_view.verticalScrollBar()
                self.assertIsNotNone(scrollbar)
                if scrollbar is not None:
                    self.assertFalse(scrollbar.isVisible())
            finally:
                dropdown.hidePopup()

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
            output_path=Path("roi_matches.csv"),
            on_output_path_changed=lambda _path: None,
            on_back=lambda: None,
        )

        with patch.object(
            view,
            "_selected_response_data",
            side_effect=AssertionError("plot size should not recompute responses"),
        ):
            view._set_plot_size(240)

        self.assertEqual(view._plot_size, 240)
        self.assertLess(
            plotting_widgets._plot_text_pixel_size(180),
            plotting_widgets._plot_text_pixel_size(260),
        )
        self.assertEqual(plotting_widgets._axis_ticks(-1.0, 1.0), (-1.0, 0.0, 1.0))

    def test_response_preview_strip_keeps_cached_panels_parented(self) -> None:
        """Confirm preview redraws do not orphan cached Qt panels.

        Inputs: a cached response preview strip rendered, cleared, then rendered
        again.
        Outputs: cached epoch panels stay parented to the strip widget, avoiding
        transient top-level widgets during Group Matching selection changes.
        """
        _ = QApplication.instance() or QApplication([])
        strip = ResponsePreviewStrip("test_preview_strip")
        plot_data = _tiny_response_plot_data()

        strip.render(
            plot_data,
            epoch_indices=(0,),
            roi_colors=(QColor("#1f77b4"),),
            plot_size=180,
        )
        panel = strip._epoch_plot_panels[0]
        self.assertIs(panel.parent(), strip.widget)
        strip.clear()
        self.assertIs(panel.parent(), strip.widget)

        strip.render(
            plot_data,
            epoch_indices=(0,),
            roi_colors=(QColor("#1f77b4"),),
            plot_size=200,
        )
        self.assertIs(panel.parent(), strip.widget)

    def test_response_preview_strip_rebuilds_after_epoch_count_shrinks(self) -> None:
        """Confirm stale epoch panels leave the visible strip.

        Inputs: a preview strip rendered with two epochs, then rendered with
        one epoch.
        Outputs: the removed epoch panel is no longer present in the layout.
        """
        _ = QApplication.instance() or QApplication([])
        strip = ResponsePreviewStrip("test_preview_strip")
        first = _tiny_response_plot_data()
        two_epoch_data = replace(
            first,
            epochs=(
                first.epochs[0],
                replace(first.epochs[0], epoch_number=2, epoch_name="Second"),
            ),
        )

        strip.render(
            two_epoch_data,
            epoch_indices=(0, 1),
            roi_colors=(QColor("#1f77b4"),),
            plot_size=180,
        )
        stale_panel = strip._epoch_plot_panels[1]
        strip.render(
            first,
            epoch_indices=(0,),
            roi_colors=(QColor("#1f77b4"),),
            plot_size=180,
        )

        layout_widgets: list[QWidget] = []
        for index in range(strip._layout.count()):
            item = strip._layout.itemAt(index)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                layout_widgets.append(widget)
        self.assertNotIn(stale_panel, layout_widgets)
        self.assertEqual(tuple(layout_widgets), (strip._epoch_plot_panels[0],))

    def test_group_matching_response_success_redraw_does_not_clear_strips(
        self,
    ) -> None:
        """Confirm successful response redraws update cached strips in place.

        Inputs: a ROI assignment view with one selected response already
        computed.
        Outputs: the success path renders plots without clearing/hiding the
        cached response strips first.
        """
        _ = QApplication.instance() or QApplication([])
        view = group_matching_roi.RoiAssignmentView(
            state=SimpleNamespace(loaded_recordings=[]),
            fov_groups={},
            current_rois={},
            output_path=Path("roi_matches.csv"),
            on_output_path_changed=lambda _path: None,
            on_back=lambda: None,
        )
        view._selected_responses = (
            group_matching_response_preview.SelectedRoiResponse(
                recording_path=Path("/recordings/first"),
                roi_label="roi_0001",
                plot_data=_tiny_response_plot_data(),
                color=QColor("#1f77b4"),
            ),
        )

        with (
            patch.object(
                view._response_strip,
                "clear",
                side_effect=AssertionError("success redraw should not clear ROI strip"),
            ),
            patch.object(
                view._mean_response_strip,
                "clear",
                side_effect=AssertionError(
                    "success redraw should not clear combined strip",
                ),
            ),
        ):
            view._render_response_preview()

        self.assertFalse(view._response_status.isHidden())
        self.assertEqual(
            view._response_status.text(),
            "Showing the selected recording trace.",
        )

    def test_group_matching_response_status_shows_all_and_partial_visibility(
        self,
    ) -> None:
        """Confirm Selected ROIs always summarizes hidden trace chips.

        Inputs: a ROI assignment view with two selected responses whose trace
        chips move from all visible to partially hidden.
        Outputs: the status text reports the current visible and hidden trace
        counts while plots still render.
        """
        _ = QApplication.instance() or QApplication([])
        first_path = Path("/recordings/first")
        second_path = Path("/recordings/second")
        view = group_matching_roi.RoiAssignmentView(
            state=SimpleNamespace(loaded_recordings=[]),
            fov_groups={},
            current_rois={},
            output_path=Path("roi_matches.csv"),
            on_output_path_changed=lambda _path: None,
            on_back=lambda: None,
        )
        view._selected_responses = (
            group_matching_response_preview.SelectedRoiResponse(
                recording_path=first_path,
                roi_label="roi_0001",
                plot_data=_tiny_response_plot_data(),
                color=QColor("#1f77b4"),
            ),
            group_matching_response_preview.SelectedRoiResponse(
                recording_path=second_path,
                roi_label="roi_0002",
                plot_data=_tiny_response_plot_data(),
                color=QColor("#ff7f0e"),
            ),
        )

        view._render_response_preview()

        self.assertFalse(view._response_status.isHidden())
        self.assertEqual(
            view._response_status.text(),
            "Showing all 2 selected recording traces.",
        )

        view._hidden_response_recordings.add(second_path)

        view._render_response_preview()

        self.assertFalse(view._response_status.isHidden())
        self.assertEqual(
            view._response_status.text(),
            "Showing 1 of 2 selected recording traces; 1 recording trace hidden.",
        )

    def test_group_matching_saved_group_selection_resets_hidden_traces(self) -> None:
        """Confirm saved-group restore starts with all selected responses visible.

        Inputs: a ROI assignment view with a hidden recording trace and one
        saved group selected from the table.
        Outputs: the hidden trace set is cleared before the restored group is
        redrawn.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            path = Path(temp_dir) / "roi_matches.csv"
            recording_path = Path(temp_dir) / "recording"
            save_manual_roi_match_rows(
                (
                    ManualRoiMatchRow(
                        fov_group_id="fov_1",
                        group_cell_id=1,
                        recording_path=recording_path,
                        roi_label="roi_0001",
                        status="matched",
                    ),
                ),
                path,
            )
            view = group_matching_roi.RoiAssignmentView(
                state=SimpleNamespace(loaded_recordings=[]),
                fov_groups={},
                current_rois={},
                output_path=path,
                on_output_path_changed=lambda _path: None,
                on_back=lambda: None,
            )
            view._fov_filter.addItem("1", "fov_1")
            view._hidden_response_recordings.add(recording_path)
            view._refresh_group_table()

            view._group_table.selectRow(0)

        self.assertEqual(view._hidden_response_recordings, set())

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
            group_matching_response_preview.visible_response_epoch_indices(plot_data),
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

    def test_group_matching_selected_roi_response_uses_full_frame_labels(self) -> None:
        """Confirm selected ROI plots analyze full-frame labels from cropped UI data.

        Inputs: a crop-native napari Labels layer with one selected ROI.
        Outputs: the Group Matching response path sends full-frame movie labels
        to the selected-response cache.
        """
        crop = SpatialCrop(
            axis0_start=1,
            axis0_stop=3,
            axis1_start=2,
            axis1_stop=5,
            original_shape=(4, 6),
            source="alignment_valid_crop",
        )
        display_labels = np.zeros((crop.shape[1], crop.shape[0]), dtype=np.int64)
        display_labels[1, 0] = 7
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(
                    Path(temp_dir),
                    movie_values=np.ones((3, 4, 6), dtype=np.float64),
                    alignment_valid_crop=crop,
                ),
            )
            loaded = LoadedNapariRecording(
                recording=recording,
                roi_save_file=Path(temp_dir) / "rois.h5",
                mean_image_layer=_FakeLayer(
                    name="mean",
                    data=np.zeros(display_labels.shape, dtype=np.float64),
                    options={},
                ),
                movie_layer=None,
                roi_labels_layer=_FakeLayer(
                    name="rois",
                    data=display_labels,
                    options={},
                ),
            )
            cache = _RecordingSelectedResponseCache()

            group_matching_response_preview.selected_roi_response_plot_data(
                loaded,
                "roi_0007",
                cache=cast(Any, cache),
                normalization_options=group_matching_roi.NormalizationOptions(),
                smoothing_options=group_matching_roi.SmoothingOptions(),
            )

        assert cache.label_image is not None
        routed_labels = np.asarray(cache.label_image, dtype=np.int64)
        self.assertEqual(routed_labels.shape, crop.original_shape)
        self.assertEqual(routed_labels[1, 3], 7)
        self.assertEqual(int(np.count_nonzero(routed_labels)), 1)

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
        self.assertEqual(widget.sizeHint().height(), 464)

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
        self.assertEqual(widget.sizeHint().height(), 304)

    def test_epoch_plot_widget_reuses_pixmap_until_display_changes(self) -> None:
        """Confirm scroll paints reuse the cached raster plot.

        Inputs: one epoch plot widget rendered twice, then changed display
        options.
        Outputs: the second render reuses the same pixmap and the display
        change invalidates it.
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
            plot_size=320,
        )

        first = widget._rendered_pixmap()
        second = widget._rendered_pixmap()
        widget.update_display(
            show_sem=False,
            roi_indices=(0,),
            roi_colors=(QColor("#ff0000"),),
            time_min=0.0,
            time_max=1.0,
            value_min=0.0,
            value_max=1.0,
            plot_size=320,
        )
        third = widget._rendered_pixmap()

        self.assertIs(first, second)
        self.assertIsNot(first, third)

    def test_dense_trace_path_preserves_extrema_per_pixel(self) -> None:
        """Confirm binned trace drawing keeps narrow peaks and troughs.

        Inputs: ten samples that all land inside one drawable x pixel.
        Outputs: the built path includes both the minimum and maximum response
        values instead of selecting one representative sample.
        """
        rect = QRectF(0.0, 0.0, 1.0, 10.0)
        path = plotting_widgets._trace_path(
            rect,
            np.linspace(0.0, 0.9, 10, dtype=np.float64),
            np.array(
                [0.0, 5.0, -4.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.5, 0.0],
                dtype=np.float64,
            ),
            0.0,
            0.9,
            -5.0,
            5.0,
        )
        y_values = tuple(
            path.elementAt(index).y for index in range(path.elementCount())
        )

        self.assertEqual(path.elementCount(), 2)
        self.assertAlmostEqual(min(y_values), 0.0)
        self.assertAlmostEqual(max(y_values), 9.0)

    def test_dense_sem_band_preserves_pixel_extent(self) -> None:
        """Confirm binned SEM bands keep the full vertical envelope.

        Inputs: ten upper/lower samples inside one drawable x pixel.
        Outputs: the path spans the maximum upper value and minimum lower value.
        """
        rect = QRectF(0.0, 0.0, 1.0, 10.0)
        path = plotting_widgets._sem_band_path(
            rect,
            np.linspace(0.0, 0.9, 10, dtype=np.float64),
            np.array(
                [1.0, 3.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.5, 0.0],
                dtype=np.float64,
            ),
            np.array(
                [-1.0, -4.0, -2.0, 0.0, -1.0, -2.0, -1.0, 0.0, -0.5, 0.0],
                dtype=np.float64,
            ),
            0.0,
            0.9,
            -5.0,
            5.0,
        )
        y_values = tuple(
            path.elementAt(index).y for index in range(path.elementCount())
        )

        self.assertAlmostEqual(min(y_values), 2.0)
        self.assertAlmostEqual(max(y_values), 9.0)

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
                _finish_response_map_worker(response_widget)

        compute_maps.assert_called_once()
        self.assertIs(response_widget._response_map_data, map_data)
        self.assertEqual(len(response_widget._response_map_area.epoch_map_widgets), 1)
        self.assertIsNone(response_widget._plot_data)

    def test_recording_load_clears_custom_workflow_results(self) -> None:
        """Confirm recording changes remove stale Custom-tab result widgets.

        Inputs: response widget with a rendered Custom-tab table, then one
        loaded converted recording.
        Outputs: the Custom-tab result area returns to its empty state.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            table_path = root / "direction_selectivity.csv"
            table_path.write_text("roi_label,dsi\nroi_0001,0.5\n", encoding="utf-8")
            recording = load_converted_recording(
                _write_converted_recording(root / "recording"),
            )
            response_widget = cast(Any, create_response_plot_widget(None))
            custom_panel = response_widget._custom_workflow_panel
            custom_panel._render_result(
                CustomResult(
                    message="ok",
                    tables=(CustomTable("Direction selectivity", table_path),),
                ),
            )

            self.assertIsNotNone(custom_panel.findChild(QTableWidget))

            response_widget.load_recording(recording)

            self.assertIsNone(custom_panel.findChild(QTableWidget))
            labels = {label.text() for label in custom_panel.findChildren(QLabel)}
            self.assertIn("No custom workflow outputs.", labels)

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
                _finish_response_map_worker(response_widget)
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
                _finish_response_map_worker(response_widget)
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
                _finish_response_map_worker(response_widget)

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


def _finish_response_map_worker(response_widget: object) -> None:
    """Run and collect one pending response-map worker job in tests."""
    worker = cast(Any, response_widget)._response_map_worker
    worker._debounce_timer.stop()
    worker._start_latest_job()
    future = worker._future
    assert future is not None
    future.result(timeout=2.0)
    worker.collect_finished_job()


class _RecordingSelectedResponseCache:
    """Selected-response cache double that records routed label images."""

    def __init__(self) -> None:
        """Create an empty cache double."""
        self.label_image: object | None = None

    def compute_selected_response(
        self,
        recording: object,
        label_image: object,
        *,
        roi_label: str,
        label_value: int,
        source_path: Path,
        response_processing_options: object,
    ) -> ResponsePlotData:
        """Record selected-response inputs and return tiny plot data."""
        del recording, roi_label, label_value, source_path, response_processing_options
        self.label_image = np.asarray(label_image, dtype=np.int64)
        return _tiny_response_plot_data()


if __name__ == "__main__":
    unittest.main()
