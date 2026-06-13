"""Napari ROI generation tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from tests.napari_support import (
    Any,
    EpochFrameWindow,
    FrameWindow,
    NapariAdapterTestCase,
    Path,
    PixelCalibrationProfileMapping,
    PixelCalibrationRow,
    QApplication,
    QCheckBox,
    QColor,
    QLabel,
    QPushButton,
    QWidget,
    RoiGenerationControls,
    RoiGenerationOptions,
    _combo_data,
    _combo_texts,
    _FakeLabelEvents,
    _FakeLayer,
    _two_roi_response_plot_data,
    _write_converted_recording,
    cast,
    create_response_plot_widget,
    date,
    generate_roi_labels,
    h5py,
    load_converted_recording,
    make_roi_set,
    np,
    patch,
    remove_roi_label_values_from_layer,
    temporary_directory,
    unittest,
    visibility_options_widget,
)
from twopy.napari.plotting.roi_generation import (
    roi_generation_edited_after_generation,
    roi_generation_metadata_from_options,
    roi_generation_options_from_metadata,
)
from twopy.napari.roi import (
    merge_roi_label_values_on_layer,
    next_roi_label_value,
    select_next_roi_label_value,
)
from twopy.roi import roi_area_pixels_from_label_image
from twopy.spatial import SpatialCrop


class NapariRoiGenerationTest(NapariAdapterTestCase):
    """Napari ROI generation tests."""

    def test_roi_generation_metadata_round_trips_mode_specific_settings(self) -> None:
        """Confirm saved ROI-generation metadata restores all generated modes.

        Inputs: representative grid, watershed, and response-watershed options.
        Outputs: metadata contains only mode-relevant settings and parses back
        to equivalent options for those settings.
        """
        cases = (
            RoiGenerationOptions(
                roi_mode="manual",
                units="pixels",
                grid_size_pixels=16,
                micron_grid_size=10.0,
                rig="",
                calibration_mode=0,
                scanner="",
                zoom=1.0,
                allow_extrapolation=True,
                watershed_min_pixels=1,
                watershed_smoothing_sigma=0.0,
            ),
            RoiGenerationOptions(
                roi_mode="grid",
                units="pixels",
                grid_size_pixels=24,
                micron_grid_size=10.0,
                rig="",
                calibration_mode=0,
                scanner="",
                zoom=1.0,
                allow_extrapolation=True,
                watershed_min_pixels=1,
                watershed_smoothing_sigma=0.0,
            ),
            RoiGenerationOptions(
                roi_mode="grid",
                units="microns",
                grid_size_pixels=16,
                micron_grid_size=8.5,
                rig="TestRig",
                calibration_mode=2,
                scanner="Galvo",
                zoom=3.25,
                allow_extrapolation=False,
                watershed_min_pixels=1,
                watershed_smoothing_sigma=0.0,
            ),
            RoiGenerationOptions(
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
            ),
            RoiGenerationOptions(
                roi_mode="response_watershed",
                units="pixels",
                grid_size_pixels=16,
                micron_grid_size=10.0,
                rig="",
                calibration_mode=0,
                scanner="",
                zoom=1.0,
                allow_extrapolation=True,
                watershed_min_pixels=1,
                watershed_smoothing_sigma=0.0,
                response_watershed_min_pixels=9,
                response_watershed_smoothing_sigma=1.25,
                response_watershed_fill_holes=False,
                response_watershed_closing_radius=2,
            ),
        )

        for options in cases:
            with self.subTest(mode=options.roi_mode, units=options.units):
                metadata = roi_generation_metadata_from_options(options)
                loaded = roi_generation_options_from_metadata(metadata)

                self.assertIsNotNone(loaded)
                if loaded is None:
                    self.fail("metadata did not parse")
                self.assertEqual(loaded.roi_mode, options.roi_mode)
                for key in metadata:
                    if key in {"roi_mode", "units"}:
                        continue
                    self.assertEqual(getattr(loaded, key), metadata[key])

    def test_roi_generation_metadata_marks_generated_masks_edited_by_hand(
        self,
    ) -> None:
        """Confirm saved provenance can mark post-generation ROI edits.

        Inputs: watershed options plus the edit marker.
        Outputs: metadata carries the marker separately from generation
        settings and the parser reads it back as a boolean.
        """
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

        metadata = roi_generation_metadata_from_options(
            options,
            edited_after_generation=True,
        )

        self.assertEqual(metadata["edited_after_generation"], True)
        self.assertTrue(roi_generation_edited_after_generation(metadata))
        self.assertFalse(
            roi_generation_edited_after_generation(
                roi_generation_metadata_from_options(options),
            )
        )

    def test_roi_visibility_options_show_color_swatches(self) -> None:
        """Confirm ROI visibility rows include color squares.

        Inputs: two ROI labels and two Qt colors.
        Outputs: option widget with swatch styles for both colors.
        """
        _ = QApplication.instance() or QApplication([])
        widget = visibility_options_widget(
            title="ROIs",
            labels=("roi_1", "roi_2"),
            visibility={"roi_1": True, "roi_2": False},
            on_change=lambda _label, _visible: None,
            colors=(QColor("#ff0000"), QColor("#0000ff")),
        )

        styles = "\n".join(child.styleSheet() for child in widget.findChildren(QWidget))
        self.assertIn("#ff0000", styles)
        self.assertIn("#0000ff", styles)

    def test_roi_tab_shows_area_pixels_for_current_labels(self) -> None:
        """Confirm the ROIs tab reports displayed Labels-layer area.

        Inputs: response plot widget with two plotted ROIs and current Labels
            pixels.
        Outputs: ROI row details show pixel counts in plot order.
        """
        _ = QApplication.instance() or QApplication([])
        layer = _FakeLayer(
            name="rois",
            data=np.array([[1, 2], [0, 2]], dtype=np.int64),
            options={},
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_roi_labels_layer(layer)

        response_widget.set_response_plot_data(
            _two_roi_response_plot_data(),
            reset_axes=True,
        )

        label_texts = {
            label.text()
            for label in response_widget.options_widget().findChildren(QLabel)
        }
        self.assertIn("area (px)", label_texts)
        self.assertIn("1 px", label_texts)
        self.assertIn("2 px", label_texts)

    def test_remove_roi_label_values_from_layer_clears_only_selected_rois(
        self,
    ) -> None:
        """Confirm ROI deletion edits only requested Labels values.

        Inputs: Labels layer with three ROI values and two selected values.
        Outputs: selected values become background while the other ROI remains.
        """
        layer = _FakeLayer(
            name="rois",
            data=np.array([[1, 2], [3, 2]], dtype=np.int64),
            options={},
        )

        removed_count = remove_roi_label_values_from_layer(layer, (1, 3))

        self.assertEqual(removed_count, 2)
        np.testing.assert_array_equal(
            layer.data,
            np.array([[0, 2], [0, 2]], dtype=np.int64),
        )

    def test_select_next_roi_label_value_uses_first_available_label(self) -> None:
        """Confirm napari paints new ROIs with the first unused label value.

        Inputs: Labels layer with values 1, 2, and 4.
        Outputs: selected paint label moves to 3 instead of overwriting an ROI.
        """
        layer = _FakeLayer(
            name="rois",
            data=np.array([[1, 2], [4, 0]], dtype=np.int64),
            options={},
            selected_label=1,
        )

        self.assertEqual(next_roi_label_value(layer), 3)

        select_next_roi_label_value(layer)

        self.assertEqual(layer.selected_label, 3)

        layer.data = np.array([[1, 2, 3], [4, 5, 0]], dtype=np.int64)

        self.assertEqual(next_roi_label_value(layer), 6)

    def test_merge_roi_label_values_on_layer_combines_selected_rois(self) -> None:
        """Confirm ROI merging keeps the first selected Labels value.

        Inputs: Labels layer with three ROI values and two selected values.
        Outputs: selected ROI pixels share the first selected value while the
        other ROI remains unchanged.
        """
        layer = _FakeLayer(
            name="rois",
            data=np.array([[1, 2], [3, 2]], dtype=np.int64),
            options={},
        )

        result = merge_roi_label_values_on_layer(layer, (1, 3))

        if result is None:
            self.fail("Expected selected ROI labels to merge.")
        self.assertEqual(result.target_label_value, 1)
        self.assertEqual(result.merged_label_values, (1, 3))
        np.testing.assert_array_equal(
            layer.data,
            np.array([[1, 2], [1, 2]], dtype=np.int64),
        )

    def test_roi_tab_remove_selected_deletes_checked_rois(self) -> None:
        """Confirm the ROI tab can delete selected Labels values.

        Inputs: response plot widget with two plotted ROIs and ROI 2 unchecked.
        Outputs: Remove Selected clears ROI 1 and leaves ROI 2 in the layer.
        """
        _ = QApplication.instance() or QApplication([])
        layer = _FakeLayer(
            name="rois",
            data=np.array([[1, 2], [0, 2]], dtype=np.int64),
            options={},
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_roi_labels_layer(layer)
        response_widget._live_controller.request_update = lambda: None
        response_widget.set_response_plot_data(
            _two_roi_response_plot_data(),
            reset_axes=True,
        )
        response_widget._roi_generation_options = RoiGenerationOptions(
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
            watershed_smoothing_sigma=0.0,
        )
        response_widget._set_roi_visibility(2, False)
        remove_button = next(
            button
            for button in response_widget.options_widget().findChildren(QPushButton)
            if button.text() == "Remove Selected"
        )

        remove_button.click()

        np.testing.assert_array_equal(
            layer.data,
            np.array([[0, 2], [0, 2]], dtype=np.int64),
        )
        self.assertEqual(response_widget._roi_labels(), ("roi_0002",))
        checkbox_texts = {
            checkbox.text()
            for checkbox in response_widget.options_widget().findChildren(QCheckBox)
        }
        self.assertNotIn("roi_0001", checkbox_texts)
        self.assertIn("roi_0002", checkbox_texts)
        self.assertTrue(response_widget._roi_generation_edited_after_generation)

    def test_roi_tab_merge_selected_combines_checked_rois(self) -> None:
        """Confirm the ROI tab can merge selected Labels values.

        Inputs: response plot widget with two plotted ROIs.
        Outputs: Merge Selected combines ROI 2 into ROI 1, removes ROI 2 from
        the option table, and requests a live response recompute.
        """
        _ = QApplication.instance() or QApplication([])
        layer = _FakeLayer(
            name="rois",
            data=np.array([[1, 2], [0, 2]], dtype=np.int64),
            options={},
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_roi_labels_layer(layer)
        update_requests: list[str] = []
        response_widget._live_controller.request_update = lambda: (
            update_requests.append("update")
        )
        response_widget.set_response_plot_data(
            _two_roi_response_plot_data(),
            reset_axes=True,
        )
        response_widget._roi_generation_options = RoiGenerationOptions(
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
            watershed_smoothing_sigma=0.0,
        )
        buttons = response_widget.options_widget().findChildren(QPushButton)
        button_texts = [button.text() for button in buttons]
        self.assertLess(
            button_texts.index("Merge Selected"),
            button_texts.index("Remove Selected"),
        )
        merge_button = next(
            button for button in buttons if button.text() == "Merge Selected"
        )

        merge_button.click()

        np.testing.assert_array_equal(
            layer.data,
            np.array([[1, 1], [0, 1]], dtype=np.int64),
        )
        self.assertEqual(response_widget._roi_labels(), ("roi_0001",))
        checkbox_texts = {
            checkbox.text()
            for checkbox in response_widget.options_widget().findChildren(QCheckBox)
        }
        self.assertIn("roi_0001", checkbox_texts)
        self.assertNotIn("roi_0002", checkbox_texts)
        self.assertEqual(update_requests, ["update"])
        self.assertTrue(response_widget._roi_generation_edited_after_generation)

    def test_roi_tab_create_grid_replaces_labels_layer(self) -> None:
        """Confirm the ROIs tab can create editable grid ROI labels.

        Inputs: a loaded recording, empty Labels layer, and one-pixel grid size.
        Outputs: the Labels layer becomes a deterministic grid in stored display
        orientation and live response recomputation is requested.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(Path(temp_dir)),
            )
            layer = _FakeLayer(
                name="rois",
                data=np.zeros((2, 2), dtype=np.int64),
                options={},
            )
            requests: list[str] = []
            response_widget = cast(Any, create_response_plot_widget(None))
            response_widget._live_controller.request_update = lambda: requests.append(
                "update",
            )
            response_widget.load_recording(recording)
            response_widget.set_roi_labels_layer(layer)
            response_widget._roi_generation_widget._roi_mode.setCurrentIndex(1)
            response_widget._roi_generation_widget._pixel_grid_size.setValue(1)
            response_widget._roi_generation_widget._create_button.click()

        np.testing.assert_array_equal(
            layer.data,
            np.array([[1, 2], [3, 4]], dtype=np.int64),
        )
        self.assertEqual(requests, ["update"])
        self.assertIn(
            "Created pixel grid ROIs",
            response_widget._update_status_label.text(),
        )

    def test_roi_tab_generation_uses_recording_zoom_metadata(self) -> None:
        """Confirm generated micron grids start from converted zoom metadata.

        Inputs: a converted recording whose acquisition metadata has zoom 2.
        Outputs: the ROIs-tab zoom control is initialized from metadata.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(Path(temp_dir)),
            )
            response_widget = cast(Any, create_response_plot_widget(None))
            response_widget.load_recording(recording)

        self.assertEqual(response_widget._roi_generation_widget._zoom.value(), 2.0)

    def test_roi_tab_create_micron_grid_reports_estimated_pixel_size(self) -> None:
        """Confirm micron-grid generation reports only the pixel-size estimate.

        Inputs: a loaded recording, empty Labels layer, and micron grid mode.
        Outputs: generated labels are created and status text reports the
        estimated microns per pixel without calibration-method prose.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(Path(temp_dir)),
            )
            layer = _FakeLayer(
                name="rois",
                data=np.zeros((2, 2), dtype=np.int64),
                options={},
            )
            response_widget = cast(Any, create_response_plot_widget(None))
            response_widget.load_recording(recording)
            response_widget.set_roi_labels_layer(layer)
            roi_widget = response_widget._roi_generation_widget
            roi_widget._roi_mode.setCurrentIndex(1)
            roi_widget._units.setCurrentIndex(1)
            roi_widget._rig.setCurrentIndex(roi_widget._rig.findText("day"))
            roi_widget._mode.setCurrentIndex(roi_widget._mode.findData(2))
            roi_widget._scanner.setCurrentIndex(roi_widget._scanner.findText("galvo"))
            roi_widget._create_button.click()

        self.assertIn(
            "Estimated pixel size:",
            response_widget._update_status_label.text(),
        )
        self.assertNotIn(
            "Created micron grid ROIs",
            response_widget._update_status_label.text(),
        )

    def test_roi_tab_grid_options_follow_selected_units(self) -> None:
        """Confirm grid controls hide options for inactive units.

        Inputs: ROIs-tab controls switched between pixel and micron grid units.
        Outputs: only the active unit's controls are shown, and extrapolation is
        enabled by default.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        roi_widget = response_widget._roi_generation_widget

        roi_widget._roi_mode.setCurrentIndex(1)

        self.assertFalse(roi_widget._pixel_grid_size.isHidden())
        self.assertTrue(roi_widget._micron_grid_size.isHidden())
        self.assertTrue(roi_widget._rig.isHidden())
        self.assertTrue(roi_widget._allow_extrapolation.isChecked())

        roi_widget._units.setCurrentIndex(1)

        self.assertTrue(roi_widget._pixel_grid_size.isHidden())
        self.assertFalse(roi_widget._micron_grid_size.isHidden())
        self.assertFalse(roi_widget._rig.isHidden())
        self.assertEqual(roi_widget._rig.currentText(), "Select rig")
        self.assertFalse(roi_widget._allow_extrapolation.isHidden())

    def test_roi_tab_generation_uses_calibration_profile_metadata(self) -> None:
        """Confirm ROIs tab can preselect an unambiguous calibration group.

        Inputs: converted metadata with a day rig and a mapped ScanImage config.
        Outputs: rig, mode, and scanner controls are initialized from metadata
        plus the profile mapping.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording_path = _write_converted_recording(Path(temp_dir))
            with h5py.File(recording_path, "a") as h5_file:
                h5_file["metadata"].attrs["configName"] = (
                    "256x128_0.5ms_fastAcquisition"
                )
                h5_file["metadata"].attrs["acq.linesPerFrame"] = 128
                h5_file["metadata"].attrs["acq.pixelsPerLine"] = 256
                h5_file["metadata"].attrs["acq.pixelTime"] = 0.0000016
                h5_file["metadata"].attrs["acq.msPerLine"] = 0.6
                h5_file["metadata"].attrs["acq.scanAngleMultiplierFast"] = 1
                h5_file["metadata"].attrs["acq.scanAngleMultiplierSlow"] = 0.57744
                h5_file["run"].attrs["rig_name"] = "day rig"
            recording = load_converted_recording(recording_path)
            response_widget = cast(Any, create_response_plot_widget(None))
            response_widget.load_recording(recording)
            roi_widget = response_widget._roi_generation_widget
            roi_widget._roi_mode.setCurrentIndex(1)

        self.assertEqual(roi_widget._rig.currentText(), "day")
        self.assertEqual(roi_widget._mode.currentData(), 2)
        self.assertEqual(roi_widget._scanner.currentText(), "galvo")
        self.assertEqual(roi_widget._status.text(), "")
        self.assertTrue(roi_widget._status.isHidden())

    def test_roi_tab_generation_selects_odorrig_mode6_calibration(self) -> None:
        """Confirm historical OdorRig recordings preselect night calibration.

        Inputs: converted metadata matching an old OdorRig mode-6 galvo
        recording with measured mode-6 pixel-size rows.
        Outputs: the rig dropdown selects ``night`` and mode/scanner select the
        matching measured calibration group instead of falling back to mode 2.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording_path = _write_converted_recording(Path(temp_dir))
            with h5py.File(recording_path, "a") as h5_file:
                h5_file["metadata"].attrs["configName"] = "128x128_1ms_6.5Hz"
                h5_file["metadata"].attrs["acq.linesPerFrame"] = 128
                h5_file["metadata"].attrs["acq.pixelsPerLine"] = 128
                h5_file["metadata"].attrs["acq.pixelTime"] = 0.0000064
                h5_file["metadata"].attrs["acq.msPerLine"] = 1.2
                h5_file["metadata"].attrs["acq.scanAngleMultiplierFast"] = 1
                h5_file["metadata"].attrs["acq.scanAngleMultiplierSlow"] = 0.57744
                h5_file["run"].attrs["rig_name"] = "OdorRig"
            recording = load_converted_recording(recording_path)
            response_widget = cast(Any, create_response_plot_widget(None))
            response_widget.load_recording(recording)
            roi_widget = response_widget._roi_generation_widget

        self.assertEqual(roi_widget._rig.currentText(), "night")
        self.assertEqual(roi_widget._mode.currentData(), 6)
        self.assertEqual(roi_widget._scanner.currentText(), "galvo")

    def test_roi_tab_unresolved_rig_does_not_default_to_first_choice(self) -> None:
        """Confirm incomplete metadata keeps calibration dropdowns unselected.

        Inputs: a recording whose config maps mode/scanner but whose rig name is
        not mapped to day or night.
        Outputs: rig remains a placeholder and micron-grid creation is disabled
        until the user chooses a rig.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording_path = _write_converted_recording(Path(temp_dir))
            with h5py.File(recording_path, "a") as h5_file:
                h5_file["metadata"].attrs["configName"] = "128x128_1ms_6.5Hz"
                h5_file["metadata"].attrs["acq.linesPerFrame"] = 128
                h5_file["metadata"].attrs["acq.pixelsPerLine"] = 128
                h5_file["metadata"].attrs["acq.pixelTime"] = 0.0000064
                h5_file["metadata"].attrs["acq.msPerLine"] = 1.2
                h5_file["metadata"].attrs["acq.scanAngleMultiplierFast"] = 1
                h5_file["metadata"].attrs["acq.scanAngleMultiplierSlow"] = 0.57744
                h5_file["run"].attrs["rig_name"] = "UnknownRig"
            recording = load_converted_recording(recording_path)
            response_widget = cast(Any, create_response_plot_widget(None))
            response_widget.load_recording(recording)
            roi_widget = response_widget._roi_generation_widget
            roi_widget._roi_mode.setCurrentIndex(1)
            roi_widget._units.setCurrentIndex(1)

        self.assertEqual(roi_widget._rig.currentText(), "Select rig")
        self.assertIsNone(roi_widget._rig.currentData())
        self.assertFalse(roi_widget._create_button.isEnabled())

        roi_widget._rig.setCurrentIndex(roi_widget._rig.findText("night"))

        self.assertEqual(roi_widget._mode.currentData(), 6)
        self.assertEqual(roi_widget._scanner.currentText(), "galvo")
        self.assertTrue(roi_widget._create_button.isEnabled())

    def test_roi_tab_generation_defaults_to_manual_mode(self) -> None:
        """Confirm the ROIs tab starts in manual Labels-editing mode.

        Inputs: a newly created response plot widget.
        Outputs: manual mode is selected and generated-ROI controls are idle.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        roi_widget = response_widget._roi_generation_widget

        self.assertEqual(roi_widget._roi_mode.currentData(), "manual")
        self.assertFalse(roi_widget._create_button.isEnabled())
        self.assertEqual(roi_widget._status.text(), "")
        self.assertTrue(roi_widget._status.isHidden())

    def test_roi_tab_response_watershed_mode_is_selectable(self) -> None:
        """Confirm response watershed appears as a generated ROI mode.

        Inputs: a loaded recording and the ROIs tab controls.
        Outputs: response-watershed options become active and selectable.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(Path(temp_dir)),
            )
            roi_widget = RoiGenerationControls(
                (), (), on_generate=lambda _options: None
            )
            roi_widget.set_recording(recording)
            roi_widget._roi_mode.setCurrentIndex(3)

        options = roi_widget.options()
        self.assertEqual(roi_widget._roi_mode.currentData(), "response_watershed")
        self.assertEqual(roi_widget._create_button.text(), "Create response watershed")
        self.assertTrue(roi_widget._create_button.isEnabled())
        self.assertTrue(roi_widget._watershed_min_pixels.isHidden())
        self.assertTrue(roi_widget._watershed_smoothing_sigma.isHidden())
        self.assertFalse(roi_widget._response_watershed_min_pixels.isHidden())
        self.assertFalse(roi_widget._response_watershed_smoothing_sigma.isHidden())
        self.assertFalse(roi_widget._response_watershed_fill_holes.isHidden())
        self.assertFalse(roi_widget._response_watershed_closing_radius.isHidden())
        self.assertEqual(options.roi_mode, "response_watershed")
        self.assertEqual(options.response_watershed_min_pixels, 5)
        self.assertEqual(options.response_watershed_smoothing_sigma, 0.0)
        self.assertTrue(options.response_watershed_fill_holes)
        self.assertEqual(options.response_watershed_closing_radius, 0)

    def test_response_watershed_generation_action_uses_epoch_windows(self) -> None:
        """Confirm response watershed delegates to the shared extraction helper.

        Inputs: response-watershed options and an explicit epoch window.
        Outputs: generated labels come from the response ROI set.
        """
        recording = cast(Any, object())
        epoch_window = EpochFrameWindow(
            window=FrameWindow(
                index=0,
                start_frame=1,
                stop_frame=3,
                label="epoch 1",
            ),
            epoch_number=1,
            epoch_name="epoch 1",
        )
        options = RoiGenerationOptions(
            roi_mode="response_watershed",
            units="pixels",
            grid_size_pixels=16,
            micron_grid_size=10.0,
            rig="",
            calibration_mode=0,
            scanner="",
            zoom=1.0,
            allow_extrapolation=True,
            watershed_min_pixels=1,
            watershed_smoothing_sigma=0.0,
            response_watershed_min_pixels=7,
            response_watershed_smoothing_sigma=1.5,
            response_watershed_fill_holes=True,
            response_watershed_closing_radius=2,
        )
        mask = np.array([[[True, False], [False, False]]], dtype=np.bool_)

        with patch(
            "twopy.napari.plotting.roi_generation.actions.response_watershed_roi_set",
            return_value=make_roi_set(mask),
        ) as response_watershed:
            generated = generate_roi_labels(
                recording,
                options,
                (),
                epoch_windows=(epoch_window,),
            )

        np.testing.assert_array_equal(
            generated.label_image,
            np.array([[1, 0], [0, 0]], dtype=np.int64),
        )
        self.assertEqual(
            generated.status_text, "Created response watershed ROIs: 1 ROIs."
        )
        response_watershed.assert_called_once()
        args, kwargs = response_watershed.call_args
        self.assertIs(args[0], recording)
        self.assertEqual(args[1], (epoch_window,))
        self.assertEqual(kwargs["min_pixels"], 7)
        self.assertEqual(kwargs["score_smoothing_sigma"], 1.5)
        self.assertTrue(kwargs["fill_holes"])
        self.assertEqual(kwargs["closing_radius"], 2)

    def test_roi_tab_create_watershed_replaces_labels_layer(self) -> None:
        """Confirm the ROIs tab can create watershed ROI labels.

        Inputs: a loaded recording, empty Labels layer, and watershed mode.
        Outputs: the Labels layer is replaced and live recomputation is
        requested.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(Path(temp_dir)),
            )
            layer = _FakeLayer(
                name="rois",
                data=np.zeros((2, 2), dtype=np.int64),
                options={},
            )
            requests: list[str] = []
            response_widget = cast(Any, create_response_plot_widget(None))
            response_widget._live_controller.request_update = lambda: requests.append(
                "update",
            )
            response_widget.load_recording(recording)
            response_widget.set_roi_labels_layer(layer)
            response_widget._roi_generation_widget._roi_mode.setCurrentIndex(2)
            response_widget._roi_generation_widget._create_button.click()

        np.testing.assert_array_equal(
            layer.data,
            np.ones((2, 2), dtype=np.int64),
        )
        self.assertEqual(requests, ["update"])
        self.assertIn(
            "Created watershed ROIs",
            response_widget._update_status_label.text(),
        )

    def test_watershed_generation_filters_inside_display_crop(self) -> None:
        """Confirm watershed minimum size applies to visible crop pixels.

        Inputs: a cropped recording where full-frame watershed would leave a
            small visible edge fragment.
        Outputs: generated watershed labels contain only ROIs meeting the
            minimum size inside the alignment-valid crop.
        """
        image = np.zeros((4, 6), dtype=np.float64)
        image[1, 0] = 10.0
        image[1, 4] = 9.0
        movie_values = np.repeat(image[np.newaxis, :, :], 3, axis=0)
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(
                    Path(temp_dir),
                    movie_values=movie_values,
                    alignment_valid_crop=SpatialCrop(
                        axis0_start=0,
                        axis0_stop=4,
                        axis1_start=3,
                        axis1_stop=6,
                        original_shape=(4, 6),
                        source="alignment_valid_crop",
                    ),
                ),
            )
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
                watershed_min_pixels=4,
                watershed_smoothing_sigma=0.0,
            )

            generated = generate_roi_labels(recording, options, ())

        np.testing.assert_array_equal(
            generated.label_image[:, :3],
            np.zeros((4, 3), dtype=np.int64),
        )
        visible_areas = roi_area_pixels_from_label_image(
            recording.alignment_valid_crop.crop_image(generated.label_image),
        )
        self.assertEqual(visible_areas, {1: 12})
        self.assertTrue(all(area >= 4 for area in visible_areas.values()))

    def test_generated_rois_rebuild_roi_table_before_live_update(self) -> None:
        """Confirm generated labels replace stale ROI table rows immediately.

        Inputs: old plot data for ``roi_0002`` and a generated label image with
            only ``roi_0001``.
        Outputs: the ROIs tab follows the current Labels layer before async live
            analysis returns new plot data.
        """
        from twopy.napari.plotting.roi_generation.actions import GeneratedRoiLabels

        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording = load_converted_recording(
                _write_converted_recording(Path(temp_dir)),
            )
            layer = _FakeLayer(
                name="rois",
                data=np.array([[2, 2], [2, 2]], dtype=np.int64),
                options={},
            )
            response_widget = cast(Any, create_response_plot_widget(None))
            response_widget._live_controller.request_update = lambda: None
            response_widget.load_recording(recording)
            response_widget.set_roi_labels_layer(layer)
            response_widget.set_response_plot_data(
                _two_roi_response_plot_data(),
                reset_axes=True,
            )
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
                watershed_smoothing_sigma=0.0,
            )

            with patch(
                "twopy.napari.plotting.docks.response_plot_widget.generate_roi_labels",
                return_value=GeneratedRoiLabels(
                    label_image=np.array([[1, 0], [0, 0]], dtype=np.int64),
                    status_text="Created watershed ROIs: 226 ROIs.",
                ),
            ):
                response_widget.create_generated_rois(options)

        checkbox_texts = {
            checkbox.text()
            for checkbox in response_widget.options_widget().findChildren(QCheckBox)
        }
        label_texts = {
            label.text()
            for label in response_widget.options_widget().findChildren(QLabel)
        }
        self.assertEqual(
            response_widget._update_status_label.text(),
            "Created watershed ROIs: 226 ROIs.",
        )
        self.assertIsNone(response_widget._plot_data)
        self.assertIn("roi_0001", checkbox_texts)
        self.assertNotIn("roi_0002", checkbox_texts)
        self.assertIn("1 px", label_texts)
        self.assertNotIn("4 px", label_texts)

    def test_manual_label_edit_rebuilds_roi_table_before_live_update(self) -> None:
        """Confirm Labels-layer edits refresh ROI rows before live analysis.

        Inputs: old plot data plus a Labels layer changed to a different label
            value.
        Outputs: the ROIs tab follows the edited Labels layer as soon as the
            Labels data event fires.
        """
        _ = QApplication.instance() or QApplication([])
        events = _FakeLabelEvents()
        layer = _FakeLayer(
            name="rois",
            data=np.array([[1, 1], [0, 0]], dtype=np.int64),
            options={},
            events=events,
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_roi_labels_layer(layer)
        response_widget.set_response_plot_data(
            _two_roi_response_plot_data(),
            reset_axes=True,
        )

        layer.data = np.array([[3, 0], [0, 0]], dtype=np.int64)
        events.data.emit()

        checkbox_texts = {
            checkbox.text()
            for checkbox in response_widget.options_widget().findChildren(QCheckBox)
        }
        label_texts = {
            label.text()
            for label in response_widget.options_widget().findChildren(QLabel)
        }
        self.assertIn("roi_0003", checkbox_texts)
        self.assertNotIn("roi_0001", checkbox_texts)
        self.assertNotIn("roi_0002", checkbox_texts)
        self.assertIn("1 px", label_texts)

    def test_roi_generation_controls_filter_calibration_choices(self) -> None:
        """Confirm calibration dropdowns expose only valid measured groups.

        Inputs: three calibration rows whose rig/mode/scanner combinations do
        not form a full Cartesian product.
        Outputs: changing rig and mode narrows dependent dropdown choices.
        """
        _ = QApplication.instance() or QApplication([])
        widget = RoiGenerationControls(
            (
                PixelCalibrationRow(
                    "day",
                    2,
                    "galvo",
                    1.0,
                    0.5,
                    date(2023, 12, 14),
                ),
                PixelCalibrationRow(
                    "day",
                    3,
                    "res",
                    1.0,
                    0.25,
                    date(2023, 12, 14),
                ),
                PixelCalibrationRow(
                    "night",
                    5,
                    "galvo",
                    1.0,
                    0.125,
                    date(2023, 12, 14),
                ),
            ),
            (
                PixelCalibrationProfileMapping(
                    "256x128_0.5ms_fastAcquisition",
                    2,
                    "galvo",
                    {},
                ),
            ),
            on_generate=lambda _options: None,
        )

        self.assertEqual(_combo_texts(widget._rig), ("Select rig", "day", "night"))
        self.assertEqual(_combo_texts(widget._mode), ("Select mode",))
        self.assertEqual(_combo_data(widget._mode), (None,))

        widget._rig.setCurrentIndex(1)

        self.assertEqual(
            _combo_texts(widget._mode),
            ("Select mode", "2: 256x128_0.5ms_fastAcquisition", "3"),
        )
        self.assertEqual(_combo_data(widget._mode), (None, 2, 3))
        widget._mode.setCurrentIndex(2)
        self.assertEqual(_combo_texts(widget._scanner), ("Select scanner", "res"))

        widget._rig.setCurrentIndex(2)

        self.assertEqual(_combo_data(widget._mode), (None, 5))
        self.assertEqual(_combo_texts(widget._scanner), ("Select scanner",))

    def test_roi_generation_restore_does_not_fall_back_to_active_profile(
        self,
    ) -> None:
        """Confirm saved micron settings do not use unrelated profile defaults.

        Inputs: controls already prefilled for one recording profile, then
        saved micron-grid settings for a missing calibration.
        Outputs: calibration fields remain unselected and generation is
        disabled instead of showing the stale recording profile.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            recording_path = _write_converted_recording(Path(temp_dir))
            with h5py.File(recording_path, "a") as h5_file:
                h5_file["metadata"].attrs["configName"] = (
                    "256x128_0.5ms_fastAcquisition"
                )
                h5_file["run"].attrs["rig_name"] = "day rig"
            recording = load_converted_recording(recording_path)
            widget = RoiGenerationControls(
                (
                    PixelCalibrationRow(
                        "day",
                        2,
                        "galvo",
                        1.0,
                        0.5,
                        date(2023, 12, 14),
                    ),
                ),
                (
                    PixelCalibrationProfileMapping(
                        "256x128_0.5ms_fastAcquisition",
                        2,
                        "galvo",
                        {},
                    ),
                ),
                on_generate=lambda _options: None,
            )
            widget.set_recording(recording)
            widget._roi_mode.setCurrentIndex(1)
            widget._units.setCurrentIndex(1)

        self.assertEqual(widget._rig.currentText(), "day")
        self.assertEqual(widget._mode.currentData(), 2)
        self.assertEqual(widget._scanner.currentText(), "galvo")

        widget.set_options(
            RoiGenerationOptions(
                roi_mode="grid",
                units="microns",
                grid_size_pixels=16,
                micron_grid_size=12.5,
                rig="missing",
                calibration_mode=9,
                scanner="resonant",
                zoom=3.0,
                allow_extrapolation=False,
                watershed_min_pixels=1,
                watershed_smoothing_sigma=0.0,
            )
        )

        self.assertIsNone(widget._rig.currentData())
        self.assertIsNone(widget._mode.currentData())
        self.assertIsNone(widget._scanner.currentData())
        self.assertEqual(
            widget._status.text(), "Saved ROI generation calibration is not available."
        )
        self.assertFalse(widget._create_button.isEnabled())

    def test_roi_tab_remove_selected_handles_empty_roi_selection(self) -> None:
        """Confirm deleting all selected ROIs leaves a stable empty ROI list.

        Inputs: response plot widget with all plotted ROIs selected.
        Outputs: Remove Selected clears the Labels layer and all ROI checkboxes
        without trying to compute y-axis bounds from empty arrays.
        """
        _ = QApplication.instance() or QApplication([])
        layer = _FakeLayer(
            name="rois",
            data=np.array([[1, 2], [0, 2]], dtype=np.int64),
            options={},
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_roi_labels_layer(layer)
        response_widget._live_controller.request_update = lambda: None
        response_widget.set_response_plot_data(
            _two_roi_response_plot_data(),
            reset_axes=True,
        )
        remove_button = next(
            button
            for button in response_widget.options_widget().findChildren(QPushButton)
            if button.text() == "Remove Selected"
        )

        remove_button.click()

        np.testing.assert_array_equal(
            layer.data,
            np.zeros((2, 2), dtype=np.int64),
        )
        self.assertEqual(response_widget._roi_labels(), ())
        checkbox_texts = {
            checkbox.text()
            for checkbox in response_widget.options_widget().findChildren(QCheckBox)
        }
        self.assertNotIn("roi_0001", checkbox_texts)
        self.assertNotIn("roi_0002", checkbox_texts)

    def test_response_status_clears_stale_roi_list(self) -> None:
        """Confirm invalid live recomputes do not leave stale ROI rows.

        Inputs: response widget with existing plot data.
        Outputs: status-only state clears the old ROI option checkboxes.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(
            _two_roi_response_plot_data(),
            reset_axes=True,
        )

        response_widget.show_response_status("No ROI labels to analyze.")

        checkbox_texts = {
            checkbox.text()
            for checkbox in response_widget.options_widget().findChildren(QCheckBox)
        }
        self.assertNotIn("roi_0001", checkbox_texts)
        self.assertNotIn("roi_0002", checkbox_texts)


if __name__ == "__main__":
    unittest.main()
