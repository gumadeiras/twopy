"""Napari loaded-recording panel tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from tests.napari_support import (
    Any,
    NapariAdapterTestCase,
    Path,
    QApplication,
    QComboBox,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSlider,
    QTableWidget,
    QWidget,
    _FakeViewer,
    _load_recording_widget,
    _write_converted_recording,
    add_twopy_magicgui_controls,
    cast,
    group_matching,
    group_matching_roi,
    load_manual_fov_group_rows,
    load_manual_roi_match_rows,
    make_roi_set,
    napari_sidebar,
    np,
    patch,
    roi_label_image_from_layer,
    save_roi_set,
    temporary_directory,
    unittest,
    write_last_recording_folder,
)


class NapariLoadedRecordingsTest(NapariAdapterTestCase):
    """Napari loaded-recording panel tests."""

    def test_loaded_recordings_panel_selects_and_unloads_layers(self) -> None:
        """Confirm multi-recording selection controls layer ownership.

        Inputs: two converted folders loaded into one fake viewer.
        Outputs: selected recording controls layer visibility, and unload
        removes only that recording's layers.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            _write_converted_recording(first, source_session_dir=first)
            _write_converted_recording(second, source_session_dir=second)
            viewer = _FakeViewer()
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            load_widget = _load_recording_widget(control_docks.load_widget)
            panel = cast(QWidget, control_docks.loaded_recordings_widget)
            loaded_list = panel.findChild(QListWidget)
            unload_buttons = {
                button.text(): button for button in panel.findChildren(QPushButton)
            }
            self.assertIn("Unload selected", unload_buttons)
            self.assertIn("Unload all", unload_buttons)
            sidebar_buttons = {
                button.text(): button
                for button in cast(QWidget, control_docks.sidebar_widget).findChildren(
                    QPushButton,
                )
            }
            self.assertIn("Open Group Matching", sidebar_buttons)
            self.assertIn("Reconvert selected", sidebar_buttons)
            self.assertIn("Save loaded list", sidebar_buttons)
            self.assertFalse(sidebar_buttons["Open Group Matching"].isEnabled())
            self.assertFalse(sidebar_buttons["Reconvert selected"].isEnabled())
            self.assertFalse(sidebar_buttons["Save loaded list"].isEnabled())

            load_widget(recording_folder=first)
            load_widget(recording_folder=second)
            viewer.labels[0].data = np.array([[1, 0], [0, 0]], dtype=np.int64)
            viewer.labels[1].data = np.array([[2, 0], [0, 0]], dtype=np.int64)

            self.assertEqual(loaded_list.count(), 2)
            self.assertEqual(loaded_list.currentRow(), 1)
            self.assertEqual(len(viewer.images), 4)
            self.assertEqual(len(viewer.labels), 2)
            self.assertFalse(viewer.images[0].visible)
            self.assertFalse(viewer.images[1].visible)
            self.assertFalse(viewer.labels[0].visible)
            self.assertTrue(viewer.images[2].visible)
            self.assertTrue(viewer.images[3].visible)
            self.assertTrue(viewer.labels[1].visible)
            self.assertTrue(sidebar_buttons["Open Group Matching"].isEnabled())
            self.assertTrue(sidebar_buttons["Reconvert selected"].isEnabled())
            self.assertTrue(sidebar_buttons["Save loaded list"].isEnabled())
            sidebar_buttons["Open Group Matching"].click()
            group_matching_button = cast(
                napari_sidebar.GroupMatchingWindowButton,
                sidebar_buttons["Open Group Matching"],
            )
            group_matching_dialog = group_matching_button._dialog
            group_matching_screen = (
                group_matching_dialog.screen() or QApplication.primaryScreen()
            )
            assert group_matching_screen is not None
            self.assertEqual(
                group_matching_dialog.geometry().height(),
                group_matching_screen.availableGeometry().height(),
            )

            fov_path = root / "fov_groups.csv"
            match_path = root / "roi_matches.csv"
            group_matching_widget = cast(QWidget, control_docks.group_matching_widget)
            fov_path_edit = group_matching_widget.findChild(
                QLineEdit,
                "fov_group_path",
            )
            match_path_edit = group_matching_widget.findChild(
                QLineEdit,
                "roi_match_path",
            )
            assert fov_path_edit is not None
            assert match_path_edit is not None
            fov_path_edit.setText(str(fov_path))
            match_path_edit.setText(str(match_path))
            contrast_sliders = group_matching_widget.findChildren(QSlider)
            self.assertEqual(len(contrast_sliders), 2)
            contrast_sliders[0].setValue(80)
            fov_note_edits = group_matching_widget.findChildren(
                QLineEdit,
                "fov_recording_note",
            )
            self.assertEqual(len(fov_note_edits), 2)
            fov_note_edits[0].setText("first field edge")
            fov_note_edits[1].setText("second field edge")
            match_buttons = {
                button.text(): button
                for button in group_matching_widget.findChildren(QPushButton)
            }
            select_buttons = [
                button
                for button in group_matching_widget.findChildren(QPushButton)
                if button.text() == "Select"
            ]
            self.assertEqual(len(select_buttons), 2)
            for button in select_buttons:
                button.click()
            match_buttons["Select none"].click()
            for button in select_buttons:
                self.assertFalse(button.isChecked())
                button.click()
            match_buttons["New FOV from selected"].click()
            match_buttons["Save FOV groups"].click()
            fov_rows = load_manual_fov_group_rows(fov_path)
            self.assertEqual(
                {row.recording_path.resolve(strict=False) for row in fov_rows},
                {first.resolve(strict=False), second.resolve(strict=False)},
            )
            self.assertEqual({row.fov_group_id for row in fov_rows}, {"fov_1"})
            self.assertEqual(
                {
                    row.recording_path.resolve(strict=False): row.note
                    for row in fov_rows
                },
                {
                    first.resolve(strict=False): "first field edge",
                    second.resolve(strict=False): "second field edge",
                },
            )
            match_buttons["Clear selected FOV"].click()
            original_note_edit_ids = {id(note_edit) for note_edit in fov_note_edits}
            with patch.object(
                group_matching.FovAssignmentView,
                "_choose_existing_fov_group_path",
                return_value=fov_path,
            ):
                match_buttons["Load FOV CSV"].click()
            fov_note_edits = group_matching_widget.findChildren(
                QLineEdit,
                "fov_recording_note",
            )
            self.assertEqual(
                {id(note_edit) for note_edit in fov_note_edits},
                original_note_edit_ids,
            )
            self.assertEqual(
                {note_edit.text() for note_edit in fov_note_edits},
                {"first field edge", "second field edge"},
            )
            with patch.object(
                group_matching.FovAssignmentView,
                "_choose_fov_group_save_path",
                return_value=root / "new_fov_groups.csv",
            ):
                match_buttons["Browse save path"].click()
            fov_path_edit.setText(str(fov_path))
            match_buttons["Save and continue to ROI assignment"].click()

            fov_filter = group_matching_widget.findChild(QComboBox, "fov_filter")
            assert fov_filter is not None
            self.assertEqual(fov_filter.currentText(), "fov_1")
            panel_widget = cast(Any, group_matching_widget)
            self.assertLessEqual(
                group_matching_dialog.geometry().height(),
                group_matching_screen.availableGeometry().height(),
            )
            self.assertEqual(
                group_matching_dialog.maximumHeight(),
                group_matching_screen.availableGeometry().height(),
            )
            roi_scroll_area = panel_widget._roi_view.findChild(
                QScrollArea,
                "roi_assignment_scroll_area",
            )
            assert roi_scroll_area is not None
            self.assertTrue(roi_scroll_area.widgetResizable())
            self.assertIs(
                roi_scroll_area.widget().findChild(QLineEdit, "roi_match_path"),
                match_path_edit,
            )
            self.assertNotIn(
                "All FOVs",
                tuple(
                    fov_filter.itemText(index) for index in range(fov_filter.count())
                ),
            )
            roi_buttons = {
                button.text(): button
                for button in group_matching_widget.findChildren(QPushButton)
            }
            roi_note_edit = group_matching_widget.findChild(
                QLineEdit,
                "roi_match_note",
            )
            assert roi_note_edit is not None
            self.assertIn("Load ROI CSV", roi_buttons)
            self.assertIn("Browse save path", roi_buttons)
            self.assertIn("Add new group", roi_buttons)
            self.assertIn("Overwrite selected group", roi_buttons)
            self.assertIn("Remove selected group", roi_buttons)
            self.assertIn("Back to FOV assignment", roi_buttons)
            self.assertIn("Save and close", roi_buttons)
            roi_cards = group_matching_widget.findChildren(
                QWidget,
                "roi_assignment_card",
            )
            self.assertEqual(len(roi_cards), 2)
            roi_selectors = [
                combo
                for combo in group_matching_widget.findChildren(QComboBox)
                if combo.objectName() == "roi_selector"
            ]
            self.assertEqual(len(roi_selectors), 2)
            for selector in roi_selectors:
                self.assertEqual(selector.count(), 2)
                selector.setCurrentIndex(1)
            self.assertEqual(
                len(
                    group_matching_widget.findChildren(
                        QWidget,
                        "roi_preview_image",
                    ),
                ),
                2,
            )
            self.assertEqual(
                len(
                    group_matching_widget.findChildren(
                        QWidget,
                        "roi_response_preview",
                    ),
                ),
                1,
            )
            loaded_list.setCurrentRow(0)
            self.assertIn(first.name, str(load_widget.recording_folder.line_edit.value))
            self.assertTrue(viewer.images[0].visible)
            self.assertTrue(viewer.images[1].visible)
            self.assertTrue(viewer.labels[0].visible)
            self.assertFalse(viewer.images[2].visible)
            self.assertFalse(viewer.images[3].visible)
            self.assertFalse(viewer.labels[1].visible)
            roi_note_edit.setText("strong match")
            roi_buttons["Add new group"].click()

            match_rows = load_manual_roi_match_rows(match_path)
            self.assertEqual(len(match_rows), 2)
            self.assertEqual(
                {row.recording_path.resolve(strict=False) for row in match_rows},
                {first.resolve(strict=False), second.resolve(strict=False)},
            )
            self.assertEqual(
                {row.roi_label for row in match_rows},
                {"roi_0001", "roi_0002"},
            )
            self.assertEqual({row.group_cell_id for row in match_rows}, {1})
            self.assertEqual({row.status for row in match_rows}, {"matched"})
            self.assertEqual({row.note for row in match_rows}, {"strong match"})
            self.assertEqual(roi_note_edit.text(), "")
            group_table = group_matching_widget.findChild(
                QTableWidget,
                "roi_match_group_table",
            )
            assert group_table is not None
            self.assertEqual(group_table.rowCount(), 1)
            roi_view = cast(Any, group_matching_widget)._roi_view
            with patch.object(
                group_matching_roi.RoiAssignmentView,
                "window",
                return_value=None,
            ):
                roi_view.close_group_matching_window()
            match_rows = load_manual_roi_match_rows(match_path)
            self.assertEqual({row.note for row in match_rows}, {"strong match"})
            roi_buttons["Clear ROI selection"].click()
            roi_selectors = [
                combo
                for combo in group_matching_widget.findChildren(QComboBox)
                if combo.objectName() == "roi_selector"
            ]
            for selector in roi_selectors:
                self.assertEqual(selector.currentData(), "")
            group_table.selectRow(0)
            roi_selectors = [
                combo
                for combo in group_matching_widget.findChildren(QComboBox)
                if combo.objectName() == "roi_selector"
            ]
            self.assertEqual(
                {selector.currentData() for selector in roi_selectors},
                {"roi_0001", "roi_0002"},
            )
            self.assertEqual(roi_note_edit.text(), "strong match")
            roi_selectors[1].setCurrentIndex(0)
            roi_note_edit.setText("first only")
            roi_buttons["Overwrite selected group"].click()
            match_rows = load_manual_roi_match_rows(match_path)
            self.assertEqual(len(match_rows), 1)
            self.assertEqual(match_rows[0].group_cell_id, 1)
            self.assertEqual(
                match_rows[0].recording_path.resolve(strict=False),
                first.resolve(strict=False),
            )
            self.assertEqual(match_rows[0].roi_label, "roi_0001")
            self.assertEqual(match_rows[0].note, "first only")
            self.assertEqual(roi_note_edit.text(), "")

            roi_selectors[1].setCurrentIndex(1)
            roi_note_edit.setText("second group")
            roi_buttons["Add new group"].click()
            match_rows = load_manual_roi_match_rows(match_path)
            self.assertEqual(len(match_rows), 3)
            self.assertEqual({row.group_cell_id for row in match_rows}, {1, 2})
            self.assertEqual(
                {row.note for row in match_rows if row.group_cell_id == 2},
                {"second group"},
            )
            self.assertEqual(roi_note_edit.text(), "")
            self.assertEqual(group_table.rowCount(), 2)
            roi_buttons["Back to FOV assignment"].click()
            self.assertIs(panel_widget._stack.currentWidget(), panel_widget._fov_view)
            match_buttons = {
                button.text(): button
                for button in group_matching_widget.findChildren(QPushButton)
            }
            match_buttons["Save and continue to ROI assignment"].click()
            self.assertIs(panel_widget._stack.currentWidget(), panel_widget._roi_view)
            roi_buttons = {
                button.text(): button
                for button in group_matching_widget.findChildren(QPushButton)
            }
            with patch.object(
                group_matching_roi.RoiAssignmentView,
                "_choose_existing_match_path",
                return_value=match_path,
            ):
                roi_buttons["Load ROI CSV"].click()
            with patch.object(
                group_matching_roi.RoiAssignmentView,
                "_choose_match_save_path",
                return_value=match_path,
            ):
                roi_buttons["Browse save path"].click()
            group_table.selectRow(1)
            roi_buttons["Remove selected group"].click()
            match_rows = load_manual_roi_match_rows(match_path)
            self.assertEqual(len(match_rows), 1)
            self.assertEqual({row.group_cell_id for row in match_rows}, {1})
            self.assertEqual(group_table.rowCount(), 1)
            self.assertEqual(roi_note_edit.text(), "")
            group_table.selectRow(0)
            roi_note_edit.setText("saved on close")
            roi_buttons["Save and close"].click()
            match_rows = load_manual_roi_match_rows(match_path)
            self.assertEqual({row.note for row in match_rows}, {"saved on close"})

            unload_buttons["Unload selected"].click()

            self.assertEqual(loaded_list.count(), 1)
            self.assertEqual(len(viewer.images), 2)
            self.assertEqual(len(viewer.labels), 1)
            self.assertTrue(viewer.images[0].visible)
            self.assertTrue(viewer.images[1].visible)
            self.assertTrue(viewer.labels[0].visible)
            remaining_item = loaded_list.item(0)
            assert remaining_item is not None
            self.assertIn(str(second), remaining_item.text())
            self.assertIn(
                second.name, str(load_widget.recording_folder.line_edit.value)
            )

            load_widget(recording_folder=first)
            self.assertEqual(loaded_list.count(), 2)
            self.assertEqual(len(viewer.images), 4)
            self.assertEqual(len(viewer.labels), 2)

            unload_buttons["Unload all"].click()

            self.assertEqual(loaded_list.count(), 0)
            self.assertFalse(sidebar_buttons["Open Group Matching"].isEnabled())
            self.assertFalse(sidebar_buttons["Reconvert selected"].isEnabled())
            self.assertEqual(len(viewer.images), 0)
            self.assertEqual(len(viewer.labels), 0)
            self.assertEqual(load_widget.recording_folder.value, Path("default"))

    def test_loading_same_recording_selects_existing_entry(self) -> None:
        """Confirm repeated loads do not duplicate the same recording.

        Inputs: two converted folders loaded into one fake viewer, then the
        first folder loaded again.
        Outputs: the existing row is selected and no extra layers are added.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            recording_path = _write_converted_recording(first)
            _write_converted_recording(second)
            viewer = _FakeViewer()
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            load_widget = _load_recording_widget(control_docks.load_widget)
            panel = cast(QWidget, control_docks.loaded_recordings_widget)
            loaded_list = panel.findChild(QListWidget)

            load_widget(recording_folder=first)
            load_widget(recording_folder=second)
            result = load_widget(recording_folder=first)

            assert loaded_list is not None
            self.assertIn(f"Already loaded {recording_path.resolve()}", str(result))
            self.assertEqual(loaded_list.count(), 2)
            self.assertEqual(loaded_list.currentRow(), 0)
            self.assertEqual(len(viewer.images), 4)
            self.assertEqual(len(viewer.labels), 2)
            self.assertTrue(viewer.images[0].visible)
            self.assertTrue(viewer.images[1].visible)
            self.assertTrue(viewer.labels[0].visible)
            self.assertFalse(viewer.images[2].visible)
            self.assertFalse(viewer.images[3].visible)
            self.assertFalse(viewer.labels[1].visible)

    def test_controls_hide_remembered_recording_folder(self) -> None:
        """Confirm the Load Recording control keeps long remembered paths hidden.

        Inputs: one folder stored in the napari state file.
        Outputs: recording-folder widget displays ``default`` instead of the
        remembered path.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            write_last_recording_folder(root)
            viewer = _FakeViewer()

            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            load_widget = _load_recording_widget(control_docks.load_widget)

            self.assertEqual(load_widget.recording_folder.value, Path("default"))

    def test_recording_folder_resolves_recording_defaults(self) -> None:
        """Confirm folder loading finds recording, movie, and ROI files.

        Inputs: converted output folder containing recording/movie/ROI files.
        Outputs: loaded layers and default ROI save path from that folder.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            roi_path = root / "rois.h5"
            save_roi_set(
                make_roi_set(np.array([[[True, False], [False, False]]])),
                roi_path,
            )
            viewer = _FakeViewer()
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            load_widget = _load_recording_widget(control_docks.load_widget)

            result = load_widget(recording_folder=root)

            self.assertIn(str(recording_path), str(result))
            self.assertEqual(len(viewer.images), 2)
            self.assertIn(root.name, str(load_widget.recording_folder.line_edit.value))
            np.testing.assert_array_equal(
                np.unique(roi_label_image_from_layer(viewer.labels[0])),
                np.array([0, 1]),
            )

    def test_recording_folder_picker_shows_short_loaded_path(self) -> None:
        """Confirm loaded recording paths show the useful folder tail.

        Inputs: long converted output path selected through the load widget.
        Outputs: picker text starts with ``...`` and ends with the source
            recording folder tail.
        """
        with temporary_directory() as temp_dir:
            root = (
                Path(temp_dir)
                / "very_long_project_name"
                / "fly"
                / "stimulus"
                / "2025"
                / "10_03"
                / "twopy"
            )
            root.mkdir(parents=True)
            _write_converted_recording(root, source_session_dir=root.parent)
            viewer = _FakeViewer()
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            load_widget = _load_recording_widget(control_docks.load_widget)

            load_widget(recording_folder=root)

            display_text = str(load_widget.recording_folder.line_edit.value)
            self.assertTrue(display_text.startswith("..."))
            self.assertTrue(display_text.endswith("/stimulus/2025/10_03/"))

    def test_roi_file_change_reuses_resolved_recording_path(self) -> None:
        """Confirm ROI file changes work after the path field is shortened.

        Inputs: long converted output path and ROI file loaded through the
            control widget.
        Outputs: changing the ROI file reloads without resolving the
            shortened display text as a filesystem path.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir) / ("long_" * 20) / "twopy"
            root.mkdir(parents=True)
            _write_converted_recording(root)
            roi_path = root / "alternate_rois.h5"
            save_roi_set(
                make_roi_set(np.array([[[True, False], [False, False]]])),
                roi_path,
            )
            viewer = _FakeViewer()
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            load_widget = _load_recording_widget(control_docks.load_widget)

            load_widget(recording_folder=root)
            load_widget.roi_file_to_load.value = roi_path

            self.assertEqual(len(viewer.images), 2)
            np.testing.assert_array_equal(
                np.unique(roi_label_image_from_layer(viewer.labels[0])),
                np.array([0, 1]),
            )


if __name__ == "__main__":
    unittest.main()
