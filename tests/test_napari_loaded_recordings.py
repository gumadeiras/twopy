"""Napari loaded-recording panel tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from qtpy.QtWidgets import QHeaderView
from tests.napari_support import (
    Any,
    NapariAdapterTestCase,
    Path,
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSlider,
    Qt,
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

from twopy.napari.group_matching import cards as group_matching_cards
from twopy.napari.group_matching import fov_assignment as group_matching_fov


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
            group_matching_button = cast(
                napari_sidebar.GroupMatchingWindowButton,
                sidebar_buttons["Open Group Matching"],
            )
            group_matching_dialog = group_matching_button._dialog
            group_matching_screen = (
                group_matching_dialog.screen() or QApplication.primaryScreen()
            )
            assert group_matching_screen is not None
            with patch.object(
                group_matching_dialog,
                "showFullScreen",
                wraps=group_matching_dialog.showFullScreen,
            ) as show_full_screen:
                sidebar_buttons["Open Group Matching"].click()
            show_full_screen.assert_not_called()
            self.assertFalse(group_matching_dialog.isFullScreen())
            dialog_geometry = group_matching_dialog.geometry()
            available_geometry = group_matching_screen.availableGeometry()
            self.assertEqual(dialog_geometry.left(), available_geometry.left())
            self.assertEqual(dialog_geometry.top(), available_geometry.top())
            self.assertEqual(dialog_geometry.width(), available_geometry.width())
            self.assertGreaterEqual(
                dialog_geometry.height(), available_geometry.height()
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
            self.assertEqual(fov_path_edit.placeholderText(), "FOV CSV path...")
            self.assertEqual(match_path_edit.placeholderText(), "ROI CSV path...")
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
            self.assertEqual(
                {note_edit.placeholderText() for note_edit in fov_note_edits},
                {"Optional note..."},
            )
            self.assertTrue(
                all(note_edit.font().italic() for note_edit in fov_note_edits)
            )
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
            match_buttons["Select all"].click()
            for button in select_buttons:
                self.assertTrue(button.isChecked())
            match_buttons["Assign new FOV ID"].click()
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
            fov_group_table = group_matching_widget.findChild(
                QTableWidget,
                "fov_assignment_table",
            )
            assert fov_group_table is not None
            overlay_labels = group_matching_widget.findChildren(
                QLabel,
                "fov_card_overlay",
            )
            fov_cards = group_matching_widget.findChildren(
                QWidget,
                "fov_recording_card",
            )
            self.assertEqual(len(fov_cards), 2)
            fov_view = cast(Any, group_matching_widget)._fov_view
            self.assertEqual(
                fov_view._card_grid_columns,
                group_matching_cards.card_columns_for_width(
                    fov_view._workspace_viewport.width(),
                    card_width=group_matching_fov.FOV_CARD_WIDTH,
                    spacing=fov_view._grid.spacing(),
                ),
            )
            for fov_card in fov_cards:
                self.assertEqual(
                    fov_card.minimumWidth(),
                    group_matching_fov.FOV_CARD_WIDTH,
                )
                self.assertEqual(
                    fov_card.maximumWidth(),
                    group_matching_fov.FOV_CARD_WIDTH,
                )
            self.assertTrue(
                all(" - FOV ID:" in label.text() for label in overlay_labels),
            )
            self.assertTrue(
                all(": FOV ID:" not in label.text() for label in overlay_labels),
            )
            left_scroll = group_matching_widget.findChild(
                QScrollArea,
                "fov_assignment_left_scroll",
            )
            assert left_scroll is not None
            left_scroll_widget = left_scroll.widget()
            assert left_scroll_widget is not None
            fov_id_label = left_scroll_widget.findChild(QLabel, "fov_id_label")
            assert fov_id_label is not None
            self.assertEqual(fov_id_label.text(), "FOV ID")
            fov_sections = {
                group_box.title()
                for group_box in left_scroll_widget.findChildren(QGroupBox)
            }
            self.assertIn("FOV file", fov_sections)
            self.assertNotIn("Decision file", fov_sections)
            self.assertEqual(
                left_scroll.horizontalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            )
            self.assertEqual(
                fov_group_table.horizontalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
            )
            horizontal_header = fov_group_table.horizontalHeader()
            assert horizontal_header is not None
            self.assertEqual(
                horizontal_header.defaultAlignment(),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            self.assertEqual(
                fov_group_table.maximumHeight(),
                (group_matching_fov._FOV_TABLE_VISIBLE_ROWS + 1)
                * group_matching_fov._FOV_TABLE_ROW_HEIGHT
                + 4,
            )
            self.assertEqual(fov_group_table.rowCount(), 1)
            fov_group_table.selectRow(0)
            for button in select_buttons:
                self.assertTrue(button.isChecked())
            fov_group_table.clearSelection()
            for button in select_buttons:
                self.assertFalse(button.isChecked())
            fov_group_table.selectRow(0)
            for button in select_buttons:
                self.assertTrue(button.isChecked())
            match_buttons["Remove assigned FOV"].click()
            self.assertEqual(fov_group_table.rowCount(), 0)
            original_note_edit_ids = {id(note_edit) for note_edit in fov_note_edits}
            with patch.object(
                group_matching.FovAssignmentView,
                "_choose_existing_fov_group_path",
                return_value=fov_path,
            ):
                match_buttons["Load FOV CSV"].click()
            status_label = left_scroll_widget.findChild(
                QLabel,
                "group_matching_caption",
            )
            assert status_label is not None
            self.assertTrue(
                status_label.text().startswith("Loaded 2 FOV assignments from "),
            )
            self.assertFalse(status_label.text().endswith("."))
            self.assertEqual(
                status_label.textInteractionFlags(),
                Qt.TextInteractionFlag.TextSelectableByMouse,
            )
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
            self.assertEqual(fov_filter.currentText(), "1")
            self.assertEqual(fov_filter.currentData(), "fov_1")
            panel_widget = cast(Any, group_matching_widget)
            self.assertGreater(
                group_matching_dialog.maximumHeight(),
                group_matching_screen.availableGeometry().height(),
            )
            roi_left_scroll = panel_widget._roi_view.findChild(
                QScrollArea,
                "roi_assignment_left_scroll",
            )
            assert roi_left_scroll is not None
            self.assertTrue(roi_left_scroll.widgetResizable())
            roi_left_widget = roi_left_scroll.widget()
            assert roi_left_widget is not None
            self.assertIs(
                roi_left_widget.findChild(QLineEdit, "roi_match_path"),
                match_path_edit,
            )
            left_layout = roi_left_widget.layout()
            assert left_layout is not None
            roi_title = roi_left_widget.findChild(QLabel, "group_matching_title")
            assert roi_title is not None
            self.assertEqual(roi_title.text(), "ROI assignment")
            roi_section_titles = [
                cast(QGroupBox, left_layout.itemAt(index).widget()).title()
                for index in range(left_layout.count())
                if isinstance(left_layout.itemAt(index).widget(), QGroupBox)
            ]
            self.assertEqual(
                roi_section_titles[:6],
                [
                    "ROI file",
                    "FOV filter",
                    "Current selection",
                    "Saved groups",
                    "Plot settings",
                    "Finish",
                ],
            )
            self.assertNotIn("Decision file", roi_section_titles)
            roi_status_label = cast(QLabel, left_layout.itemAt(3).widget())
            self.assertEqual(roi_status_label.objectName(), "group_matching_caption")
            self.assertEqual(
                cast(QGroupBox, left_layout.itemAt(4).widget()).title(),
                "FOV filter",
            )
            self.assertEqual(
                roi_status_label.textInteractionFlags(),
                Qt.TextInteractionFlag.TextSelectableByMouse,
            )
            self.assertIn(
                "FOV ID",
                [
                    label.text()
                    for label in roi_left_widget.findChildren(
                        QLabel,
                        "fov_id_label",
                    )
                ],
            )
            plot_toggles = {
                checkbox.text(): checkbox
                for checkbox in roi_left_widget.findChildren(QCheckBox)
                if checkbox.objectName()
                in {"show_roi_responses", "show_combined_response"}
            }
            self.assertEqual(
                set(plot_toggles),
                {"ROI responses", "Combined response"},
            )
            roi_response_preview = group_matching_widget.findChild(
                QWidget,
                "roi_response_preview",
            )
            combined_response_preview = group_matching_widget.findChild(
                QWidget,
                "roi_mean_response_preview",
            )
            assert roi_response_preview is not None
            assert combined_response_preview is not None
            fov_filter_view = fov_filter.view()
            assert fov_filter_view is not None
            self.assertGreaterEqual(fov_filter_view.minimumWidth(), 128)
            roi_workspace_scroll = panel_widget._roi_view.findChild(
                QScrollArea,
                "roi_assignment_workspace_scroll",
            )
            assert roi_workspace_scroll is not None
            self.assertTrue(roi_workspace_scroll.widgetResizable())
            roi_workspace_widget = roi_workspace_scroll.widget()
            assert roi_workspace_widget is not None
            right_layout = roi_workspace_widget.layout()
            assert right_layout is not None
            first_section = right_layout.itemAt(0).widget()
            second_section = right_layout.itemAt(1).widget()
            third_section = right_layout.itemAt(2).widget()
            fourth_section = right_layout.itemAt(3).widget()
            self.assertIsInstance(first_section, QGroupBox)
            self.assertIsInstance(second_section, QGroupBox)
            self.assertIsInstance(third_section, QGroupBox)
            self.assertIsInstance(fourth_section, QGroupBox)
            self.assertEqual(
                cast(QGroupBox, first_section).title(),
                "Selected ROIs",
            )
            self.assertEqual(
                cast(QGroupBox, second_section).title(),
                "ROI responses",
            )
            self.assertEqual(
                cast(QGroupBox, third_section).title(),
                "Combined responses",
            )
            self.assertEqual(
                cast(QGroupBox, fourth_section).title(),
                "ROI cards",
            )
            plot_toggles["ROI responses"].click()
            self.assertTrue(cast(QGroupBox, second_section).isHidden())
            plot_toggles["ROI responses"].click()
            plot_toggles["Combined response"].click()
            self.assertTrue(cast(QGroupBox, third_section).isHidden())
            plot_toggles["Combined response"].click()
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
            self.assertEqual(roi_note_edit.placeholderText(), "Optional note...")
            self.assertTrue(roi_note_edit.font().italic())
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
            roi_view = cast(Any, group_matching_widget)._roi_view
            self.assertEqual(
                roi_view._card_grid_columns,
                group_matching_roi._roi_card_columns_for_width(
                    roi_workspace_scroll.viewport().width(),
                    spacing=roi_view._grid.spacing(),
                ),
            )
            for roi_card in roi_cards:
                self.assertEqual(
                    roi_card.minimumWidth(),
                    group_matching_roi._ROI_CARD_WIDTH,
                )
                self.assertEqual(
                    roi_card.maximumWidth(),
                    group_matching_roi._ROI_CARD_WIDTH,
                )
                self.assertEqual(
                    roi_card.minimumHeight(),
                    group_matching_roi._ROI_CARD_HEIGHT,
                )
                roi_overlay = roi_card.findChild(QLabel, "fov_card_overlay")
                assert roi_overlay is not None
                self.assertIn(" - FOV ID: 1", roi_overlay.text())
                self.assertNotIn(": FOV ID:", roi_overlay.text())
                roi_preview_image = roi_card.findChild(QLabel, "roi_preview_image")
                assert roi_preview_image is not None
                self.assertEqual(
                    roi_preview_image.width(),
                    group_matching_roi.THUMBNAIL_SIZE,
                )
            roi_selectors = [
                combo
                for combo in group_matching_widget.findChildren(QComboBox)
                if combo.objectName() == "roi_selector"
            ]
            self.assertEqual(len(roi_selectors), 2)
            for selector in roi_selectors:
                self.assertEqual(selector.count(), 2)
                self.assertTrue(selector.itemText(1).isdecimal())
                self.assertNotIn("roi_", selector.itemText(1))
                self.assertEqual(
                    selector.height(),
                    group_matching_roi._ROI_CONTROL_HEIGHT,
                )
                selector.setCurrentIndex(1)
            roi_trace_chips = [
                label
                for label in group_matching_widget.findChildren(QLabel)
                if label.text() == "ROI" and label.height() == selector.height()
            ]
            self.assertEqual(len(roi_trace_chips), 2)
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
            self.assertEqual(
                len(
                    group_matching_widget.findChildren(
                        QWidget,
                        "roi_mean_response_preview",
                    ),
                ),
                1,
            )
            self.assertEqual(roi_view._plot_size, 180)
            self.assertEqual(
                roi_view._smoothing_widget._smoothing_method.itemText(1),
                "moving average",
            )
            self.assertEqual(
                roi_view._smoothing_widget._smoothing_method.maxVisibleItems(),
                3,
            )
            self.assertFalse(roi_view._normalization_widget._epoch.isEnabled())
            response_epoch_titles = [
                label.text()
                for label in roi_response_preview.findChildren(QLabel)
                if ": " in label.text()
            ]
            roi_view._set_plot_size(200)
            self.assertEqual(
                [
                    label.text()
                    for label in roi_response_preview.findChildren(QLabel)
                    if ": " in label.text()
                ],
                response_epoch_titles,
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
            roi_table_header = group_table.horizontalHeader()
            assert roi_table_header is not None
            self.assertEqual(
                roi_table_header.defaultAlignment(),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            self.assertEqual(
                group_table.maximumHeight(),
                (group_matching_roi._ROI_TABLE_VISIBLE_ROWS + 1)
                * group_matching_roi._ROI_TABLE_ROW_HEIGHT
                + 4,
            )
            self.assertEqual(
                roi_table_header.sectionResizeMode(0),
                QHeaderView.ResizeMode.Interactive,
            )
            self.assertEqual(
                roi_table_header.sectionResizeMode(1),
                QHeaderView.ResizeMode.Interactive,
            )
            self.assertEqual(group_table.rowCount(), 1)
            self.assertEqual(
                cast(QGroupBox, first_section).title(),
                "Selected ROIs - Group 1",
            )
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
            group_table.clearSelection()
            for selector in roi_selectors:
                self.assertEqual(selector.currentData(), "")
            self.assertEqual(
                cast(QGroupBox, first_section).title(),
                "Selected ROIs",
            )
            group_table.selectRow(0)
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
