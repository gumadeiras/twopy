"""Napari Load tab tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from tests.napari_support import (
    Any,
    ConvertedRecording,
    NapariAdapterTestCase,
    Path,
    QApplication,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QWidget,
    _FakeViewer,
    _load_recording_widget,
    _write_converted_recording,
    _write_source_recording_shape,
    add_twopy_magicgui_controls,
    cast,
    chdir,
    csv,
    load_converted_recording,
    make_manual_fov_group_rows,
    napari_controls,
    np,
    patch,
    process_qt_events_until,
    roi_label_image_from_layer,
    save_manual_fov_group_rows,
    temporary_directory,
    unittest,
)
from twopy.napari.loaded_recordings_csv import (
    load_recording_paths_csv,
    write_loaded_recordings_csv,
)
from twopy.napari.output_routing import NapariOutputRoute
from twopy.napari.session import LoadedNapariRecording
from twopy.napari.state import (
    read_last_recording_csv_folder,
    read_last_recording_folder,
    write_last_recording_folder,
)


class NapariLoadTabTest(NapariAdapterTestCase):
    """Napari Load tab tests."""

    def test_recording_folder_selection_loads_recording_layers(self) -> None:
        """Confirm selecting a recording folder populates an empty viewer.

        Inputs: fake viewer, control dock, and tiny converted recording folder.
        Outputs: mean image, movie preview, and ROI Labels layer.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            roi_save_file = root / "drawn_rois.h5"
            _write_converted_recording(root)
            viewer = _FakeViewer()

            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=roi_save_file,
            )
            load_widget = _load_recording_widget(control_docks.load_widget)

            load_widget.recording_folder.value = root
            process_qt_events_until(lambda: len(viewer.images) == 2)

            self.assertEqual(len(viewer.images), 2)
            self.assertEqual(len(viewer.labels), 1)
            self.assertEqual(np.asarray(viewer.images[1].data).shape[0], 3)
            np.testing.assert_array_equal(
                roi_label_image_from_layer(viewer.labels[0]),
                np.zeros((2, 2), dtype=np.int64),
            )

    def test_manual_load_button_loads_multiple_selected_folders(self) -> None:
        """Confirm manual loading accepts several selected recording folders.

        Inputs: fake viewer, control dock, and two tiny converted folders
        returned by the manual chooser.
        Outputs: both folders are loaded into the same viewer and list panel.
        """
        _ = QApplication.instance() or QApplication([])
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
            panel = cast(QWidget, control_docks.loaded_recordings_widget)
            loaded_list = panel.findChild(QListWidget)
            load_panel = cast(QWidget, control_docks.load_widget)
            load_buttons = {
                button.text(): button for button in load_panel.findChildren(QPushButton)
            }

            with patch.object(
                napari_controls,
                "_choose_recording_paths",
                return_value=(first, second),
            ):
                load_buttons["Load manually"].click()
                process_qt_events_until(lambda: loaded_list.count() == 2)

            assert loaded_list is not None
            self.assertEqual(loaded_list.count(), 2)
            self.assertEqual(loaded_list.currentRow(), 1)
            self.assertEqual(len(viewer.images), 4)
            self.assertEqual(len(viewer.labels), 2)

    def test_load_tab_saves_and_loads_recording_list_csv(self) -> None:
        """Confirm a saved loaded-recordings CSV can be loaded manually.

        Inputs: two converted folders loaded into one viewer, then exported as
        a Load-tab CSV and selected from a fresh Load CSV list dialog.
        Outputs: the CSV lists both recording paths and reloads both rows.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            _write_converted_recording(first, source_session_dir=first)
            _write_converted_recording(second, source_session_dir=second)
            csv_path = root / "loaded_recordings.csv"
            viewer = _FakeViewer()
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            load_widget = _load_recording_widget(control_docks.load_widget)
            load_widget(recording_folder=first)
            process_qt_events_until(lambda: len(viewer.images) == 2)
            load_widget(recording_folder=second)
            process_qt_events_until(lambda: len(viewer.images) == 4)
            sidebar_buttons = {
                button.text(): button
                for button in cast(QWidget, control_docks.sidebar_widget).findChildren(
                    QPushButton,
                )
            }

            with patch.object(
                napari_controls,
                "_choose_loaded_recordings_csv_save_path",
                return_value=csv_path,
            ):
                sidebar_buttons["Save loaded list"].click()

            with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(
                [Path(row["recording_path"]).resolve(strict=False) for row in rows],
                [
                    first.resolve(strict=False),
                    second.resolve(strict=False),
                ],
            )
            self.assertTrue(
                rows[0]["recording_data_path"].endswith("recording_data.h5")
            )
            fov_path = root / "fov_groups.csv"
            save_manual_fov_group_rows(
                make_manual_fov_group_rows(
                    {
                        first: "fov_1",
                        second: "fov_2",
                    },
                    notes={
                        first: "first loaded-list note",
                        second: "second loaded-list note",
                    },
                ),
                fov_path,
            )

            fresh_viewer = _FakeViewer()
            fresh_docks = add_twopy_magicgui_controls(
                fresh_viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            fresh_load_panel = cast(QWidget, fresh_docks.load_widget)
            fresh_loaded_panel = cast(QWidget, fresh_docks.loaded_recordings_widget)
            fresh_loaded_list = fresh_loaded_panel.findChild(QListWidget)
            load_buttons = {
                button.text(): button
                for button in fresh_load_panel.findChildren(QPushButton)
            }

            with patch.object(
                napari_controls,
                "_choose_recording_csv_paths",
                return_value=(csv_path,),
            ):
                load_buttons["Load CSV list"].click()
                process_qt_events_until(lambda: fresh_loaded_list.count() == 2)

            assert fresh_loaded_list is not None
            self.assertEqual(fresh_loaded_list.count(), 2)
            self.assertEqual(fresh_loaded_list.currentRow(), 1)
            self.assertEqual(len(fresh_viewer.images), 4)
            self.assertEqual(len(fresh_viewer.labels), 2)
            fresh_group_matching_widget = cast(
                QWidget,
                fresh_docks.group_matching_widget,
            )
            fov_path_edit = fresh_group_matching_widget.findChild(
                QLineEdit,
                "fov_group_path",
            )
            assert fov_path_edit is not None
            self.assertEqual(Path(fov_path_edit.text()), fov_path)
            fov_note_edits = fresh_group_matching_widget.findChildren(
                QLineEdit,
                "fov_recording_note",
            )
            self.assertEqual(
                {note_edit.text() for note_edit in fov_note_edits},
                {"first loaded-list note", "second loaded-list note"},
            )

    def test_load_csv_list_retargets_auto_group_matching_paths_after_unload_all(
        self,
    ) -> None:
        """Confirm auto group-matching CSV paths follow each loaded-list folder."""
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)

            def write_recording_list(folder_name: str) -> tuple[Path, Path, Path]:
                folder = root / folder_name
                first = folder / "first"
                second = folder / "second"
                first.mkdir(parents=True)
                second.mkdir()
                _write_converted_recording(first, source_session_dir=first)
                _write_converted_recording(second, source_session_dir=second)
                csv_path = folder / "loaded_recordings.csv"
                with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
                    writer = csv.DictWriter(
                        csv_file,
                        fieldnames=("recording_path", "recording_data_path"),
                    )
                    writer.writeheader()
                    for recording in (first, second):
                        writer.writerow(
                            {
                                "recording_path": str(recording),
                                "recording_data_path": str(
                                    recording / "twopy" / "recording_data.h5"
                                ),
                            },
                        )
                save_manual_fov_group_rows(
                    make_manual_fov_group_rows({first: "fov_1", second: "fov_1"}),
                    folder / "fov_groups.csv",
                )
                return csv_path, folder / "fov_groups.csv", folder / "roi_matches.csv"

            first_csv, first_fov, first_roi = write_recording_list("first_set")
            second_csv, second_fov, second_roi = write_recording_list("second_set")
            viewer = _FakeViewer()
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            load_panel = cast(QWidget, control_docks.load_widget)
            loaded_panel = cast(QWidget, control_docks.loaded_recordings_widget)
            group_matching_widget = cast(QWidget, control_docks.group_matching_widget)
            load_buttons = {
                button.text(): button for button in load_panel.findChildren(QPushButton)
            }
            unload_buttons = {
                button.text(): button
                for button in loaded_panel.findChildren(QPushButton)
            }
            fov_path_edit = group_matching_widget.findChild(
                QLineEdit,
                "fov_group_path",
            )
            roi_path_edit = group_matching_widget.findChild(
                QLineEdit,
                "roi_match_path",
            )
            assert fov_path_edit is not None
            assert roi_path_edit is not None

            with patch.object(
                napari_controls,
                "_choose_recording_csv_paths",
                return_value=(first_csv,),
            ):
                load_buttons["Load CSV list"].click()
                process_qt_events_until(lambda: Path(fov_path_edit.text()) == first_fov)

            self.assertEqual(Path(fov_path_edit.text()), first_fov)
            self.assertEqual(Path(roi_path_edit.text()), first_roi)

            unload_buttons["Unload selected"].click()
            unload_buttons["Unload selected"].click()

            with patch.object(
                napari_controls,
                "_choose_recording_csv_paths",
                return_value=(second_csv,),
            ):
                load_buttons["Load CSV list"].click()
                process_qt_events_until(
                    lambda: Path(fov_path_edit.text()) == second_fov
                )

            self.assertEqual(Path(fov_path_edit.text()), second_fov)
            self.assertEqual(Path(roi_path_edit.text()), second_roi)

            unload_buttons["Unload all"].click()

            with patch.object(
                napari_controls,
                "_choose_recording_csv_paths",
                return_value=(first_csv,),
            ):
                load_buttons["Load CSV list"].click()
                process_qt_events_until(lambda: Path(fov_path_edit.text()) == first_fov)

            self.assertEqual(Path(fov_path_edit.text()), first_fov)
            self.assertEqual(Path(roi_path_edit.text()), first_roi)

            manual_roi_path = root / "manual_roi_matches.csv"
            roi_path_edit.setText(str(manual_roi_path))
            roi_path_edit.editingFinished.emit()
            unload_buttons["Unload all"].click()

            with patch.object(
                napari_controls,
                "_choose_recording_csv_paths",
                return_value=(second_csv,),
            ):
                load_buttons["Load CSV list"].click()
                process_qt_events_until(
                    lambda: Path(fov_path_edit.text()) == second_fov
                )

            self.assertEqual(Path(fov_path_edit.text()), second_fov)
            self.assertEqual(Path(roi_path_edit.text()), manual_roi_path)

    def test_load_csv_list_targets_group_matching_paths_when_files_are_missing(
        self,
    ) -> None:
        """Confirm missing group-matching files still become default save paths.

        Inputs: a loaded-recordings CSV in a folder without FOV or ROI match
            CSV files.
        Outputs: Group Matching points its FOV and ROI path fields at that
            folder so the first save writes beside the recording-list CSV.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = root / "recording"
            recording.mkdir()
            _write_converted_recording(recording, source_session_dir=recording)
            csv_path = root / "loaded_recordings.csv"
            expected_fov_path = root / "fov_groups.csv"
            expected_roi_path = root / "roi_matches.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=("recording_path", "recording_data_path"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "recording_path": str(recording),
                        "recording_data_path": str(
                            recording / "twopy" / "recording_data.h5"
                        ),
                    },
                )
            viewer = _FakeViewer()
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            load_panel = cast(QWidget, control_docks.load_widget)
            loaded_panel = cast(QWidget, control_docks.loaded_recordings_widget)
            loaded_list = loaded_panel.findChild(QListWidget)
            group_matching_widget = cast(QWidget, control_docks.group_matching_widget)
            fov_view = cast(Any, group_matching_widget)._fov_view
            load_buttons = {
                button.text(): button for button in load_panel.findChildren(QPushButton)
            }
            status_label = fov_view.findChild(
                QLabel,
                "group_matching_caption",
            )
            fov_path_edit = group_matching_widget.findChild(
                QLineEdit,
                "fov_group_path",
            )
            roi_path_edit = group_matching_widget.findChild(
                QLineEdit,
                "roi_match_path",
            )
            assert loaded_list is not None
            assert status_label is not None
            assert fov_path_edit is not None
            assert roi_path_edit is not None

            with patch.object(
                napari_controls,
                "_choose_recording_csv_paths",
                return_value=(csv_path,),
            ):
                load_buttons["Load CSV list"].click()
                process_qt_events_until(
                    lambda: Path(fov_path_edit.text()) == expected_fov_path
                )

            self.assertEqual(loaded_list.count(), 1)
            self.assertEqual(Path(fov_path_edit.text()), expected_fov_path)
            self.assertEqual(Path(roi_path_edit.text()), expected_roi_path)
            self.assertIn(
                f"FOV CSV will be saved to {expected_fov_path}",
                status_label.text(),
            )
            self.assertFalse(expected_fov_path.exists())
            self.assertFalse(expected_roi_path.exists())

    def test_load_csv_list_uses_separate_remembered_folder(self) -> None:
        """Confirm CSV-list dialogs do not overwrite manual-load memory.

        Inputs: separate manual and CSV folders persisted through the Load tab.
        Outputs: CSV dialogs reopen at the CSV folder, while manual recording
        memory remains on the source-recording folder.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            manual_folder = root / "manual_recordings"
            csv_folder = root / "csv_lists"
            recording = root / "recording"
            manual_folder.mkdir()
            csv_folder.mkdir()
            recording.mkdir()
            _write_converted_recording(recording, source_session_dir=recording)
            write_last_recording_folder(manual_folder)
            csv_path = csv_folder / "loaded_recordings.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=("recording_path", "recording_data_path"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "recording_path": str(recording),
                        "recording_data_path": str(
                            recording / "twopy" / "recording_data.h5"
                        ),
                    },
                )
            viewer = _FakeViewer()
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            load_panel = cast(QWidget, control_docks.load_widget)
            load_buttons = {
                button.text(): button for button in load_panel.findChildren(QPushButton)
            }

            with patch.object(
                napari_controls.QFileDialog,
                "getOpenFileNames",
                return_value=([str(csv_path)], ""),
            ) as choose_csv:
                load_buttons["Load CSV list"].click()
                process_qt_events_until(lambda: len(viewer.images) == 2)

            first_call_args = choose_csv.call_args.args
            self.assertEqual(first_call_args[2], "")
            self.assertEqual(read_last_recording_folder(), manual_folder.resolve())
            self.assertEqual(read_last_recording_csv_folder(), csv_folder.resolve())

            with patch.object(
                napari_controls.QFileDialog,
                "getOpenFileNames",
                return_value=([], ""),
            ) as choose_csv_again:
                load_buttons["Load CSV list"].click()

            second_call_args = choose_csv_again.call_args.args
            self.assertEqual(second_call_args[2], str(csv_folder.resolve()))

    def test_loaded_recording_csv_prefers_valid_recording_data_path(self) -> None:
        """Confirm saved CSVs reload valid converted files directly.

        Inputs: a CSV row with both the source recording path and an existing
            ``recording_data_path`` file.
        Outputs: CSV expansion returns the converted file path so batch reloads
            skip slower source-path resolution.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            converted_dir = root / "published"
            converted_path = _write_converted_recording(
                converted_dir,
                source_session_dir=source_dir,
            )
            csv_path = root / "loaded_recordings.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=("recording_path", "recording_data_path"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "recording_path": str(source_dir),
                        "recording_data_path": str(converted_path),
                    },
                )

            self.assertEqual(load_recording_paths_csv(csv_path), (converted_path,))

    def test_load_tab_csv_converts_from_recording_path_when_h5_missing(
        self,
    ) -> None:
        """Confirm CSV loading falls back when the HDF5 file is missing.

        Inputs: a CSV row with a valid source recording path and a missing
            ``recording_data_path`` value.
        Outputs: loading runs conversion from the source path and opens the new
            converted files.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            missing_recording_data_path = root / "missing" / "recording_data.h5"
            csv_path = root / "loaded_recordings.csv"
            _write_source_recording_shape(source_dir)
            with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=("recording_path", "recording_data_path"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "recording_path": str(source_dir),
                        "recording_data_path": str(missing_recording_data_path),
                    },
                )

            def write_loadable_conversion(
                source: Path,
                output_dir: Path | None = None,
                **_kwargs: object,
            ) -> ConvertedRecording:
                destination = output_dir or source / "twopy"
                destination.mkdir(parents=True, exist_ok=True)
                recording_path = _write_converted_recording(
                    destination,
                    source_session_dir=source,
                )
                return ConvertedRecording(
                    path=recording_path,
                    movie_path=destination / "aligned_movie.h5",
                    source_session_dir=source,
                    movie_shape=(3, 2, 2),
                    mean_image_start_frame=0,
                    mean_image_stop_frame=3,
                )

            viewer = _FakeViewer()
            original_cwd = Path.cwd()
            try:
                chdir(root)
                control_docks = add_twopy_magicgui_controls(
                    viewer,
                    roi_labels_layer=None,
                    roi_save_file=Path("unused.h5"),
                )
                load_panel = cast(QWidget, control_docks.load_widget)
                loaded_panel = cast(QWidget, control_docks.loaded_recordings_widget)
                loaded_list = loaded_panel.findChild(QListWidget)
                load_buttons = {
                    button.text(): button
                    for button in load_panel.findChildren(QPushButton)
                }

                with (
                    patch.dict(
                        "os.environ",
                        {"TWOPY_CONFIG": str(root / "missing-config.yml")},
                    ),
                    patch.object(
                        napari_controls,
                        "_choose_recording_csv_paths",
                        return_value=(csv_path,),
                    ),
                    patch(
                        "twopy.napari.loading.convert_recording_to_twopy",
                        side_effect=write_loadable_conversion,
                    ) as convert,
                ):
                    load_buttons["Load CSV list"].click()
                    process_qt_events_until(lambda: loaded_list.count() == 1)
            finally:
                chdir(original_cwd)

            assert loaded_list is not None
            convert.assert_called_once_with(source_dir.resolve())
            self.assertEqual(loaded_list.count(), 1)
            loaded_item = loaded_list.item(0)
            assert loaded_item is not None
            self.assertEqual(loaded_item.text(), str(source_dir))
            self.assertEqual(len(viewer.images), 2)
            self.assertEqual(len(viewer.labels), 1)
            self.assertFalse(missing_recording_data_path.exists())

    def test_load_tab_saved_recording_list_uses_source_paths_for_cached_loads(
        self,
    ) -> None:
        """Confirm saved loaded-recording CSVs do not expose cache paths.

        Inputs: source recording with converted data already in the analysis
            cache.
        Outputs: the loaded list and saved CSV use the source session path while
            the audit column records the published HDF5 path.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            cache_dir = root / "cache" / "fly" / "stim" / "2023" / "10_17"
            publish_dir = root / "publish" / "fly" / "stim" / "2023" / "10_17"
            csv_path = root / "loaded_recordings.csv"
            _write_source_recording_shape(source_dir)
            cache_dir.mkdir(parents=True)
            _write_converted_recording(
                cache_dir,
                source_session_dir=source_dir,
            )
            (root / "config.yml").write_text(
                f"database_path: {root / 'db'}\n"
                "data_paths:\n"
                f"  - {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {root / 'cache'}\n"
                f"analysis_output: {root / 'publish'}\n",
                encoding="utf-8",
            )
            viewer = _FakeViewer()
            original_cwd = Path.cwd()
            try:
                chdir(root)
                control_docks = add_twopy_magicgui_controls(
                    viewer,
                    roi_labels_layer=None,
                    roi_save_file=Path("unused.h5"),
                )
                load_widget = _load_recording_widget(control_docks.load_widget)
                load_widget(recording_folder=source_dir)
                process_qt_events_until(lambda: len(viewer.images) == 2)
                picker_text = str(load_widget.recording_folder.line_edit.value)
                panel = cast(QWidget, control_docks.loaded_recordings_widget)
                loaded_list = panel.findChild(QListWidget)
                sidebar_buttons = {
                    button.text(): button
                    for button in cast(
                        QWidget,
                        control_docks.sidebar_widget,
                    ).findChildren(QPushButton)
                }

                with patch.object(
                    napari_controls,
                    "_choose_loaded_recordings_csv_save_path",
                    return_value=csv_path,
                ):
                    sidebar_buttons["Save loaded list"].click()
            finally:
                chdir(original_cwd)

            assert loaded_list is not None
            loaded_item = loaded_list.item(0)
            assert loaded_item is not None
            self.assertIn(source_dir.name, picker_text)
            self.assertNotIn(str(root / "cache"), picker_text)
            self.assertEqual(loaded_item.text(), str(source_dir))
            with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(rows[0]["recording_path"], str(source_dir))
            self.assertEqual(
                Path(rows[0]["recording_data_path"]).resolve(strict=False),
                (publish_dir / "recording_data.h5").resolve(strict=False),
            )

    def test_loaded_recording_csv_uses_loaded_output_route(self) -> None:
        """Confirm CSV writing uses the route resolved at load time.

        Inputs: one loaded recording with a local cache folder and a final
            source-local output route.
        Outputs: the reusable CSV points at the route's output HDF5 file.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "external-source"
            local_dir = root / "cache"
            publish_dir = source_dir / "twopy"
            recording_path = _write_converted_recording(
                local_dir,
                source_session_dir=source_dir,
            )
            recording = load_converted_recording(recording_path)
            csv_path = root / "loaded_recordings.csv"

            write_loaded_recordings_csv(
                (
                    LoadedNapariRecording(
                        recording=recording,
                        output_route=NapariOutputRoute(
                            local_root=local_dir,
                            publish_root=publish_dir,
                        ),
                        roi_save_file=local_dir / "rois.h5",
                        mean_image_layer=object(),
                        movie_layer=None,
                        roi_labels_layer=None,
                    ),
                ),
                csv_path,
            )

            with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(rows[0]["recording_path"], str(source_dir))
            self.assertEqual(
                Path(rows[0]["recording_data_path"]),
                publish_dir / "recording_data.h5",
            )

    def test_load_tab_warns_when_using_cache_for_unavailable_source(self) -> None:
        """Confirm cache-only source loads are visible to the user.

        Inputs: missing source recording folder with matching cached converted
            files.
        Outputs: loading succeeds from cache and shows one source-unavailable
            warning dialog.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            cache_dir = root / "cache" / "fly" / "stim" / "2023" / "10_17"
            cache_dir.mkdir(parents=True)
            cached_recording_path = _write_converted_recording(
                cache_dir,
                source_session_dir=source_dir,
            )
            (root / "config.yml").write_text(
                f"database_path: {root / 'db'}\n"
                "data_paths:\n"
                f"  - {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {root / 'cache'}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            viewer = _FakeViewer()
            original_cwd = Path.cwd()
            try:
                chdir(root)
                control_docks = add_twopy_magicgui_controls(
                    viewer,
                    roi_labels_layer=None,
                    roi_save_file=Path("unused.h5"),
                )
                load_widget = _load_recording_widget(control_docks.load_widget)
                with patch.object(napari_controls.QMessageBox, "warning") as warning:
                    result = load_widget(recording_folder=source_dir)
                    process_qt_events_until(lambda: warning.called)
            finally:
                chdir(original_cwd)

            self.assertEqual(result, "Loading started.")
            warning.assert_called_once()
            _parent, title, message = warning.call_args.args
            self.assertEqual(title, "Source path unavailable")
            self.assertIn(str(source_dir), message)
            self.assertIn(str(cached_recording_path.resolve()), message)
            self.assertIn("will not sync back", message)


if __name__ == "__main__":
    unittest.main()
