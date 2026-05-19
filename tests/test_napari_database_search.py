"""Napari database search tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from tests.napari_support import (
    ExperimentFavoriteErrorDialog,
    ExperimentSearchDialog,
    ExperimentSearchErrorDialog,
    ExperimentSearchFavorite,
    ExperimentSearchFilters,
    NapariAdapterTestCase,
    Path,
    QAbstractItemView,
    QApplication,
    QDialog,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    RecordingLoadErrorDialog,
    RecordingLoadFailure,
    RecordingLoadResult,
    _record_loaded_paths,
    _tree_leaf_items,
    _write_database,
    load_database_search_favorites,
    patch,
    replace_database_search_favorite,
    save_database_search_favorites,
    sqlite3,
    temporary_directory,
    unittest,
)


class NapariDatabaseSearchTest(NapariAdapterTestCase):
    """Napari database search tests."""

    def test_database_search_dialog_loads_selected_hierarchy_paths(self) -> None:
        """Confirm DB search selections resolve to configured source paths.

        Inputs: a temporary config, Clark-lab-style DB, and dialog callback.
        Outputs: selecting a hierarchy node sends all descendant source paths to
        the loader callback.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            database_dir = root / "db"
            data_dir = root / "data"
            database_dir.mkdir()
            data_dir.mkdir()
            _write_database(database_dir / "experimentLog.db")
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {database_dir}\n"
                "data_paths:\n"
                f"  - {data_dir}\n"
                "database_access: direct\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            loaded_paths: list[tuple[Path, ...]] = []
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: _record_loaded_paths(
                    loaded_paths,
                    paths,
                ),
                config_path=config_path,
            )

            dialog._user_filter.setText("Gus")
            dialog._date_filter.setText("2023/10/17")
            dialog.search()

            self.assertIsNotNone(dialog.findChild(QSplitter))
            self.assertEqual(
                dialog._tree.selectionMode(),
                QAbstractItemView.SelectionMode.ExtendedSelection,
            )
            self.assertEqual(dialog._user_filter.placeholderText(), "gustavo...")
            self.assertEqual(dialog._cell_type_filter.placeholderText(), "ALPN...")
            self.assertEqual(dialog._sensor_filter.placeholderText(), "g6f...")
            self.assertEqual(
                dialog._stimulus_filter.placeholderText(),
                "bars_singles_stim=5s_bars=3x400ms_dt=200ms_int=20...",
            )
            self.assertEqual(dialog._date_filter.placeholderText(), "2025-10-22...")
            self.assertEqual(dialog._date_filter.text(), "2023-10-17")
            self.assertEqual(dialog._tree.columnWidth(1), 72)
            self.assertEqual(dialog._user_filter.textMargins().left(), 6)
            self.assertEqual(dialog._user_filter.textMargins().right(), 6)
            self.assertFalse(dialog._user_filter.font().italic())
            self.assertTrue(dialog._cell_type_filter.font().italic())
            self.assertNotIn(
                "Typing...",
                {label.text() for label in dialog.findChildren(QLabel)},
            )
            dialog._cell_type_filter.setText("LN6")
            self.assertFalse(dialog._cell_type_filter.font().italic())
            dialog._cell_type_filter.clear()
            self.assertTrue(dialog._cell_type_filter.font().italic())
            bottom_buttons = [
                button.text()
                for button in dialog.findChildren(QPushButton)
                if button.text() in {"Load selected", "Close"}
            ]
            self.assertEqual(bottom_buttons, ["Load selected", "Close"])
            search_buttons = {
                button.text(): button
                for button in dialog.findChildren(QPushButton)
                if button.text() in {"Search", "Save favorite..."}
            }
            self.assertIn(
                "background-color: #f2f2f2", search_buttons["Search"].styleSheet()
            )
            self.assertIn(
                "background-color: #2d7dd2",
                search_buttons["Save favorite..."].styleSheet(),
            )

            result = dialog._tree.topLevelItem(0)
            assert result is not None
            result.setSelected(True)
            dialog.load_selected()

            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
            self.assertEqual(
                loaded_paths,
                [(data_dir / "genotype" / "stimulus" / "2023" / "10_17" / "10_02_49",)],
            )

    def test_database_search_favorites_round_trip_and_deduplicate(self) -> None:
        """Confirm search favorites persist normalized filter sets.

        Inputs: two favorites with equivalent filters and different names.
        Outputs: one persisted favorite with stripped text and normalized date.
        """
        with temporary_directory() as temp_dir:
            favorites_path = Path(temp_dir) / "favorites.yml"
            first = ExperimentSearchFavorite(
                name="first",
                filters=ExperimentSearchFilters(
                    user=" Gus ",
                    cell_type="ALPN",
                    stimulus=" bars ",
                    date="2023/10/17",
                ),
            )
            renamed = ExperimentSearchFavorite(
                name="renamed",
                filters=ExperimentSearchFilters(
                    user="Gus",
                    cell_type="ALPN",
                    stimulus="bars",
                    date="2023-10-17",
                ),
            )

            favorites = replace_database_search_favorite((), first)
            favorites = replace_database_search_favorite(favorites, renamed)
            save_database_search_favorites(favorites, favorites_path)

            loaded = load_database_search_favorites(favorites_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].name, "renamed")
            self.assertEqual(loaded[0].filters.user, "Gus")
            self.assertEqual(loaded[0].filters.cell_type, "ALPN")
            self.assertEqual(loaded[0].filters.stimulus, "bars")
            self.assertEqual(loaded[0].filters.date, "2023-10-17")

    def test_database_search_dialog_saves_uses_and_removes_favorites(self) -> None:
        """Confirm favorite buttons manage reusable search filters.

        Inputs: a search dialog with a temporary favorites file.
        Outputs: Save persists filters, Use restores them and searches, and
        Remove deletes only the selected favorite.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            database_dir = root / "db"
            data_dir = root / "data"
            database_dir.mkdir()
            data_dir.mkdir()
            _write_database(database_dir / "experimentLog.db")
            config_path = root / "config.yml"
            favorites_path = root / "favorites.yml"
            config_path.write_text(
                f"database_path: {database_dir}\n"
                "data_paths:\n"
                f"  - {data_dir}\n"
                "database_access: direct\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: RecordingLoadResult(
                    loaded_count=len(paths),
                ),
                config_path=config_path,
                favorites_path=favorites_path,
            )

            self.assertFalse(dialog._save_favorite_button.isEnabled())
            self.assertFalse(dialog._use_favorite_button.isEnabled())
            self.assertFalse(dialog._remove_favorite_button.isEnabled())

            dialog._user_filter.setText("Gus")
            dialog._stimulus_filter.setText("combo")
            dialog._date_filter.setText("2023/10/17")
            with patch.object(
                dialog,
                "_favorite_name_from_user",
                return_value="Gus combo",
            ):
                dialog.save_current_favorite()

            self.assertEqual(dialog._favorites_list.count(), 1)
            self.assertTrue(dialog._use_favorite_button.isEnabled())
            self.assertTrue(dialog._remove_favorite_button.isEnabled())
            loaded = load_database_search_favorites(favorites_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].filters.date, "2023-10-17")

            dialog._user_filter.clear()
            dialog._stimulus_filter.clear()
            dialog._date_filter.clear()
            searches: list[bool] = []
            with patch.object(
                dialog,
                "search",
                side_effect=lambda: searches.append(True),
            ):
                dialog.use_selected_favorite()

            self.assertEqual(dialog._user_filter.text(), "Gus")
            self.assertEqual(dialog._stimulus_filter.text(), "combo")
            self.assertEqual(dialog._date_filter.text(), "2023-10-17")
            self.assertEqual(searches, [True])

            dialog.remove_selected_favorite()

            self.assertEqual(dialog._favorites_list.count(), 0)
            self.assertFalse(dialog._use_favorite_button.isEnabled())
            self.assertFalse(dialog._remove_favorite_button.isEnabled())
            self.assertEqual(load_database_search_favorites(favorites_path), ())

    def test_database_search_dialog_keeps_memory_when_favorite_save_fails(
        self,
    ) -> None:
        """Confirm failed favorite writes do not update in-memory state.

        Inputs: one loaded favorite and a simulated disk-write failure.
        Outputs: the dialog still shows and uses the last successfully saved
        favorite list.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            favorites_path = Path(temp_dir) / "favorites.yml"
            saved = ExperimentSearchFavorite(
                name="saved",
                filters=ExperimentSearchFilters(user="Gus"),
            )
            save_database_search_favorites((saved,), favorites_path)
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: RecordingLoadResult(
                    loaded_count=len(paths),
                ),
                favorites_path=favorites_path,
            )

            dialog._stimulus_filter.setText("combo")
            errors: list[Exception] = []
            with (
                patch(
                    "twopy.napari.database_search.save_database_search_favorites",
                    side_effect=OSError("disk full"),
                ),
                patch.object(
                    dialog,
                    "_favorite_name_from_user",
                    return_value="new favorite",
                ),
                patch.object(
                    dialog,
                    "_show_favorite_error",
                    side_effect=lambda error: errors.append(error),
                ),
            ):
                dialog.save_current_favorite()

            self.assertEqual(len(errors), 1)
            self.assertEqual(dialog._favorites, (saved,))
            self.assertEqual(dialog._favorites_list.count(), 1)
            item = dialog._favorites_list.item(0)
            assert item is not None
            self.assertEqual(item.text(), "saved")

    def test_database_search_dialog_keeps_memory_when_favorite_remove_fails(
        self,
    ) -> None:
        """Confirm failed favorite removal does not update in-memory state.

        Inputs: one selected favorite and a simulated disk-write failure.
        Outputs: the selected favorite remains visible and usable.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            favorites_path = Path(temp_dir) / "favorites.yml"
            saved = ExperimentSearchFavorite(
                name="saved",
                filters=ExperimentSearchFilters(user="Gus"),
            )
            save_database_search_favorites((saved,), favorites_path)
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: RecordingLoadResult(
                    loaded_count=len(paths),
                ),
                favorites_path=favorites_path,
            )
            dialog._favorites_list.setCurrentRow(0)

            errors: list[Exception] = []
            with (
                patch(
                    "twopy.napari.database_search.save_database_search_favorites",
                    side_effect=OSError("disk full"),
                ),
                patch.object(
                    dialog,
                    "_show_favorite_error",
                    side_effect=lambda error: errors.append(error),
                ),
            ):
                dialog.remove_selected_favorite()

            self.assertEqual(len(errors), 1)
            self.assertEqual(dialog._favorites, (saved,))
            self.assertEqual(dialog._favorites_list.count(), 1)
            self.assertTrue(dialog._use_favorite_button.isEnabled())
            self.assertTrue(dialog._remove_favorite_button.isEnabled())

    def test_database_search_dialog_blocks_favorite_writes_after_load_failure(
        self,
    ) -> None:
        """Confirm corrupt favorites are not overwritten as an empty list.

        Inputs: an invalid favorites YAML file and edited search filters.
        Outputs: save stays disabled and direct save reports the original load
        error without rewriting the corrupt file.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            favorites_path = Path(temp_dir) / "favorites.yml"
            original_text = "favorites: invalid\n"
            favorites_path.write_text(original_text, encoding="utf-8")
            with patch.object(
                ExperimentFavoriteErrorDialog,
                "exec",
                return_value=QDialog.DialogCode.Rejected,
            ):
                dialog = ExperimentSearchDialog(
                    on_load_recording_paths=lambda paths: RecordingLoadResult(
                        loaded_count=len(paths),
                    ),
                    favorites_path=favorites_path,
                )

            dialog._user_filter.setText("Gus")
            self.assertFalse(dialog._save_favorite_button.isEnabled())
            errors: list[Exception] = []
            with patch.object(
                dialog,
                "_show_favorite_error",
                side_effect=lambda error: errors.append(error),
            ):
                dialog.save_current_favorite()

            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], ValueError)
            self.assertEqual(favorites_path.read_text(encoding="utf-8"), original_text)

    def test_database_search_dialog_loads_multiple_selected_lines(self) -> None:
        """Confirm DB search can load several selected result rows at once.

        Inputs: two valid experiment rows and a multi-selected search tree.
        Outputs: Load selected sends both source folders to the loader callback.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            database_dir = root / "db"
            data_dir = root / "data"
            database_dir.mkdir()
            data_dir.mkdir()
            database_path = database_dir / "experimentLog.db"
            _write_database(database_path)
            connection = sqlite3.connect(database_path)
            try:
                connection.execute(
                    """
                    INSERT INTO stimulusPresentation
                        (
                            stimulusPresentationId,
                            fly,
                            relativeDataPath,
                            stimulusFunction,
                            date,
                            dataQuality
                        )
                    VALUES
                        (
                            20006,
                            10923,
                            'genotype\\stimulus\\2023\\10_17\\10_03_00',
                            'combo_stim_singles',
                            '2023-10-17 10:03:00',
                            1
                        )
                    """,
                )
                connection.commit()
            finally:
                connection.close()
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {database_dir}\n"
                "data_paths:\n"
                f"  - {data_dir}\n"
                "database_access: direct\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            loaded_paths: list[tuple[Path, ...]] = []
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: _record_loaded_paths(
                    loaded_paths,
                    paths,
                ),
                config_path=config_path,
            )
            dialog._user_filter.setText("Gus")
            dialog.search()

            root_item = dialog._tree.topLevelItem(0)
            assert root_item is not None
            leaves = _tree_leaf_items(root_item)
            self.assertEqual(len(leaves), 2)
            for item in leaves:
                item.setSelected(True)
            dialog.load_selected()

            self.assertEqual(
                loaded_paths,
                [
                    (
                        data_dir
                        / "genotype"
                        / "stimulus"
                        / "2023"
                        / "10_17"
                        / "10_02_49",
                        data_dir
                        / "genotype"
                        / "stimulus"
                        / "2023"
                        / "10_17"
                        / "10_03_00",
                    )
                ],
            )

    def test_database_load_error_dialog_shows_scrollable_failed_paths(self) -> None:
        """Confirm load failures get a readable scrollable path list.

        Inputs: structured load result with two failed recording paths.
        Outputs: an error dialog with a clear summary and plain scrollable
        failure details.
        """
        _ = QApplication.instance() or QApplication([])
        result = RecordingLoadResult(
            loaded_count=1,
            failures=(
                RecordingLoadFailure(
                    path=Path("/missing/first"),
                    message="Could not find recording_data.h5.",
                ),
                RecordingLoadFailure(
                    path=Path("/missing/second"),
                    message="Missing source files.",
                ),
            ),
        )

        dialog = RecordingLoadErrorDialog(result)
        summary_text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        details = dialog.findChild(QPlainTextEdit)
        assert details is not None

        self.assertIn("Loaded 1 recordings", summary_text)
        self.assertIn("Failed to load 2 recordings", summary_text)
        self.assertTrue(details.isReadOnly())
        self.assertIn("/missing/first", details.toPlainText())
        self.assertIn("Could not find recording_data.h5.", details.toPlainText())
        self.assertIn("/missing/second", details.toPlainText())
        self.assertIn("Missing source files.", details.toPlainText())

    def test_database_search_failure_shows_error_dialog(self) -> None:
        """Confirm database search failures do not look like empty results.

        Inputs: dialog pointed at a missing config file.
        Outputs: search clears the tree and reports the actual error.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            errors: list[Exception] = []
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: RecordingLoadResult(
                    loaded_count=len(paths),
                ),
                config_path=Path(temp_dir) / "missing.yml",
            )

            with patch.object(
                dialog,
                "_show_search_error",
                side_effect=lambda error: errors.append(error),
            ):
                dialog.search()

            self.assertEqual(dialog._tree.topLevelItemCount(), 0)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], FileNotFoundError)

    def test_database_search_error_dialog_shows_details(self) -> None:
        """Confirm search errors get readable scrollable details.

        Inputs: a representative search exception.
        Outputs: a dialog with a clear summary and exception details.
        """
        _ = QApplication.instance() or QApplication([])
        dialog = ExperimentSearchErrorDialog(FileNotFoundError("missing config.yml"))
        summary_text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        details = dialog.findChild(QPlainTextEdit)
        assert details is not None

        self.assertIn("Could not search", summary_text)
        self.assertTrue(details.isReadOnly())
        self.assertIn("FileNotFoundError", details.toPlainText())
        self.assertIn("missing config.yml", details.toPlainText())

    def test_database_search_loads_valid_paths_when_one_row_is_bad(self) -> None:
        """Confirm one bad DB path does not block valid selected siblings.

        Inputs: one valid DB result and one unsafe relativeDataPath.
        Outputs: the valid source path is loaded and the bad row is reported.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            database_dir = root / "db"
            data_dir = root / "data"
            database_dir.mkdir()
            data_dir.mkdir()
            database_path = database_dir / "experimentLog.db"
            _write_database(database_path)
            connection = sqlite3.connect(database_path)
            try:
                connection.execute(
                    """
                    INSERT INTO stimulusPresentation
                        (
                            stimulusPresentationId,
                            fly,
                            relativeDataPath,
                            stimulusFunction,
                            date,
                            dataQuality
                        )
                    VALUES
                        (
                            20006,
                            10923,
                            '..\\escape\\2023\\10_17\\10_03_00',
                            'combo_stim_singles',
                            '2023-10-17 10:03:00',
                            1
                        )
                    """,
                )
                connection.commit()
            finally:
                connection.close()
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {database_dir}\n"
                "data_paths:\n"
                f"  - {data_dir}\n"
                "database_access: direct\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            loaded_paths: list[tuple[Path, ...]] = []
            shown_results: list[RecordingLoadResult] = []
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: _record_loaded_paths(
                    loaded_paths,
                    paths,
                ),
                config_path=config_path,
            )
            dialog._user_filter.setText("Gus")
            dialog.search()

            result = dialog._tree.topLevelItem(0)
            assert result is not None
            result.setSelected(True)
            with patch.object(
                dialog,
                "_show_load_errors",
                side_effect=lambda load_result: shown_results.append(load_result),
            ):
                dialog.load_selected()

            self.assertEqual(
                loaded_paths,
                [(data_dir / "genotype" / "stimulus" / "2023" / "10_17" / "10_02_49",)],
            )
            self.assertEqual(len(shown_results), 1)
            self.assertEqual(shown_results[0].loaded_count, 1)
            self.assertEqual(len(shown_results[0].failures), 1)
            self.assertIn("relativeDataPath", shown_results[0].failures[0].message)


if __name__ == "__main__":
    unittest.main()
