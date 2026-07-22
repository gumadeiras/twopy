"""Napari database search tests.

Inputs: shared fake napari state and tiny converted recordings.
Outputs: assertions for one napari workflow area.
"""

from napari.settings import get_settings
from napari.settings._fields import Theme as NapariThemeId
from napari.utils.theme import get_theme
from qtpy.QtCore import QAbstractItemModel, QEvent, QModelIndex, Qt
from qtpy.QtGui import QKeyEvent

from tests.napari_support import (
    ExperimentFavoriteEditDialog,
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
    update_database_search_favorite,
)
from twopy.napari.database_search import _TreeBranchStyle


class NapariDatabaseSearchTest(NapariAdapterTestCase):
    """Napari database search tests."""

    def test_database_search_filter_enter_only_searches(self) -> None:
        """Confirm Return in a database-search filter does not save favorites.

        Inputs: a dialog with a focused filter field containing saveable text.
        Outputs: pressing Return emits only the search action, not the dialog's
        first enabled button action.
        """

        class ProbeDialog(ExperimentSearchDialog):
            """Record actions reached through Qt signal wiring."""

            def __init__(self) -> None:
                """Create a search dialog that records activated actions."""
                self.events: list[str] = []
                super().__init__(
                    on_load_recording_paths=lambda paths: RecordingLoadResult(
                        loaded_count=len(paths),
                    ),
                )

            def search(self) -> None:
                """Record a database-search action."""
                self.events.append("search")

            def save_current_favorite(self) -> None:
                """Record a save-favorite action."""
                self.events.append("save")

        _ = QApplication.instance() or QApplication([])
        dialog = ProbeDialog()

        dialog._user_filter.setText("Gus")
        dialog._user_filter.setFocus()
        QApplication.sendEvent(
            dialog._user_filter,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Return,
                Qt.KeyboardModifier.NoModifier,
            ),
        )

        self.assertEqual(dialog.events, ["search"])

    def test_database_tree_chevron_follows_napari_theme(self) -> None:
        """Confirm branch chevrons use readable colors in light and dark themes.

        Inputs: one database dialog while napari changes theme.
        Outputs: branch color matching the active napari text color.
        """
        application = QApplication.instance() or QApplication([])
        settings = get_settings()
        original_theme = settings.appearance.theme
        dialog = ExperimentSearchDialog(
            on_load_recording_paths=lambda paths: RecordingLoadResult(
                loaded_count=len(paths),
            ),
        )
        try:
            settings.appearance.theme = NapariThemeId("light")
            application.processEvents()
            self.assertEqual(
                dialog._tree_branch_style._color.name(),
                get_theme("light").text.as_hex(),
            )

            settings.appearance.theme = NapariThemeId("dark")
            application.processEvents()
            self.assertEqual(
                dialog._tree_branch_style._color.name(),
                get_theme("dark").text.as_hex(),
            )
        finally:
            settings.appearance.theme = original_theme
            application.processEvents()

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

            splitter = dialog.findChild(QSplitter)
            if splitter is None:
                self.fail("Database search dialog did not create its splitter.")
            self.assertEqual(splitter.objectName(), "database_search_splitter")
            self.assertEqual(splitter.handleWidth(), 12)
            self.assertEqual(
                dialog._tree.selectionMode(),
                QAbstractItemView.SelectionMode.ExtendedSelection,
            )
            self.assertEqual(dialog._tree.objectName(), "database_experiment_tree")
            self.assertIsInstance(dialog._tree_branch_style, _TreeBranchStyle)
            header_style = dialog.styleSheet().split(
                "QTreeWidget#database_experiment_tree QHeaderView {",
                maxsplit=1,
            )[1]
            header_section_style = dialog.styleSheet().split(
                "QTreeWidget#database_experiment_tree QHeaderView::section {",
                maxsplit=1,
            )[1]
            self.assertIn(
                "border-radius: 0",
                header_style.split("}", maxsplit=1)[0],
            )
            self.assertIn(
                "border-radius: 0",
                header_section_style.split("}", maxsplit=1)[0],
            )
            self.assertEqual(
                dialog._favorites_list.objectName(),
                "database_favorites_list",
            )
            self.assertTrue(dialog._favorites_list.uniformItemSizes())
            self.assertEqual(dialog._favorites_list.spacing(), 0)
            self.assertGreaterEqual(dialog._favorites_list.minimumHeight(), 140)
            self.assertEqual(
                dialog._favorites_list.verticalScrollMode(),
                QAbstractItemView.ScrollMode.ScrollPerPixel,
            )
            self.assertEqual(
                dialog._favorites_list.horizontalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
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
            self.assertEqual(bottom_buttons, ["Close", "Load selected"])
            search_buttons = {
                button.text(): button
                for button in dialog.findChildren(QPushButton)
                if button.text() in {"Search", "Save as favorite..."}
            }
            self.assertEqual(
                search_buttons["Search"].property("twopyRole"),
                "primary",
            )
            self.assertEqual(
                search_buttons["Save as favorite..."].property("twopyRole"),
                "secondary",
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

    def test_database_search_favorite_edit_keeps_edited_row_unique(self) -> None:
        """Confirm editing one favorite removes duplicates outside that row.

        Inputs: three favorites and an edit that gives the last row the first
        row's display name.
        Outputs: the edited row remains while the duplicate name row is removed.
        """
        first, second, third = _sample_ordered_favorites()
        edited = ExperimentSearchFavorite(
            name="first",
            filters=ExperimentSearchFilters(sensor="g8"),
        )

        updated = update_database_search_favorite((first, second, third), 2, edited)

        self.assertEqual(
            [favorite.name for favorite in updated],
            ["second", "first"],
        )
        self.assertEqual(updated[1].filters.sensor, "g8")

    def test_database_search_favorite_editor_routes_fields(self) -> None:
        """Confirm the favorite editor maps widgets to favorite data.

        Inputs: one favorite with every editable filter field populated.
        Outputs: the editor shows normalized starting text and returns the
        edited raw text in the matching favorite fields.
        """
        _ = QApplication.instance() or QApplication([])
        original = ExperimentSearchFavorite(
            name=" original ",
            filters=ExperimentSearchFilters(
                user=" Gus ",
                cell_type=" ALPN ",
                sensor=" g6f ",
                stimulus=" bars ",
                date="2024/01/02",
            ),
        )
        dialog = ExperimentFavoriteEditDialog(original)

        self.assertEqual(dialog._name.text(), "original")
        self.assertEqual(dialog._user.text(), "Gus")
        self.assertEqual(dialog._cell_type.text(), "ALPN")
        self.assertEqual(dialog._sensor.text(), "g6f")
        self.assertEqual(dialog._stimulus.text(), "bars")
        self.assertEqual(dialog._date.text(), "2024-01-02")

        dialog._name.setText(" edited ")
        dialog._user.setText(" user ")
        dialog._cell_type.setText(" cell ")
        dialog._sensor.setText(" sensor ")
        dialog._stimulus.setText(" stimulus ")
        dialog._date.setText(" 2025/03/04 ")

        favorite = dialog.favorite()
        self.assertEqual(favorite.name, " edited ")
        self.assertEqual(favorite.filters.user, " user ")
        self.assertEqual(favorite.filters.cell_type, " cell ")
        self.assertEqual(favorite.filters.sensor, " sensor ")
        self.assertEqual(favorite.filters.stimulus, " stimulus ")
        self.assertEqual(favorite.filters.date, " 2025/03/04 ")

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
            self.assertFalse(dialog._edit_favorite_button.isEnabled())
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
            self.assertTrue(dialog._edit_favorite_button.isEnabled())
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
            self.assertFalse(dialog._edit_favorite_button.isEnabled())
            self.assertFalse(dialog._remove_favorite_button.isEnabled())
            self.assertEqual(load_database_search_favorites(favorites_path), ())

    def test_database_search_dialog_edits_selected_favorite(self) -> None:
        """Confirm Edit updates the selected favorite name and filters.

        Inputs: two saved favorites and an accepted edit dialog.
        Outputs: the selected row is replaced, filters are normalized, and the
        YAML file reloads the edited favorite.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            favorites_path = Path(temp_dir) / "favorites.yml"
            first, second, _third = _sample_ordered_favorites()
            save_database_search_favorites((first, second), favorites_path)
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: RecordingLoadResult(
                    loaded_count=len(paths),
                ),
                favorites_path=favorites_path,
            )
            dialog._favorites_list.setCurrentRow(1)
            edited = ExperimentSearchFavorite(
                name="renamed",
                filters=ExperimentSearchFilters(
                    user="  Gustavo ",
                    sensor=" g8 ",
                    date="2024/01/02",
                ),
            )

            with (
                patch.object(
                    ExperimentFavoriteEditDialog,
                    "exec",
                    return_value=QDialog.DialogCode.Accepted,
                ),
                patch.object(
                    ExperimentFavoriteEditDialog,
                    "favorite",
                    return_value=edited,
                ),
            ):
                dialog.edit_selected_favorite()

            self.assertEqual(
                [favorite.name for favorite in dialog._favorites],
                ["first", "renamed"],
            )
            self.assertEqual(dialog._favorites[1].filters.user, "Gustavo")
            self.assertEqual(dialog._favorites[1].filters.sensor, "g8")
            self.assertEqual(dialog._favorites[1].filters.date, "2024-01-02")
            self.assertEqual(_favorite_list_names(dialog), ["first", "renamed"])
            self.assertEqual(dialog._favorites_list.currentRow(), 1)
            loaded = load_database_search_favorites(favorites_path)
            self.assertEqual(loaded, dialog._favorites)

    def test_database_search_dialog_reorders_favorites_by_dragged_rows(
        self,
    ) -> None:
        """Confirm dragged favorite rows become the persisted order.

        Inputs: three saved favorites and the Search database favorites list.
        Outputs: moving the visible bottom row to the top updates memory, YAML,
        and the next dialog instance in that same visible order.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            favorites_path = Path(temp_dir) / "favorites.yml"
            first, second, third = _sample_ordered_favorites()
            save_database_search_favorites((first, second, third), favorites_path)
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: RecordingLoadResult(
                    loaded_count=len(paths),
                ),
                favorites_path=favorites_path,
            )

            self.assertEqual(
                dialog._favorites_list.dragDropMode(),
                QAbstractItemView.DragDropMode.InternalMove,
            )
            moved = _favorite_list_model(dialog).moveRows(
                QModelIndex(),
                2,
                1,
                QModelIndex(),
                0,
            )

            self.assertTrue(moved)
            self.assertEqual(
                [favorite.name for favorite in dialog._favorites],
                ["third", "first", "second"],
            )
            reloaded_names = [
                favorite.name
                for favorite in load_database_search_favorites(favorites_path)
            ]
            self.assertEqual(reloaded_names, ["third", "first", "second"])
            self.assertEqual(dialog._favorites_list.currentRow(), 0)

            fresh_dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: RecordingLoadResult(
                    loaded_count=len(paths),
                ),
                favorites_path=favorites_path,
            )
            self.assertEqual(
                _favorite_list_names(fresh_dialog),
                ["third", "first", "second"],
            )

    def test_database_search_dialog_reverts_reorder_when_save_fails(self) -> None:
        """Confirm failed favorite reorder writes do not change dialog state.

        Inputs: three saved favorites and a simulated YAML write failure.
        Outputs: the dialog re-renders the previous order and leaves the local
        favorites file unchanged.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            favorites_path = Path(temp_dir) / "favorites.yml"
            first, second, third = _sample_ordered_favorites()
            save_database_search_favorites((first, second, third), favorites_path)
            original_text = favorites_path.read_text(encoding="utf-8")
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: RecordingLoadResult(
                    loaded_count=len(paths),
                ),
                favorites_path=favorites_path,
            )

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
                moved = _favorite_list_model(dialog).moveRows(
                    QModelIndex(),
                    2,
                    1,
                    QModelIndex(),
                    0,
                )

            self.assertTrue(moved)
            self.assertEqual(len(errors), 1)
            self.assertEqual(dialog._favorites, (first, second, third))
            self.assertEqual(
                _favorite_list_names(dialog),
                ["first", "second", "third"],
            )
            self.assertEqual(dialog._favorites_list.currentRow(), 2)
            self.assertEqual(favorites_path.read_text(encoding="utf-8"), original_text)

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
            self.assertTrue(dialog._edit_favorite_button.isEnabled())
            self.assertTrue(dialog._remove_favorite_button.isEnabled())

    def test_database_search_dialog_keeps_memory_when_favorite_edit_fails(
        self,
    ) -> None:
        """Confirm failed favorite edits do not update in-memory state.

        Inputs: one selected favorite and a simulated disk-write failure.
        Outputs: the selected favorite remains visible, usable, and unchanged.
        """
        _ = QApplication.instance() or QApplication([])
        with temporary_directory() as temp_dir:
            favorites_path = Path(temp_dir) / "favorites.yml"
            saved = ExperimentSearchFavorite(
                name="saved",
                filters=ExperimentSearchFilters(user="Gus"),
            )
            save_database_search_favorites((saved,), favorites_path)
            original_text = favorites_path.read_text(encoding="utf-8")
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: RecordingLoadResult(
                    loaded_count=len(paths),
                ),
                favorites_path=favorites_path,
            )
            dialog._favorites_list.setCurrentRow(0)
            edited = ExperimentSearchFavorite(
                name="edited",
                filters=ExperimentSearchFilters(sensor="g8"),
            )

            errors: list[Exception] = []
            with (
                patch(
                    "twopy.napari.database_search.save_database_search_favorites",
                    side_effect=OSError("disk full"),
                ),
                patch.object(
                    ExperimentFavoriteEditDialog,
                    "exec",
                    return_value=QDialog.DialogCode.Accepted,
                ),
                patch.object(
                    ExperimentFavoriteEditDialog,
                    "favorite",
                    return_value=edited,
                ),
                patch.object(
                    dialog,
                    "_show_favorite_error",
                    side_effect=lambda error: errors.append(error),
                ),
            ):
                dialog.edit_selected_favorite()

            self.assertEqual(len(errors), 1)
            self.assertEqual(dialog._favorites, (saved,))
            self.assertEqual(_favorite_list_names(dialog), ["saved"])
            self.assertEqual(favorites_path.read_text(encoding="utf-8"), original_text)

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
        Outputs: pressing Return in the focused result tree sends both source
        folders to the loader callback.
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
            dialog._tree.setCurrentItem(leaves[0])
            for item in leaves:
                item.setSelected(True)
            dialog._tree.setFocus()
            QApplication.sendEvent(
                dialog._tree,
                QKeyEvent(
                    QEvent.Type.KeyPress,
                    Qt.Key.Key_Return,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )
            QApplication.processEvents()

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


def _sample_ordered_favorites() -> tuple[
    ExperimentSearchFavorite,
    ExperimentSearchFavorite,
    ExperimentSearchFavorite,
]:
    """Return three distinct favorites for order-sensitive dialog tests."""
    return (
        ExperimentSearchFavorite(
            name="first",
            filters=ExperimentSearchFilters(user="Gus"),
        ),
        ExperimentSearchFavorite(
            name="second",
            filters=ExperimentSearchFilters(cell_type="ALPN"),
        ),
        ExperimentSearchFavorite(
            name="third",
            filters=ExperimentSearchFilters(stimulus="combo"),
        ),
    )


def _favorite_list_names(dialog: ExperimentSearchDialog) -> list[str]:
    """Return the visible favorite row labels from a search dialog."""
    names: list[str] = []
    for row in range(dialog._favorites_list.count()):
        item = dialog._favorites_list.item(row)
        assert item is not None
        names.append(item.text())
    return names


def _favorite_list_model(dialog: ExperimentSearchDialog) -> QAbstractItemModel:
    """Return the Qt model backing the favorites list."""
    model = dialog._favorites_list.model()
    assert model is not None
    return model


if __name__ == "__main__":
    unittest.main()
