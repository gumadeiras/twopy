"""Tests for the shared napari load-path dialog.

Inputs: separate manual and CSV starting folders.
Outputs: assertions that both load actions use one dialog implementation.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QListView,
    QTreeView,
)

from tests.tempdir import temporary_directory
from twopy.napari import load_picker


class NapariLoadPickerTest(unittest.TestCase):
    """Shared napari load-path dialog tests."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create the Qt application required by real file dialogs."""
        super().setUpClass()
        cls._application = QApplication.instance() or QApplication([])

    def test_load_actions_use_one_dialog_with_separate_start_folders(self) -> None:
        """Confirm both load actions use the shared dialog configuration.

        Inputs: different manual and CSV folders.
        Outputs: two shared-dialog calls with the correct folder and path mode.
        """
        manual_folder = Path("/manual")
        csv_folder = Path("/csv")

        with patch.object(
            load_picker,
            "_choose_existing_paths",
            return_value=(),
        ) as choose:
            load_picker.choose_recording_folders(manual_folder)
            load_picker.choose_loaded_recording_csvs(csv_folder)

        manual_call, csv_call = choose.call_args_list
        self.assertEqual(manual_call.kwargs["start_folder"], manual_folder)
        self.assertEqual(
            manual_call.kwargs["file_mode"],
            QFileDialog.FileMode.Directory,
        )
        self.assertTrue(manual_call.kwargs["show_directories_only"])
        self.assertEqual(csv_call.kwargs["start_folder"], csv_folder)
        self.assertEqual(
            csv_call.kwargs["file_mode"],
            QFileDialog.FileMode.ExistingFiles,
        )
        self.assertFalse(csv_call.kwargs["show_directories_only"])
        self.assertEqual(
            csv_call.kwargs["name_filters"],
            ("CSV files (*.csv)", "All files (*)"),
        )

    def test_manual_dialog_has_the_shared_configuration(self) -> None:
        """Confirm the manual picker uses the real shared dialog settings."""
        with temporary_directory() as temp_dir:
            start_folder = Path(temp_dir)
            dialog = load_picker._build_load_dialog(
                title="Load recording folders",
                start_folder=start_folder,
                file_mode=QFileDialog.FileMode.Directory,
                show_directories_only=True,
            )

            self.assertEqual(dialog.objectName(), "twopy_load_path_dialog")
            self.assertNotEqual(dialog.styleSheet(), "")
            self.assertEqual(dialog.windowTitle(), "Load recording folders")
            self.assertEqual(dialog.fileMode(), QFileDialog.FileMode.Directory)
            self.assertTrue(dialog.testOption(QFileDialog.Option.DontUseNativeDialog))
            self.assertTrue(dialog.testOption(QFileDialog.Option.ShowDirsOnly))
            self.assertEqual(
                dialog.labelText(QFileDialog.DialogLabel.Accept),
                "Load",
            )
            self.assertEqual(
                Path(dialog.directory().absolutePath()),
                start_folder.resolve(),
            )
            self._assert_file_views_allow_multiple_selection(dialog)

    def test_csv_dialog_has_the_shared_configuration(self) -> None:
        """Confirm the CSV picker uses the real shared dialog settings."""
        with temporary_directory() as temp_dir:
            start_folder = Path(temp_dir)
            dialog = load_picker._build_load_dialog(
                title="Load recording lists",
                start_folder=start_folder,
                file_mode=QFileDialog.FileMode.ExistingFiles,
                show_directories_only=False,
                name_filters=("CSV files (*.csv)", "All files (*)"),
            )

            self.assertEqual(dialog.objectName(), "twopy_load_path_dialog")
            self.assertNotEqual(dialog.styleSheet(), "")
            self.assertEqual(dialog.windowTitle(), "Load recording lists")
            self.assertEqual(
                dialog.fileMode(),
                QFileDialog.FileMode.ExistingFiles,
            )
            self.assertTrue(dialog.testOption(QFileDialog.Option.DontUseNativeDialog))
            self.assertFalse(dialog.testOption(QFileDialog.Option.ShowDirsOnly))
            self.assertEqual(
                dialog.nameFilters(),
                ["CSV files (*.csv)", "All files (*)"],
            )
            self.assertEqual(dialog.selectedNameFilter(), "CSV files (*.csv)")
            self.assertEqual(
                Path(dialog.directory().absolutePath()),
                start_folder.resolve(),
            )
            self._assert_file_views_allow_multiple_selection(dialog)

    def test_selected_browse_folder_uses_the_selection_parent(self) -> None:
        """Confirm a picker reopens beside the last selected item.

        Inputs: one selected recording folder.
        Outputs: its containing browse folder.
        """
        selected = Path("/recordings/session_a")

        self.assertEqual(
            load_picker.selected_browse_folder((selected,)),
            Path("/recordings"),
        )
        self.assertIsNone(load_picker.selected_browse_folder(()))

    def _assert_file_views_allow_multiple_selection(
        self,
        dialog: QFileDialog,
    ) -> None:
        """Confirm both file-content views allow multiple selected paths."""
        list_view = dialog.findChild(QListView, "listView")
        tree_view = dialog.findChild(QTreeView, "treeView")
        self.assertIsNotNone(list_view)
        self.assertIsNotNone(tree_view)
        assert list_view is not None
        assert tree_view is not None
        self.assertEqual(
            list_view.selectionMode(),
            QAbstractItemView.SelectionMode.ExtendedSelection,
        )
        self.assertEqual(
            tree_view.selectionMode(),
            QAbstractItemView.SelectionMode.ExtendedSelection,
        )


if __name__ == "__main__":
    unittest.main()
