"""Tests for the shared napari load-path dialog.

Inputs: separate manual and CSV starting folders.
Outputs: assertions that both load actions use one dialog implementation.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

from qtpy.QtWidgets import QFileDialog

from twopy.napari import load_picker


class NapariLoadPickerTest(unittest.TestCase):
    """Shared napari load-path dialog tests."""

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


if __name__ == "__main__":
    unittest.main()
