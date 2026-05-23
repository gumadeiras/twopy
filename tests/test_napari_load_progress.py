"""Tests for the napari recording-load progress dialog.

Inputs: long recording paths and a Qt application.
Outputs: assertions that progress text remains visible in the default dialog.
"""

from tests.napari_support import Path, QApplication, unittest
from twopy.napari.load_progress import RecordingLoadProgressDialog


class RecordingLoadProgressDialogTest(unittest.TestCase):
    """Tests for the asynchronous recording-load progress dialog."""

    def test_long_path_sets_enough_label_height(self) -> None:
        """Confirm long paths are not clipped by the default dialog height.

        Inputs: a long source path similar to microscope output locations.
        Outputs: the path label keeps the full text and reserves wrapped height.
        """
        _ = QApplication.instance() or QApplication([])
        long_path = Path(
            "/Volumes/DataNAS_2/2p_microscope_data/"
            "dh146gal4_w-uasdsf_+-orcolexa_lexaopcs"
            "chrimsondtomato_+/2026_05_23/very_long_recording_name"
        )
        dialog = RecordingLoadProgressDialog(
            total_count=10,
            on_cancel_pending=lambda: None,
        )

        dialog.update_progress(
            completed_count=1,
            total_count=10,
            current_path=long_path,
            phase="Reading recording data...",
        )

        path_label = dialog._path
        self.assertEqual(path_label.text(), str(long_path))
        self.assertGreaterEqual(path_label.height(), path_label.minimumHeight())
        self.assertGreater(
            path_label.minimumHeight(),
            path_label.fontMetrics().height(),
        )


if __name__ == "__main__":
    unittest.main()
