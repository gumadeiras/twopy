"""Tests for whole-recording file inspection.

Inputs: a small temporary recording with MATLAB, TIFF, text, CSV, and ZIP files.
Outputs: assertions that twopy classifies and summarizes recognized files.
"""

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import scipy.io
import tifffile

from twopy.inspection import inspect_recording_files


class InspectRecordingFilesTest(unittest.TestCase):
    """Tests file-type inspection for one recording folder."""

    def test_inspects_all_recognized_file_types(self) -> None:
        """Confirm recognized recording files produce typed summaries.

        Inputs: a temporary folder containing representative source files.
        Outputs: assertions for kind detection and type-specific summaries.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            scipy.io.savemat(session_dir / "imageDescription.mat", {"zoom": [[2.0]]})
            tifffile.imwrite(session_dir / "movie.tif", [[[1, 2], [3, 4]]])
            (session_dir / "metadata.txt").write_text("animal=A\ntrial=1\n")
            (session_dir / "textStimData.csv").write_text("trial,odor\n1,a\n")
            with ZipFile(session_dir / "filebackup.zip", "w") as archive:
                archive.writestr("inside.txt", "backup")

            inspection = inspect_recording_files(session_dir)

            by_name = {str(file.relative_path): file for file in inspection.files}
            tiff = by_name["movie.tif"].tiff
            text_preview = by_name["metadata.txt"].text_preview
            self.assertEqual(by_name["imageDescription.mat"].kind, "matlab")
            assert tiff is not None
            self.assertEqual(tiff.series_count, 1)
            assert text_preview is not None
            self.assertEqual(text_preview[0], "animal=A")
            self.assertEqual(by_name["textStimData.csv"].kind, "csv")
            self.assertEqual(by_name["filebackup.zip"].zip_entries, ("inside.txt",))


if __name__ == "__main__":
    unittest.main()
