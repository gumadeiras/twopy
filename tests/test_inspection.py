"""Tests for whole-recording file inspection.

Inputs: a small temporary recording with MATLAB, TIFF, text, CSV, and ZIP files.
Outputs: assertions that twopy classifies and summarizes recognized files.
"""

import csv
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
            tifffile.imwrite(
                session_dir / "movie.tif",
                [[[1, 2], [3, 4]]],
                description=(
                    "state.configName='test_config'\n"
                    "state.acq.zoomFactor=9\n"
                    "state.acq.frameRate=13.0208333333333\n"
                    "state.motor.absZPosition=-8503.4\n"
                ),
            )
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
            self.assertEqual(tiff.page_count, 1)
            self.assertEqual(tiff.tags["ImageWidth"], "2")
            self.assertEqual(tiff.scanimage_state["configName"], "test_config")
            self.assertEqual(tiff.scanimage_state["acq.zoomFactor"], "9")
            assert text_preview is not None
            self.assertEqual(text_preview[0], "animal=A")
            self.assertEqual(by_name["textStimData.csv"].kind, "csv")
            self.assertEqual(by_name["filebackup.zip"].zip_entries, ("inside.txt",))

    def test_writes_tiff_metadata_csv(self) -> None:
        """Confirm TIFF metadata can be written as a simple CSV table.

        Inputs: a temporary recording with one TIFF that has ScanImage metadata.
        Outputs: a CSV row with TIFF layout and selected ScanImage fields.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            csv_path = session_dir / "metadata" / "tiff_metadata.csv"
            tifffile.imwrite(
                session_dir / "movie.tif",
                [[[1, 2], [3, 4]]],
                description=(
                    "state.configName='test_config'\n"
                    "state.acq.pixelsPerLine=2\n"
                    "state.acq.linesPerFrame=2\n"
                    "state.acq.zoomFactor=9\n"
                    "state.motor.absZPosition=-8503.4\n"
                ),
            )

            inspect_recording_files(session_dir, tiff_metadata_csv=csv_path)

            with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["relative_path"], "movie.tif")
            self.assertEqual(rows[0]["page_count"], "1")
            self.assertEqual(rows[0]["scanimage_config_name"], "test_config")
            self.assertEqual(rows[0]["scanimage_acq_zoom_factor"], "9")
            self.assertEqual(rows[0]["scanimage_motor_abs_z_position"], "-8503.4")


if __name__ == "__main__":
    unittest.main()
