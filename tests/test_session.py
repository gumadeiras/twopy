"""Tests for two-photon session path discovery.

Inputs: temporary session folders built with the expected microscope file names.
Outputs: assertions that discovery returns the right typed paths and errors.
"""

import unittest
from pathlib import Path

from tests.tempdir import temporary_directory

from twopy.session import discover_session_files


class DiscoverSessionFilesTest(unittest.TestCase):
    """Tests the auditable session-folder contract used before analysis."""

    def test_discovers_required_files_and_optional_saved_analysis(self) -> None:
        """Confirm discovery returns required paths and optional MATLAB analysis.

        Inputs: a temporary session folder with all required files.
        Outputs: assertions that flexible and fixed filenames are discovered.
        """
        with temporary_directory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_required_session(session_dir)
            (session_dir / "savedAnalysis").mkdir()

            files = discover_session_files(session_dir)

            self.assertEqual(files.session_dir, session_dir)
            self.assertEqual(files.stimulus_data_dir, session_dir / "stimulusData")
            self.assertEqual(files.raw_tif.name, "stimulus_name_changes_001.tif")
            self.assertEqual(
                files.alignment_text.name,
                "stimulus_name_changes_001_ch1_disinterleaved_alignment.txt",
            )
            self.assertEqual(
                files.stimulus_filebackup_zip,
                session_dir / "stimulusData" / "filebackup.zip",
            )
            self.assertEqual(files.saved_analysis_dir, session_dir / "savedAnalysis")

    def test_saved_analysis_is_optional(self) -> None:
        """Confirm a new unanalyzed session is still valid.

        Inputs: a temporary session folder without ``savedAnalysis``.
        Outputs: an assertion that the optional path is represented as ``None``.
        """
        with temporary_directory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_required_session(session_dir)

            files = discover_session_files(session_dir)

            self.assertIsNone(files.saved_analysis_dir)

    def test_missing_required_file_raises_clear_error(self) -> None:
        """Confirm missing required data stops discovery early.

        Inputs: a temporary session folder missing ``alignedMovie.mat``.
        Outputs: an assertion that the error names the missing file.
        """
        with temporary_directory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_required_session(session_dir)
            (session_dir / "alignedMovie.mat").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "aligned movie"):
                discover_session_files(session_dir)

    def test_multiple_raw_tifs_raise_clear_error(self) -> None:
        """Confirm ambiguous raw movies are not guessed.

        Inputs: a temporary session folder with two TIFF movies.
        Outputs: an assertion that discovery reports the ambiguity.
        """
        with temporary_directory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_required_session(session_dir)
            (session_dir / "second_movie.tif").touch()

            with self.assertRaisesRegex(ValueError, "raw TIFF movie"):
                discover_session_files(session_dir)

    def test_ignores_macos_appledouble_sidecars(self) -> None:
        """Confirm mounted-volume sidecars do not make discovery ambiguous.

        Inputs: a valid session folder plus ``._`` sidecar files created by
            macOS on some network and external volumes.
        Outputs: discovered raw movie and alignment paths ignore sidecars.
        """
        with temporary_directory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_required_session(session_dir)
            (session_dir / "._stimulus_name_changes_001.tif").touch()
            (
                session_dir
                / "._stimulus_name_changes_001_ch1_disinterleaved_alignment.txt"
            ).touch()

            files = discover_session_files(session_dir)

            self.assertEqual(files.raw_tif.name, "stimulus_name_changes_001.tif")
            self.assertEqual(
                files.alignment_text.name,
                "stimulus_name_changes_001_ch1_disinterleaved_alignment.txt",
            )

    def _write_required_session(self, session_dir: Path) -> None:
        """Create the smallest valid session-folder fixture.

        Args:
            session_dir: Temporary folder that should receive fixture files.

        Returns:
            None. The function creates files and folders on disk for the test.

        The fixture mirrors the stable contract from real microscope output
        while keeping names short enough for a test reader to inspect quickly.
        """
        (session_dir / "stimulusData").mkdir()
        (session_dir / "alignedMovie.mat").touch()
        (session_dir / "stimulus_name_changes_001.tif").touch()
        (
            session_dir / "stimulus_name_changes_001_ch1_disinterleaved_alignment.txt"
        ).touch()
        (session_dir / "defaultAlignChannel.txt").touch()
        (session_dir / "highResPd.mat").touch()
        (session_dir / "imageDescription.mat").touch()
        (session_dir / "imagingResPd.mat").touch()
        (session_dir / "stimulusData" / "filebackup.zip").touch()


if __name__ == "__main__":
    unittest.main()
