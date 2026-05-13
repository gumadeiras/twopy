"""Tests for shared twopy output filename constants.

Inputs: filename constants and path helper functions.
Outputs: assertions that default recording, analysis, and export paths stay
stable.
"""

import unittest
from pathlib import Path

from twopy.analysis.workflow import _resolve_output_path, _resolve_summary_csv_path
from twopy.filenames import (
    ALIGNED_MOVIE_FILENAME,
    ANALYSIS_OUTPUT_FILENAME,
    CSV_EXPORTS_DIRNAME,
    EXPORTS_DIRNAME,
    PLOT_EXPORTS_DIRNAME,
    PLOT_WITH_ROIS_EXPORTS_DIRNAME,
    RECORDING_DATA_FILENAME,
    RECORDING_ROI_OVERLAY_EXPORT_STEM,
    RECORDING_VIEW_EXPORT_STEM,
    RESPONSE_SUMMARY_GROUPED_FILENAME,
    RESPONSE_SUMMARY_TRIALS_FILENAME,
    ROI_FILENAME,
    ROI_VIEW_EXPORT_STEM,
)
from twopy.napari.plotting.docks.paths import (
    resolved_analysis_path,
    resolved_roi_save_file,
)


class FilenameConstantsTest(unittest.TestCase):
    """Tests twopy-owned filenames used across native modules."""

    def test_core_filename_values_are_stable(self) -> None:
        """Confirm the user-visible twopy output names do not drift.

        Inputs: shared filename constants.
        Outputs: assertions for the converted, ROI, analysis, and export names.
        """
        self.assertEqual(RECORDING_DATA_FILENAME, "recording_data.h5")
        self.assertEqual(ALIGNED_MOVIE_FILENAME, "aligned_movie.h5")
        self.assertEqual(ROI_FILENAME, "rois.h5")
        self.assertEqual(ANALYSIS_OUTPUT_FILENAME, "analysis_outputs.h5")
        self.assertEqual(EXPORTS_DIRNAME, "exports")
        self.assertEqual(CSV_EXPORTS_DIRNAME, "csvs")
        self.assertEqual(PLOT_EXPORTS_DIRNAME, "plots")
        self.assertEqual(PLOT_WITH_ROIS_EXPORTS_DIRNAME, "plots_with_rois")
        self.assertEqual(
            RESPONSE_SUMMARY_TRIALS_FILENAME,
            "response_summary_trials.csv",
        )
        self.assertEqual(
            RESPONSE_SUMMARY_GROUPED_FILENAME,
            "response_summary_grouped.csv",
        )
        self.assertEqual(RECORDING_VIEW_EXPORT_STEM, "recording_view")
        self.assertEqual(ROI_VIEW_EXPORT_STEM, "roi_view")
        self.assertEqual(
            RECORDING_ROI_OVERLAY_EXPORT_STEM,
            "recording_roi_overlay",
        )

    def test_analysis_workflow_uses_shared_default_names(self) -> None:
        """Confirm script analysis defaults compose paths from shared names.

        Inputs: a synthetic converted recording path and no explicit outputs.
        Outputs: default analysis HDF5 and CSV paths.
        """
        recording_path = Path("/tmp/session") / RECORDING_DATA_FILENAME
        output_path = _resolve_output_path(recording_path, None)

        self.assertEqual(output_path, Path("/tmp/session") / ANALYSIS_OUTPUT_FILENAME)
        self.assertEqual(
            _resolve_summary_csv_path(
                output_path=output_path,
                requested_path=None,
                write_summary_csv=True,
                filename=RESPONSE_SUMMARY_TRIALS_FILENAME,
            ),
            Path("/tmp/session")
            / EXPORTS_DIRNAME
            / CSV_EXPORTS_DIRNAME
            / RESPONSE_SUMMARY_TRIALS_FILENAME,
        )

    def test_napari_empty_defaults_use_shared_names(self) -> None:
        """Confirm empty napari state uses the same default HDF5 filenames.

        Inputs: no loaded recording and no explicit output path.
        Outputs: relative default analysis and ROI save paths.
        """
        self.assertEqual(
            resolved_analysis_path(None, None),
            Path(ANALYSIS_OUTPUT_FILENAME),
        )
        self.assertEqual(resolved_roi_save_file(None, None), Path(ROI_FILENAME))


if __name__ == "__main__":
    unittest.main()
