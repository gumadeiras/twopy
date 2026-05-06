"""Tests for saving twopy analysis outputs.

Inputs: small ROI, trace, dF/F, window, and grouped-response objects.
Outputs: inspectable HDF5 files and compact CSV summaries.
"""

import csv
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from twopy.analysis.background_subtraction import BackgroundCorrectedRoiTraces
from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.persistence import (
    ANALYSIS_OUTPUT_FILE_FORMAT,
    save_analysis_outputs,
)
from twopy.analysis.responses import group_delta_f_over_f_by_epoch
from twopy.analysis.trials import EpochFrameWindow, FrameWindow
from twopy.roi import make_roi_set


class PersistenceTest(unittest.TestCase):
    """Tests analysis HDF5 and CSV persistence."""

    def test_saves_analysis_outputs_and_response_summary(self) -> None:
        """Confirm analysis objects are saved in auditable HDF5 groups.

        Inputs: one ROI set, traces, dF/F, epoch window, and grouped response.
        Outputs: one HDF5 file plus one CSV summary row per ROI.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            h5_path = root / "analysis_outputs.h5"
            csv_path = root / "response_summary.csv"
            roi_set = make_roi_set(
                np.array(
                    [
                        [[True, False], [False, False]],
                        [[False, False], [False, True]],
                    ],
                ),
                labels=("roi_1", "roi_2"),
            )
            traces = self._traces()
            dff = self._dff()
            windows = (EpochFrameWindow(FrameWindow(0, 0, 2, "gray"), 1, "Gray"),)
            grouped = group_delta_f_over_f_by_epoch(
                dff,
                windows,
                data_rate_hz=2.0,
            )

            save_analysis_outputs(
                h5_path,
                roi_set=roi_set,
                traces=traces,
                dff=dff,
                epoch_windows=windows,
                grouped_responses=grouped,
                response_summary_csv=csv_path,
            )

            with h5py.File(h5_path, "r") as h5_file:
                self.assertEqual(
                    h5_file.attrs["twopy_format"],
                    ANALYSIS_OUTPUT_FILE_FORMAT,
                )
                self.assertEqual(h5_file["roi_set/masks"].shape, (2, 2, 2))
                self.assertEqual(h5_file["traces/raw_values"].shape, (3, 2))
                self.assertEqual(h5_file["dff/values"].shape, (3, 2))
                self.assertEqual(h5_file["epoch_windows/start_frame"][0], 0)
                trial_group = h5_file["responses/trials/trial_0001"]
                self.assertEqual(trial_group.attrs["epoch_name"], "Gray")
                np.testing.assert_array_equal(
                    trial_group["values"][()],
                    np.array([[0.0, 1.0], [2.0, 3.0]]),
                )

            with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["epoch_name"], "Gray")
            self.assertEqual(rows[0]["roi_label"], "roi_1")
            self.assertEqual(float(rows[0]["mean_response"]), 1.0)

    def test_csv_requires_grouped_responses(self) -> None:
        """Confirm summary CSV is not written without response groups.

        Inputs: a requested CSV path and no grouped response object.
        Outputs: a clear validation error.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = Path(temp_dir) / "analysis_outputs.h5"
            with self.assertRaisesRegex(ValueError, "grouped_responses"):
                save_analysis_outputs(
                    h5_path,
                    response_summary_csv=Path(temp_dir) / "summary.csv",
                )
            self.assertFalse(h5_path.exists())

    def _traces(self) -> BackgroundCorrectedRoiTraces:
        """Create small background-corrected traces.

        Args:
            None.

        Returns:
            ``BackgroundCorrectedRoiTraces`` with three frames and two ROIs.
        """
        raw = np.arange(6, dtype=np.float64).reshape(3, 2) + 10.0
        background = np.ones_like(raw)
        return BackgroundCorrectedRoiTraces(
            raw_values=raw,
            background_values=background,
            corrected_values=raw - background,
            labels=("roi_1", "roi_2"),
            start_frame=0,
            stop_frame=3,
            statistic="mean",
            method="movie_global_percentile",
            metadata={"method": "movie_global_percentile", "percentile": 10.0},
        )

    def _dff(self) -> RoiDeltaFOverF:
        """Create small dF/F output.

        Args:
            None.

        Returns:
            ``RoiDeltaFOverF`` with three frames and two ROIs.
        """
        values = np.arange(6, dtype=np.float64).reshape(3, 2)
        return RoiDeltaFOverF(
            fluorescence=values + 10.0,
            baseline=np.full_like(values, 10.0),
            values=values,
            labels=("roi_1", "roi_2"),
            start_frame=0,
            stop_frame=3,
            tau=0.0,
            amplitudes=np.array([10.0, 10.0]),
            interleave_frame_numbers=np.array([0.5]),
            interleave_fluorescence=np.array([[10.0, 10.0]]),
            metadata={"method": "test", "fit_mode": "robust"},
        )


if __name__ == "__main__":
    unittest.main()
