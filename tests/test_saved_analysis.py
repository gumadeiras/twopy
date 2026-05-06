"""Tests for read-only parity helpers.

Inputs: tiny HDF5-backed MATLAB-style ``lastRoi`` files.
Outputs: arrays used by parity checks against twopy analysis.
"""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from twopy.parity import load_saved_analysis_last_roi


class SavedAnalysisTest(unittest.TestCase):
    """Tests read-only saved-analysis loading."""

    def test_loads_last_roi_fields_for_parity_checks(self) -> None:
        """Confirm key ``lastRoi`` arrays are loaded from references.

        Inputs: a synthetic saved-analysis file with ROI masks, epoch metadata,
        and one referenced ROI trace matrix.
        Outputs: typed arrays with traces shaped frame-by-ROI.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "saved.mat"
            self._write_saved_analysis(path)

            loaded = load_saved_analysis_last_roi(path)

            np.testing.assert_array_equal(
                loaded.roi_mask_initial,
                np.array([[1.0, 0.0], [0.0, 2.0]]),
            )
            np.testing.assert_array_equal(loaded.epoch_list, np.array([1, 1, 2]))
            np.testing.assert_array_equal(loaded.epoch_start_times[0], np.array([1, 3]))
            np.testing.assert_array_equal(loaded.epoch_durations[0], np.array([2, 2]))
            np.testing.assert_array_equal(
                loaded.time_by_rois_initial[0],
                np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
            )

    def _write_saved_analysis(self, path: Path) -> None:
        """Write a small HDF5 file with MATLAB cell-reference structure.

        Args:
            path: Destination file path.

        Returns:
            None.
        """
        with h5py.File(path, "w") as h5_file:
            refs = h5_file.create_group("#refs#")
            start_times = refs.create_dataset("start_times", data=np.array([[1, 3]]))
            durations = refs.create_dataset("durations", data=np.array([[2, 2]]))
            traces = refs.create_dataset(
                "traces",
                data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            )

            last_roi = h5_file.create_group("lastRoi")
            last_roi.create_dataset(
                "roiMaskInitial",
                data=np.array([[1.0, 0.0], [0.0, 2.0]]),
            )
            last_roi.create_dataset("epochList", data=np.array([[1, 1, 2]]))
            epoch_start_times = last_roi.create_dataset(
                "epochStartTimes",
                shape=(1, 1),
                dtype=h5py.ref_dtype,
            )
            epoch_start_times[0, 0] = start_times.ref
            epoch_durations = last_roi.create_dataset(
                "epochDurations",
                shape=(1, 1),
                dtype=h5py.ref_dtype,
            )
            epoch_durations[0, 0] = durations.ref
            time_by_rois = last_roi.create_dataset(
                "timeByRoisInitial",
                shape=(1, 1),
                dtype=h5py.ref_dtype,
            )
            time_by_rois[0, 0] = traces.ref


if __name__ == "__main__":
    unittest.main()
