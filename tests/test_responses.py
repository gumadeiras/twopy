"""Tests for grouping ROI dF/F responses by epoch and trial.

Inputs: tiny dF/F matrices and stimulus-labeled frame windows.
Outputs: grouped trial arrays and compact response summaries.
"""

import unittest

import numpy as np

from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.responses import (
    group_delta_f_over_f_by_epoch,
    summarize_grouped_responses,
)
from twopy.analysis.trials import EpochFrameWindow, FrameWindow


class ResponseGroupingTest(unittest.TestCase):
    """Tests response grouping helpers."""

    def test_groups_dff_by_epoch_and_trial(self) -> None:
        """Confirm windows slice dF/F and count trials within each epoch.

        Inputs: four frames, two ROIs, and three epoch windows.
        Outputs: three grouped trials with one-based trial indices per epoch.
        """
        dff = self._dff()
        windows = (
            EpochFrameWindow(FrameWindow(0, 10, 12, "gray_a"), 1, "Gray"),
            EpochFrameWindow(FrameWindow(1, 12, 13, "odor"), 2, "Odor"),
            EpochFrameWindow(FrameWindow(2, 13, 14, "gray_b"), 1, "Gray"),
        )

        grouped = group_delta_f_over_f_by_epoch(
            dff,
            windows,
            data_rate_hz=2.0,
        )

        self.assertEqual(grouped.roi_labels, ("roi_1", "roi_2"))
        self.assertEqual([trial.trial_index for trial in grouped.trials], [1, 1, 2])
        np.testing.assert_array_equal(
            grouped.trials[0].values,
            np.array([[0.0, 1.0], [2.0, 3.0]]),
        )
        np.testing.assert_array_equal(grouped.trials[0].frame_numbers, [10, 11])
        np.testing.assert_allclose(grouped.trials[0].time_seconds, [0.0, 0.5])

    def test_summarizes_each_trial_and_roi(self) -> None:
        """Confirm compact summaries are one row per trial and ROI.

        Inputs: grouped responses with two ROIs.
        Outputs: mean, peak, and minimum response metrics.
        """
        grouped = group_delta_f_over_f_by_epoch(
            self._dff(),
            (EpochFrameWindow(FrameWindow(0, 10, 12, "gray"), 1, "Gray"),),
            data_rate_hz=2.0,
        )

        summaries = summarize_grouped_responses(grouped)

        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0].roi_label, "roi_1")
        self.assertEqual(summaries[0].frame_count, 2)
        self.assertEqual(summaries[0].mean_response, 1.0)
        self.assertEqual(summaries[0].peak_response, 2.0)
        self.assertEqual(summaries[0].min_response, 0.0)

    def test_groups_with_prestimulus_frames(self) -> None:
        """Confirm grouping can include baseline frames before each window.

        Inputs: dF/F trace and one window with one second of pre-window context.
        Outputs: response values and time axis start before stimulus onset.
        """
        grouped = group_delta_f_over_f_by_epoch(
            self._dff(),
            (EpochFrameWindow(FrameWindow(0, 12, 14, "odor"), 2, "Odor"),),
            data_rate_hz=2.0,
            pre_window_seconds=1.0,
        )

        trial = grouped.trials[0]
        self.assertEqual(trial.start_frame, 10)
        self.assertEqual(trial.stop_frame, 14)
        np.testing.assert_array_equal(trial.frame_numbers, [10, 11, 12, 13])
        np.testing.assert_allclose(trial.time_seconds, [-1.0, -0.5, 0.0, 0.5])
        np.testing.assert_array_equal(trial.values, self._dff().values)

    def test_groups_with_poststimulus_frames(self) -> None:
        """Confirm grouping can include frames after each window.

        Inputs: dF/F trace and one window with one second of post-window
            context.
        Outputs: response time axis extends past the epoch offset.
        """
        grouped = group_delta_f_over_f_by_epoch(
            self._dff(),
            (EpochFrameWindow(FrameWindow(0, 10, 12, "odor"), 2, "Odor"),),
            data_rate_hz=2.0,
            post_window_seconds=1.0,
        )

        trial = grouped.trials[0]
        self.assertEqual(trial.start_frame, 10)
        self.assertEqual(trial.stop_frame, 14)
        np.testing.assert_array_equal(trial.frame_numbers, [10, 11, 12, 13])
        np.testing.assert_allclose(trial.time_seconds, [0.0, 0.5, 1.0, 1.5])
        np.testing.assert_array_equal(trial.values, self._dff().values)

    def test_rejects_window_outside_dff_range(self) -> None:
        """Confirm grouping does not silently slice missing frames.

        Inputs: a dF/F trace covering frames ten through fourteen and an early
        window.
        Outputs: a clear validation error.
        """
        with self.assertRaisesRegex(ValueError, "outside dF/F range"):
            group_delta_f_over_f_by_epoch(
                self._dff(),
                (EpochFrameWindow(FrameWindow(0, 9, 11, "early"), 1, "Gray"),),
                data_rate_hz=2.0,
            )

    def test_rejects_negative_prestimulus_duration(self) -> None:
        """Confirm pre-window duration cannot be negative.

        Inputs: negative pre-window duration.
        Outputs: clear validation error.
        """
        with self.assertRaisesRegex(ValueError, "pre_window_seconds"):
            group_delta_f_over_f_by_epoch(
                self._dff(),
                (EpochFrameWindow(FrameWindow(0, 10, 12, "gray"), 1, "Gray"),),
                data_rate_hz=2.0,
                pre_window_seconds=-1.0,
            )

    def test_rejects_negative_poststimulus_duration(self) -> None:
        """Confirm post-window duration cannot be negative.

        Inputs: negative post-window duration.
        Outputs: clear validation error.
        """
        with self.assertRaisesRegex(ValueError, "post_window_seconds"):
            group_delta_f_over_f_by_epoch(
                self._dff(),
                (EpochFrameWindow(FrameWindow(0, 10, 12, "gray"), 1, "Gray"),),
                data_rate_hz=2.0,
                post_window_seconds=-1.0,
            )

    def _dff(self) -> RoiDeltaFOverF:
        """Create a small dF/F object for grouping tests.

        Args:
            None.

        Returns:
            ``RoiDeltaFOverF`` covering absolute frames ten through fourteen.
        """
        values = np.arange(8, dtype=np.float64).reshape(4, 2)
        return RoiDeltaFOverF(
            fluorescence=values + 10.0,
            baseline=np.full_like(values, 10.0),
            values=values,
            labels=("roi_1", "roi_2"),
            start_frame=10,
            stop_frame=14,
            tau=0.0,
            amplitudes=np.array([10.0, 10.0]),
            baseline_frame_numbers=np.array([10.5]),
            baseline_fluorescence=np.array([[10.0, 10.0]]),
            metadata={"method": "test"},
        )


if __name__ == "__main__":
    unittest.main()
