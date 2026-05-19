"""Tests for grouping ROI dF/F responses by epoch and trial.

Inputs: tiny dF/F matrices and stimulus-labeled frame windows.
Outputs: grouped trial arrays and compact response summaries.
"""

import unittest
from dataclasses import replace

import numpy as np

from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.responses import (
    GroupedRoiResponses,
    RoiResponseTrial,
    finite_mean_and_sem,
    group_delta_f_over_f_by_epoch,
    summarize_grouped_responses,
    validate_grouped_roi_responses,
)
from twopy.analysis.trials import EpochFrameWindow, FrameWindow


class ResponseGroupingTest(unittest.TestCase):
    """Tests response grouping helpers."""

    def test_finite_mean_and_sem_uses_sample_sem(self) -> None:
        """Confirm shared mean/SEM helper ignores invalid values consistently.

        Inputs: repeated values with invalid values and one single-sample timepoint.
        Outputs: finite means plus sample SEM, with zero SEM for one sample.
        """
        values = np.array(
            [
                [1.0, 4.0, np.nan],
                [3.0, np.nan, 5.0],
                [5.0, 8.0, np.inf],
            ],
            dtype=np.float64,
        )

        means, sems = finite_mean_and_sem(values, axis=0)

        np.testing.assert_allclose(means, np.array([3.0, 6.0, 5.0]))
        np.testing.assert_allclose(sems, np.array([2.0 / np.sqrt(3.0), 2.0, 0.0]))

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

    def test_validates_grouped_response_contract(self) -> None:
        """Confirm well-formed grouped responses pass shared validation.

        Inputs: one trial with matching value, time, frame, and ROI axes.
        Outputs: no validation error.
        """
        validate_grouped_roi_responses(_grouped_response_contract())

    def test_rejects_grouped_response_with_invalid_data_rate(self) -> None:
        """Confirm grouped responses require a finite positive frame rate.

        Inputs: grouped responses with zero and NaN frame rates.
        Outputs: clear validation errors before downstream math runs.
        """
        for data_rate_hz in (0.0, np.nan):
            with self.subTest(data_rate_hz=data_rate_hz):
                grouped = replace(
                    _grouped_response_contract(),
                    data_rate_hz=float(data_rate_hz),
                )
                with self.assertRaisesRegex(ValueError, "data_rate_hz"):
                    validate_grouped_roi_responses(grouped)

    def test_rejects_grouped_response_with_invalid_trial_array_rank(self) -> None:
        """Confirm trial response values must stay frame-by-ROI matrices.

        Inputs: one trial with a three-dimensional value array.
        Outputs: clear validation error naming the values shape.
        """
        grouped = _replace_grouped_trial(
            _grouped_response_contract(),
            values=np.zeros((2, 2, 1), dtype=np.float64),
        )

        with self.assertRaisesRegex(ValueError, "trial values"):
            validate_grouped_roi_responses(grouped)

    def test_rejects_grouped_response_with_wrong_roi_width(self) -> None:
        """Confirm trial ROI columns must match grouped ROI labels.

        Inputs: two ROI labels and a one-column response matrix.
        Outputs: clear validation error naming the label mismatch.
        """
        grouped = _replace_grouped_trial(
            _grouped_response_contract(),
            values=np.zeros((2, 1), dtype=np.float64),
        )

        with self.assertRaisesRegex(ValueError, "ROI labels"):
            validate_grouped_roi_responses(grouped)

    def test_rejects_grouped_response_with_mismatched_time_axis(self) -> None:
        """Confirm trial time vectors must match response frame rows.

        Inputs: two response frames and one relative time sample.
        Outputs: clear validation error before CSV or QC alignment.
        """
        grouped = _replace_grouped_trial(
            _grouped_response_contract(),
            time_seconds=np.array([0.0], dtype=np.float64),
        )

        with self.assertRaisesRegex(ValueError, "time_seconds"):
            validate_grouped_roi_responses(grouped)

    def test_rejects_grouped_response_with_mismatched_frame_axis(self) -> None:
        """Confirm trial frame-number vectors must match response frame rows.

        Inputs: two response frames and one absolute frame number.
        Outputs: clear validation error before persisted data is trusted.
        """
        grouped = _replace_grouped_trial(
            _grouped_response_contract(),
            frame_numbers=np.array([10], dtype=np.int64),
        )

        with self.assertRaisesRegex(ValueError, "frame_numbers"):
            validate_grouped_roi_responses(grouped)

    def test_rejects_grouped_response_with_wrong_frame_range(self) -> None:
        """Confirm trial frame ranges must describe the stored response rows.

        Inputs: two response frames and a three-frame half-open range.
        Outputs: clear validation error naming the frame range.
        """
        grouped = _replace_grouped_trial(
            _grouped_response_contract(),
            stop_frame=13,
        )

        with self.assertRaisesRegex(ValueError, "frame range"):
            validate_grouped_roi_responses(grouped)

    def test_rejects_grouped_response_with_noncontiguous_frame_numbers(
        self,
    ) -> None:
        """Confirm frame numbers must match the stored half-open range.

        Inputs: response rows with absolute frame numbers from another range.
        Outputs: clear validation error before time-series alignment.
        """
        grouped = _replace_grouped_trial(
            _grouped_response_contract(),
            frame_numbers=np.array([20, 21], dtype=np.int64),
        )

        with self.assertRaisesRegex(ValueError, "frame_numbers"):
            validate_grouped_roi_responses(grouped)

    def test_rejects_grouped_response_with_nonfinite_time(self) -> None:
        """Confirm relative time axes cannot contain non-finite samples.

        Inputs: response rows with a NaN time sample.
        Outputs: clear validation error before plotting or CSV column naming.
        """
        grouped = _replace_grouped_trial(
            _grouped_response_contract(),
            time_seconds=np.array([0.0, np.nan], dtype=np.float64),
        )

        with self.assertRaisesRegex(ValueError, "time_seconds"):
            validate_grouped_roi_responses(grouped)

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


def _grouped_response_contract() -> GroupedRoiResponses:
    """Create one valid grouped response object for contract tests."""
    trial = RoiResponseTrial(
        epoch_number=1,
        epoch_name="Gray",
        trial_index=1,
        window_index=0,
        start_frame=10,
        stop_frame=12,
        frame_numbers=np.array([10, 11], dtype=np.int64),
        time_seconds=np.array([0.0, 0.5], dtype=np.float64),
        values=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
    )
    return GroupedRoiResponses(
        roi_labels=("roi_1", "roi_2"),
        data_rate_hz=2.0,
        trials=(trial,),
    )


def _replace_grouped_trial(
    grouped: GroupedRoiResponses,
    **changes: object,
) -> GroupedRoiResponses:
    """Return grouped responses with changed fields on the first trial."""
    trial = replace(grouped.trials[0], **changes)
    return replace(grouped, trials=(trial,))


if __name__ == "__main__":
    unittest.main()
