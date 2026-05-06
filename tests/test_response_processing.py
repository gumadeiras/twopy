"""Tests for core response post-processing helpers.

Inputs: tiny dF/F objects, grouped ROI responses, and processing options.
Outputs: processed arrays, validation errors, and auditable correlation masks.
"""

import unittest

import numpy as np

from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.response_processing import (
    CorrelationFilterOptions,
    LowPassFilterOptions,
    ResponseProcessingOptions,
    SmoothingOptions,
    nan_aware_moving_average,
    process_grouped_roi_responses,
    process_roi_delta_f_over_f,
)
from twopy.analysis.responses import GroupedRoiResponses, RoiResponseTrial


class ResponseProcessingTest(unittest.TestCase):
    """Tests response-processing contracts and numeric behavior."""

    def test_moving_average_ignores_nan_samples(self) -> None:
        """Confirm smoothing averages finite samples without filling all-NaN gaps.

        Inputs: one trace with a motion-masked NaN sample.
        Outputs: smoothed finite neighbors and preserved NaN-only windows.
        """
        values = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float64)

        smoothed = nan_aware_moving_average(values, window_frames=3)

        np.testing.assert_allclose(smoothed, np.array([1.0, 2.0, 3.0, 3.0]))

        all_nan = nan_aware_moving_average(
            np.array([np.nan], dtype=np.float64),
            window_frames=3,
        )
        self.assertTrue(np.isnan(all_nan[0]))

    def test_processes_continuous_dff_before_grouping(self) -> None:
        """Confirm dF/F processing preserves raw audit arrays and labels.

        Inputs: one continuous dF/F object and moving-average options.
        Outputs: a new dF/F object with processed values only.
        """
        dff = _dff(
            np.array(
                [
                    [0.0, 10.0],
                    [2.0, 12.0],
                    [4.0, 14.0],
                ],
                dtype=np.float64,
            ),
        )
        options = ResponseProcessingOptions(
            smoothing=SmoothingOptions(method="moving_average", window_frames=3),
        )

        processed = process_roi_delta_f_over_f(
            dff,
            options=options,
            data_rate_hz=2.0,
        )

        np.testing.assert_allclose(
            processed.values,
            np.array(
                [
                    [1.0, 11.0],
                    [2.0, 12.0],
                    [3.0, 13.0],
                ],
                dtype=np.float64,
            ),
        )
        np.testing.assert_array_equal(processed.fluorescence, dff.fluorescence)
        self.assertEqual(processed.labels, dff.labels)
        self.assertEqual(
            processed.metadata["response_processing_smoothing_method"],
            "moving_average",
        )

    def test_low_pass_options_validate_cutoff_against_nyquist(self) -> None:
        """Confirm enabled low-pass filters reject impossible cutoffs.

        Inputs: Butterworth options with cutoff at the Nyquist frequency.
        Outputs: a clear ``ValueError`` before filtering starts.
        """
        dff = _dff(np.arange(12, dtype=np.float64).reshape(6, 2))
        options = ResponseProcessingOptions(
            low_pass=LowPassFilterOptions(
                method="butterworth",
                cutoff_hz=1.0,
                order=2,
            ),
        )

        with self.assertRaisesRegex(ValueError, "below Nyquist"):
            process_roi_delta_f_over_f(dff, options=options, data_rate_hz=2.0)

    def test_correlation_filter_scores_and_masks_inconsistent_roi(self) -> None:
        """Confirm correlation QC masks ROIs without removing columns.

        Inputs: two repeated trials where one ROI reverses its response.
        Outputs: correlation scores and NaN values for the excluded ROI.
        """
        grouped = _grouped_responses(
            trial_values=(
                np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
                np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float64),
            ),
        )
        options = ResponseProcessingOptions(
            correlation_filter=CorrelationFilterOptions(
                reference="epoch_mean",
                minimum_correlation=0.5,
            ),
        )

        result = process_grouped_roi_responses(grouped, options=options)

        self.assertIsNotNone(result.correlation_scores)
        if result.correlation_scores is None:
            raise AssertionError("correlation scores should be present")
        np.testing.assert_array_equal(
            result.correlation_scores.included_mask,
            np.array([True, False]),
        )
        self.assertEqual(result.grouped_responses.roi_labels, ("roi_1", "roi_2"))
        for trial in result.grouped_responses.trials:
            self.assertTrue(np.all(np.isnan(trial.values[:, 1])))
            self.assertTrue(np.all(np.isfinite(trial.values[:, 0])))

    def test_correlation_filter_can_use_epoch_peak_reference(self) -> None:
        """Confirm peak-reference QC scores trials against the peak trial.

        Inputs: two repeated trials where one ROI has a reversed response shape.
        Outputs: peak-reference scores that exclude only the inconsistent ROI.
        """
        grouped = _grouped_responses(
            trial_values=(
                np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]], dtype=np.float64),
                np.array([[0.0, 8.0], [1.0, 4.0], [2.0, 0.0]], dtype=np.float64),
            ),
        )
        options = ResponseProcessingOptions(
            correlation_filter=CorrelationFilterOptions(
                reference="epoch_peak",
                minimum_correlation=0.5,
            ),
        )

        result = process_grouped_roi_responses(grouped, options=options)

        self.assertIsNotNone(result.correlation_scores)
        if result.correlation_scores is None:
            raise AssertionError("correlation scores should be present")
        self.assertEqual(result.correlation_scores.reference, "epoch_peak")
        np.testing.assert_array_equal(
            result.correlation_scores.included_mask,
            np.array([True, False]),
        )

    def test_correlation_filter_uses_epoch_relative_window(self) -> None:
        """Confirm correlation QC restricts scoring to selected seconds.

        Inputs: repeated trials with mismatched early samples and matched late
        samples.
        Outputs: passing scores when the correlation window selects late samples.
        """
        grouped = _grouped_responses(
            trial_values=(
                np.array(
                    [[100.0, 100.0], [100.0, 100.0], [0.0, 0.0], [1.0, 1.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[-100.0, -100.0], [-100.0, -100.0], [0.0, 0.0], [1.0, 1.0]],
                    dtype=np.float64,
                ),
            ),
        )
        options = ResponseProcessingOptions(
            correlation_filter=CorrelationFilterOptions(
                reference="epoch_mean",
                minimum_correlation=0.5,
                window_seconds=(2.0, None),
            ),
        )

        result = process_grouped_roi_responses(grouped, options=options)

        self.assertIsNotNone(result.correlation_scores)
        if result.correlation_scores is None:
            raise AssertionError("correlation scores should be present")
        self.assertEqual(result.correlation_scores.window_seconds, (2.0, None))
        np.testing.assert_array_equal(
            result.correlation_scores.included_mask,
            np.array([True, True]),
        )


def _dff(values: np.ndarray) -> RoiDeltaFOverF:
    """Return a minimal dF/F object for response-processing tests."""
    float_values = values.astype(np.float64, copy=False)
    return RoiDeltaFOverF(
        fluorescence=float_values + 10.0,
        baseline=np.full(float_values.shape, 10.0, dtype=np.float64),
        values=float_values,
        labels=tuple(f"roi_{index + 1}" for index in range(float_values.shape[1])),
        start_frame=0,
        stop_frame=float_values.shape[0],
        tau=0.0,
        amplitudes=np.ones(float_values.shape[1], dtype=np.float64),
        interleave_frame_numbers=np.array([0.0], dtype=np.float64),
        interleave_fluorescence=np.ones((1, float_values.shape[1]), dtype=np.float64),
        metadata={"method": "test"},
    )


def _grouped_responses(
    *,
    trial_values: tuple[np.ndarray, ...],
) -> GroupedRoiResponses:
    """Return grouped responses with one epoch and repeated trials."""
    trials = tuple(
        RoiResponseTrial(
            epoch_number=2,
            epoch_name="Odor",
            trial_index=index + 1,
            window_index=index,
            start_frame=index * 3,
            stop_frame=index * 3 + values.shape[0],
            frame_numbers=np.arange(values.shape[0], dtype=np.int64) + index * 3,
            time_seconds=np.arange(values.shape[0], dtype=np.float64),
            values=values.astype(np.float64, copy=False),
        )
        for index, values in enumerate(trial_values)
    )
    return GroupedRoiResponses(
        roi_labels=("roi_1", "roi_2"),
        data_rate_hz=1.0,
        trials=trials,
    )


if __name__ == "__main__":
    unittest.main()
