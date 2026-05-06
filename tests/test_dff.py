"""Tests for ROI-level delta F over F calculation.

Inputs: background-corrected ROI fluorescence traces and interleave windows.
Outputs: dF/F traces, fitted baselines, and validation errors.
"""

import unittest

import numpy as np

from twopy.analysis.background_subtraction import BackgroundCorrectedRoiTraces
from twopy.analysis.dff import compute_roi_delta_f_over_f
from twopy.analysis.trials import FrameWindow


class DeltaFOverFTest(unittest.TestCase):
    """Tests ROI-level dF/F helpers."""

    def test_computes_roi_level_dff_from_gray_interleaves(self) -> None:
        """Confirm baseline is fit from interleaves and applied per ROI.

        Inputs: corrected ROI fluorescence with two gray windows.
        Outputs: ROI dF/F values where interleaves are baseline and stimulus
        frames rise above baseline.
        """
        traces = self._traces(
            np.array(
                [
                    [10.0, 20.0],
                    [10.0, 20.0],
                    [20.0, 20.0],
                    [30.0, 20.0],
                    [10.0, 20.0],
                    [10.0, 20.0],
                ],
            ),
        )

        result = compute_roi_delta_f_over_f(
            traces,
            (
                FrameWindow(0, 0, 2, "gray_1"),
                FrameWindow(1, 4, 6, "gray_2"),
            ),
            data_rate_hz=2.0,
            seconds_interleave_use=None,
        )

        np.testing.assert_allclose(result.baseline, np.tile([[10.0, 20.0]], (6, 1)))
        np.testing.assert_allclose(
            result.values,
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
            ),
        )
        self.assertEqual(result.metadata["method"], "shared_tau_roi_amplitude")
        self.assertEqual(result.metadata["fit_mode"], "robust")

    def test_uses_last_requested_seconds_of_each_interleave(self) -> None:
        """Confirm interleave baseline uses the end of the gray window.

        Inputs: one four-frame interleave and a two-frame-per-second data rate.
        Outputs: baseline from only the final two frames when one second is
        requested.
        """
        traces = self._traces(np.array([[1.0], [2.0], [10.0], [14.0]]))

        result = compute_roi_delta_f_over_f(
            traces,
            (FrameWindow(0, 0, 4, "gray"),),
            data_rate_hz=2.0,
            seconds_interleave_use=1.0,
        )

        np.testing.assert_allclose(result.amplitudes, np.array([12.0]))
        np.testing.assert_allclose(result.baseline, np.full((4, 1), 12.0))
        np.testing.assert_allclose(result.interleave_fluorescence, np.array([[12.0]]))

    def test_source_bounds_fit_mode_records_and_runs(self) -> None:
        """Confirm source-bounds mode is available for audit comparisons.

        Inputs: positive interleave fluorescence where source-style bounds are
        ordered.
        Outputs: dF/F result with metadata recording ``source_bounds``.
        """
        traces = self._traces(
            np.array(
                [
                    [10.0],
                    [10.0],
                    [20.0],
                    [10.0],
                    [10.0],
                ],
            ),
        )

        result = compute_roi_delta_f_over_f(
            traces,
            (
                FrameWindow(0, 0, 2, "gray_1"),
                FrameWindow(1, 3, 5, "gray_2"),
            ),
            data_rate_hz=2.0,
            seconds_interleave_use=None,
            fit_mode="source_bounds",
        )

        self.assertEqual(result.metadata["fit_mode"], "source_bounds")
        self.assertEqual(result.values.shape, (5, 1))

    def test_robust_fit_mode_filters_nonpositive_interleave_sample(self) -> None:
        """Confirm robust mode tolerates one nonpositive baseline sample.

        Inputs: one nonpositive and two positive gray-window averages.
        Outputs: finite dF/F values from the remaining positive fit samples.
        """
        traces = self._traces(np.array([[10.0], [-1.0], [10.0], [20.0], [10.0]]))

        result = compute_roi_delta_f_over_f(
            traces,
            (
                FrameWindow(0, 0, 1, "gray_1"),
                FrameWindow(1, 1, 2, "gray_2"),
                FrameWindow(2, 4, 5, "gray_3"),
            ),
            data_rate_hz=1.0,
            seconds_interleave_use=None,
            fit_mode="robust",
        )

        self.assertTrue(np.isfinite(result.values).all())

    def test_source_bounds_fit_mode_rejects_nonpositive_interleave_sample(self) -> None:
        """Confirm source-bounds mode does not hide nonpositive fit samples.

        Inputs: one nonpositive gray-window average.
        Outputs: a clear validation error.
        """
        traces = self._traces(np.array([[10.0], [-1.0], [10.0]]))

        with self.assertRaisesRegex(ValueError, "positive interleave"):
            compute_roi_delta_f_over_f(
                traces,
                (
                    FrameWindow(0, 0, 1, "gray_1"),
                    FrameWindow(1, 1, 2, "gray_2"),
                    FrameWindow(2, 2, 3, "gray_3"),
                ),
                data_rate_hz=1.0,
                seconds_interleave_use=None,
                fit_mode="source_bounds",
            )

    def test_rejects_interleave_window_outside_trace_range(self) -> None:
        """Confirm baseline windows must be inside the trace frame range.

        Inputs: traces covering frames two through five and a window starting
        before frame two.
        Outputs: a clear validation error.
        """
        traces = BackgroundCorrectedRoiTraces(
            raw_values=np.ones((3, 1)),
            background_values=np.zeros((3, 1)),
            corrected_values=np.ones((3, 1)),
            labels=("roi",),
            start_frame=2,
            stop_frame=5,
            statistic="mean",
            method="none",
            metadata={"method": "none"},
        )

        with self.assertRaisesRegex(ValueError, "outside trace range"):
            compute_roi_delta_f_over_f(
                traces,
                (FrameWindow(0, 1, 3, "early"),),
                data_rate_hz=2.0,
            )

    def test_nonzero_start_frame_uses_absolute_frame_coordinates(self) -> None:
        """Confirm cropped traces keep movie-frame coordinates for fitting.

        Inputs: an exponential trace covering absolute frames ten through
        thirteen.
        Outputs: interleave frame numbers and baseline in absolute movie-frame
        coordinates.
        """
        start_frame = 10
        absolute_frame_numbers = np.arange(11, 15, dtype=np.float64)
        fluorescence = 7.0 * np.exp(0.004 * absolute_frame_numbers)
        traces = self._traces(fluorescence[:, None], start_frame=start_frame)

        result = compute_roi_delta_f_over_f(
            traces,
            (
                FrameWindow(0, 10, 11, "gray_start"),
                FrameWindow(1, 13, 14, "gray_end"),
            ),
            data_rate_hz=1.0,
            seconds_interleave_use=None,
        )

        np.testing.assert_allclose(
            result.interleave_frame_numbers,
            np.array([11.5, 14.5]),
        )
        expected_baseline = result.amplitudes[None, :] * np.exp(
            result.tau * absolute_frame_numbers[:, None],
        )
        np.testing.assert_allclose(result.baseline, expected_baseline)

    def _traces(
        self,
        values: np.ndarray,
        *,
        start_frame: int = 0,
    ) -> BackgroundCorrectedRoiTraces:
        """Create background-corrected traces from known fluorescence values.

        Args:
            values: Corrected ROI fluorescence shaped ``(frames, rois)``.
            start_frame: Absolute frame index for ``values[0]``.

        Returns:
            ``BackgroundCorrectedRoiTraces`` suitable for dF/F tests.
        """
        return BackgroundCorrectedRoiTraces(
            raw_values=values,
            background_values=np.zeros_like(values),
            corrected_values=values,
            labels=tuple(f"roi_{index + 1}" for index in range(values.shape[1])),
            start_frame=start_frame,
            stop_frame=start_frame + values.shape[0],
            statistic="mean",
            method="none",
            metadata={"method": "none"},
        )


if __name__ == "__main__":
    unittest.main()
