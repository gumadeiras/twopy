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
    """Tests MATLAB-style ROI-level dF/F helpers."""

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

    def _traces(self, values: np.ndarray) -> BackgroundCorrectedRoiTraces:
        """Create background-corrected traces from known fluorescence values.

        Args:
            values: Corrected ROI fluorescence shaped ``(frames, rois)``.

        Returns:
            ``BackgroundCorrectedRoiTraces`` suitable for dF/F tests.
        """
        return BackgroundCorrectedRoiTraces(
            raw_values=values,
            background_values=np.zeros_like(values),
            corrected_values=values,
            labels=tuple(f"roi_{index + 1}" for index in range(values.shape[1])),
            start_frame=0,
            stop_frame=values.shape[0],
            statistic="mean",
            method="none",
            metadata={"method": "none"},
        )


if __name__ == "__main__":
    unittest.main()
