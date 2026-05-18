"""Tests for splitting ROI traces into response windows.

Inputs: explicit frame boundaries and ROI trace arrays.
Outputs: frame windows and sliced ROI response arrays.
"""

import unittest

import numpy as np

from twopy import (
    default_baseline_epoch_number,
    is_baseline_epoch_name,
    make_frame_windows,
    no_baseline_epoch_frame_windows,
    select_baseline_frame_windows,
    split_traces_by_frame_windows,
)
from twopy.analysis.trials import EpochFrameWindow, FrameWindow
from twopy.roi import RoiTraces


class ResponsesTest(unittest.TestCase):
    """Tests response frame-window helpers."""

    def test_makes_frame_windows_from_boundaries(self) -> None:
        """Confirm neighboring frame boundaries become half-open windows.

        Inputs: three ordered frame boundaries.
        Outputs: two labeled frame windows.
        """
        windows = make_frame_windows(
            [1, 4, 7],
            frame_count=10,
            labels=("trial_a", "trial_b"),
        )

        self.assertEqual(
            windows,
            (
                FrameWindow(0, 1, 4, "trial_a"),
                FrameWindow(1, 4, 7, "trial_b"),
            ),
        )

    def test_splits_traces_by_frame_windows(self) -> None:
        """Confirm ROI traces are sliced by absolute frame windows.

        Inputs: ROI traces covering frames 1 through 6 and two windows.
        Outputs: one response array per window.
        """
        traces = RoiTraces(
            values=np.arange(12, dtype=np.float64).reshape(6, 2),
            labels=("roi_1", "roi_2"),
            start_frame=1,
            stop_frame=7,
            statistic="mean",
        )
        windows = (
            FrameWindow(0, 2, 4, "trial_a"),
            FrameWindow(1, 4, 7, "trial_b"),
        )

        responses = split_traces_by_frame_windows(traces, windows)

        self.assertEqual(responses[0].window.label, "trial_a")
        np.testing.assert_array_equal(
            responses[0].values,
            np.array([[2.0, 3.0], [4.0, 5.0]]),
        )
        np.testing.assert_array_equal(
            responses[1].values,
            np.array([[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]),
        )

    def test_rejects_unordered_boundaries(self) -> None:
        """Confirm invalid boundaries fail before slicing traces.

        Inputs: repeated frame boundaries.
        Outputs: a clear validation error.
        """
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            make_frame_windows([1, 1, 3], frame_count=5)

    def test_rejects_window_outside_trace_range(self) -> None:
        """Confirm response windows must be inside extracted trace frames.

        Inputs: traces for frames 5 through 8 and a window starting at 4.
        Outputs: a clear validation error.
        """
        traces = RoiTraces(
            values=np.zeros((3, 1), dtype=np.float64),
            labels=("roi_1",),
            start_frame=5,
            stop_frame=8,
            statistic="mean",
        )

        with self.assertRaisesRegex(ValueError, "outside trace range"):
            split_traces_by_frame_windows(
                traces,
                (FrameWindow(0, 4, 6, "early"),),
            )

    def test_default_baseline_epoch_uses_gray_like_name(self) -> None:
        """Confirm baseline epoch selection uses the shared name rule."""
        epoch_names = {1: "Odor A", 2: "Grey screen", 3: "Odor B"}

        self.assertEqual(default_baseline_epoch_number(epoch_names), 2)

    def test_default_baseline_epoch_accepts_interleave_name(self) -> None:
        """Confirm interleave text is accepted without gray spelling."""
        epoch_names = {1: "Odor A", 2: "Baseline Interleave"}

        self.assertEqual(default_baseline_epoch_number(epoch_names), 2)

    def test_default_baseline_epoch_falls_back_to_one(self) -> None:
        """Confirm unknown epoch names preserve the historical default."""
        self.assertEqual(default_baseline_epoch_number({2: "Odor A"}), 1)

    def test_baseline_epoch_name_predicate_is_shared(self) -> None:
        """Confirm the gray/interleave string rule stays explicit."""
        self.assertTrue(is_baseline_epoch_name("Gray Interleave"))
        self.assertTrue(is_baseline_epoch_name("Grey screen"))
        self.assertTrue(is_baseline_epoch_name("Baseline Interleave"))
        self.assertFalse(is_baseline_epoch_name("Odor A"))

    def test_selects_baseline_windows_by_multiple_epoch_numbers(self) -> None:
        """Confirm parity can reuse native baseline window selection."""
        epoch_windows = (
            self._epoch_window(1, 0, 5, "Gray Interleave"),
            self._epoch_window(2, 5, 10, "Odor A"),
            self._epoch_window(3, 10, 15, "Gray Interleave"),
        )

        selected = select_baseline_frame_windows(
            epoch_windows,
            epoch_number=None,
            epoch_numbers=(3, 1),
        )

        self.assertEqual(
            [(window.start_frame, window.stop_frame) for window in selected],
            [(0, 5), (10, 15)],
        )

    def test_no_baseline_epoch_uses_continuous_epoch_span(self) -> None:
        """Confirm no-baseline-epoch windows cover selected epochs."""
        epoch_windows = (
            self._epoch_window(1, 0, 5, "Probe"),
            self._epoch_window(2, 5, 10, "Stim A"),
            self._epoch_window(1, 10, 15, "Probe"),
            self._epoch_window(3, 15, 20, "Stim B"),
        )

        selected = no_baseline_epoch_frame_windows(
            epoch_windows,
            start_epoch_number=2,
        )

        self.assertEqual(
            [(window.start_frame, window.stop_frame) for window in selected],
            [(5, 20)],
        )

    def _epoch_window(
        self,
        epoch_number: int,
        start_frame: int,
        stop_frame: int,
        epoch_name: str,
    ) -> EpochFrameWindow:
        """Create one labeled epoch frame window."""
        return EpochFrameWindow(
            FrameWindow(epoch_number - 1, start_frame, stop_frame, epoch_name),
            epoch_number,
            epoch_name,
        )


if __name__ == "__main__":
    unittest.main()
