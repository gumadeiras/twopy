"""Tests for splitting ROI traces into response windows.

Inputs: explicit frame boundaries, photodiode-aligned event frames, and ROI
trace arrays.
Outputs: frame windows and sliced ROI response arrays.
"""

import unittest

import numpy as np

from twopy import (
    frame_windows_from_photodiode_alignment,
    make_frame_windows,
    split_traces_by_frame_windows,
)
from twopy.analysis.trials import FrameWindow
from twopy.roi import RoiTraces
from twopy.synchronization import (
    AlignedPhotodiodeEvent,
    PhotodiodeAlignment,
    PhotodiodeEvent,
    PhotodiodeEventSet,
)


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

    def test_makes_frame_windows_from_photodiode_alignment(self) -> None:
        """Confirm paired photodiode events define response windows.

        Inputs: three paired events assigned to imaging frames.
        Outputs: two frame windows between consecutive events.
        """
        alignment = self._alignment((2, 5, 8))

        windows = frame_windows_from_photodiode_alignment(
            alignment,
            frame_count=10,
        )

        self.assertEqual(
            [(window.start_frame, window.stop_frame) for window in windows],
            [(2, 5), (5, 8)],
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

    def _alignment(self, frames: tuple[int, ...]) -> PhotodiodeAlignment:
        """Create photodiode alignment with paired events at selected frames.

        Args:
            frames: Imaging-frame index for each paired event.

        Returns:
            ``PhotodiodeAlignment`` with simple one-sample events.
        """
        high_res_events = tuple(
            PhotodiodeEvent(index * 10, index * 10 + 1, float(index * 10), 1, 1.0)
            for index in range(len(frames))
        )
        imaging_events = tuple(
            PhotodiodeEvent(frame, frame + 1, float(frame), 1, 1.0) for frame in frames
        )
        return PhotodiodeAlignment(
            high_res=PhotodiodeEventSet(
                signal_name="high_res_pd",
                events=high_res_events,
                threshold=0.5,
                polarity="above",
            ),
            imaging_res=PhotodiodeEventSet(
                signal_name="imaging_res_pd",
                events=imaging_events,
                threshold=0.5,
                polarity="above",
            ),
            paired_events=tuple(
                AlignedPhotodiodeEvent(
                    high_res_event=high_res_event,
                    imaging_res_event=imaging_event,
                    imaging_frame=imaging_event.start_sample,
                )
                for high_res_event, imaging_event in zip(
                    high_res_events,
                    imaging_events,
                    strict=True,
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
