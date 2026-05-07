"""Tests for splitting ROI traces into response windows.

Inputs: explicit frame boundaries, photodiode-aligned event frames, and ROI
trace arrays.
Outputs: frame windows and sliced ROI response arrays.
"""

import unittest
from pathlib import Path

import numpy as np

from twopy import (
    default_baseline_epoch_number,
    frame_windows_from_photodiode_alignment,
    is_baseline_epoch_name,
    make_frame_windows,
    map_stimulus_epochs_to_frame_windows,
    select_baseline_frame_windows,
    split_traces_by_frame_windows,
)
from twopy.analysis.trials import EpochFrameWindow, FrameWindow
from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData
from twopy.roi import RoiTraces
from twopy.spatial import full_frame_crop
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

    def test_maps_stimulus_epoch_runs_to_frame_windows(self) -> None:
        """Confirm stimulus epoch runs pair with photodiode windows by order.

        Inputs: three stimulus epoch runs and four paired photodiode events.
        Outputs: three frame windows labeled with epoch numbers and names.
        """
        recording = self._recording_with_stimulus_epochs(
            np.array([1, 1, 2, 2, 1, 1, 0], dtype=np.float64),
        )
        alignment = self._alignment((0, 5, 10, 15))

        epoch_windows = map_stimulus_epochs_to_frame_windows(recording, alignment)

        self.assertEqual(
            [(item.epoch_number, item.epoch_name) for item in epoch_windows],
            [(1, "Gray Interleave"), (2, "LR20"), (1, "Gray Interleave")],
        )
        self.assertEqual(
            [
                (item.window.start_frame, item.window.stop_frame)
                for item in epoch_windows
            ],
            [(0, 5), (5, 10), (10, 15)],
        )
        selected = select_baseline_frame_windows(
            epoch_windows,
            epoch_name="Gray Interleave",
            epoch_number=None,
        )
        self.assertEqual(
            [(window.start_frame, window.stop_frame) for window in selected],
            [(0, 5), (10, 15)],
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

    def test_rejects_epoch_window_count_mismatch(self) -> None:
        """Confirm epoch-to-frame mapping fails when counts disagree.

        Inputs: two stimulus epoch runs and one photodiode frame window.
        Outputs: a clear count-mismatch error.
        """
        recording = self._recording_with_stimulus_epochs(
            np.array([1, 1, 2, 2], dtype=np.float64),
        )

        with self.assertRaisesRegex(ValueError, "epoch runs"):
            map_stimulus_epochs_to_frame_windows(recording, self._alignment((0, 5)))

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

    def _recording_with_stimulus_epochs(
        self,
        epoch_numbers: np.ndarray,
    ) -> RecordingData:
        """Create a minimal recording with stimulus epoch metadata.

        Args:
            epoch_numbers: Stimulus epoch numbers written as ``stimulus/data``.

        Returns:
            ``RecordingData`` with enough fields for epoch-window mapping.
        """
        frame_count = 20
        return RecordingData(
            path=Path("/tmp/recording_data.h5"),
            movie=ConvertedMovie(
                path=Path("/tmp/aligned_movie.h5"),
                dataset_name="movie/aligned",
                shape=(frame_count, 2, 2),
                dtype="float64",
            ),
            source_session_dir=Path("/tmp/source"),
            acquisition_metadata={},
            run_metadata={},
            synchronization_metadata={},
            stimulus_data=np.column_stack(
                (epoch_numbers, self._flash_values_for_epochs(epoch_numbers)),
            ),
            stimulus_data_column_names=("epoch_number", "photodiode_flash"),
            stimulus_parameters=(
                {"epochName": "Gray Interleave"},
                {"epochName": "LR20"},
            ),
            stimulus_function_lookup={},
            stimulus_specific_columns={},
            imaging_res_pd=np.zeros(frame_count, dtype=np.float64),
            high_res_pd=np.zeros(frame_count, dtype=np.float64),
            mean_image=np.zeros((2, 2), dtype=np.float64),
            alignment_valid_crop=full_frame_crop((2, 2)),
            alignment_shift_pixels=np.zeros(frame_count, dtype=np.float64),
            motion_artifact_mask=np.zeros(frame_count, dtype=np.bool_),
            frame_counts=FrameCountAudit(
                aligned_movie_frames=frame_count,
                imaging_res_pd_samples=frame_count,
                acquisition_number_of_frames=frame_count,
                imaging_res_pd_minus_movie=0,
                acquisition_minus_movie=0,
            ),
        )

    def _flash_values_for_epochs(self, epoch_numbers: np.ndarray) -> np.ndarray:
        """Create one synthetic flash row at each epoch-run start.

        Args:
            epoch_numbers: Stimulus epoch numbers.

        Returns:
            Flash values with the same length.
        """
        flash_values = np.zeros_like(epoch_numbers)
        if epoch_numbers.size == 0:
            return flash_values
        run_starts = np.r_[
            0,
            np.flatnonzero(epoch_numbers[1:] != epoch_numbers[:-1]) + 1,
        ]
        flash_values[run_starts] = 1.0
        return flash_values


if __name__ == "__main__":
    unittest.main()
