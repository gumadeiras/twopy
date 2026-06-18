"""Tests for native recording timing resolution.

Inputs: synthetic converted recordings with stimulus and photodiode evidence.
Outputs: one audited timing boundary for native analysis callers.
"""

import unittest

import numpy as np

from tests.recording_data import minimal_recording_data
from twopy.analysis.timing import resolve_recording_timing
from twopy.analysis.trials import EpochFrameWindow
from twopy.converted import RecordingData


class RecordingTimingTest(unittest.TestCase):
    """Tests the native ``RecordingData`` to epoch-window timing boundary."""

    def test_resolves_classified_photodiode_timing_when_flash_evidence_exists(
        self,
    ) -> None:
        """Confirm active flash evidence selects classified boundary timing."""
        recording = self._classified_recording(
            epoch_numbers=np.array([1, 1, 2, 2, 1, 1, 0], dtype=np.float64),
            flash_values=np.array([1, 0, 1, 0, 1, 0, 1], dtype=np.float64),
        )

        timing = resolve_recording_timing(recording)

        self.assertEqual(timing.source, "classified_photodiode")
        self.assertEqual(timing.frame_rate_hz, 10.0)
        self.assertEqual(timing.metadata["event_count"], 4)
        self.assertEqual(
            _window_summary(timing.epoch_windows),
            [
                (0, 10, "Gray Interleave"),
                (10, 20, "LR20"),
                (20, 30, "Gray Interleave"),
            ],
        )

    def test_falls_back_to_interpolation_without_flash_contract(self) -> None:
        """Confirm recordings without active flash rows use interpolation."""
        recording = self._interpolation_recording()

        timing = resolve_recording_timing(recording)

        self.assertEqual(timing.source, "interpolated_epochs")
        self.assertEqual(timing.metadata["start_frame"], 1)
        self.assertEqual(timing.metadata["stop_frame"], 8)
        self.assertEqual(
            _window_summary(timing.epoch_windows),
            [
                (1, 3, "Gray Interleave"),
                (3, 5, "LR20"),
                (5, 8, "Gray Interleave"),
            ],
        )

    def test_falls_back_to_interpolation_for_flash_train_protocol(self) -> None:
        """Confirm old multi-pulse flash trains do not select classified timing."""
        recording = self._interpolation_recording(
            epoch_values=np.array([1, 1, 1, 2, 2, 2, 2], dtype=np.float64),
            flash_values=np.array([1, 0, 1, 0, 1, 0, 1], dtype=np.float64),
        )

        timing = resolve_recording_timing(recording)

        self.assertEqual(timing.source, "interpolated_epochs")
        self.assertEqual(timing.metadata["window_count"], 2)
        self.assertEqual(
            _window_summary(timing.epoch_windows),
            [
                (1, 4, "Gray Interleave"),
                (4, 8, "LR20"),
            ],
        )

    def test_falls_back_when_flash_count_matches_but_rows_are_not_boundaries(
        self,
    ) -> None:
        """Confirm equal-count flash trains still need boundary row evidence."""
        recording = self._interpolation_recording(
            epoch_values=np.array([1, 1, 1, 2, 2, 2, 2], dtype=np.float64),
            flash_values=np.array([1, 0, 1, 0, 1, 0, 0], dtype=np.float64),
        )

        timing = resolve_recording_timing(recording)

        self.assertEqual(timing.source, "interpolated_epochs")
        self.assertEqual(
            _window_summary(timing.epoch_windows),
            [
                (1, 4, "Gray Interleave"),
                (4, 8, "LR20"),
            ],
        )

    def test_rejects_incomplete_boundary_flash_evidence(self) -> None:
        """Confirm missing boundary flashes fail instead of interpolating."""
        recording = self._interpolation_recording(
            epoch_values=np.array([1, 1, 2, 2, 1, 1, 1], dtype=np.float64),
            flash_values=np.array([1, 0, 1, 0, 1, 0, 0], dtype=np.float64),
        )

        with self.assertRaisesRegex(ValueError, "Incomplete photodiode"):
            resolve_recording_timing(recording)

    def test_rejects_inconsistent_classified_photodiode_evidence(self) -> None:
        """Confirm applicable but inconsistent boundary evidence fails loudly."""
        recording = self._classified_recording(
            epoch_numbers=np.array([1, 1, 2, 2, 0], dtype=np.float64),
            flash_values=np.array([1, 0, 1, 0, 1], dtype=np.float64),
        )

        with self.assertRaisesRegex(ValueError, "classified photodiode timing"):
            resolve_recording_timing(recording)

    def _classified_recording(
        self,
        *,
        epoch_numbers: np.ndarray,
        flash_values: np.ndarray,
    ) -> RecordingData:
        """Create a recording with active boundary-flash evidence."""
        stimulus_data = np.column_stack((epoch_numbers, flash_values))
        imaging_res_pd = np.zeros(40, dtype=np.float64)
        imaging_res_pd[[0, 10, 20, 30]] = 1.0
        high_res_pd = np.zeros(400, dtype=np.float64)
        high_res_pd[0:5] = 1.0
        high_res_pd[100:102] = 1.0
        high_res_pd[200:202] = 1.0
        high_res_pd[300:306] = 1.0
        return minimal_recording_data(
            frame_count=40,
            acquisition_metadata={"acq.frameRate": 10.0},
            stimulus_data=stimulus_data,
            stimulus_data_column_names=("epoch_number", "photodiode_flash"),
            stimulus_parameters=(
                {"epochName": "Gray Interleave"},
                {"epochName": "LR20"},
            ),
            imaging_res_pd=imaging_res_pd,
            high_res_pd=high_res_pd,
        )

    def _interpolation_recording(
        self,
        *,
        epoch_values: np.ndarray | None = None,
        flash_values: np.ndarray | None = None,
    ) -> RecordingData:
        """Create a recording with only interpolation timing evidence."""
        high_res_pd = np.zeros(100, dtype=np.float64)
        high_res_pd[10] = 1.0
        high_res_pd[80:84] = 1.0
        stimulus_columns = [
            np.arange(7, dtype=np.float64),
            np.arange(1, 8, dtype=np.float64),
            (
                np.array([1, 1, 2, 2, 1, 1, 1], dtype=np.float64)
                if epoch_values is None
                else epoch_values
            ),
        ]
        column_names = [
            "time_seconds",
            "stimulus_frame_number",
            "epoch_number",
        ]
        if flash_values is not None:
            stimulus_columns.append(flash_values)
            column_names.append("photodiode_flash")
        stimulus_data = np.column_stack(stimulus_columns)
        return minimal_recording_data(
            frame_count=10,
            acquisition_metadata={"acq.frameRate": 1.0},
            stimulus_data=stimulus_data,
            stimulus_data_column_names=tuple(column_names),
            stimulus_parameters=(
                {"epochName": "Gray Interleave"},
                {"epochName": "LR20"},
            ),
            high_res_pd=high_res_pd,
        )


def _window_summary(
    windows: tuple[EpochFrameWindow, ...],
) -> list[tuple[int, int, str]]:
    """Return the frame/name fields that timing tests compare."""
    return [
        (window.window.start_frame, window.window.stop_frame, window.epoch_name)
        for window in windows
    ]


if __name__ == "__main__":
    unittest.main()
