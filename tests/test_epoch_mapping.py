"""Tests for stimulus epoch interpolation onto imaging frames.

Inputs: converted stimulus rows and high-rate photodiode bounds.
Outputs: per-frame epoch labels and stimulus-labeled frame windows.
"""

import unittest

import numpy as np

from tests.recording_data import minimal_recording_data
from twopy.analysis.epoch_mapping import interpolate_stimulus_epochs_to_frame_windows
from twopy.converted import RecordingData


class EpochMappingTest(unittest.TestCase):
    """Tests Psycho5-style stimulus epoch interpolation."""

    def test_interpolates_epochs_without_imaging_event_pairing(self) -> None:
        """Confirm epoch mapping uses high-rate stimulus bounds directly.

        Inputs: two high-rate photodiode flashes, no imaging-resolution flashes,
        and seven positive stimulus epoch rows.
        Outputs: frame windows labeled by nearest interpolated stimulus epochs.
        """
        recording = self._recording()

        mapping = interpolate_stimulus_epochs_to_frame_windows(recording)

        self.assertEqual(mapping.start_frame, 1)
        self.assertEqual(mapping.stop_frame, 8)
        np.testing.assert_array_equal(
            mapping.epoch_numbers,
            np.array([1, 1, 2, 2, 1, 1, 1]),
        )
        self.assertEqual(
            [
                (item.window.start_frame, item.window.stop_frame, item.epoch_name)
                for item in mapping.windows
            ],
            [
                (1, 3, "Gray Interleave"),
                (3, 5, "LR20"),
                (5, 8, "Gray Interleave"),
            ],
        )

    def _recording(self) -> RecordingData:
        """Create a minimal recording for epoch interpolation.

        Args:
            None.

        Returns:
            ``RecordingData`` with high-rate photodiode start/end flashes and
            stimulus epoch rows.
        """
        frame_count = 10
        high_res_pd = np.zeros(100, dtype=np.float64)
        high_res_pd[10] = 1.0
        high_res_pd[80:84] = 1.0
        stimulus_data = np.column_stack(
            (
                np.arange(7, dtype=np.float64),
                np.arange(1, 8, dtype=np.float64),
                np.array([1, 1, 2, 2, 1, 1, 1], dtype=np.float64),
            ),
        )
        return minimal_recording_data(
            frame_count=frame_count,
            acquisition_metadata={"acq.frameRate": 1.0},
            stimulus_data=stimulus_data,
            stimulus_data_column_names=(
                "time_seconds",
                "stimulus_frame_number",
                "epoch_number",
            ),
            stimulus_parameters=(
                {"epochName": "Gray Interleave"},
                {"epochName": "LR20"},
            ),
            high_res_pd=high_res_pd,
        )


if __name__ == "__main__":
    unittest.main()
