"""Tests for stimulus epoch interpolation onto imaging frames.

Inputs: converted stimulus rows and high-rate photodiode bounds.
Outputs: per-frame epoch labels and stimulus-labeled frame windows.
"""

import unittest
from pathlib import Path

import numpy as np

from twopy import interpolate_stimulus_epochs_to_frame_windows
from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData
from twopy.spatial import full_frame_crop


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
        return RecordingData(
            path=Path("/tmp/recording_data.h5"),
            movie=ConvertedMovie(
                path=Path("/tmp/aligned_movie.h5"),
                dataset_name="movie/aligned",
                shape=(frame_count, 2, 2),
                dtype="float64",
            ),
            source_session_dir=Path("/tmp/source"),
            acquisition_metadata={"acq.frameRate": 1.0},
            run_metadata={},
            synchronization_metadata={},
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
            stimulus_function_lookup={},
            stimulus_specific_columns={},
            imaging_res_pd=np.zeros(frame_count, dtype=np.float64),
            high_res_pd=high_res_pd,
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


if __name__ == "__main__":
    unittest.main()
