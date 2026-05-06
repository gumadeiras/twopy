"""Tests for photodiode event classification.

Inputs: synthetic converted stimulus tables and paired photodiode events.
Outputs: classified stimulus events and imaging-frame windows with epoch
metadata.
"""

import unittest
from pathlib import Path

import numpy as np

from twopy import classify_recording_photodiode_events
from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData
from twopy.synchronization import (
    AlignedPhotodiodeEvent,
    PhotodiodeAlignment,
    PhotodiodeEvent,
    PhotodiodeEventSet,
)


class PhotodiodeClassificationTest(unittest.TestCase):
    """Tests semantic classification of photodiode timing events."""

    def test_classifies_start_transitions_end_and_epoch_windows(self) -> None:
        """Confirm boundary flashes become typed stimulus timing windows.

        Inputs: four paired photodiode events and three positive epoch runs.
        Outputs: start, transition, transition, end event labels plus three
        epoch windows in imaging-frame coordinates.
        """
        recording = self._recording(
            epoch_numbers=np.array([1, 1, 2, 2, 1, 1, 0], dtype=np.float64),
            flash_values=np.array([1, 0, 1, 0, 1, 0, 1], dtype=np.float64),
        )
        alignment = self._alignment(
            imaging_frames=(0, 10, 20, 30),
            high_res_durations=(5, 2, 2, 6),
        )

        timing = classify_recording_photodiode_events(recording, alignment)

        self.assertEqual(
            [event.event_type for event in timing.events],
            [
                "stimulus_start",
                "trial_transition",
                "trial_transition",
                "stimulus_end",
            ],
        )
        self.assertEqual(
            [event.duration_class for event in timing.events],
            ["long", "short", "short", "long"],
        )
        self.assertEqual(
            [
                (window.start_frame, window.stop_frame, window.epoch_name)
                for window in timing.windows
            ],
            [
                (0, 10, "Gray Interleave"),
                (10, 20, "LR20"),
                (20, 30, "Gray Interleave"),
            ],
        )
        self.assertEqual(timing.metadata["stimulus_flash_segment_count"], 4)

    def test_rejects_photodiode_flash_count_mismatch(self) -> None:
        """Confirm measured photodiode events must match flash-column events.

        Inputs: three paired photodiode events but only two stimulus flash
        segments.
        Outputs: a clear count-mismatch error.
        """
        recording = self._recording(
            epoch_numbers=np.array([1, 1, 2, 2, 0], dtype=np.float64),
            flash_values=np.array([1, 0, 1, 0, 0], dtype=np.float64),
        )

        with self.assertRaisesRegex(ValueError, "flash requests"):
            classify_recording_photodiode_events(
                recording,
                self._alignment(
                    imaging_frames=(0, 10, 20),
                    high_res_durations=(5, 2, 6),
                ),
            )

    def test_rejects_epoch_boundaries_that_do_not_match_flashes(self) -> None:
        """Confirm stimulus epoch runs must begin at flash rows.

        Inputs: matching event and flash counts but a delayed epoch transition.
        Outputs: a clear stimulus-row contract error.
        """
        recording = self._recording(
            epoch_numbers=np.array([1, 1, 1, 2, 0], dtype=np.float64),
            flash_values=np.array([1, 0, 1, 0, 1], dtype=np.float64),
        )

        with self.assertRaisesRegex(ValueError, "does not stop"):
            classify_recording_photodiode_events(
                recording,
                self._alignment(
                    imaging_frames=(0, 10, 20),
                    high_res_durations=(5, 2, 6),
                ),
            )

    def _alignment(
        self,
        *,
        imaging_frames: tuple[int, ...],
        high_res_durations: tuple[int, ...],
    ) -> PhotodiodeAlignment:
        """Create paired photodiode events with chosen durations.

        Args:
            imaging_frames: Imaging frame assigned to each event.
            high_res_durations: High-rate duration for each event.

        Returns:
            ``PhotodiodeAlignment`` suitable for classification tests.
        """
        high_res_events = tuple(
            PhotodiodeEvent(
                start_sample=index * 100,
                stop_sample=(index * 100) + duration,
                center_sample=float((index * 100) + ((duration - 1) / 2.0)),
                duration_samples=duration,
                peak_value=1.0,
            )
            for index, duration in enumerate(high_res_durations)
        )
        imaging_events = tuple(
            PhotodiodeEvent(frame, frame + 1, float(frame), 1, 1.0)
            for frame in imaging_frames
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

    def _recording(
        self,
        *,
        epoch_numbers: np.ndarray,
        flash_values: np.ndarray,
    ) -> RecordingData:
        """Create a minimal recording with stimulus timing columns.

        Args:
            epoch_numbers: Converted ``epoch_number`` stimulus column.
            flash_values: Converted ``photodiode_flash`` stimulus column.

        Returns:
            ``RecordingData`` with enough fields for classification.
        """
        stimulus_data = np.column_stack((epoch_numbers, flash_values))
        frame_count = 40
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
            stimulus_data=stimulus_data,
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
