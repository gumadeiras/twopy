"""Tests for photodiode event detection and frame alignment.

Inputs: synthetic photodiode vectors with known event positions.
Outputs: detected events and high-resolution events paired to imaging frames.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from twopy import (
    detect_photodiode_events,
    detect_recording_photodiode_events,
    pair_photodiode_events_to_imaging_frames,
)
from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData
from twopy.spatial import full_frame_crop
from twopy.synchronization import PhotodiodeEvent


class SynchronizationTest(unittest.TestCase):
    """Tests photodiode synchronization primitives."""

    def test_detects_photodiode_events(self) -> None:
        """Confirm thresholding returns contiguous flash segments.

        Inputs: one synthetic photodiode vector with two flashes.
        Outputs: start/stop samples and event widths.
        """
        event_set = detect_photodiode_events(
            np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0]),
            threshold=0.5,
            signal_name="high_res_pd",
        )

        self.assertEqual(event_set.threshold, 0.5)
        self.assertEqual(
            [(event.start_sample, event.stop_sample) for event in event_set.events],
            [(2, 4), (6, 7)],
        )
        self.assertEqual(event_set.events[0].duration_samples, 2)

    def test_merges_close_event_segments(self) -> None:
        """Confirm short gaps inside a flash can be merged deliberately.

        Inputs: one signal with a one-sample dip inside an event.
        Outputs: one merged event spanning the dip.
        """
        event_set = detect_photodiode_events(
            np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
            threshold=0.5,
            merge_gap_samples=1,
        )

        self.assertEqual(len(event_set.events), 1)
        self.assertEqual(event_set.events[0].start_sample, 1)
        self.assertEqual(event_set.events[0].stop_sample, 4)

    def test_pairs_recording_events_to_imaging_frames(self) -> None:
        """Confirm high-resolution events pair with imaging-frame events.

        Inputs: converted recording photodiode vectors with two events each.
        Outputs: paired events assigned to imaging frames by event order.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(
                Path(temp_dir),
                high_res_pd=np.array(
                    [0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0],
                    dtype=np.float64,
                ),
                imaging_res_pd=np.array(
                    [0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
                    dtype=np.float64,
                ),
            )

            alignment = detect_recording_photodiode_events(
                recording,
                high_res_threshold=0.5,
                imaging_res_threshold=0.5,
            )

            self.assertEqual(len(alignment.paired_events), 2)
            self.assertEqual(
                [event.imaging_frame for event in alignment.paired_events],
                [1, 4],
            )

    def test_rejects_event_count_mismatch(self) -> None:
        """Confirm pairing fails instead of silently dropping events.

        Inputs: two high-resolution events and one imaging-resolution event.
        Outputs: a clear validation error.
        """
        high_res_events = (
            PhotodiodeEvent(1, 2, 1.0, 1, 1.0),
            PhotodiodeEvent(4, 5, 4.0, 1, 1.0),
        )
        imaging_res_events = (PhotodiodeEvent(3, 4, 3.0, 1, 1.0),)

        with self.assertRaisesRegex(ValueError, "different counts"):
            pair_photodiode_events_to_imaging_frames(
                high_res_events,
                imaging_res_events,
            )

    def _recording(
        self,
        root: Path,
        *,
        high_res_pd: np.ndarray,
        imaging_res_pd: np.ndarray,
    ) -> RecordingData:
        """Create a minimal loaded recording for synchronization tests.

        Args:
            root: Temporary directory used for placeholder paths.
            high_res_pd: Synthetic high-resolution photodiode signal.
            imaging_res_pd: Synthetic imaging-resolution photodiode signal.

        Returns:
            ``RecordingData`` object with the provided photodiode vectors.
        """
        return RecordingData(
            path=root / "recording_data.h5",
            movie=ConvertedMovie(
                path=root / "aligned_movie.h5",
                dataset_name="movie/aligned",
                shape=(len(imaging_res_pd), 2, 2),
                dtype="float64",
            ),
            source_session_dir=root / "source",
            acquisition_metadata={},
            run_metadata={},
            synchronization_metadata={},
            stimulus_data=np.zeros((0, 0), dtype=np.float64),
            stimulus_data_column_names=(),
            stimulus_parameters=(),
            stimulus_function_lookup={},
            stimulus_specific_columns={},
            imaging_res_pd=imaging_res_pd,
            high_res_pd=high_res_pd,
            mean_image=np.zeros((2, 2), dtype=np.float64),
            alignment_valid_crop=full_frame_crop((2, 2)),
            frame_counts=FrameCountAudit(
                aligned_movie_frames=len(imaging_res_pd),
                imaging_res_pd_samples=len(imaging_res_pd),
                acquisition_number_of_frames=len(imaging_res_pd) - 1,
                imaging_res_pd_minus_movie=0,
                acquisition_minus_movie=-1,
            ),
        )


if __name__ == "__main__":
    unittest.main()
