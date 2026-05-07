"""Tests for post-dF/F motion artifact masking.

Inputs: computed ROI dF/F values and converted motion masks.
Outputs: dF/F values with high-motion frames marked as NaN.
"""

import unittest
from pathlib import Path

import numpy as np

from twopy import apply_motion_artifact_mask_to_delta_f_over_f
from twopy.analysis.dff import RoiDeltaFOverF
from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData
from twopy.spatial import full_frame_crop


class MotionArtifactMaskTest(unittest.TestCase):
    """Tests motion artifact masking after dF/F."""

    def test_masks_absolute_frame_range(self) -> None:
        """Confirm motion mask is sliced by absolute dF/F frame coordinates.

        Inputs: dF/F values for frames one through four and a full-recording
        motion mask.
        Outputs: only the local frame corresponding to absolute frame two is
        set to NaN.
        """
        dff = self._dff()
        recording = self._recording()

        masked = apply_motion_artifact_mask_to_delta_f_over_f(dff, recording)

        expected = dff.values.copy()
        expected[1, :] = np.nan
        np.testing.assert_allclose(masked.values, expected)
        self.assertEqual(masked.metadata["motion_artifact_masked_frame_count"], 1)
        np.testing.assert_array_equal(masked.baseline, dff.baseline)

    def _dff(self) -> RoiDeltaFOverF:
        """Create a small dF/F result for motion-mask tests.

        Returns:
            ``RoiDeltaFOverF`` covering absolute frames one through four.
        """
        values = np.arange(6, dtype=np.float64).reshape(3, 2)
        return RoiDeltaFOverF(
            fluorescence=np.ones((3, 2), dtype=np.float64),
            baseline=np.ones((3, 2), dtype=np.float64),
            values=values,
            labels=("roi_1", "roi_2"),
            start_frame=1,
            stop_frame=4,
            tau=0.0,
            amplitudes=np.ones(2, dtype=np.float64),
            baseline_frame_numbers=np.array([1.0], dtype=np.float64),
            baseline_fluorescence=np.ones((1, 2), dtype=np.float64),
            metadata={"method": "test"},
        )

    def _recording(self) -> RecordingData:
        """Create a minimal recording with one high-motion frame.

        Returns:
            ``RecordingData`` with ``motion_artifact_mask[2]`` set.
        """
        frame_count = 5
        motion_mask = np.array([False, False, True, False, False])
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
            stimulus_data=np.zeros((0, 0), dtype=np.float64),
            stimulus_data_column_names=(),
            stimulus_parameters=(),
            stimulus_function_lookup={},
            stimulus_specific_columns={},
            imaging_res_pd=np.zeros(frame_count, dtype=np.float64),
            high_res_pd=np.zeros(frame_count, dtype=np.float64),
            mean_image=np.zeros((2, 2), dtype=np.float64),
            alignment_valid_crop=full_frame_crop((2, 2)),
            alignment_shift_pixels=np.zeros(frame_count, dtype=np.float64),
            motion_artifact_mask=motion_mask,
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
