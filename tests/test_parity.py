"""Tests for saved-analysis parity adapters.

Inputs: small converted-recording fixtures and synthetic saved-analysis objects.
Outputs: ROI sets, dF/F comparisons, and validation errors.
"""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from twopy.analysis.trials import FrameWindow
from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData
from twopy.parity import (
    SavedAnalysisLastRoi,
    compare_saved_analysis_delta_f_over_f,
    roi_set_from_saved_analysis,
    saved_analysis_epoch_windows,
)
from twopy.spatial import SpatialCrop


class ParityTest(unittest.TestCase):
    """Tests saved-analysis parity helpers."""

    def test_embeds_crop_domain_saved_roi_mask_as_full_frame_roi_set(self) -> None:
        """Confirm saved crop masks become normal full-frame twopy ROI masks.

        Inputs: a crop-shaped saved ROI label image and a recording crop.
        Outputs: full-frame ROI masks with labels sorted by saved ROI number.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(Path(temp_dir))
            saved = self._saved_analysis(
                roi_mask=np.array([[1.0, 0.0], [2.0, 2.0]]),
                saved_values=np.zeros((4, 2), dtype=np.float64),
            )

            converted = roi_set_from_saved_analysis(saved, recording)

            self.assertEqual(converted.spatial_source, "alignment_valid_crop")
            self.assertEqual(converted.source_label_values, (1, 2))
            self.assertEqual(converted.roi_set.masks.shape, (2, 3, 3))
            self.assertEqual(
                converted.roi_set.labels,
                ("saved_roi_0001", "saved_roi_0002"),
            )
            self.assertTrue(converted.roi_set.masks[0, 1, 1])
            self.assertTrue(converted.roi_set.masks[1, 2, 1])
            self.assertTrue(converted.roi_set.masks[1, 2, 2])

    def test_converts_saved_epoch_windows_to_absolute_frame_windows(self) -> None:
        """Confirm one-based saved starts become zero-based movie windows.

        Inputs: saved starts and durations inside a trace beginning at frame ten.
        Outputs: half-open absolute ``FrameWindow`` objects.
        """
        saved = self._saved_analysis(
            roi_mask=np.array([[1.0]]),
            saved_values=np.zeros((8, 1), dtype=np.float64),
            epoch_starts=np.array([1, 5]),
            epoch_durations=np.array([2, 3]),
        )

        windows = saved_analysis_epoch_windows(
            saved,
            epoch_number=1,
            trace_start_frame=10,
            trace_stop_frame=18,
        )

        self.assertEqual(
            windows,
            (
                FrameWindow(0, 10, 12, "saved_epoch_0001_0001"),
                FrameWindow(1, 14, 17, "saved_epoch_0001_0002"),
            ),
        )

    def test_compares_dff_against_saved_trace_matrix(self) -> None:
        """Confirm the comparator runs the normal twopy analysis path.

        Inputs: a constant converted movie, saved ROI labels, and zero saved
        dF/F values.
        Outputs: zero comparison error because constant fluorescence has a
        constant baseline.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(Path(temp_dir))
            saved = self._saved_analysis(
                roi_mask=np.array([[1.0, 0.0], [2.0, 2.0]]),
                saved_values=np.zeros((4, 2), dtype=np.float64),
                epoch_starts=np.array([1, 4]),
                epoch_durations=np.array([1, 1]),
            )

            comparison = compare_saved_analysis_delta_f_over_f(
                recording,
                saved,
                background_method="none",
                seconds_interleave_use=None,
                trace_start_frame=0,
                data_rate_hz=1.0,
                chunk_frames=2,
            )

            self.assertEqual(comparison.computed.values.shape, (4, 2))
            np.testing.assert_allclose(comparison.computed.values, 0.0)
            np.testing.assert_allclose(comparison.absolute_error, 0.0)
            self.assertEqual(comparison.mean_absolute_error, 0.0)

    def _saved_analysis(
        self,
        *,
        roi_mask: np.ndarray,
        saved_values: np.ndarray,
        epoch_starts: np.ndarray | None = None,
        epoch_durations: np.ndarray | None = None,
    ) -> SavedAnalysisLastRoi:
        """Create a synthetic saved-analysis object.

        Args:
            roi_mask: Saved ROI label image.
            saved_values: Saved trace matrix shaped ``(frames, rois)``.
            epoch_starts: Optional one-based saved interleave starts.
            epoch_durations: Optional saved interleave durations.

        Returns:
            ``SavedAnalysisLastRoi`` for parity tests.
        """
        starts = (
            np.array([1], dtype=np.int64)
            if epoch_starts is None
            else epoch_starts.astype(np.int64)
        )
        durations = (
            np.array([saved_values.shape[0]], dtype=np.int64)
            if epoch_durations is None
            else epoch_durations.astype(np.int64)
        )
        return SavedAnalysisLastRoi(
            path=Path("/tmp/saved.mat"),
            roi_mask_initial=roi_mask.astype(np.float64),
            epoch_list=np.ones(saved_values.shape[0], dtype=np.int64),
            epoch_start_times=(starts,),
            epoch_durations=(durations,),
            time_by_rois_initial=(saved_values.astype(np.float64),),
        )

    def _recording(self, root: Path) -> RecordingData:
        """Create a tiny converted recording with a constant movie.

        Args:
            root: Temporary directory for the movie HDF5 file.

        Returns:
            ``RecordingData`` with enough fields for parity extraction.
        """
        movie_path = root / "aligned_movie.h5"
        with h5py.File(movie_path, "w") as h5_file:
            h5_file.create_dataset("movie/aligned", data=np.full((4, 3, 3), 10.0))

        frame_count = 4
        return RecordingData(
            path=root / "recording_data.h5",
            movie=ConvertedMovie(
                path=movie_path,
                dataset_name="movie/aligned",
                shape=(frame_count, 3, 3),
                dtype="float64",
            ),
            source_session_dir=root / "source",
            acquisition_metadata={"acq.frameRate": 1.0},
            run_metadata={},
            synchronization_metadata={},
            stimulus_data=np.zeros((1, 2), dtype=np.float64),
            stimulus_data_column_names=("time_seconds", "epoch_number"),
            stimulus_parameters=({"epochName": "Gray Interleave"},),
            stimulus_function_lookup={},
            stimulus_specific_columns={},
            imaging_res_pd=np.zeros(frame_count, dtype=np.float64),
            high_res_pd=np.zeros(frame_count, dtype=np.float64),
            mean_image=np.full((3, 3), 10.0, dtype=np.float64),
            alignment_valid_crop=SpatialCrop(
                axis0_start=1,
                axis0_stop=3,
                axis1_start=1,
                axis1_stop=3,
                original_shape=(3, 3),
                source="test_crop",
            ),
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
