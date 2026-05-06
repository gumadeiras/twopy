"""Tests for GUI-independent ROI storage and trace extraction.

Inputs: small ROI masks and a tiny converted aligned movie.
Outputs: saved ROI files and fluorescence traces with known expected values.
"""

import tempfile
import unittest
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from twopy import (
    extract_roi_traces,
    load_roi_set,
    make_roi_set,
    make_roi_set_from_label_image,
    roi_set_to_label_image,
    save_roi_set,
)
from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData
from twopy.roi import TraceStatistic
from twopy.spatial import full_frame_crop


class RoiTest(unittest.TestCase):
    """Tests ROI validation, persistence, and trace extraction."""

    def test_saves_and_loads_roi_set(self) -> None:
        """Confirm ROI masks and labels round-trip through HDF5.

        Inputs: two boolean ROI masks and labels.
        Outputs: a saved ROI file and loaded masks matching the source.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            roi_path = Path(temp_dir) / "rois.h5"
            masks = np.array(
                [
                    [[True, True], [False, False]],
                    [[False, False], [False, True]],
                ],
            )
            roi_set = make_roi_set(masks, labels=("top", "corner"))

            save_roi_set(roi_set, roi_path)
            loaded = load_roi_set(roi_path)

            self.assertEqual(loaded.labels, ("top", "corner"))
            np.testing.assert_array_equal(loaded.masks, masks)

    def test_rejects_saved_roi_masks_with_wrong_dtype(self) -> None:
        """Confirm ROI files store masks as boolean arrays.

        Inputs: ROI HDF5 file whose ``masks`` dataset was rewritten as integers.
        Outputs: clear validation error before returning a ``RoiSet``.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            roi_path = Path(temp_dir) / "rois.h5"
            roi_set = make_roi_set(np.ones((1, 2, 2), dtype=bool))
            save_roi_set(roi_set, roi_path)
            with h5py.File(roi_path, "r+") as h5_file:
                del h5_file["masks"]
                h5_file.create_dataset(
                    "masks",
                    data=np.ones((1, 2, 2), dtype=np.int64),
                )

            with self.assertRaisesRegex(ValueError, "masks must have dtype"):
                load_roi_set(roi_path)

    def test_extracts_mean_roi_traces_from_movie_chunks(self) -> None:
        """Confirm trace extraction streams chunks and averages ROI pixels.

        Inputs: two ROI masks over a three-frame converted movie.
        Outputs: a frame-by-ROI trace matrix for the requested frame range.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = self._write_recording(root)
            roi_set = make_roi_set(
                np.array(
                    [
                        [[True, True], [False, False]],
                        [[False, False], [False, True]],
                    ],
                ),
                labels=("top", "corner"),
            )

            traces = extract_roi_traces(
                recording,
                roi_set,
                start_frame=1,
                stop_frame=3,
                chunk_frames=1,
            )

            self.assertEqual(traces.labels, ("top", "corner"))
            self.assertEqual(traces.start_frame, 1)
            self.assertEqual(traces.stop_frame, 3)
            np.testing.assert_array_equal(
                traces.values,
                np.array([[4.5, 7.0], [8.5, 11.0]]),
            )

    def test_rejects_empty_roi_mask(self) -> None:
        """Confirm ROIs must contain pixels.

        Inputs: one empty ROI mask.
        Outputs: a validation error instead of a NaN trace later.
        """
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            make_roi_set(np.zeros((1, 2, 2), dtype=bool))

    def test_rejects_roi_shape_that_does_not_match_movie(self) -> None:
        """Confirm ROI masks must use aligned movie spatial coordinates.

        Inputs: a converted movie and a ROI mask with a different shape.
        Outputs: a clear validation error.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._write_recording(Path(temp_dir))
            roi_set = make_roi_set(np.ones((1, 3, 2), dtype=bool))

            with self.assertRaisesRegex(ValueError, "does not match movie"):
                extract_roi_traces(recording, roi_set)

    def test_rejects_unimplemented_trace_statistic(self) -> None:
        """Confirm trace metadata cannot claim an unsupported statistic.

        Inputs: an untyped-script-style ``median`` statistic request.
        Outputs: a clear validation error before trace extraction.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._write_recording(Path(temp_dir))
            roi_set = make_roi_set(np.ones((1, 2, 2), dtype=bool))

            with self.assertRaisesRegex(ValueError, "Unknown trace statistic"):
                extract_roi_traces(
                    recording,
                    roi_set,
                    statistic=cast(TraceStatistic, "median"),
                )

    def test_converts_label_image_to_roi_set_and_back(self) -> None:
        """Confirm label images become deterministic ROI masks.

        Inputs: one integer label image with background and two ROI labels.
        Outputs: a ``RoiSet`` plus a one-based label image representation.
        """
        roi_set = make_roi_set_from_label_image(
            np.array([[0, 2], [5, 5]]),
            label_prefix="drawn",
        )

        self.assertEqual(roi_set.labels, ("drawn_0002", "drawn_0005"))
        np.testing.assert_array_equal(
            roi_set.masks,
            np.array(
                [
                    [[False, True], [False, False]],
                    [[False, False], [True, True]],
                ],
            ),
        )
        np.testing.assert_array_equal(
            roi_set_to_label_image(roi_set),
            np.array([[0, 1], [2, 2]]),
        )

    def test_rejects_overlapping_roi_set_for_label_image(self) -> None:
        """Confirm overlapping masks cannot silently collapse into labels.

        Inputs: two ROI masks that share one pixel.
        Outputs: a clear validation error before display or saving.
        """
        roi_set = make_roi_set(np.array([[[True]], [[True]]]))

        with self.assertRaisesRegex(ValueError, "overlap"):
            roi_set_to_label_image(roi_set)

    def _write_recording(
        self,
        root: Path,
    ) -> RecordingData:
        """Create a minimal loaded recording object backed by HDF5 movie data.

        Args:
            root: Temporary directory that receives the movie file.

        Returns:
            ``RecordingData`` object that can be passed to ROI extraction.
        """
        movie_values = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
        movie_path = root / "aligned_movie.h5"
        with h5py.File(movie_path, "w") as h5_file:
            h5_file.attrs["twopy_format"] = "aligned-movie"
            h5_file.create_dataset(
                "movie/aligned",
                data=movie_values,
            )

        return RecordingData(
            path=root / "recording_data.h5",
            movie=ConvertedMovie(
                path=movie_path,
                dataset_name="movie/aligned",
                shape=movie_values.shape,
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
            imaging_res_pd=np.zeros(movie_values.shape[0], dtype=np.float64),
            high_res_pd=np.zeros(0, dtype=np.float64),
            mean_image=movie_values.mean(axis=0),
            alignment_valid_crop=full_frame_crop(movie_values.shape[1:]),
            alignment_shift_pixels=np.zeros(movie_values.shape[0], dtype=np.float64),
            motion_artifact_mask=np.zeros(movie_values.shape[0], dtype=np.bool_),
            frame_counts=FrameCountAudit(
                aligned_movie_frames=movie_values.shape[0],
                imaging_res_pd_samples=movie_values.shape[0],
                acquisition_number_of_frames=2,
                imaging_res_pd_minus_movie=0,
                acquisition_minus_movie=2 - movie_values.shape[0],
            ),
        )


if __name__ == "__main__":
    unittest.main()
