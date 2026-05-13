"""Tests for response-driven watershed ROI extraction.

Inputs: small converted movies with repeated stimulus windows.
Outputs: ROIs generated from repeatable stimulus-locked response score images.
"""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import twopy.response_roi_extraction as response_roi_extraction
from twopy import (
    EpochFrameWindow,
    FrameWindow,
    extract_response_watershed_rois,
    response_watershed_roi_set,
    roi_set_to_label_image,
)
from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData
from twopy.spatial import SpatialCrop, full_frame_crop


class ResponseRoiExtractionTest(unittest.TestCase):
    """Tests native response-watershed ROI discovery."""

    def test_response_watershed_splits_repeatable_epoch_response_fields(self) -> None:
        """Confirm response watershed segments repeatable stimulus-tuned pixels.

        Inputs: a movie with two repeated epoch response fields and dim
        unresponsive background.
        Outputs: two ROIs over the response fields plus auditable score maps.
        """
        movie = np.full((19, 5, 7), 10.0, dtype=np.float64)
        windows = _alternating_epoch_windows()
        epoch_one_mask = np.zeros((5, 7), dtype=np.bool_)
        epoch_one_mask[1:3, 1:3] = True
        epoch_two_mask = np.zeros((5, 7), dtype=np.bool_)
        epoch_two_mask[2:4, 4:6] = True
        for window in windows:
            response_mask = (
                epoch_one_mask if window.epoch_number == 1 else epoch_two_mask
            )
            movie[
                window.window.start_frame : window.window.stop_frame,
                response_mask,
            ] += 5.0

        with tempfile.TemporaryDirectory() as directory:
            recording = _recording(Path(directory), movie)

            extraction = extract_response_watershed_rois(
                recording,
                windows,
                min_pixels=4,
                minimum_score_percentile=0.0,
                within_roi_percentile=0.0,
            )

        self.assertTrue(extraction.score_images.reliability_available)
        self.assertEqual(extraction.selected_epoch_numbers, (1, 2))
        self.assertEqual(
            extraction.roi_set.labels,
            ("response_watershed_0001", "response_watershed_0002"),
        )
        label_image = roi_set_to_label_image(extraction.roi_set)
        self.assertEqual(label_image.shape, (5, 7))
        self.assertEqual(set(label_image[epoch_one_mask]), {1})
        self.assertEqual(set(label_image[epoch_two_mask]), {2})
        self.assertEqual(int(np.count_nonzero(label_image)), 8)
        self.assertGreater(float(np.max(extraction.score_images.coherence_image)), 0.0)

    def test_response_watershed_roi_set_returns_only_roi_set(self) -> None:
        """Confirm the convenience API returns masks without score maps.

        Inputs: the same repeated response movie used by the full extraction
        path.
        Outputs: a plain ``RoiSet`` for callers that do not need audit images.
        """
        movie = np.full((19, 5, 7), 10.0, dtype=np.float64)
        windows = _alternating_epoch_windows()
        movie[1:3, 1:3, 1:3] += 5.0
        movie[7:9, 1:3, 1:3] += 5.0
        movie[13:15, 1:3, 1:3] += 5.0
        movie[4:6, 2:4, 4:6] += 5.0
        movie[10:12, 2:4, 4:6] += 5.0
        movie[16:18, 2:4, 4:6] += 5.0

        with tempfile.TemporaryDirectory() as directory:
            recording = _recording(Path(directory), movie)

            roi_set = response_watershed_roi_set(
                recording,
                windows,
                min_pixels=4,
                minimum_score_percentile=0.0,
                within_roi_percentile=0.0,
            )

        self.assertEqual(roi_set.masks.shape, (2, 5, 7))

    def test_response_watershed_keeps_repeatable_broad_epoch_responses(self) -> None:
        """Confirm reliability does not require epoch-selective responses.

        Inputs: one field responding repeatably to every selected epoch.
        Outputs: a valid response-watershed ROI over that broad response field.
        """
        movie = np.full((19, 5, 7), 10.0, dtype=np.float64)
        windows = _alternating_epoch_windows()
        response_mask = np.zeros((5, 7), dtype=np.bool_)
        response_mask[1:3, 2:4] = True
        for window in windows:
            movie[
                window.window.start_frame : window.window.stop_frame,
                response_mask,
            ] += 5.0

        with tempfile.TemporaryDirectory() as directory:
            recording = _recording(Path(directory), movie)

            extraction = extract_response_watershed_rois(
                recording,
                windows,
                min_pixels=4,
                minimum_score_percentile=0.0,
                within_roi_percentile=0.0,
            )

        self.assertTrue(extraction.score_images.reliability_available)
        self.assertEqual(extraction.roi_set.masks.shape, (1, 5, 7))
        np.testing.assert_array_equal(extraction.roi_set.masks[0], response_mask)

    def test_response_watershed_embeds_alignment_crop_masks(self) -> None:
        """Confirm crop-domain score maps still produce full-frame ROI masks.

        Inputs: a recording with a smaller alignment-valid crop.
        Outputs: full-frame masks with all ROI pixels inside the crop.
        """
        movie = np.full((19, 5, 7), 10.0, dtype=np.float64)
        windows = _alternating_epoch_windows()
        movie[1:3, 1:3, 1:3] += 5.0
        movie[7:9, 1:3, 1:3] += 5.0
        movie[13:15, 1:3, 1:3] += 5.0
        movie[4:6, 2:4, 4:6] += 5.0
        movie[10:12, 2:4, 4:6] += 5.0
        movie[16:18, 2:4, 4:6] += 5.0
        crop = SpatialCrop(
            axis0_start=1,
            axis0_stop=4,
            axis1_start=1,
            axis1_stop=6,
            original_shape=(5, 7),
            source="test",
        )

        with tempfile.TemporaryDirectory() as directory:
            recording = _recording(Path(directory), movie, crop=crop)

            extraction = extract_response_watershed_rois(
                recording,
                windows,
                min_pixels=4,
                minimum_score_percentile=0.0,
                within_roi_percentile=0.0,
            )

        self.assertEqual(extraction.score_images.score_image.shape, crop.shape)
        self.assertEqual(extraction.roi_set.masks.shape[1:], (5, 7))
        self.assertFalse(np.any(extraction.roi_set.masks[:, 0, :]))
        self.assertFalse(np.any(extraction.roi_set.masks[:, :, 0]))

    def test_response_watershed_splits_trimmed_disconnected_components(self) -> None:
        """Confirm score trimming cannot produce sparse disconnected ROIs.

        Inputs: one watershed basin whose retained high-score pixels split into
        two 8-disconnected components after per-basin trimming.
        Outputs: each retained component becomes a separate contiguous ROI.
        """
        score_image = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 9.0, 8.0, 0.1, 0.0],
                [0.0, 8.0, 7.0, 0.1, 0.0],
                [0.0, 0.1, 0.1, 7.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        roi_set = response_roi_extraction._roi_set_from_score_image(
            score_image,
            score_smoothing_sigma=0.0,
            minimum_score_percentile=0.0,
            within_roi_percentile=50.0,
            min_pixels=1,
            max_pixels=None,
            label_prefix="response_watershed",
        )

        self.assertEqual(roi_set.masks.shape, (2, 5, 5))
        self.assertEqual(
            tuple(int(np.sum(mask)) for mask in roi_set.masks),
            (4, 1),
        )
        self.assertFalse(np.any(roi_set.masks[0] & roi_set.masks[1]))


def _alternating_epoch_windows() -> tuple[EpochFrameWindow, ...]:
    """Return six one-second-baselined alternating epoch windows."""
    windows: list[EpochFrameWindow] = []
    for index, start in enumerate((1, 4, 7, 10, 13, 16)):
        epoch_number = 1 if index % 2 == 0 else 2
        windows.append(
            EpochFrameWindow(
                window=FrameWindow(
                    index=index,
                    start_frame=start,
                    stop_frame=start + 2,
                    label=f"epoch {epoch_number}",
                ),
                epoch_number=epoch_number,
                epoch_name=f"epoch {epoch_number}",
            )
        )
    return tuple(windows)


def _recording(
    root: Path,
    movie_values: np.ndarray,
    *,
    crop: SpatialCrop | None = None,
) -> RecordingData:
    """Create a minimal loaded recording backed by a temporary movie file."""
    movie_path = root / "aligned_movie.h5"
    with h5py.File(movie_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = "aligned-movie"
        h5_file.create_dataset("movie/aligned", data=movie_values)
    spatial_crop = full_frame_crop(movie_values.shape[1:]) if crop is None else crop
    return RecordingData(
        path=root / "recording_data.h5",
        movie=ConvertedMovie(
            path=movie_path,
            dataset_name="movie/aligned",
            shape=movie_values.shape,
            dtype="float64",
        ),
        source_session_dir=root / "source",
        acquisition_metadata={"acq.frameRate": 1.0},
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
        alignment_valid_crop=spatial_crop,
        alignment_shift_pixels=np.zeros(movie_values.shape[0], dtype=np.float64),
        motion_artifact_mask=np.zeros(movie_values.shape[0], dtype=np.bool_),
        frame_counts=FrameCountAudit(
            aligned_movie_frames=movie_values.shape[0],
            imaging_res_pd_samples=movie_values.shape[0],
            acquisition_number_of_frames=movie_values.shape[0],
            imaging_res_pd_minus_movie=0,
            acquisition_minus_movie=0,
        ),
    )


if __name__ == "__main__":
    unittest.main()
