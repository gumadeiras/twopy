"""Tests for stimulus response heatmap computation.

Inputs: small converted movies with explicit epoch windows.
Outputs: foreground-masked dF/F response maps for pixel and window modes.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from twopy import (
    EpochFrameWindow,
    FrameWindow,
    ResponseMapOptions,
    compute_recording_response_maps,
    load_response_map_data,
    save_response_map_data,
)
from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData
from twopy.spatial import SpatialCrop, full_frame_crop


class ResponseMapsTest(unittest.TestCase):
    """Tests response-map computation independent from napari."""

    def test_pixel_mode_default_uses_two_pixel_smoothing(self) -> None:
        """Confirm pixel heatmaps default to a broader smoothing sigma.

        Inputs: default response-map options.
        Outputs: pixel smoothing sigma is two pixels.
        """
        self.assertEqual(ResponseMapOptions().pixel_smoothing_sigma, 2.0)

    def test_pixel_response_maps_average_epoch_trials(self) -> None:
        """Confirm pixel maps report stimulus-locked dF/F by epoch.

        Inputs: one small movie with two stimulus patches and explicit baseline
        frames.
        Outputs: one positive map per epoch with expected dF/F in the patch.
        """
        movie = np.full((7, 4, 4), 10.0, dtype=np.float64)
        movie[1:3, 1:3, 1:3] = 15.0
        movie[4:6, 2:4, 2:4] = 20.0
        windows = (
            EpochFrameWindow(FrameWindow(0, 1, 3, "epoch_1:A"), 1, "A"),
            EpochFrameWindow(FrameWindow(1, 4, 6, "epoch_2:B"), 2, "B"),
        )

        with tempfile.TemporaryDirectory() as directory:
            maps = compute_recording_response_maps(
                _recording(Path(directory), movie),
                epoch_windows=windows,
                options=ResponseMapOptions(
                    pixel_smoothing_sigma=0.0,
                    foreground_percentile=0.0,
                ),
            )

        self.assertEqual(tuple(epoch.epoch_name for epoch in maps.epochs), ("A", "B"))
        np.testing.assert_allclose(maps.epochs[0].response_values[1:3, 1:3], 0.5)
        np.testing.assert_allclose(maps.epochs[1].response_values[2:4, 2:4], 1.0)
        self.assertGreaterEqual(maps.response_scale, 1.0)

    def test_response_maps_keep_signed_inhibition_and_normalize(self) -> None:
        """Confirm heatmaps preserve inhibition on a shared -1 to 1 scale.

        Inputs: one movie with a response drop and a response increase.
        Outputs: negative and positive maps normalized by the largest magnitude.
        """
        movie = np.full((7, 4, 4), 10.0, dtype=np.float64)
        movie[1:3, 1:3, 1:3] = 5.0
        movie[4:6, 2:4, 2:4] = 20.0
        windows = (
            EpochFrameWindow(FrameWindow(0, 1, 3, "epoch_1:A"), 1, "A"),
            EpochFrameWindow(FrameWindow(1, 4, 6, "epoch_2:B"), 2, "B"),
        )

        with tempfile.TemporaryDirectory() as directory:
            maps = compute_recording_response_maps(
                _recording(Path(directory), movie),
                epoch_windows=windows,
                options=ResponseMapOptions(
                    pixel_smoothing_sigma=0.0,
                    foreground_percentile=0.0,
                ),
            )

        self.assertEqual(maps.response_scale, 1.0)
        np.testing.assert_allclose(maps.epochs[0].response_values[1:3, 1:3], -0.5)
        np.testing.assert_allclose(maps.epochs[1].response_values[2:4, 2:4], 1.0)
        self.assertLessEqual(
            float(np.nanmax(np.abs(maps.epochs[0].response_values))), 1.0
        )
        self.assertLessEqual(
            float(np.nanmax(np.abs(maps.epochs[1].response_values))), 1.0
        )

    def test_window_mode_covers_trailing_edges(self) -> None:
        """Confirm window starts include the bottom and right crop edges.

        Inputs: a 5x5 movie with a 3x3 window and stride 2.
        Outputs: every foreground pixel receives a finite response value.
        """
        movie = np.full((4, 5, 5), 10.0, dtype=np.float64)
        movie[1:3, 3:5, 3:5] = 15.0
        windows = (EpochFrameWindow(FrameWindow(0, 1, 3, "epoch_1:A"), 1, "A"),)

        with tempfile.TemporaryDirectory() as directory:
            maps = compute_recording_response_maps(
                _recording(Path(directory), movie),
                epoch_windows=windows,
                options=ResponseMapOptions(
                    mode="window",
                    window_size_pixels=3,
                    window_stride_pixels=2,
                    foreground_percentile=0.0,
                ),
            )

        self.assertEqual(maps.epochs[0].response_values.shape, (5, 5))
        self.assertTrue(np.all(np.isfinite(maps.epochs[0].response_values)))
        self.assertGreater(float(maps.epochs[0].response_values[-1, -1]), 0.0)

    def test_window_stride_cannot_exceed_window_size(self) -> None:
        """Confirm invalid window settings fail loudly."""
        movie = np.full((4, 5, 5), 10.0, dtype=np.float64)
        windows = (EpochFrameWindow(FrameWindow(0, 1, 3, "epoch_1:A"), 1, "A"),)

        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "cannot exceed"),
        ):
            compute_recording_response_maps(
                _recording(Path(directory), movie),
                epoch_windows=windows,
                options=ResponseMapOptions(
                    mode="window",
                    window_size_pixels=3,
                    window_stride_pixels=4,
                ),
            )

    def test_response_maps_stream_windows_without_full_movie_read(self) -> None:
        """Confirm heatmaps do not load the full aligned movie at once.

        Inputs: one small movie and an epoch window that can be read in bounded
        frame batches.
        Outputs: response-map computation succeeds without calling the
        ``read_frames`` escape hatch.
        """
        movie = np.full((4, 5, 5), 10.0, dtype=np.float64)
        movie[1:3, 2:4, 2:4] = 15.0
        windows = (EpochFrameWindow(FrameWindow(0, 1, 3, "epoch_1:A"), 1, "A"),)

        with tempfile.TemporaryDirectory() as directory:
            recording = _recording(Path(directory), movie)
            with patch.object(
                ConvertedMovie,
                "read_frames",
                side_effect=AssertionError("full movie read"),
            ):
                maps = compute_recording_response_maps(
                    recording,
                    epoch_windows=windows,
                    options=ResponseMapOptions(
                        pixel_smoothing_sigma=0.0,
                        foreground_percentile=0.0,
                    ),
                )

        self.assertEqual(len(maps.epochs), 1)
        np.testing.assert_allclose(maps.epochs[0].response_values[2:4, 2:4], 1.0)

    def test_response_maps_skip_epochs_without_preceding_baseline(self) -> None:
        """Confirm an initial epoch does not disable later heatmaps.

        Inputs: one movie whose first gray epoch starts at frame zero, followed
        by a stimulus epoch with enough preceding baseline context.
        Outputs: heatmaps are computed for the later stimulus epoch only.
        """
        movie = np.full((5, 4, 4), 10.0, dtype=np.float64)
        movie[2:4, 1:3, 1:3] = 15.0
        windows = (
            EpochFrameWindow(FrameWindow(0, 0, 1, "epoch_1:gray"), 1, "gray"),
            EpochFrameWindow(FrameWindow(1, 2, 4, "epoch_2:Odor"), 2, "Odor"),
        )

        with tempfile.TemporaryDirectory() as directory:
            maps = compute_recording_response_maps(
                _recording(Path(directory), movie),
                epoch_windows=windows,
                options=ResponseMapOptions(
                    pixel_smoothing_sigma=0.0,
                    foreground_percentile=0.0,
                ),
            )

        self.assertEqual(tuple(epoch.epoch_name for epoch in maps.epochs), ("Odor",))
        self.assertEqual(maps.response_scale, 0.5)
        np.testing.assert_allclose(maps.epochs[0].response_values[1:3, 1:3], 1.0)

    def test_response_maps_fail_when_no_response_epoch_has_baseline(self) -> None:
        """Confirm heatmaps fail loudly when no auditable baseline exists.

        Inputs: one non-baseline epoch starting at frame zero.
        Outputs: a clear error instead of fabricating baseline data.
        """
        movie = np.full((3, 4, 4), 10.0, dtype=np.float64)
        windows = (EpochFrameWindow(FrameWindow(0, 0, 2, "epoch_1:Odor"), 1, "Odor"),)

        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "preceding baseline"),
        ):
            compute_recording_response_maps(
                _recording(Path(directory), movie),
                epoch_windows=windows,
                options=ResponseMapOptions(foreground_percentile=0.0),
            )

    def test_response_map_data_round_trips_hdf5(self) -> None:
        """Confirm response heatmap files preserve maps and audit metadata.

        Inputs: one computed response map saved to ``response_heatmaps.h5``.
        Outputs: loaded options, crop, scale, epoch metadata, and arrays match.
        """
        movie = np.full((4, 5, 5), 10.0, dtype=np.float64)
        movie[1:3, 2:4, 2:4] = 15.0
        windows = (EpochFrameWindow(FrameWindow(0, 1, 3, "epoch_1:Odor"), 1, "Odor"),)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            maps = compute_recording_response_maps(
                _recording(root, movie),
                epoch_windows=windows,
                options=ResponseMapOptions(
                    mode="window",
                    window_size_pixels=3,
                    window_stride_pixels=1,
                    foreground_percentile=0.0,
                ),
            )
            path = root / "response_heatmaps.h5"
            save_response_map_data(path, maps)
            loaded = load_response_map_data(path)

        self.assertEqual(loaded.options, maps.options)
        self.assertEqual(loaded.spatial_crop, maps.spatial_crop)
        self.assertEqual(loaded.response_scale, maps.response_scale)
        np.testing.assert_allclose(loaded.mean_image, maps.mean_image)
        self.assertEqual(loaded.epochs[0].epoch_name, "Odor")
        self.assertEqual(loaded.epochs[0].epoch_number, 1)
        self.assertEqual(loaded.epochs[0].trial_count, 1)
        np.testing.assert_allclose(
            loaded.epochs[0].response_values,
            maps.epochs[0].response_values,
        )


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
