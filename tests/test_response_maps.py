"""Tests for stimulus response heatmap computation.

Inputs: small converted movies with explicit epoch windows.
Outputs: foreground-masked dF/F response maps for pixel and window modes.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from tests.converted_files import write_aligned_movie_file
from tests.recording_data import minimal_recording_data
from tests.tempdir import temporary_directory
from twopy import (
    EpochFrameWindow,
    FrameWindow,
    ResponseMapData,
    ResponseMapOptions,
    compute_recording_response_maps,
    load_response_map_data,
    save_response_map_data,
)
from twopy.converted import ConvertedMovie, RecordingData
from twopy.spatial import SpatialCrop, full_frame_crop
from twopy.spatial_orientation import (
    SPATIAL_ORIENTATION_ATTR,
    TWOPY_SPATIAL_ORIENTATION,
)


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

        with temporary_directory() as directory:
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

        with temporary_directory() as directory:
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

        with temporary_directory() as directory:
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
            temporary_directory() as directory,
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

        with temporary_directory() as directory:
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

        with temporary_directory() as directory:
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
            temporary_directory() as directory,
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

        with temporary_directory() as directory:
            root = Path(directory)
            maps = _response_map_data(root, movie, windows)
            path = root / "response_heatmaps.h5"
            save_response_map_data(path, maps)
            with h5py.File(path, "r") as h5_file:
                self.assertEqual(
                    h5_file.attrs[SPATIAL_ORIENTATION_ATTR],
                    TWOPY_SPATIAL_ORIENTATION,
                )
                self.assertEqual(
                    h5_file["mean_image"].attrs[SPATIAL_ORIENTATION_ATTR],
                    TWOPY_SPATIAL_ORIENTATION,
                )
                self.assertEqual(
                    h5_file["epochs/0000/response_values"].attrs[
                        SPATIAL_ORIENTATION_ATTR
                    ],
                    TWOPY_SPATIAL_ORIENTATION,
                )
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

    def test_response_map_data_rejects_missing_file_orientation(self) -> None:
        """Confirm response heatmap files must carry current orientation."""
        with temporary_directory() as directory:
            path = _write_response_map_file(Path(directory))
            with h5py.File(path, "r+") as h5_file:
                del h5_file.attrs[SPATIAL_ORIENTATION_ATTR]

            with self.assertRaisesRegex(ValueError, "spatial orientation"):
                load_response_map_data(path)

    def test_response_map_data_rejects_missing_mean_image_orientation(self) -> None:
        """Confirm persisted heatmap mean images must carry orientation."""
        with temporary_directory() as directory:
            path = _write_response_map_file(Path(directory))
            with h5py.File(path, "r+") as h5_file:
                del h5_file["mean_image"].attrs[SPATIAL_ORIENTATION_ATTR]

            with self.assertRaisesRegex(ValueError, "mean_image"):
                load_response_map_data(path)

    def test_response_map_data_rejects_missing_epoch_orientation(self) -> None:
        """Confirm persisted epoch response arrays must carry orientation."""
        with temporary_directory() as directory:
            path = _write_response_map_file(Path(directory))
            with h5py.File(path, "r+") as h5_file:
                del h5_file["epochs/0000/response_values"].attrs[
                    SPATIAL_ORIENTATION_ATTR
                ]

            with self.assertRaisesRegex(ValueError, "response_values"):
                load_response_map_data(path)

    def test_response_map_data_rejects_mean_image_shape_mismatch(self) -> None:
        """Confirm persisted heatmap backgrounds match the recorded crop."""
        with temporary_directory() as directory:
            path = _write_response_map_file(Path(directory))
            with h5py.File(path, "r+") as h5_file:
                del h5_file["mean_image"]
                mean_image = h5_file.create_dataset(
                    "mean_image",
                    data=np.zeros((5, 4), dtype=np.float64),
                )
                mean_image.attrs[SPATIAL_ORIENTATION_ATTR] = TWOPY_SPATIAL_ORIENTATION

            with self.assertRaisesRegex(ValueError, "mean_image shape"):
                load_response_map_data(path)

    def test_response_map_data_rejects_epoch_shape_mismatch(self) -> None:
        """Confirm persisted epoch response arrays match the mean image."""
        with temporary_directory() as directory:
            path = _write_response_map_file(Path(directory))
            with h5py.File(path, "r+") as h5_file:
                del h5_file["epochs/0000/response_values"]
                response_values = h5_file["epochs/0000"].create_dataset(
                    "response_values",
                    data=np.zeros((5, 4), dtype=np.float64),
                )
                response_values.attrs[SPATIAL_ORIENTATION_ATTR] = (
                    TWOPY_SPATIAL_ORIENTATION
                )

            with self.assertRaisesRegex(ValueError, "response_values shape"):
                load_response_map_data(path)


def _recording(
    root: Path,
    movie_values: np.ndarray,
    *,
    crop: SpatialCrop | None = None,
) -> RecordingData:
    """Create a minimal loaded recording backed by a temporary movie file."""
    movie_path = root / "aligned_movie.h5"
    write_aligned_movie_file(movie_path, movie_values)
    spatial_crop = full_frame_crop(movie_values.shape[1:]) if crop is None else crop
    return minimal_recording_data(
        root=root,
        movie_shape=movie_values.shape,
        acquisition_metadata={"acq.frameRate": 1.0},
        mean_image=movie_values.mean(axis=0),
        alignment_valid_crop=spatial_crop,
    )


def _response_map_data(
    root: Path,
    movie: np.ndarray,
    windows: tuple[EpochFrameWindow, ...],
) -> ResponseMapData:
    """Create one response-map object with non-default persisted settings."""
    return compute_recording_response_maps(
        _recording(root, movie),
        epoch_windows=windows,
        options=ResponseMapOptions(
            mode="window",
            window_size_pixels=3,
            window_stride_pixels=1,
            foreground_percentile=0.0,
        ),
    )


def _write_response_map_file(root: Path) -> Path:
    """Write a small response heatmap file and return its path."""
    movie = np.full((4, 5, 5), 10.0, dtype=np.float64)
    movie[1:3, 2:4, 2:4] = 15.0
    windows = (EpochFrameWindow(FrameWindow(0, 1, 3, "epoch_1:Odor"), 1, "Odor"),)
    path = root / "response_heatmaps.h5"
    save_response_map_data(path, _response_map_data(root, movie, windows))
    return path


if __name__ == "__main__":
    unittest.main()
