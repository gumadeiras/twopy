"""Tests for loading converted twopy HDF5 recordings.

Inputs: tiny synthetic ``recording_data.h5`` and ``aligned_movie.h5`` files.
Outputs: assertions that analysis-facing objects load and validate them.
"""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from twopy import load_converted_recording


class ConvertedRecordingTest(unittest.TestCase):
    """Tests converted-recording loading for analysis code."""

    def test_loads_converted_recording_and_lazy_movie(self) -> None:
        """Confirm converted HDF5 files become typed Python objects.

        Inputs: a minimal converted recording with metadata, stimulus,
        photodiode, frame-count audit, mean image, and aligned movie.
        Outputs: assertions on loaded values and chunked movie reads.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_converted_recording(root)

            recording = load_converted_recording(root / "recording_data.h5")

            self.assertEqual(recording.path, root / "recording_data.h5")
            self.assertEqual(recording.movie.path, root / "aligned_movie.h5")
            self.assertEqual(recording.movie.shape, (3, 2, 2))
            self.assertEqual(recording.acquisition_metadata["acq.frameRate"], 10.0)
            self.assertEqual(recording.run_metadata["rig_name"], "OdorRig")
            self.assertEqual(recording.stimulus_parameters[1]["epochName"], "LR20")
            self.assertEqual(
                recording.stimulus_timeline_column_names,
                ("time_seconds", "stimulus_frame_number", "epoch_number"),
            )
            self.assertEqual(recording.frame_counts.acquisition_minus_movie, -1)
            np.testing.assert_array_equal(
                recording.mean_image,
                np.array([[4.0, 5.0], [6.0, 7.0]]),
            )
            np.testing.assert_array_equal(
                recording.movie.read_frames(1, 3),
                np.array([[[4.0, 5.0], [6.0, 7.0]], [[8.0, 9.0], [10.0, 11.0]]]),
            )
            batches = tuple(recording.movie.iter_frame_batches(chunk_frames=2))
            self.assertEqual(
                [(start, stop) for start, stop, _ in batches], [(0, 2), (2, 3)]
            )
            np.testing.assert_array_equal(
                batches[0][2],
                np.arange(8, dtype=np.float64).reshape(2, 2, 2),
            )

    def test_rejects_missing_movie_file(self) -> None:
        """Confirm the loader fails loudly when movie data is absent.

        Inputs: a converted recording manifest without ``aligned_movie.h5``.
        Outputs: a clear validation error before analysis starts.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_recording_data_file(root / "recording_data.h5")

            with self.assertRaisesRegex(ValueError, "Missing converted aligned movie"):
                load_converted_recording(root / "recording_data.h5")

    def test_rejects_imaging_photodiode_length_mismatch(self) -> None:
        """Confirm loaded recordings keep one PD sample per movie frame.

        Inputs: converted files where ``imaging_res_pd`` is too short.
        Outputs: a clear validation error.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_aligned_movie_file(root / "aligned_movie.h5")
            self._write_recording_data_file(
                root / "recording_data.h5",
                imaging_res_pd=np.array([0.0, 1.0]),
            )

            with self.assertRaisesRegex(ValueError, "one sample per aligned movie"):
                load_converted_recording(root / "recording_data.h5")

    def _write_converted_recording(self, root: Path) -> None:
        """Write both converted HDF5 files for a small recording.

        Args:
            root: Temporary directory that receives the files.

        Returns:
            None.
        """
        self._write_aligned_movie_file(root / "aligned_movie.h5")
        self._write_recording_data_file(root / "recording_data.h5")

    def _write_aligned_movie_file(self, path: Path) -> None:
        """Write a tiny converted aligned movie file.

        Args:
            path: Destination ``aligned_movie.h5`` path.

        Returns:
            None.
        """
        with h5py.File(path, "w") as h5_file:
            h5_file.attrs["twopy_format"] = "aligned-movie"
            h5_file.create_dataset(
                "movie/aligned",
                data=np.arange(12, dtype=np.float64).reshape(3, 2, 2),
                compression="gzip",
            )

    def _write_recording_data_file(
        self,
        path: Path,
        *,
        imaging_res_pd: np.ndarray | None = None,
    ) -> None:
        """Write a tiny converted recording manifest file.

        Args:
            path: Destination ``recording_data.h5`` path.
            imaging_res_pd: Optional frame-resolution photodiode vector.

        Returns:
            None.
        """
        with h5py.File(path, "w") as h5_file:
            h5_file.attrs["twopy_format"] = "converted-recording"
            h5_file.attrs["source_session_dir"] = "/source/recording"

            movie_group = h5_file.create_group("movie")
            movie_group.attrs["aligned_movie_file"] = "aligned_movie.h5"
            movie_group.attrs["aligned_movie_dataset"] = "movie/aligned"
            movie_group.attrs["aligned_movie_shape"] = (3, 2, 2)
            movie_group.attrs["aligned_movie_dtype"] = "float64"
            movie_group.create_dataset(
                "mean_image",
                data=np.array([[4.0, 5.0], [6.0, 7.0]]),
            )

            metadata_group = h5_file.create_group("metadata")
            metadata_group.attrs["acq.frameRate"] = 10.0
            metadata_group.attrs["acq.zoomFactor"] = 2.0

            run_group = h5_file.create_group("run")
            run_group.attrs["rig_name"] = "OdorRig"
            run_group.attrs["run_number"] = 1

            stimulus_group = h5_file.create_group("stimulus")
            stimulus_group.attrs["clock_source"] = "stimulus presentation computer"
            stimulus_group.create_dataset(
                "timeline",
                data=np.array(
                    [[0.0, 1.0, 1.0], [0.1, 2.0, 2.0], [0.2, 3.0, 2.0]],
                ),
            )
            stimulus_group.create_dataset(
                "timeline_column_names",
                data=np.asarray(
                    ("time_seconds", "stimulus_frame_number", "epoch_number"),
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )
            stimulus_group.create_dataset(
                "parameters_json",
                data=('[{"epochName": "Gray Interleave"}, {"epochName": "LR20"}]'),
            )

            photodiode_group = h5_file.create_group("photodiode")
            photodiode_group.attrs["sync_signal"] = "photodiode"
            photodiode_group.create_dataset(
                "imaging_res_pd",
                data=(
                    np.array([0.0, 1.0, 0.0])
                    if imaging_res_pd is None
                    else imaging_res_pd
                ),
            )
            photodiode_group.create_dataset(
                "high_res_pd",
                data=np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
            )

            frame_counts = h5_file.create_group("frame_counts")
            frame_counts.attrs["aligned_movie_frames"] = 3
            frame_counts.attrs["imaging_res_pd_samples"] = 3
            frame_counts.attrs["acquisition_number_of_frames"] = 2
            frame_counts.attrs["imaging_res_pd_minus_movie"] = 0
            frame_counts.attrs["acquisition_minus_movie"] = -1


if __name__ == "__main__":
    unittest.main()
