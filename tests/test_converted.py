"""Tests for loading converted twopy HDF5 recordings.

Inputs: tiny synthetic ``recording_data.h5`` and ``aligned_movie.h5`` files.
Outputs: assertions that analysis-facing objects load and validate them.
"""

import unittest
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from tests.tempdir import temporary_directory

from twopy import load_converted_recording, recording_frame_rate_hz
from twopy.converted import RecordingData


class ConvertedRecordingTest(unittest.TestCase):
    """Tests converted-recording loading for analysis code."""

    def test_loads_converted_recording_and_lazy_movie(self) -> None:
        """Confirm converted HDF5 files become typed Python objects.

        Inputs: a minimal converted recording with metadata, stimulus,
        photodiode, frame-count audit, mean image, and aligned movie.
        Outputs: assertions on loaded values and chunked movie reads.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            self._write_converted_recording(root)

            recording = load_converted_recording(root / "recording_data.h5")

            self.assertEqual(recording.path, root / "recording_data.h5")
            self.assertEqual(recording.movie.path, root / "aligned_movie.h5")
            self.assertEqual(recording.movie.shape, (3, 2, 2))
            self.assertEqual(recording.acquisition_metadata["acq.frameRate"], 10.0)
            self.assertEqual(recording_frame_rate_hz(recording), 10.0)
            self.assertEqual(recording.run_metadata["rig_name"], "OdorRig")
            self.assertEqual(recording.stimulus_parameters[1]["epochName"], "LR20")
            self.assertEqual(
                recording.stimulus_function_lookup,
                {"62002": "LEDMovingBars"},
            )
            columns = cast(
                list[dict[str, object]],
                recording.stimulus_specific_columns["62002"]["columns"],
            )
            self.assertEqual(
                columns[0]["source_expression"],
                "p.antenna",
            )
            self.assertEqual(
                recording.stimulus_data_column_names,
                ("time_seconds", "stimulus_frame_number", "epoch_number"),
            )
            self.assertEqual(recording.frame_counts.acquisition_minus_movie, -1)
            self.assertEqual(recording.alignment_valid_crop.shape, (1, 2))
            self.assertEqual(recording.alignment_valid_crop.axis0_start, 1)
            np.testing.assert_array_equal(
                recording.alignment_shift_pixels,
                np.array([0.0, 6.0, 0.0]),
            )
            np.testing.assert_array_equal(
                recording.motion_artifact_mask,
                np.array([False, True, False]),
            )
            np.testing.assert_array_equal(
                recording.mean_image,
                np.array([[4.0, 5.0], [6.0, 7.0]]),
            )
            np.testing.assert_array_equal(
                recording.movie.read_frames(
                    0,
                    1,
                    spatial_crop=recording.alignment_valid_crop,
                ),
                np.array([[[2.0, 3.0]]]),
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
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            self._write_recording_data_file(root / "recording_data.h5")

            with self.assertRaisesRegex(ValueError, "Missing converted aligned movie"):
                load_converted_recording(root / "recording_data.h5")

    def test_converted_movie_rejects_invalid_frame_ranges(self) -> None:
        """Confirm lazy converted movie reads share frame-range validation.

        Inputs: loaded converted movie and negative or empty frame ranges.
        Outputs: clear validation errors before HDF5 slicing.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            self._write_converted_recording(root)
            recording = load_converted_recording(root / "recording_data.h5")

            with self.assertRaisesRegex(ValueError, "converted movie frame range"):
                recording.movie.read_frames(-1, 2)
            with self.assertRaisesRegex(ValueError, "converted movie frame range"):
                tuple(recording.movie.iter_frame_batches(start=2, stop=2))

    def test_rejects_imaging_photodiode_length_mismatch(self) -> None:
        """Confirm loaded recordings keep one PD sample per movie frame.

        Inputs: converted files where ``imaging_res_pd`` is too short.
        Outputs: a clear validation error.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            self._write_aligned_movie_file(root / "aligned_movie.h5")
            self._write_recording_data_file(
                root / "recording_data.h5",
                imaging_res_pd=np.array([0.0, 1.0]),
            )

            with self.assertRaisesRegex(ValueError, "one sample per aligned movie"):
                load_converted_recording(root / "recording_data.h5")

    def test_recording_frame_rate_rejects_missing_metadata(self) -> None:
        """Confirm frame-rate parsing fails near the recording contract.

        Inputs: a minimal loaded recording without ``acq.frameRate``.
        Outputs: a clear metadata error.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = self._recording_with_acquisition_metadata(root, {})

            with self.assertRaisesRegex(ValueError, "acq.frameRate"):
                recording_frame_rate_hz(recording)

    def test_recording_frame_rate_rejects_non_numeric_metadata(self) -> None:
        """Confirm frame-rate metadata must be numeric.

        Inputs: a minimal loaded recording with a nonnumeric frame rate.
        Outputs: a clear metadata error.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording = self._recording_with_acquisition_metadata(
                root,
                {"acq.frameRate": object()},
            )

            with self.assertRaisesRegex(ValueError, "must be numeric"):
                recording_frame_rate_hz(recording)

    def test_rejects_malformed_stimulus_specific_column_metadata(self) -> None:
        """Confirm nested stimulus metadata is validated when loading.

        Inputs: converted recording whose stimulus-specific metadata has a
        non-dictionary value for one ``stimtype``.
        Outputs: clear validation error before stimulus helpers consume it.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            self._write_aligned_movie_file(root / "aligned_movie.h5")
            self._write_recording_data_file(
                root / "recording_data.h5",
                stimulus_specific_columns_json='{"62002": []}',
            )

            with self.assertRaisesRegex(
                ValueError,
                "stimulus/stimulus_specific_columns_json",
            ):
                load_converted_recording(root / "recording_data.h5")

    def test_rejects_converted_array_with_wrong_dtype(self) -> None:
        """Confirm converted arrays keep their documented dtype.

        Inputs: Converted recording where ``stimulus/data`` is integer-typed.
        Outputs: clear validation error before recording data is returned.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            self._write_aligned_movie_file(root / "aligned_movie.h5")
            self._write_recording_data_file(root / "recording_data.h5")
            with h5py.File(root / "recording_data.h5", "r+") as h5_file:
                del h5_file["stimulus/data"]
                h5_file["stimulus"].create_dataset(
                    "data",
                    data=np.zeros((3, 3), dtype=np.int64),
                )

            with self.assertRaisesRegex(ValueError, "stimulus/data must have dtype"):
                load_converted_recording(root / "recording_data.h5")

    def test_rejects_converted_array_with_wrong_rank(self) -> None:
        """Confirm converted arrays keep their documented rank.

        Inputs: Converted recording where ``mean_image`` is one-dimensional.
        Outputs: clear validation error before recording data is returned.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            self._write_aligned_movie_file(root / "aligned_movie.h5")
            self._write_recording_data_file(root / "recording_data.h5")
            with h5py.File(root / "recording_data.h5", "r+") as h5_file:
                del h5_file["movie/mean_image"]
                h5_file["movie"].create_dataset(
                    "mean_image",
                    data=np.zeros(4, dtype=np.float64),
                )

            with self.assertRaisesRegex(ValueError, "movie/mean_image must have 2"):
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

    def _recording_with_acquisition_metadata(
        self,
        root: Path,
        acquisition_metadata: dict[str, object],
    ) -> RecordingData:
        """Create a minimal recording object with custom acquisition metadata.

        Args:
            root: Temporary directory used for placeholder paths.
            acquisition_metadata: Metadata dictionary to attach.

        Returns:
            ``RecordingData`` suitable for metadata-helper tests.
        """
        self._write_converted_recording(root)
        recording = load_converted_recording(root / "recording_data.h5")
        return RecordingData(
            path=recording.path,
            movie=recording.movie,
            source_session_dir=recording.source_session_dir,
            acquisition_metadata=acquisition_metadata,
            run_metadata=recording.run_metadata,
            synchronization_metadata=recording.synchronization_metadata,
            stimulus_data=recording.stimulus_data,
            stimulus_data_column_names=recording.stimulus_data_column_names,
            stimulus_parameters=recording.stimulus_parameters,
            stimulus_function_lookup=recording.stimulus_function_lookup,
            stimulus_specific_columns=recording.stimulus_specific_columns,
            imaging_res_pd=recording.imaging_res_pd,
            high_res_pd=recording.high_res_pd,
            mean_image=recording.mean_image,
            alignment_valid_crop=recording.alignment_valid_crop,
            alignment_shift_pixels=recording.alignment_shift_pixels,
            motion_artifact_mask=recording.motion_artifact_mask,
            frame_counts=recording.frame_counts,
        )

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
        stimulus_specific_columns_json: str | None = None,
    ) -> None:
        """Write a tiny converted recording manifest file.

        Args:
            path: Destination ``recording_data.h5`` path.
            imaging_res_pd: Optional frame-resolution photodiode vector.
            stimulus_specific_columns_json: Optional stimulus metadata JSON.

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
            crop_group = movie_group.create_group("alignment_valid_crop")
            crop_group.attrs["source"] = "alignment_valid_crop"
            crop_group.attrs["axis0_start"] = 1
            crop_group.attrs["axis0_stop"] = 2
            crop_group.attrs["axis1_start"] = 0
            crop_group.attrs["axis1_stop"] = 2
            crop_group.attrs["original_shape"] = (2, 2)
            crop_group.create_dataset(
                "alignment_shift_pixels",
                data=np.array([0.0, 6.0, 0.0]),
            )
            crop_group.create_dataset(
                "motion_artifact_mask",
                data=np.array([False, True, False]),
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
                "data",
                data=np.array(
                    [[0.0, 1.0, 1.0], [0.1, 2.0, 2.0], [0.2, 3.0, 2.0]],
                ),
            )
            stimulus_group.create_dataset(
                "data_column_names",
                data=np.asarray(
                    ("time_seconds", "stimulus_frame_number", "epoch_number"),
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )
            stimulus_group.create_dataset(
                "parameters_json",
                data=(
                    '[{"epochName": "Gray Interleave", "stimtype": 62002}, '
                    '{"epochName": "LR20", "stimtype": 62002}]'
                ),
            )
            stimulus_group.create_dataset(
                "function_lookup_json",
                data='{"62002": "LEDMovingBars"}',
            )
            stimulus_group.create_dataset(
                "stimulus_specific_columns_json",
                data=stimulus_specific_columns_json
                or (
                    '{"62002": {"stimtype": "62002", "function": "LEDMovingBars", '
                    '"source_path": "stimfunctions/LEDMovingBars.m", "columns": ['
                    '{"mat_slot": 1, "column_name": "stimulus_specific_01", '
                    '"source_expression": "p.antenna", "source_line": 4}]}}'
                ),
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
