"""Tests for converting microscope source files into twopy HDF5 files.

Inputs: a tiny synthetic two-photon session folder.
Outputs: assertions that conversion creates analysis-owned HDF5 datasets, with
the large aligned movie stored in a separate file.
"""

import json
import sqlite3
import unittest
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
import scipy.io

from tests.tempdir import temporary_directory
from twopy.analysis_cache import copy_converted_files_to_publish
from twopy.conversion import convert_recording_to_twopy, load_source_conversion_inputs
from twopy.conversion.orientation import orient_source_array_to_twopy
from twopy.spatial_orientation import (
    SPATIAL_ORIENTATION_ATTR,
    TWOPY_SPATIAL_ORIENTATION,
)


class ConversionTest(unittest.TestCase):
    """Tests the source-to-twopy conversion boundary."""

    def test_loads_source_conversion_inputs(self) -> None:
        """Confirm source files can be loaded for conversion.

        Inputs: a temporary recording with aligned movie, metadata, stimulus,
        and photodiode files.
        Outputs: assertions that each conversion input is available as Python.
        """
        with temporary_directory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_session(session_dir)

            loaded = load_source_conversion_inputs(session_dir)

            self.assertEqual(loaded.aligned_movie.shape, (3, 2, 2))
            self.assertEqual(loaded.aligned_movie.dtype, "float64")
            np.testing.assert_array_equal(
                loaded.aligned_movie.read_frames(1, 2),
                np.array([[[4.0, 5.0], [6.0, 7.0]]]),
            )
            np.testing.assert_array_equal(
                loaded.aligned_movie.mean_image(),
                np.array([[4.0, 5.0], [6.0, 7.0]]),
            )
            np.testing.assert_array_equal(
                loaded.aligned_movie.mean_image(start_frame=1, stop_frame=3),
                np.array([[6.0, 7.0], [8.0, 9.0]]),
            )
            self.assertEqual(loaded.acquisition.fields["acq.frameRate"], 2.0)
            self.assertEqual(loaded.acquisition.fields["acq.zoomFactor"], 2.0)
            self.assertEqual(loaded.alignment_crop.crop.shape, (2, 2))
            self.assertEqual(loaded.alignment_crop.alignment_frame_start, 0)
            self.assertEqual(loaded.alignment_crop.alignment_frame_stop, 2)
            np.testing.assert_array_equal(
                loaded.alignment_crop.motion_artifact_mask,
                np.array([False, False, False]),
            )
            self.assertEqual(loaded.run.fields["rig_name"], "OdorRig")
            self.assertEqual(loaded.run.fields["run_number"], 1)
            self.assertEqual(loaded.frame_counts.aligned_movie_frames, 3)
            self.assertEqual(loaded.frame_counts.imaging_res_pd_samples, 3)
            self.assertEqual(loaded.frame_counts.acquisition_number_of_frames, 3)
            self.assertEqual(loaded.frame_counts.imaging_res_pd_minus_movie, 0)
            self.assertEqual(loaded.frame_counts.acquisition_minus_movie, 0)
            self.assertEqual(
                loaded.stimulus_parameters.epochs[0]["epochName"],
                "Gray Interleave",
            )
            self.assertEqual(
                loaded.stimulus_code.function_lookup,
                {"62002": "LEDMovingBars"},
            )
            self.assertEqual(
                loaded.stimulus_code.stimulus_specific_columns["62002"]["function"],
                "LEDMovingBars",
            )
            np.testing.assert_array_equal(
                loaded.stimulus_data.epoch_numbers,
                np.array([1, 2, 2]),
            )
            self.assertEqual(
                loaded.stimulus_data.column_names,
                (
                    "time_seconds",
                    "stimulus_frame_number",
                    "epoch_number",
                ),
            )
            np.testing.assert_array_equal(
                loaded.photodiode.imaging_res_pd,
                np.array([0.0, 1.0, 0.0]),
            )

    def test_loads_older_chosenparams_stimulus_parameters(self) -> None:
        """Confirm older recordings without ``stimParams.mat`` still load.

        Inputs: a temporary recording where stimulus epoch parameters are saved
        as ``chosenparams.mat`` variable ``params``.
        Outputs: loaded epoch parameters and decoded stimulus-code metadata.
        """
        with temporary_directory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_session(session_dir)
            stimulus_dir = session_dir / "stimulusData"
            stim_params_path = stimulus_dir / "stimParams.mat"
            raw_params = scipy.io.loadmat(
                stim_params_path,
                squeeze_me=True,
                struct_as_record=False,
            )["stimParams"]
            stim_params_path.unlink()
            scipy.io.savemat(stimulus_dir / "chosenparams.mat", {"params": raw_params})

            loaded = load_source_conversion_inputs(session_dir)

            self.assertEqual(loaded.stimulus_parameters.path.name, "chosenparams.mat")
            self.assertEqual(
                loaded.stimulus_parameters.epochs[0]["epochName"],
                "Gray Interleave",
            )
            self.assertEqual(
                loaded.stimulus_code.function_lookup,
                {"62002": "LEDMovingBars"},
            )

    def test_labels_stable_stimulus_data_slots_from_matlab_writer(self) -> None:
        """Confirm ``stimData`` labels follow the backed-up MATLAB writer.

        Inputs: a temporary recording with the full 35-column stimulus table
        written by the stimulus computer.
        Outputs: column labels for the stable columns and generic labels for
        stimulus-specific slots that depend on the MATLAB stimulus function.
        """
        with temporary_directory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_session(session_dir, stimulus_data_column_count=35)

            loaded = load_source_conversion_inputs(session_dir)

            self.assertEqual(
                loaded.stimulus_data.column_names[0:3],
                (
                    "time_seconds",
                    "stimulus_frame_number",
                    "epoch_number",
                ),
            )
            self.assertEqual(
                loaded.stimulus_data.column_names[3:13],
                (
                    "closed_loop_01",
                    "closed_loop_02",
                    "closed_loop_03",
                    "closed_loop_04",
                    "closed_loop_05",
                    "closed_loop_06",
                    "closed_loop_07",
                    "closed_loop_08",
                    "closed_loop_09",
                    "closed_loop_10",
                ),
            )
            self.assertEqual(
                loaded.stimulus_data.column_names[13:35],
                (
                    "stimulus_specific_01",
                    "stimulus_specific_02",
                    "stimulus_specific_03",
                    "stimulus_specific_04",
                    "stimulus_specific_05",
                    "stimulus_specific_06",
                    "stimulus_specific_07",
                    "stimulus_specific_08",
                    "stimulus_specific_09",
                    "stimulus_specific_10",
                    "stimulus_specific_11",
                    "stimulus_specific_12",
                    "stimulus_specific_13",
                    "stimulus_specific_14",
                    "stimulus_specific_15",
                    "stimulus_specific_16",
                    "stimulus_specific_17",
                    "stimulus_specific_18",
                    "stimulus_specific_19",
                    "stimulus_specific_20",
                    "photodiode_flash",
                    "trailing_empty",
                ),
            )

    def test_computes_alignment_valid_crop_from_stimulus_bounded_offsets(self) -> None:
        """Confirm conversion saves the alignment-valid spatial crop.

        Inputs: a temporary recording with known alignment offsets during the
        stimulus-bounded frame range.
        Outputs: crop bounds that remove only the invalid aligned border.
        """
        with temporary_directory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_session(
                session_dir,
                movie_values=np.zeros((3, 4, 4), dtype=np.float64),
                alignment_data=np.array(
                    [
                        [2.0, 1.0, 0.0, 0.0, 2.0, 1.0],
                        [-2.0, -1.0, 0.0, 0.0, -2.0, -1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                ),
            )

            loaded = load_source_conversion_inputs(session_dir)

            self.assertEqual(loaded.alignment_crop.crop.axis0_start, 0)
            self.assertEqual(loaded.alignment_crop.crop.axis0_stop, 4)
            self.assertEqual(loaded.alignment_crop.crop.axis1_start, 1)
            self.assertEqual(loaded.alignment_crop.crop.axis1_stop, 3)
            self.assertEqual(loaded.alignment_crop.x_cutoff_pixels, 2)
            self.assertEqual(loaded.alignment_crop.y_cutoff_pixels, 1)

    def test_converts_recording_to_twopy_hdf5(self) -> None:
        """Confirm conversion writes twopy-owned datasets and mean image.

        Inputs: a temporary source recording and output directory.
        Outputs: HDF5 datasets for metadata, stimulus, photodiode, full-movie
        mean image, and a separate aligned movie file.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / "recording"
            output_dir = root / "output"
            self._write_session(session_dir)

            converted = convert_recording_to_twopy(session_dir, output_dir)

            self.assertEqual(converted.path, output_dir / "recording_data.h5")
            self.assertEqual(converted.movie_path, output_dir / "aligned_movie.h5")
            self.assertEqual(converted.mean_image_start_frame, 0)
            self.assertEqual(converted.mean_image_stop_frame, 3)
            with h5py.File(converted.path, "r") as h5_file:
                self.assertEqual(h5_file.attrs["twopy_format"], "converted-recording")
                self.assertEqual(
                    h5_file["movie"].attrs["aligned_movie_file"],
                    "aligned_movie.h5",
                )
                self.assertEqual(
                    h5_file["movie"].attrs["aligned_movie_dataset"],
                    "movie/aligned",
                )
                np.testing.assert_array_equal(
                    h5_file["movie"].attrs["aligned_movie_shape"],
                    np.array([3, 2, 2]),
                )
                np.testing.assert_array_equal(
                    h5_file["movie/mean_image"][()],
                    np.array([[4.0, 6.0], [5.0, 7.0]]),
                )
                self.assertEqual(
                    h5_file["movie"].attrs[SPATIAL_ORIENTATION_ATTR],
                    TWOPY_SPATIAL_ORIENTATION,
                )
                self.assertEqual(
                    h5_file["movie/mean_image"].attrs[SPATIAL_ORIENTATION_ATTR],
                    TWOPY_SPATIAL_ORIENTATION,
                )
                self.assertIsNone(h5_file["movie/mean_image"].compression)
                self.assertEqual(h5_file["movie/mean_image"].attrs["start_frame"], 0)
                self.assertEqual(h5_file["movie/mean_image"].attrs["stop_frame"], 3)
                self.assertEqual(
                    h5_file["movie/alignment_valid_crop"].attrs["source"],
                    "alignment_valid_crop",
                )
                self.assertEqual(
                    h5_file["movie/alignment_valid_crop"].attrs["axis0_start"],
                    0,
                )
                self.assertEqual(
                    h5_file["movie/alignment_valid_crop"].attrs["axis0_stop"],
                    2,
                )
                self.assertEqual(
                    h5_file["movie/alignment_valid_crop"].attrs["alignment_frame_stop"],
                    2,
                )
                np.testing.assert_array_equal(
                    h5_file["movie/alignment_valid_crop/alignment_shift_pixels"][()],
                    np.array([0.0, 0.0, 0.0]),
                )
                np.testing.assert_array_equal(
                    h5_file["movie/alignment_valid_crop/motion_artifact_mask"][()],
                    np.array([False, False, False]),
                )
                self.assertEqual(h5_file["metadata"].attrs["acq.frameRate"], 2.0)
                self.assertEqual(h5_file["run"].attrs["rig_name"], "OdorRig")
                self.assertEqual(h5_file["run"].attrs["genotype"], "test_genotype")
                self.assertEqual(h5_file["run"].attrs["run_number"], 1)
                self.assertEqual(
                    h5_file["stimulus"].attrs["clock_source"],
                    "stimulus presentation computer",
                )
                self.assertEqual(h5_file["stimulus/data"].shape, (3, 3))
                self.assertEqual(
                    tuple(
                        value.decode("utf-8")
                        for value in h5_file["stimulus/data_column_names"][()]
                    ),
                    ("time_seconds", "stimulus_frame_number", "epoch_number"),
                )
                self.assertEqual(h5_file["photodiode/imaging_res_pd"].shape, (3,))
                self.assertEqual(
                    h5_file["photodiode"].attrs["sync_signal"],
                    "photodiode",
                )
                self.assertIn(
                    "independent computers",
                    h5_file["photodiode"].attrs["clock_relationship"],
                )
                self.assertIn(
                    "trial transitions",
                    h5_file["photodiode"].attrs["event_encoding"],
                )
                self.assertIn(
                    "decode photodiode events",
                    h5_file["photodiode"].attrs["analysis_rule"],
                )
                self.assertEqual(
                    h5_file["photodiode/imaging_res_pd"].attrs["sample_domain"],
                    "one sample per aligned imaging frame",
                )
                self.assertEqual(
                    h5_file["photodiode/high_res_pd"].attrs["sample_domain"],
                    "high-rate photodiode signal for precise event detection",
                )
                self.assertEqual(
                    h5_file["frame_counts"].attrs["aligned_movie_frames"],
                    3,
                )
                self.assertEqual(
                    h5_file["frame_counts"].attrs["imaging_res_pd_samples"],
                    3,
                )
                self.assertEqual(
                    h5_file["frame_counts"].attrs["acquisition_number_of_frames"],
                    3,
                )
                self.assertEqual(
                    h5_file["frame_counts"].attrs["imaging_res_pd_minus_movie"],
                    0,
                )
                self.assertEqual(
                    h5_file["frame_counts"].attrs["acquisition_minus_movie"],
                    0,
                )
                parameters = json.loads(h5_file["stimulus/parameters_json"][()])
                self.assertEqual(parameters[1]["epochName"], "LR20")
                function_lookup = json.loads(
                    h5_file["stimulus/function_lookup_json"][()],
                )
                self.assertEqual(function_lookup, {"62002": "LEDMovingBars"})
                stimulus_columns = json.loads(
                    h5_file["stimulus/stimulus_specific_columns_json"][()],
                )
                self.assertEqual(
                    stimulus_columns["62002"]["columns"][0]["source_expression"],
                    "p.antenna",
                )
                self.assertNotIn("aligned", h5_file["movie"])

            with h5py.File(converted.movie_path, "r") as h5_file:
                self.assertEqual(h5_file.attrs["twopy_format"], "aligned-movie")
                self.assertEqual(
                    h5_file["movie/aligned"].attrs[SPATIAL_ORIENTATION_ATTR],
                    TWOPY_SPATIAL_ORIENTATION,
                )
                np.testing.assert_array_equal(
                    h5_file["movie/aligned"][()],
                    orient_source_array_to_twopy(
                        np.arange(12, dtype=np.float64).reshape(3, 2, 2),
                    ),
                )

    def test_conversion_stores_movie_in_python_image_order(self) -> None:
        """Confirm conversion orients non-square source movies once at write time.

        Inputs: a non-square source movie whose axes make transposition visible.
        Outputs: converted movie and mean image in Python image order matching
        MATLAB display.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / "recording"
            output_dir = root / "output"
            source_movie = np.array(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                    [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                ],
            )
            self._write_session(session_dir, movie_values=source_movie)

            converted = convert_recording_to_twopy(session_dir, output_dir)

            expected_movie = orient_source_array_to_twopy(source_movie)
            self.assertEqual(converted.movie_shape, expected_movie.shape)
            with h5py.File(converted.path, "r") as recording_file:
                np.testing.assert_array_equal(
                    recording_file["movie"].attrs["aligned_movie_shape"],
                    np.array(expected_movie.shape),
                )
                np.testing.assert_array_equal(
                    recording_file["movie/mean_image"][()],
                    expected_movie.mean(axis=0),
                )
                np.testing.assert_array_equal(
                    recording_file["movie/alignment_valid_crop"].attrs[
                        "original_shape"
                    ],
                    np.array(expected_movie.shape[1:]),
                )
            with h5py.File(converted.movie_path, "r") as movie_file:
                np.testing.assert_array_equal(
                    movie_file["movie/aligned"][()],
                    expected_movie,
                )

    def test_conversion_persists_database_hemisphere_metadata(self) -> None:
        """Confirm conversion saves database fly-eye metadata into run fields.

        Inputs: a source recording path with a matching Clark-style DB row.
        Outputs: converted HDF5 run metadata includes ``hemisphere`` and ``eye``.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            database_dir = root / "db"
            database_dir.mkdir()
            session_dir = (
                data_dir / "genotype" / "stimulus" / "2023" / "10_17" / "10_02_49"
            )
            output_dir = root / "output"
            config_path = root / "config.yml"
            self._write_session(session_dir)
            self._write_database_log(database_dir / "experimentLog.db")
            config_path.write_text(
                "\n".join(
                    (
                        f"database_path: {database_dir}",
                        f"data_paths:\n  - {data_dir}",
                        "database_access: direct",
                        "analysis_output: source",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            converted = convert_recording_to_twopy(
                session_dir,
                output_dir,
                config_path=config_path,
            )

            with h5py.File(converted.path, "r") as h5_file:
                self.assertEqual(h5_file["run"].attrs["hemisphere"], "left")
                self.assertEqual(h5_file["run"].attrs["eye"], "left")

    def test_explicit_output_conversion_allows_missing_config(self) -> None:
        """Confirm explicit output conversion does not require machine config.

        Inputs: a valid source recording, explicit output directory, and a
        missing config file.
        Outputs: converted files are written under the explicit output.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / "recording"
            output_dir = root / "output"
            self._write_session(session_dir)

            converted = convert_recording_to_twopy(
                session_dir,
                output_dir,
                config_path=root / "missing-config.yml",
            )

            self.assertEqual(converted.path.parent, output_dir)

    def test_explicit_output_conversion_rejects_invalid_config(self) -> None:
        """Confirm malformed config fails even with explicit output.

        Inputs: a valid source recording, explicit output directory, and a
        malformed config file.
        Outputs: conversion raises the config parse error.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / "recording"
            output_dir = root / "output"
            config_path = root / "config.yml"
            self._write_session(session_dir)
            config_path.write_text("database_path: [broken\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Could not parse twopy config"):
                convert_recording_to_twopy(
                    session_dir,
                    output_dir,
                    config_path=config_path,
                )

    def test_converts_mean_image_for_requested_frame_range(self) -> None:
        """Confirm conversion can compute a range-limited mean image.

        Inputs: a temporary source recording and a selected frame range.
        Outputs: a mean image over only that frame range.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / "recording"
            output_dir = root / "output"
            self._write_session(session_dir)

            converted = convert_recording_to_twopy(
                session_dir,
                output_dir,
                mean_start_frame=1,
                mean_stop_frame=3,
            )

            with h5py.File(converted.path, "r") as h5_file:
                np.testing.assert_array_equal(
                    h5_file["movie/mean_image"][()],
                    np.array([[6.0, 8.0], [7.0, 9.0]]),
                )
                self.assertEqual(h5_file["movie/mean_image"].attrs["start_frame"], 1)
                self.assertEqual(h5_file["movie/mean_image"].attrs["stop_frame"], 3)

    def test_rejects_invalid_conversion_mean_image_frame_range(self) -> None:
        """Confirm conversion validates requested mean-image frame bounds.

        Inputs: a temporary source recording and an empty mean-image range.
        Outputs: clear validation error before writing converted files.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            session_dir = root / "recording"
            output_dir = root / "output"
            self._write_session(session_dir)

            with self.assertRaisesRegex(ValueError, "frame range for mean image"):
                convert_recording_to_twopy(
                    session_dir,
                    output_dir,
                    mean_start_frame=2,
                    mean_stop_frame=2,
                )

    def test_conversion_uses_configured_analysis_output_by_default(self) -> None:
        """Confirm conversion defaults to ``analysis_output`` from config.

        Inputs: a temporary config using mirrored output routing and no explicit
        conversion output directory.
        Outputs: converted files under the configured mirrored output root.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            session_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            output_root = root / "analysis"
            config_path = root / "config.yml"
            self._write_session(session_dir)
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                f"data_paths:\n  - {data_root}\n"
                "database_access: copy\n"
                "analysis_caching: false\n"
                f"analysis_output: {output_root}\n",
                encoding="utf-8",
            )

            converted = convert_recording_to_twopy(
                session_dir,
                config_path=config_path,
            )

            expected_dir = output_root / "fly" / "stim" / "2023" / "10_17"
            self.assertEqual(converted.path, expected_dir / "recording_data.h5")
            self.assertEqual(converted.movie_path, expected_dir / "aligned_movie.h5")
            self.assertTrue(converted.path.is_file())
            self.assertTrue(converted.movie_path.is_file())

    def test_conversion_uses_cached_work_dir_when_analysis_caching_enabled(
        self,
    ) -> None:
        """Confirm conversion writes only to the local cache when enabled.

        Inputs: a temporary config with cache and publish roots.
        Outputs: converted files under the cache root only.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            session_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            output_root = root / "publish"
            cache_root = root / "cache"
            config_path = root / "config.yml"
            self._write_session(session_dir)
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                f"data_paths:\n  - {data_root}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {cache_root}\n"
                f"analysis_output: {output_root}\n",
                encoding="utf-8",
            )

            converted = convert_recording_to_twopy(
                session_dir,
                config_path=config_path,
            )

            expected_dir = cache_root / "fly" / "stim" / "2023" / "10_17"
            published_dir = output_root / "fly" / "stim" / "2023" / "10_17"
            self.assertEqual(converted.path, expected_dir / "recording_data.h5")
            self.assertEqual(converted.movie_path, expected_dir / "aligned_movie.h5")
            self.assertTrue(converted.path.is_file())
            self.assertFalse((published_dir / "recording_data.h5").exists())
            self.assertFalse((published_dir / "aligned_movie.h5").exists())

    def test_converted_file_publish_does_not_require_analysis_caching(self) -> None:
        """Confirm converted HDF5 publish copies are not tied to cache policy.

        Inputs: explicit local conversion output plus a config with caching off.
        Outputs: converted files are copied to the configured publish root.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            session_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            local_output = root / "local"
            output_root = root / "publish"
            config_path = root / "config.yml"
            self._write_session(session_dir)
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                f"data_paths:\n  - {data_root}\n"
                "database_access: copy\n"
                "analysis_caching: false\n"
                f"analysis_output: {output_root}\n",
                encoding="utf-8",
            )
            converted = convert_recording_to_twopy(session_dir, output_dir=local_output)

            result = copy_converted_files_to_publish(
                recording_data_path=converted.path,
                movie_path=converted.movie_path,
                source_session_dir=converted.source_session_dir,
                config_path=config_path,
            )

            published_dir = output_root / "fly" / "stim" / "2023" / "10_17"
            self.assertIsNotNone(result)
            self.assertTrue((published_dir / "recording_data.h5").is_file())
            self.assertTrue((published_dir / "aligned_movie.h5").is_file())

    def test_allows_observed_one_frame_acquisition_metadata_offset(self) -> None:
        """Confirm sampled ScanImage frame-count offset is audited, not hidden.

        Inputs: a temporary recording where ``acq.numberOfFrames`` is one less
        than the aligned movie and imaging-resolution photodiode.
        Outputs: loaded audit values that preserve the one-frame offset.
        """
        with temporary_directory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_session(session_dir, acquisition_frame_count=2)

            loaded = load_source_conversion_inputs(session_dir)

            self.assertEqual(loaded.frame_counts.aligned_movie_frames, 3)
            self.assertEqual(loaded.frame_counts.imaging_res_pd_samples, 3)
            self.assertEqual(loaded.frame_counts.acquisition_number_of_frames, 2)
            self.assertEqual(loaded.frame_counts.acquisition_minus_movie, -1)

    def test_rejects_imaging_photodiode_frame_count_mismatch(self) -> None:
        """Confirm frame-resolution photodiode must match movie frames.

        Inputs: a temporary recording where ``imagingResPd`` has too few samples.
        Outputs: a clear failure before conversion writes twopy files.
        """
        with temporary_directory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_session(
                session_dir,
                imaging_res_pd=np.array([0.0, 1.0]),
            )

            with self.assertRaisesRegex(ValueError, "imagingResPd"):
                load_source_conversion_inputs(session_dir)

    def test_rejects_unexpected_acquisition_frame_count_offset(self) -> None:
        """Confirm acquisition frame metadata cannot drift by more than one.

        Inputs: a temporary recording where acquisition metadata is two frames
        short of the aligned movie.
        Outputs: a clear failure before conversion writes twopy files.
        """
        with temporary_directory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_session(session_dir, acquisition_frame_count=1)

            with self.assertRaisesRegex(ValueError, "acq.numberOfFrames"):
                load_source_conversion_inputs(session_dir)

    def test_rejects_ambiguous_long_end_flashes(self) -> None:
        """Confirm conversion does not guess when end flashes are ambiguous.

        Inputs: a temporary recording with three long high-rate photodiode
        flashes.
        Outputs: a clear ambiguity error before conversion writes files.
        """
        with temporary_directory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_session(
                session_dir,
                movie_values=np.zeros((4, 2, 2), dtype=np.float64),
                high_res_pd=np.array(
                    [
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                    dtype=np.float64,
                ),
            )

            with self.assertRaisesRegex(ValueError, "Ambiguous stimulus end"):
                load_source_conversion_inputs(session_dir)

    def _write_session(
        self,
        session_dir: Path,
        *,
        acquisition_frame_count: int = 3,
        imaging_res_pd: np.ndarray | None = None,
        stimulus_data_column_count: int = 3,
        movie_values: np.ndarray | None = None,
        alignment_data: np.ndarray | None = None,
        high_res_pd: np.ndarray | None = None,
    ) -> None:
        """Create a minimal valid source recording fixture.

        Args:
            session_dir: Temporary folder that receives fixture files.
            acquisition_frame_count: ``acq.numberOfFrames`` metadata value.
            imaging_res_pd: Optional imaging-resolution photodiode vector.
            stimulus_data_column_count: Number of ``stimData`` columns to write.
            movie_values: Optional aligned movie values.
            alignment_data: Optional alignment text table.
            high_res_pd: Optional high-resolution photodiode vector.

        Returns:
            None. The function writes all required source files to disk.
        """
        stimulus_dir = session_dir / "stimulusData"
        stimulus_dir.mkdir(parents=True)
        aligned_movie = (
            np.arange(12, dtype=np.float64).reshape(3, 2, 2)
            if movie_values is None
            else movie_values
        )

        with h5py.File(session_dir / "alignedMovie.mat", "w") as mat_file:
            mat_file.create_dataset(
                "imgFrames_ch1",
                data=aligned_movie,
                chunks=(1, aligned_movie.shape[1], aligned_movie.shape[2]),
                compression="gzip",
            )

        scipy.io.savemat(
            session_dir / "imageDescription.mat",
            {
                "state": {
                    "configName": "test_config",
                    "software": {"version": 3.8},
                    "acq": {
                        "linesPerFrame": aligned_movie.shape[2],
                        "pixelsPerLine": aligned_movie.shape[1],
                        "numberOfFrames": acquisition_frame_count,
                        "numberOfChannelsSave": 2,
                        "frameRate": 2.0,
                        "zoomFactor": 2.0,
                        "pixelTime": 0.1,
                        "msPerLine": 1.0,
                        "zStepSize": 1.0,
                        "scanAngleMultiplierFast": 1.0,
                        "scanAngleMultiplierSlow": 1.0,
                    },
                    "motor": {
                        "absXPosition": 1.0,
                        "absYPosition": 2.0,
                        "absZPosition": 3.0,
                    },
                },
            },
        )
        scipy.io.savemat(
            stimulus_dir / "stimParams.mat",
            {
                "stimParams": np.array(
                    [
                        {
                            "epochName": "Gray Interleave",
                            "stimtype": 62002,
                            "intensity": 0.0,
                        },
                        {
                            "epochName": "LR20",
                            "stimtype": 62002,
                            "intensity": 20.0,
                        },
                    ],
                    dtype=object,
                ),
            },
        )
        scipy.io.savemat(
            stimulus_dir / "stimdata.mat",
            {
                "stimData": self._stimulus_data_table(stimulus_data_column_count),
            },
        )
        scipy.io.savemat(
            stimulus_dir / "runDetails.mat",
            {
                "flyId": 123,
                "genotype": "test_genotype",
                "rigName": "OdorRig",
                "rigTemperature": 0,
                "runNumber": 1,
            },
        )
        (stimulus_dir / "textStimData.csv").write_text(
            "Time,FrameNumber,Epoch,Flash\n0.0,1,1,1\n0.1,2,2,0\n0.2,3,2,0\n",
            encoding="utf-8",
        )
        self._write_filebackup_zip(stimulus_dir / "filebackup.zip")
        scipy.io.savemat(
            session_dir / "imagingResPd.mat",
            {
                "imagingResPd": (
                    self._default_imaging_res_pd(aligned_movie.shape[0])
                    if imaging_res_pd is None
                    else imaging_res_pd
                ),
            },
        )
        scipy.io.savemat(
            session_dir / "highResPd.mat",
            {
                "highResPd": (
                    np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
                    if high_res_pd is None
                    else high_res_pd
                ),
            },
        )

        (session_dir / "stimulus_name_changes_001.tif").touch()
        np.savetxt(
            session_dir / "stimulus_name_changes_001_ch1_disinterleaved_alignment.txt",
            (
                np.zeros((aligned_movie.shape[0], 6), dtype=np.float64)
                if alignment_data is None
                else alignment_data
            ),
            delimiter="\t",
        )
        (session_dir / "defaultAlignChannel.txt").write_text("1\n", encoding="utf-8")

    def _write_database_log(self, path: Path) -> None:
        """Create the minimal database row conversion uses for fly-eye metadata."""
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                """
                CREATE TABLE fly (
                    flyId INTEGER PRIMARY KEY,
                    genotype TEXT,
                    cellType TEXT,
                    fluorescentProtein TEXT,
                    eye TEXT,
                    surgeon TEXT
                )
                """,
            )
            connection.execute(
                """
                CREATE TABLE stimulusPresentation (
                    stimulusPresentationId INTEGER PRIMARY KEY,
                    fly INT,
                    relativeDataPath TEXT,
                    stimulusFunction TEXT,
                    date TEXT,
                    dataQuality INT
                )
                """,
            )
            connection.execute(
                """
                INSERT INTO fly
                    (flyId, genotype, cellType, fluorescentProtein, eye, surgeon)
                VALUES
                    (10923, 'genotype', 'cell', 'sensor', 'left', 'Gustavo')
                """,
            )
            connection.execute(
                """
                INSERT INTO stimulusPresentation
                    (
                        stimulusPresentationId,
                        fly,
                        relativeDataPath,
                        stimulusFunction,
                        date,
                        dataQuality
                    )
                VALUES
                    (
                        20005,
                        10923,
                        'genotype\\stimulus\\2023\\10_17\\10_02_49',
                        'stimulus',
                        '2023-10-17 10:02:49',
                        1
                    )
                """,
            )
            connection.commit()
        finally:
            connection.close()

    def _write_filebackup_zip(self, path: Path) -> None:
        """Write the tiny stimulus-code backup required for conversion.

        Args:
            path: Destination ``filebackup.zip`` path.

        Returns:
            None. The zip contains the lookup and one stimulus function.
        """
        with ZipFile(path, "w") as archive:
            archive.writestr("paramfiles/stimulus_lookup.txt", "62002,LEDMovingBars\n")
            archive.writestr(
                "stimfunctions/LEDMovingBars.m",
                "\n".join(
                    (
                        "function stimData = LEDMovingBars(Q)",
                        "p = Q.stims.currParam;",
                        "stimData = Q.stims.stimData;",
                        "stimData.mat(1) = p.antenna;",
                        "stimData.mat(2) = p.intensity; % epoch intensity",
                        "end",
                    ),
                ),
            )

    def _default_imaging_res_pd(self, frame_count: int) -> np.ndarray:
        """Return a small default frame-resolution photodiode vector.

        Args:
            frame_count: Number of aligned movie frames.

        Returns:
            One photodiode sample per frame.
        """
        values = np.zeros(frame_count, dtype=np.float64)
        if frame_count > 1:
            values[1] = 1.0
        return values

    def _stimulus_data_table(self, column_count: int) -> np.ndarray:
        """Build a small ``stimData`` table with a requested column count.

        Args:
            column_count: Number of stimulus table columns to return.

        Returns:
            Three stimulus rows. The first three columns are time, stimulus
            frame number, and epoch number; later columns are zero placeholders.
        """
        data = np.zeros((3, column_count), dtype=np.float64)
        data[:, 0:3] = np.array(
            [[0.0, 1.0, 1.0], [0.1, 2.0, 2.0], [0.2, 3.0, 2.0]],
        )
        return data


if __name__ == "__main__":
    unittest.main()
