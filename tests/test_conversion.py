"""Tests for converting microscope source files into twopy HDF5 files.

Inputs: a tiny synthetic two-photon session folder.
Outputs: assertions that conversion creates analysis-owned HDF5 datasets, with
the large aligned movie stored in a separate file.
"""

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from twopy.conversion import convert_recording_to_twopy, load_source_conversion_inputs


class ConversionTest(unittest.TestCase):
    """Tests the source-to-twopy conversion boundary."""

    def test_loads_source_conversion_inputs(self) -> None:
        """Confirm source files can be loaded for conversion.

        Inputs: a temporary recording with aligned movie, metadata, stimulus,
        and photodiode files.
        Outputs: assertions that each conversion input is available as Python.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
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
            self.assertEqual(loaded.acquisition.fields["acq.frameRate"], 10.0)
            self.assertEqual(loaded.acquisition.fields["acq.zoomFactor"], 2.0)
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

    def test_converts_recording_to_twopy_hdf5(self) -> None:
        """Confirm conversion writes twopy-owned datasets and mean image.

        Inputs: a temporary source recording and output directory.
        Outputs: HDF5 datasets for metadata, stimulus, photodiode, full-movie
        mean image, and a separate aligned movie file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
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
                    np.array([[4.0, 5.0], [6.0, 7.0]]),
                )
                self.assertIsNone(h5_file["movie/mean_image"].compression)
                self.assertEqual(h5_file["movie/mean_image"].attrs["start_frame"], 0)
                self.assertEqual(h5_file["movie/mean_image"].attrs["stop_frame"], 3)
                self.assertEqual(h5_file["metadata"].attrs["acq.frameRate"], 10.0)
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
                self.assertNotIn("aligned", h5_file["movie"])

            with h5py.File(converted.movie_path, "r") as h5_file:
                self.assertEqual(h5_file.attrs["twopy_format"], "aligned-movie")
                np.testing.assert_array_equal(
                    h5_file["movie/aligned"][()],
                    np.arange(12, dtype=np.float64).reshape(3, 2, 2),
                )

    def test_converts_mean_image_for_requested_frame_range(self) -> None:
        """Confirm conversion can compute a range-limited mean image.

        Inputs: a temporary source recording and a selected frame range.
        Outputs: a mean image over only that frame range.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
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
                    np.array([[6.0, 7.0], [8.0, 9.0]]),
                )
                self.assertEqual(h5_file["movie/mean_image"].attrs["start_frame"], 1)
                self.assertEqual(h5_file["movie/mean_image"].attrs["stop_frame"], 3)

    def test_conversion_uses_configured_analysis_output_by_default(self) -> None:
        """Confirm conversion defaults to ``analysis_output`` from config.

        Inputs: a temporary config using mirrored output routing and no explicit
        conversion output directory.
        Outputs: converted files under the configured mirrored output root.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            session_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            output_root = root / "analysis"
            config_path = root / "config.yml"
            self._write_session(session_dir)
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root}\n"
                "database_access: copy\n"
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

    def test_allows_observed_one_frame_acquisition_metadata_offset(self) -> None:
        """Confirm sampled ScanImage frame-count offset is audited, not hidden.

        Inputs: a temporary recording where ``acq.numberOfFrames`` is one less
        than the aligned movie and imaging-resolution photodiode.
        Outputs: loaded audit values that preserve the one-frame offset.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
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
        with tempfile.TemporaryDirectory() as temp_dir:
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
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir)
            self._write_session(session_dir, acquisition_frame_count=1)

            with self.assertRaisesRegex(ValueError, "acq.numberOfFrames"):
                load_source_conversion_inputs(session_dir)

    def _write_session(
        self,
        session_dir: Path,
        *,
        acquisition_frame_count: int = 3,
        imaging_res_pd: np.ndarray | None = None,
    ) -> None:
        """Create a minimal valid source recording fixture.

        Args:
            session_dir: Temporary folder that receives fixture files.
            acquisition_frame_count: ``acq.numberOfFrames`` metadata value.
            imaging_res_pd: Optional imaging-resolution photodiode vector.

        Returns:
            None. The function writes all required source files to disk.
        """
        stimulus_dir = session_dir / "stimulusData"
        stimulus_dir.mkdir(parents=True)

        with h5py.File(session_dir / "alignedMovie.mat", "w") as mat_file:
            mat_file.create_dataset(
                "imgFrames_ch1",
                data=np.arange(12, dtype=np.float64).reshape(3, 2, 2),
                chunks=(1, 2, 2),
                compression="gzip",
            )

        scipy.io.savemat(
            session_dir / "imageDescription.mat",
            {
                "state": {
                    "configName": "test_config",
                    "software": {"version": 3.8},
                    "acq": {
                        "linesPerFrame": 2,
                        "pixelsPerLine": 2,
                        "numberOfFrames": acquisition_frame_count,
                        "numberOfChannelsSave": 2,
                        "frameRate": 10.0,
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
                            "intensity": 0.0,
                        },
                        {
                            "epochName": "LR20",
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
                "stimData": np.array(
                    [[0.0, 1.0, 1.0], [0.1, 2.0, 2.0], [0.2, 3.0, 2.0]],
                ),
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
        scipy.io.savemat(
            session_dir / "imagingResPd.mat",
            {
                "imagingResPd": (
                    np.array([0.0, 1.0, 0.0])
                    if imaging_res_pd is None
                    else imaging_res_pd
                ),
            },
        )
        scipy.io.savemat(
            session_dir / "highResPd.mat",
            {"highResPd": np.array([0.0, 0.2, 1.0, 0.2, 0.0])},
        )

        (session_dir / "stimulus_name_changes_001.tif").touch()
        (
            session_dir / "stimulus_name_changes_001_ch1_disinterleaved_alignment.txt"
        ).touch()
        (session_dir / "defaultAlignChannel.txt").write_text("1\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
