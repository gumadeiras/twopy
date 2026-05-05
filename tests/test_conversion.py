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
            self.assertEqual(
                loaded.stimulus_parameters.epochs[0]["epochName"],
                "Gray Interleave",
            )
            np.testing.assert_array_equal(
                loaded.stimulus_timeline.epoch_numbers,
                np.array([1, 2, 2]),
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
                self.assertEqual(h5_file["movie/mean_image"].attrs["start_frame"], 0)
                self.assertEqual(h5_file["movie/mean_image"].attrs["stop_frame"], 3)
                self.assertEqual(h5_file["metadata"].attrs["acq.frameRate"], 10.0)
                self.assertEqual(
                    h5_file["stimulus"].attrs["clock_source"],
                    "stimulus presentation computer",
                )
                self.assertEqual(h5_file["stimulus/timeline"].shape, (3, 3))
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

    def _write_session(self, session_dir: Path) -> None:
        """Create a minimal valid source recording fixture.

        Args:
            session_dir: Temporary folder that receives fixture files.

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
                        "numberOfFrames": 3,
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
            session_dir / "imagingResPd.mat",
            {"imagingResPd": np.array([0.0, 1.0, 0.0])},
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
