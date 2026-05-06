"""Tests for analysis-time ROI background correction.

Inputs: small converted movies, ROI masks, and background-correction settings.
Outputs: raw, background, and corrected ROI traces with known expected values.
"""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from twopy import extract_background_corrected_roi_traces, make_roi_set
from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData


class BackgroundSubtractionTest(unittest.TestCase):
    """Tests auditable background-corrected ROI trace extraction."""

    def test_extracts_uncorrected_background_trace_outputs(self) -> None:
        """Confirm the no-background mode keeps raw traces inspectable.

        Inputs: one ROI over a tiny converted movie.
        Outputs: raw and corrected traces that match, plus zero background.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._write_recording(Path(temp_dir))
            roi_set = make_roi_set(
                np.array([[[True, False], [False, False]]]),
                labels=("single_pixel",),
            )

            traces = extract_background_corrected_roi_traces(
                recording,
                roi_set,
                method="none",
                chunk_frames=2,
            )

            self.assertEqual(traces.method, "none")
            np.testing.assert_array_equal(
                traces.raw_values,
                np.array([[0.0], [4.0], [8.0]]),
            )
            np.testing.assert_array_equal(traces.background_values, np.zeros((3, 1)))
            np.testing.assert_array_equal(traces.corrected_values, traces.raw_values)

    def test_extracts_movie_global_percentile_corrected_traces(self) -> None:
        """Confirm MATLAB-style movie background subtraction is pixel-level.

        Inputs: one ROI, one dim background pixel, and framewise background
        values.
        Outputs: raw ROI values, repeated background values, and corrected ROI
        values after pixel-level subtraction and clamping.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            movie = np.array(
                [
                    [[10.0, 4.0], [6.0, 1.0]],
                    [[12.0, 6.0], [6.0, 2.0]],
                    [[8.0, 1.0], [6.0, 3.0]],
                ],
            )
            recording = self._write_recording(Path(temp_dir), movie=movie)
            roi_set = make_roi_set(
                np.array([[[True, True], [False, False]]]),
                labels=("top",),
            )

            traces = extract_background_corrected_roi_traces(
                recording,
                roi_set,
                method="movie_global_percentile",
                global_percentile=30.0,
                chunk_frames=1,
            )

            self.assertEqual(traces.method, "movie_global_percentile")
            self.assertEqual(traces.metadata["background_pixel_count"], 1)
            np.testing.assert_array_equal(
                traces.raw_values,
                np.array([[7.0], [9.0], [4.5]]),
            )
            np.testing.assert_array_equal(
                traces.background_values,
                np.array([[1.0], [2.0], [3.0]]),
            )
            np.testing.assert_array_equal(
                traces.corrected_values,
                np.array([[6.0], [7.0], [2.5]]),
            )

    def test_movie_global_percentile_clamps_pixels_before_roi_mean(self) -> None:
        """Confirm global movie correction clamps each pixel before averaging.

        Inputs: one ROI with one pixel below the background trace.
        Outputs: corrected ROI value larger than raw-minus-background because
        the low pixel is clamped to zero before averaging.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            movie = np.array(
                [
                    [[10.0, 1.0], [4.0, 9.0]],
                    [[10.0, 20.0], [4.0, 9.0]],
                ],
            )
            recording = self._write_recording(Path(temp_dir), movie=movie)
            roi_set = make_roi_set(
                np.array([[[True, True], [False, False]]]),
                labels=("top",),
            )

            traces = extract_background_corrected_roi_traces(
                recording,
                roi_set,
                method="movie_global_percentile",
                global_percentile=30.0,
            )

            np.testing.assert_array_equal(
                traces.background_values,
                np.array([[4.0], [4.0]]),
            )
            np.testing.assert_array_equal(traces.raw_values, np.array([[5.5], [15.0]]))
            np.testing.assert_array_equal(
                traces.corrected_values,
                np.array([[3.0], [11.0]]),
            )

    def test_extracts_roi_local_y_corrected_traces(self) -> None:
        """Confirm local-y correction subtracts one background trace per ROI.

        Inputs: two ROIs at different image rows and row-local background
        pixels outside all ROIs.
        Outputs: ROI traces corrected by their own nearby background rows.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            movie = np.array(
                [
                    [
                        [10.0, 2.0],
                        [4.0, 4.0],
                        [50.0, 20.0],
                        [8.0, 8.0],
                    ],
                    [
                        [12.0, 3.0],
                        [5.0, 5.0],
                        [55.0, 25.0],
                        [9.0, 9.0],
                    ],
                ],
            )
            recording = self._write_recording(Path(temp_dir), movie=movie)
            roi_set = make_roi_set(
                np.array(
                    [
                        [
                            [True, False],
                            [False, False],
                            [False, False],
                            [False, False],
                        ],
                        [
                            [False, False],
                            [False, False],
                            [True, False],
                            [False, False],
                        ],
                    ],
                ),
                labels=("top", "lower"),
            )

            traces = extract_background_corrected_roi_traces(
                recording,
                roi_set,
                method="roi_local_percentile_y",
                local_y_radius=0,
            )

            np.testing.assert_array_equal(
                traces.raw_values,
                np.array([[10.0, 50.0], [12.0, 55.0]]),
            )
            np.testing.assert_array_equal(
                traces.background_values,
                np.array([[2.0, 20.0], [3.0, 25.0]]),
            )
            np.testing.assert_array_equal(
                traces.corrected_values,
                np.array([[8.0, 30.0], [9.0, 30.0]]),
            )

    def _write_recording(
        self,
        root: Path,
        *,
        movie: npt.NDArray[np.float64] | None = None,
    ) -> RecordingData:
        """Create a minimal loaded recording object backed by HDF5 movie data.

        Args:
            root: Temporary directory that receives the movie file.
            movie: Optional movie values shaped ``(frames, x, y)``.

        Returns:
            ``RecordingData`` object that can be passed to background correction.
        """
        movie_values = (
            np.arange(12, dtype=np.float64).reshape(3, 2, 2) if movie is None else movie
        )
        movie_path = root / "aligned_movie.h5"
        with h5py.File(movie_path, "w") as h5_file:
            h5_file.attrs["twopy_format"] = "aligned-movie"
            h5_file.create_dataset("movie/aligned", data=movie_values)

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
