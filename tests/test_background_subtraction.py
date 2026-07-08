"""Tests for analysis-time ROI background correction.

Inputs: small converted movies, ROI masks, and background-correction settings.
Outputs: raw, background, and corrected ROI traces with known expected values.
"""

import unittest
from concurrent.futures import CancelledError
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from tests.converted_files import write_aligned_movie_file
from tests.tempdir import temporary_directory
from twopy import extract_background_corrected_roi_traces, make_roi_set
from twopy.analysis.background_subtraction import BackgroundCorrectionMethod
from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData
from twopy.roi import TraceStatistic
from twopy.spatial import SpatialCrop, full_frame_crop


class BackgroundSubtractionTest(unittest.TestCase):
    """Tests auditable background-corrected ROI trace extraction."""

    def test_extracts_uncorrected_background_trace_outputs(self) -> None:
        """Confirm the no-background mode keeps raw traces inspectable.

        Inputs: one ROI over a tiny converted movie.
        Outputs: raw and corrected crop-domain traces that match, plus zero
        background.
        """
        with temporary_directory() as temp_dir:
            movie = np.array(
                [
                    [[99.0, 99.0], [10.0, 20.0]],
                    [[99.0, 99.0], [12.0, 22.0]],
                    [[99.0, 99.0], [14.0, 24.0]],
                ],
            )
            crop = SpatialCrop(
                axis0_start=1,
                axis0_stop=2,
                axis1_start=0,
                axis1_stop=2,
                original_shape=(2, 2),
                source="alignment_valid_crop",
            )
            recording = self._write_recording(Path(temp_dir), movie=movie, crop=crop)
            roi_set = make_roi_set(
                np.array([[[False, False], [True, False]]]),
                labels=("single_pixel",),
            )

            traces = extract_background_corrected_roi_traces(
                recording,
                roi_set,
                method="none",
                chunk_frames=2,
            )

            self.assertEqual(traces.method, "none")
            self.assertEqual(traces.metadata["spatial_domain"], "alignment_valid_crop")
            self.assertEqual(traces.metadata["spatial_axis0_start"], 1)
            np.testing.assert_array_equal(
                traces.raw_values,
                np.array([[10.0], [12.0], [14.0]]),
            )
            np.testing.assert_array_equal(traces.background_values, np.zeros((3, 1)))
            np.testing.assert_array_equal(traces.corrected_values, traces.raw_values)

    def test_none_method_can_explicitly_use_full_frame(self) -> None:
        """Confirm scripts can request full-frame raw analysis traces.

        Inputs: an ROI in the invalid border and explicit ``full_frame`` domain.
        Outputs: raw traces from the full movie frame.
        """
        with temporary_directory() as temp_dir:
            movie = np.array(
                [
                    [[1.0, 2.0], [10.0, 20.0]],
                    [[3.0, 4.0], [12.0, 22.0]],
                ],
            )
            crop = SpatialCrop(
                axis0_start=1,
                axis0_stop=2,
                axis1_start=0,
                axis1_stop=2,
                original_shape=(2, 2),
                source="alignment_valid_crop",
            )
            recording = self._write_recording(Path(temp_dir), movie=movie, crop=crop)
            roi_set = make_roi_set(
                np.array([[[True, False], [False, False]]]),
                labels=("border",),
            )

            traces = extract_background_corrected_roi_traces(
                recording,
                roi_set,
                method="none",
                spatial_domain="full_frame",
            )

            self.assertEqual(traces.metadata["spatial_domain"], "full_frame")
            np.testing.assert_array_equal(traces.raw_values, np.array([[1.0], [3.0]]))

    def test_extracts_movie_global_percentile_corrected_traces(self) -> None:
        """Confirm global movie background subtraction is pixel-level.

        Inputs: one ROI, one dim background pixel, and framewise background
        values.
        Outputs: raw ROI values, repeated background values, and corrected ROI
        values after pixel-level subtraction and clamping.
        """
        with temporary_directory() as temp_dir:
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
        with temporary_directory() as temp_dir:
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

    def test_movie_global_percentile_defaults_to_alignment_valid_crop(self) -> None:
        """Confirm global background uses the saved crop by default.

        Inputs: a movie with a very dim invalid-border pixel outside the crop and
        a dim valid pixel inside the crop.
        Outputs: background values from the crop, not the full frame.
        """
        with temporary_directory() as temp_dir:
            movie = np.array(
                [
                    [
                        [0.0, 100.0, 100.0],
                        [20.0, 20.0, 20.0],
                        [30.0, 30.0, 30.0],
                        [5.0, 50.0, 50.0],
                    ],
                    [
                        [0.0, 100.0, 100.0],
                        [22.0, 22.0, 22.0],
                        [32.0, 32.0, 32.0],
                        [7.0, 50.0, 50.0],
                    ],
                ],
            )
            crop = SpatialCrop(
                axis0_start=1,
                axis0_stop=4,
                axis1_start=0,
                axis1_stop=3,
                original_shape=(4, 3),
                source="alignment_valid_crop",
            )
            recording = self._write_recording(Path(temp_dir), movie=movie, crop=crop)
            roi_set = make_roi_set(
                np.array(
                    [
                        [
                            [False, False, False],
                            [True, False, False],
                            [False, False, False],
                            [False, False, False],
                        ],
                    ],
                ),
                labels=("inside_crop",),
            )

            traces = extract_background_corrected_roi_traces(
                recording,
                roi_set,
                method="movie_global_percentile",
                global_percentile=20.0,
            )

            self.assertEqual(traces.metadata["spatial_domain"], "alignment_valid_crop")
            self.assertEqual(traces.metadata["spatial_axis0_start"], 1)
            np.testing.assert_array_equal(
                traces.background_values,
                np.array([[5.0], [7.0]]),
            )
            np.testing.assert_array_equal(
                traces.corrected_values,
                np.array([[15.0], [15.0]]),
            )

    def test_extracts_movie_y_stripe_percentile_corrected_traces(self) -> None:
        """Confirm framewise y-stripe background subtraction is per stripe.

        Inputs: two ROIs in different image rows and per-frame row backgrounds.
        Outputs: each ROI is corrected by the low-percentile value from its own
        frame stripe.
        """
        with temporary_directory() as temp_dir:
            movie = np.array(
                [
                    [
                        [10.0, 2.0],
                        [99.0, 99.0],
                        [50.0, 20.0],
                        [99.0, 99.0],
                    ],
                    [
                        [12.0, 4.0],
                        [99.0, 99.0],
                        [55.0, 25.0],
                        [99.0, 99.0],
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
                method="movie_y_stripe_percentile",
                stripe_y_height=1,
                stripe_percentile=50.0,
            )

            self.assertEqual(traces.method, "movie_y_stripe_percentile")
            self.assertEqual(traces.metadata["stripe_y_height"], 1)
            self.assertEqual(traces.metadata["stripe_count"], 4)
            np.testing.assert_array_equal(
                traces.raw_values,
                np.array([[10.0, 50.0], [12.0, 55.0]]),
            )
            np.testing.assert_array_equal(
                traces.background_values,
                np.array([[6.0, 35.0], [8.0, 40.0]]),
            )
            np.testing.assert_array_equal(
                traces.corrected_values,
                np.array([[4.0, 15.0], [4.0, 15.0]]),
            )

    def test_crop_domain_rejects_roi_pixels_outside_crop(self) -> None:
        """Confirm crop-domain analysis never drops ROI pixels silently.

        Inputs: an ROI in the invalid border outside the alignment-valid crop.
        Outputs: a validation error naming the spatial domain.
        """
        with temporary_directory() as temp_dir:
            movie = np.ones((2, 3, 3), dtype=np.float64)
            crop = SpatialCrop(
                axis0_start=1,
                axis0_stop=3,
                axis1_start=0,
                axis1_stop=3,
                original_shape=(3, 3),
                source="alignment_valid_crop",
            )
            recording = self._write_recording(Path(temp_dir), movie=movie, crop=crop)
            roi_set = make_roi_set(
                np.array(
                    [
                        [
                            [True, False, False],
                            [False, False, False],
                            [False, False, False],
                        ],
                    ],
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "outside the selected spatial domain",
            ):
                extract_background_corrected_roi_traces(
                    recording,
                    roi_set,
                    method="none",
                )

    def test_rejects_unknown_background_method(self) -> None:
        """Confirm invalid runtime method strings get public API errors.

        Inputs: an untyped-script-style background correction method.
        Outputs: a clear validation error before branch-specific work starts.
        """
        with temporary_directory() as temp_dir:
            recording = self._write_recording(Path(temp_dir))
            roi_set = make_roi_set(np.ones((1, 2, 2), dtype=bool))

            with self.assertRaisesRegex(
                ValueError,
                "Unknown background correction method",
            ):
                extract_background_corrected_roi_traces(
                    recording,
                    roi_set,
                    method=cast(BackgroundCorrectionMethod, "median_filter"),
                )

    def test_rejects_unimplemented_trace_statistic(self) -> None:
        """Confirm corrected traces cannot claim an unsupported statistic.

        Inputs: an untyped-script-style ``median`` statistic request.
        Outputs: a clear validation error before trace extraction.
        """
        with temporary_directory() as temp_dir:
            recording = self._write_recording(Path(temp_dir))
            roi_set = make_roi_set(np.ones((1, 2, 2), dtype=bool))

            with self.assertRaisesRegex(ValueError, "Unknown trace statistic"):
                extract_background_corrected_roi_traces(
                    recording,
                    roi_set,
                    statistic=cast(TraceStatistic, "median"),
                )

    def test_rejects_invalid_trace_frame_range(self) -> None:
        """Confirm background trace extraction validates frame bounds.

        Inputs: a converted movie and an empty trace frame range.
        Outputs: clear validation error before background branch work starts.
        """
        with temporary_directory() as temp_dir:
            recording = self._write_recording(Path(temp_dir))
            roi_set = make_roi_set(np.ones((1, 2, 2), dtype=bool))

            with self.assertRaisesRegex(ValueError, "trace frame range"):
                extract_background_corrected_roi_traces(
                    recording,
                    roi_set,
                    start_frame=1,
                    stop_frame=1,
                )

    def test_cancellation_callback_stops_before_trace_extraction(self) -> None:
        """Confirm interactive cancellation propagates before movie streaming.

        Inputs: a converted movie, one ROI, and a callback that raises.
        Outputs: the callback error reaches the caller unchanged.
        """
        with temporary_directory() as temp_dir:
            recording = self._write_recording(Path(temp_dir))
            roi_set = make_roi_set(np.ones((1, 2, 2), dtype=bool))

            def raise_cancelled() -> None:
                raise CancelledError()

            with self.assertRaises(CancelledError):
                extract_background_corrected_roi_traces(
                    recording,
                    roi_set,
                    check_cancelled=raise_cancelled,
                )

    def test_extracts_roi_y_stripe_corrected_traces(self) -> None:
        """Confirm ROI y-stripe correction subtracts one background trace per ROI.

        Inputs: two ROIs at different image rows and row-local background
        pixels outside all ROIs.
        Outputs: ROI traces corrected by their own nearby background rows.
        """
        with temporary_directory() as temp_dir:
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
                method="roi_y_stripe_percentile",
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

    def test_roi_y_stripe_rejects_roi_masks_without_background_pixels(self) -> None:
        """Confirm ROI y-stripe correction explains dense grid incompatibility.

        Inputs: two non-overlapping ROIs that cover every movie pixel.
        Outputs: clear guidance that ROI-local subtraction needs unlabeled
        background pixels instead of a cryptic per-ROI candidate error.
        """
        with temporary_directory() as temp_dir:
            recording = self._write_recording(Path(temp_dir))
            roi_set = make_roi_set(
                np.array(
                    [
                        [[True, True], [False, False]],
                        [[False, False], [True, True]],
                    ],
                ),
                labels=("top", "bottom"),
            )

            with self.assertRaisesRegex(
                ValueError,
                "needs unlabeled background pixels outside all ROIs",
            ):
                extract_background_corrected_roi_traces(
                    recording,
                    roi_set,
                    method="roi_y_stripe_percentile",
                )

    def _write_recording(
        self,
        root: Path,
        *,
        movie: npt.NDArray[np.float64] | None = None,
        crop: SpatialCrop | None = None,
    ) -> RecordingData:
        """Create a minimal loaded recording object backed by HDF5 movie data.

        Args:
            root: Temporary directory that receives the movie file.
            movie: Optional movie values shaped ``(frames, x, y)``.
            crop: Optional alignment-valid crop. When omitted, the crop is full
                frame.

        Returns:
            ``RecordingData`` object that can be passed to background correction.
        """
        movie_values = (
            np.arange(12, dtype=np.float64).reshape(3, 2, 2) if movie is None else movie
        )
        alignment_crop = (
            full_frame_crop(movie_values.shape[1:]) if crop is None else crop
        )
        movie_path = root / "aligned_movie.h5"
        write_aligned_movie_file(movie_path, movie_values)

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
            alignment_valid_crop=alignment_crop,
            alignment_offset_pixels=np.zeros(
                (movie_values.shape[0], 2),
                dtype=np.float64,
            ),
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
