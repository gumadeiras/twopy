"""Tests for psycho5 comparison-only parity helpers.

Inputs: tiny converted recordings, ROI masks, and external-analysis assumptions.
Outputs: parity response computations and selector checks.
"""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from twopy.analysis.trials import EpochFrameWindow, FrameWindow
from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData, load_converted_recording
from twopy.parity import (
    compute_psycho5_default_recording_responses,
    psycho5_default_interleave_epoch_numbers,
)
from twopy.roi import make_roi_set
from twopy.spatial import full_frame_crop


class Psycho5ParityTest(unittest.TestCase):
    """Tests for parity-only psycho5 comparison helpers."""

    def test_psycho5_default_interleave_uses_next_epoch_first(self) -> None:
        """Confirm psycho5 ``nextEpoch`` metadata has highest precedence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(
                Path(temp_dir),
                stimulus_parameters=(
                    {"epochName": "Gray Interleave"},
                    {"epochName": "Odor A"},
                    {"epochName": "Odor B", "nextEpoch": 2.0},
                ),
            )

            self.assertEqual(psycho5_default_interleave_epoch_numbers(recording), (2,))

    def test_psycho5_default_interleave_reverses_epoch_one_gray_list(self) -> None:
        """Confirm multiple exact gray interleaves preserve psycho5 ordering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(
                Path(temp_dir),
                stimulus_parameters=(
                    {"epochName": "Gray Interleave"},
                    {"epochName": "Odor A"},
                    {"epochName": "Gray Interleave"},
                ),
            )

            self.assertEqual(
                psycho5_default_interleave_epoch_numbers(recording),
                (3, 1),
            )

    def test_psycho5_default_response_parity_uses_log_linear_full_interleave(
        self,
    ) -> None:
        """Confirm parity response computation uses comparison-specific defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = self._write_converted_recording(
                root,
                stimulus_parameters_json=(
                    '[{"epochName":"Odor A"}, '
                    '{"epochName":"Gray Interleave", "nextEpoch": 2}]'
                ),
            )
            recording = load_converted_recording(recording_path)
            roi_set = make_roi_set(
                np.array([[[True, False], [False, False]]]),
                labels=("roi_1",),
            )
            windows = (
                EpochFrameWindow(FrameWindow(0, 0, 2, "odor"), 1, "Odor A"),
                EpochFrameWindow(
                    FrameWindow(1, 2, 4, "gray"),
                    2,
                    "Gray Interleave",
                ),
            )

            result = compute_psycho5_default_recording_responses(
                recording,
                roi_set,
                epoch_windows=windows,
                background_method="none",
                apply_motion_mask=False,
                chunk_frames=2,
            )

            self.assertEqual(result.baseline_windows, (windows[1].window,))
            self.assertEqual(result.dff.metadata["baseline_sample_seconds"], "full")
            self.assertEqual(result.dff.metadata["fit_mode"], "log_linear")

    def test_psycho5_no_true_interleave_parity_uses_continuous_span(
        self,
    ) -> None:
        """Confirm psycho5 ``noTrueInterleave`` parity uses one baseline span."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = self._write_converted_recording(
                root,
                stimulus_parameters_json=(
                    '[{"epochName":"Probe"}, '
                    '{"epochName":"Stim A"}, '
                    '{"epochName":"Stim B", "nextEpoch": 2}]'
                ),
            )
            recording = load_converted_recording(recording_path)
            roi_set = make_roi_set(
                np.array([[[True, False], [False, False]]]),
                labels=("roi_1",),
            )
            windows = (
                EpochFrameWindow(FrameWindow(0, 0, 1, "probe"), 1, "Probe"),
                EpochFrameWindow(FrameWindow(1, 1, 2, "stim_a"), 2, "Stim A"),
                EpochFrameWindow(FrameWindow(2, 2, 4, "stim_b"), 3, "Stim B"),
            )

            result = compute_psycho5_default_recording_responses(
                recording,
                roi_set,
                epoch_windows=windows,
                no_true_interleave=True,
                background_method="none",
                apply_motion_mask=False,
                chunk_frames=2,
            )

            self.assertEqual(
                result.baseline_windows,
                (FrameWindow(0, 1, 4, "no_baseline_epoch_from_epoch_0002"),),
            )
            self.assertEqual(result.dff.metadata["baseline_mode"], "no_baseline_epoch")
            self.assertEqual(result.dff.metadata["baseline_sample_seconds"], "full")
            self.assertEqual(result.dff.metadata["fit_mode"], "log_linear")

    def _recording(
        self,
        root: Path,
        *,
        stimulus_parameters: tuple[dict[str, object], ...],
    ) -> RecordingData:
        """Create a minimal loaded recording with selected stimulus parameters."""
        return RecordingData(
            path=root / "recording_data.h5",
            movie=ConvertedMovie(
                path=root / "aligned_movie.h5",
                dataset_name="movie/aligned",
                shape=(3, 2, 2),
                dtype="float64",
            ),
            source_session_dir=root / "source",
            acquisition_metadata={},
            run_metadata={},
            synchronization_metadata={},
            stimulus_data=np.zeros((3, 2), dtype=np.float64),
            stimulus_data_column_names=(),
            stimulus_parameters=stimulus_parameters,
            stimulus_function_lookup={},
            stimulus_specific_columns={},
            imaging_res_pd=np.zeros(3, dtype=np.float64),
            high_res_pd=np.zeros(0, dtype=np.float64),
            mean_image=np.zeros((2, 2), dtype=np.float64),
            alignment_valid_crop=full_frame_crop((2, 2)),
            alignment_shift_pixels=np.zeros(3, dtype=np.float64),
            motion_artifact_mask=np.zeros(3, dtype=np.bool_),
            frame_counts=FrameCountAudit(
                aligned_movie_frames=3,
                imaging_res_pd_samples=3,
                acquisition_number_of_frames=3,
                imaging_res_pd_minus_movie=0,
                acquisition_minus_movie=0,
            ),
        )

    def _write_converted_recording(
        self,
        root: Path,
        *,
        stimulus_parameters_json: str,
    ) -> Path:
        """Write a minimal converted recording and movie pair."""
        movie_values = np.full((4, 2, 2), 10.0, dtype=np.float64)
        movie_values[2:, :, :] = 20.0
        movie_path = root / "aligned_movie.h5"
        with h5py.File(movie_path, "w") as h5_file:
            h5_file.attrs["twopy_format"] = "aligned-movie"
            h5_file.create_dataset("movie/aligned", data=movie_values)

        recording_path = root / "recording_data.h5"
        with h5py.File(recording_path, "w") as h5_file:
            h5_file.attrs["twopy_format"] = "converted-recording"
            h5_file.attrs["source_session_dir"] = str(root / "source")
            movie_group = h5_file.create_group("movie")
            movie_group.attrs["aligned_movie_file"] = "aligned_movie.h5"
            movie_group.attrs["aligned_movie_dataset"] = "movie/aligned"
            movie_group.attrs["aligned_movie_shape"] = movie_values.shape
            movie_group.attrs["aligned_movie_dtype"] = "float64"
            movie_group.create_dataset("mean_image", data=movie_values.mean(axis=0))
            crop_group = movie_group.create_group("alignment_valid_crop")
            crop_group.attrs["source"] = "full_frame"
            crop_group.attrs["axis0_start"] = 0
            crop_group.attrs["axis0_stop"] = 2
            crop_group.attrs["axis1_start"] = 0
            crop_group.attrs["axis1_stop"] = 2
            crop_group.attrs["original_shape"] = (2, 2)
            crop_group.create_dataset(
                "alignment_shift_pixels",
                data=np.zeros(4, dtype=np.float64),
            )
            crop_group.create_dataset(
                "motion_artifact_mask",
                data=np.zeros(4, dtype=np.bool_),
            )
            metadata_group = h5_file.create_group("metadata")
            metadata_group.attrs["acq.frameRate"] = 1.0
            h5_file.create_group("run")
            stimulus_group = h5_file.create_group("stimulus")
            stimulus_group.create_dataset("data", data=np.zeros((1, 2)))
            stimulus_group.create_dataset(
                "data_column_names",
                data=np.asarray(
                    ("time_seconds", "epoch_number"),
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )
            stimulus_group.create_dataset(
                "parameters_json",
                data=stimulus_parameters_json,
            )
            stimulus_group.create_dataset("function_lookup_json", data="{}")
            stimulus_group.create_dataset(
                "stimulus_specific_columns_json",
                data="{}",
            )
            photodiode_group = h5_file.create_group("photodiode")
            photodiode_group.create_dataset("imaging_res_pd", data=np.zeros(4))
            photodiode_group.create_dataset("high_res_pd", data=np.zeros(4))
            frame_counts = h5_file.create_group("frame_counts")
            frame_counts.attrs["aligned_movie_frames"] = 4
            frame_counts.attrs["imaging_res_pd_samples"] = 4
            frame_counts.attrs["acquisition_number_of_frames"] = 4
            frame_counts.attrs["imaging_res_pd_minus_movie"] = 0
            frame_counts.attrs["acquisition_minus_movie"] = 0
        return recording_path


if __name__ == "__main__":
    unittest.main()
