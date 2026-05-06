"""Tests for the script-facing response-analysis workflow.

Inputs: tiny converted recordings, ROI sets, and explicit epoch windows.
Outputs: analysis objects plus persisted HDF5 and CSV files.
"""

import csv
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from twopy import (
    analyze_recording_responses,
    compute_recording_responses,
    load_analysis_outputs,
    load_converted_recording,
    make_roi_set,
    save_roi_set,
)
from twopy.analysis.trials import EpochFrameWindow, FrameWindow


class WorkflowTest(unittest.TestCase):
    """Tests end-to-end script workflow behavior."""

    def test_runs_response_analysis_and_reloads_outputs(self) -> None:
        """Confirm one call computes, groups, saves, and reloads outputs.

        Inputs: a converted recording, ROI HDF5 path, and explicit epoch
        windows.
        Outputs: dF/F, grouped responses, analysis HDF5, and CSV summary.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = self._write_converted_recording(root)
            roi_path = root / "rois.h5"
            save_roi_set(
                make_roi_set(
                    np.array(
                        [
                            [[True, False], [False, False]],
                            [[False, False], [False, True]],
                        ],
                    ),
                    labels=("roi_1", "roi_2"),
                ),
                roi_path,
            )
            windows = (
                EpochFrameWindow(FrameWindow(0, 0, 2, "gray"), 1, "Gray"),
                EpochFrameWindow(FrameWindow(1, 2, 4, "odor"), 2, "Odor"),
            )

            run = analyze_recording_responses(
                recording_path,
                roi_path,
                epoch_windows=windows,
                background_method="none",
                seconds_interleave_use=None,
                apply_motion_mask=False,
                fit_mode="robust",
                chunk_frames=2,
            )

            self.assertEqual(run.output_path, root / "analysis_outputs.h5")
            self.assertEqual(
                run.response_summary_csv_path,
                root / "response_summary.csv",
            )
            self.assertEqual(len(run.grouped_responses.trials), 2)
            self.assertEqual(run.interleave_windows, (windows[0].window,))
            np.testing.assert_allclose(run.dff.values[:2, :], 0.0)

            loaded = load_analysis_outputs(run.output_path)
            self.assertIsNotNone(loaded.dff)
            self.assertIsNotNone(loaded.grouped_responses)
            if loaded.grouped_responses is not None:
                self.assertEqual(len(loaded.grouped_responses.trials), 2)

            self.assertIsNotNone(run.response_summary_csv_path)
            summary_csv_path = run.response_summary_csv_path
            if summary_csv_path is None:
                raise AssertionError("workflow should write a response summary CSV")
            with summary_csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 4)

    def test_compute_recording_responses_does_not_write_outputs(self) -> None:
        """Confirm in-memory response computation has no persistence side effect.

        Inputs: a converted recording object, ROI set, and explicit epoch
        windows.
        Outputs: response objects in memory and no analysis files on disk.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = self._write_converted_recording(root)
            recording = load_converted_recording(recording_path)
            roi_set = make_roi_set(
                np.array([[[True, False], [False, False]]]),
                labels=("roi_1",),
            )
            windows = (
                EpochFrameWindow(FrameWindow(0, 0, 2, "gray"), 1, "Gray"),
                EpochFrameWindow(FrameWindow(1, 2, 4, "odor"), 2, "Odor"),
            )

            result = compute_recording_responses(
                recording,
                roi_set,
                epoch_windows=windows,
                background_method="none",
                seconds_interleave_use=None,
                apply_motion_mask=False,
                fit_mode="robust",
                chunk_frames=2,
            )

            self.assertEqual(len(result.grouped_responses.trials), 2)
            self.assertFalse((root / "analysis_outputs.h5").exists())
            self.assertFalse((root / "response_summary.csv").exists())

    def _write_converted_recording(self, root: Path) -> Path:
        """Write a minimal converted recording and movie pair.

        Args:
            root: Temporary directory that receives HDF5 files.

        Returns:
            Path to ``recording_data.h5``.
        """
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
            stimulus_group.create_dataset(
                "data",
                data=np.zeros((1, 2), dtype=np.float64),
            )
            stimulus_group.create_dataset(
                "data_column_names",
                data=np.asarray(
                    ("time_seconds", "epoch_number"),
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )
            stimulus_group.create_dataset(
                "parameters_json",
                data='[{"epochName":"Gray"}]',
            )
            stimulus_group.create_dataset("function_lookup_json", data="{}")
            stimulus_group.create_dataset(
                "stimulus_specific_columns_json",
                data="{}",
            )
            photodiode_group = h5_file.create_group("photodiode")
            photodiode_group.create_dataset(
                "imaging_res_pd",
                data=np.zeros(4, dtype=np.float64),
            )
            photodiode_group.create_dataset(
                "high_res_pd",
                data=np.zeros(4, dtype=np.float64),
            )
            frame_counts = h5_file.create_group("frame_counts")
            frame_counts.attrs["aligned_movie_frames"] = 4
            frame_counts.attrs["imaging_res_pd_samples"] = 4
            frame_counts.attrs["acquisition_number_of_frames"] = 4
            frame_counts.attrs["imaging_res_pd_minus_movie"] = 0
            frame_counts.attrs["acquisition_minus_movie"] = 0
        return recording_path


if __name__ == "__main__":
    unittest.main()
