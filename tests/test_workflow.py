"""Tests for the script-facing response-analysis workflow.

Inputs: tiny converted recordings, ROI sets, and explicit epoch windows.
Outputs: analysis objects plus persisted HDF5 and response CSV files.
"""

import csv
import unittest
from pathlib import Path

import numpy as np

from tests.converted_files import write_converted_recording_files
from tests.tempdir import temporary_directory
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
        Outputs: dF/F, grouped responses, analysis HDF5, and response CSVs.
        """
        with temporary_directory() as temp_dir:
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
                baseline_sample_seconds=None,
                apply_motion_mask=False,
                fit_mode="direct_bounded_tau",
                chunk_frames=2,
                response_window_auto=True,
            )

            self.assertEqual(run.output_path, root / "analysis_outputs.h5")
            self.assertEqual(
                run.response_summary_trials_csv_path,
                root / "exports" / "csvs" / "response_summary_trials.csv",
            )
            self.assertEqual(
                run.response_summary_grouped_csv_path,
                root / "exports" / "csvs" / "response_summary_grouped.csv",
            )
            self.assertEqual(len(run.grouped_responses.trials), 2)
            self.assertTrue(run.grouped_responses.response_window_auto)
            self.assertEqual(run.baseline_windows, (windows[0].window,))
            self.assertEqual(run.dff.metadata["baseline_epoch_number"], 1)
            np.testing.assert_allclose(run.dff.values[:2, :], 0.0)

            loaded = load_analysis_outputs(run.output_path)
            self.assertIsNotNone(loaded.dff)
            self.assertIsNotNone(loaded.grouped_responses)
            self.assertEqual(loaded.baseline_windows, (windows[0].window,))
            if loaded.dff is not None:
                self.assertEqual(loaded.dff.metadata["baseline_epoch_number"], 1)
            if loaded.grouped_responses is not None:
                self.assertEqual(len(loaded.grouped_responses.trials), 2)
                self.assertTrue(loaded.grouped_responses.response_window_auto)

            self.assertIsNotNone(run.response_summary_trials_csv_path)
            summary_csv_path = run.response_summary_trials_csv_path
            if summary_csv_path is None:
                raise AssertionError("workflow should write a response CSV")
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
        with temporary_directory() as temp_dir:
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
                baseline_sample_seconds=None,
                apply_motion_mask=False,
                fit_mode="direct_bounded_tau",
                chunk_frames=2,
            )

            self.assertEqual(len(result.grouped_responses.trials), 2)
            self.assertFalse((root / "analysis_outputs.h5").exists())
            self.assertFalse((root / "response_summary_trials.csv").exists())
            self.assertFalse((root / "response_summary_grouped.csv").exists())
            self.assertFalse((root / "exports" / "csvs").exists())

    def test_compute_recording_responses_defaults_to_named_baseline_epoch(
        self,
    ) -> None:
        """Confirm script defaults use the shared gray/interleave baseline rule.

        Inputs: a recording whose gray-like epoch is not epoch one.
        Outputs: baseline windows and metadata select the named baseline epoch.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = self._write_converted_recording(
                root,
                stimulus_parameters_json=(
                    '[{"epochName":"Odor A"}, {"epochName":"Gray Interleave"}]'
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

            result = compute_recording_responses(
                recording,
                roi_set,
                epoch_windows=windows,
                background_method="none",
                baseline_sample_seconds=None,
                fit_mode="log_linear",
                apply_motion_mask=False,
                chunk_frames=2,
            )

            self.assertEqual(result.baseline_windows, (windows[1].window,))
            self.assertEqual(result.dff.metadata["baseline_epoch_number"], 2)

    def test_compute_recording_responses_honors_explicit_epoch_one_baseline(
        self,
    ) -> None:
        """Confirm explicit epoch-one selection overrides the shared default.

        Inputs: a recording with epoch two named as gray interleave and an
        explicit request for baseline epoch one.
        Outputs: baseline windows come from epoch one.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = self._write_converted_recording(
                root,
                stimulus_parameters_json=(
                    '[{"epochName":"Odor A"}, {"epochName":"Gray Interleave"}]'
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

            result = compute_recording_responses(
                recording,
                roi_set,
                epoch_windows=windows,
                baseline_epoch_number=1,
                background_method="none",
                baseline_sample_seconds=None,
                fit_mode="log_linear",
                apply_motion_mask=False,
                chunk_frames=2,
            )

            self.assertEqual(result.baseline_windows, (windows[0].window,))
            self.assertEqual(result.dff.metadata["baseline_epoch_number"], 1)

    def test_compute_recording_responses_accepts_explicit_baseline_windows(
        self,
    ) -> None:
        """Confirm callers can provide baseline windows directly.

        Inputs: explicit epoch windows and an explicit baseline window.
        Outputs: dF/F metadata and trace range from the normal workflow.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = self._write_converted_recording(root)
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

            result = compute_recording_responses(
                recording,
                roi_set,
                epoch_windows=windows,
                baseline_windows=(windows[1].window,),
                background_method="none",
                baseline_sample_seconds=None,
                fit_mode="log_linear",
                apply_motion_mask=False,
                chunk_frames=2,
            )

            self.assertEqual(result.baseline_windows, (windows[1].window,))
            self.assertEqual(result.traces.start_frame, 0)
            self.assertEqual(result.traces.stop_frame, 4)
            self.assertEqual(result.dff.metadata["baseline_sample_seconds"], "full")
            self.assertEqual(result.dff.metadata["fit_mode"], "log_linear")

    def test_compute_recording_responses_no_baseline_epoch_uses_span(
        self,
    ) -> None:
        """Confirm no-baseline-epoch mode fits over one continuous span.

        Inputs: explicit epoch windows with a selected first stimulus epoch.
        Outputs: one baseline window spans that epoch and all later epochs.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            recording_path = self._write_converted_recording(
                root,
                stimulus_parameters_json=(
                    '[{"epochName":"Probe"}, {"epochName":"Stim A"}, '
                    '{"epochName":"Stim B"}]'
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

            result = compute_recording_responses(
                recording,
                roi_set,
                epoch_windows=windows,
                baseline_mode="no_baseline_epoch",
                baseline_epoch_number=2,
                background_method="none",
                baseline_sample_seconds=None,
                fit_mode="log_linear",
                apply_motion_mask=False,
                chunk_frames=2,
            )

            self.assertEqual(
                result.baseline_windows,
                (FrameWindow(0, 1, 4, "no_baseline_epoch_from_epoch_0002"),),
            )
            self.assertEqual(result.dff.metadata["baseline_mode"], "no_baseline_epoch")
            self.assertEqual(result.dff.metadata["baseline_epoch_number"], 2)

    def _write_converted_recording(
        self,
        root: Path,
        *,
        stimulus_parameters_json: str | None = None,
    ) -> Path:
        """Write a minimal converted recording and movie pair.

        Args:
            root: Temporary directory that receives HDF5 files.
            stimulus_parameters_json: Optional JSON stimulus parameter list.

        Returns:
            Path to ``recording_data.h5``.
        """
        movie_values = np.full((4, 2, 2), 10.0, dtype=np.float64)
        movie_values[2:, :, :] = 20.0
        return write_converted_recording_files(
            root,
            movie_values=movie_values,
            acquisition_metadata={"acq.frameRate": 1.0},
            stimulus_parameters_json=(
                stimulus_parameters_json or '[{"epochName":"Gray"}]'
            ),
        )


if __name__ == "__main__":
    unittest.main()
