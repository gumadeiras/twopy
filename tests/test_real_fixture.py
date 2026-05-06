"""Regression tests for a tiny real-data converted fixture.

Inputs: committed HDF5 files derived from the example converted recording.
Outputs: stable loader and workflow assertions over real pixel values.

The fixture is spatially and temporally cropped from the example recording so
tests stay small while still exercising real movie values and metadata.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from twopy import analyze_recording_responses, load_converted_recording, load_roi_set
from twopy.analysis.persistence import load_analysis_outputs
from twopy.analysis.response_processing import (
    ResponseProcessingOptions,
    SmoothingOptions,
)
from twopy.analysis.responses import summarize_grouped_responses
from twopy.analysis.trials import EpochFrameWindow, FrameWindow

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "example_recording_small"


class RealDataFixtureTest(unittest.TestCase):
    """Tests the committed cropped real-data fixture."""

    def test_loads_small_real_converted_recording(self) -> None:
        """Confirm the real-data fixture keeps the converted HDF5 contract.

        Inputs: committed ``recording_data.h5`` and ``aligned_movie.h5``.
        Outputs: loaded metadata and deterministic real movie values.
        """
        recording = load_converted_recording(FIXTURE_ROOT / "recording_data.h5")

        self.assertEqual(recording.movie.shape, (60, 24, 24))
        self.assertEqual(recording.run_metadata["rig_name"], "OdorRig")
        self.assertEqual(recording.stimulus_data.shape, (120, 35))
        self.assertEqual(recording.frame_counts.aligned_movie_frames, 60)
        self.assertEqual(recording.frame_counts.imaging_res_pd_minus_movie, 0)
        self.assertAlmostEqual(float(recording.mean_image.mean()), 20.735185185185188)
        self.assertAlmostEqual(float(recording.movie.read_frames(0, 1).max()), 472.0)

    def test_runs_response_workflow_on_small_real_fixture(self) -> None:
        """Confirm workflow outputs stay stable on real pixel data.

        Inputs: cropped real movie fixture, three ROI masks, and explicit epoch
        windows.
        Outputs: grouped responses with known dF/F summary values.
        """
        windows = (
            EpochFrameWindow(FrameWindow(0, 0, 20, "gray_1"), 1, "Gray Interleave"),
            EpochFrameWindow(FrameWindow(1, 20, 40, "odor_1"), 2, "Odor"),
            EpochFrameWindow(FrameWindow(2, 40, 60, "gray_2"), 1, "Gray Interleave"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = analyze_recording_responses(
                FIXTURE_ROOT / "recording_data.h5",
                load_roi_set(FIXTURE_ROOT / "rois.h5"),
                output_path=root / "analysis_outputs.h5",
                response_summary_trials_csv_path=root / "response_summary_trials.csv",
                response_summary_grouped_csv_path=root / "response_summary_grouped.csv",
                epoch_windows=windows,
                background_method="none",
                seconds_interleave_use=None,
                apply_motion_mask=True,
                fit_mode="robust",
                chunk_frames=11,
            )

            self.assertEqual(run.traces.raw_values.shape, (60, 3))
            self.assertEqual(run.dff.values.shape, (60, 3))
            self.assertEqual(len(run.grouped_responses.trials), 3)
            self.assertAlmostEqual(run.dff.tau, 0.01)
            np.testing.assert_allclose(
                run.traces.raw_values[0],
                np.array([2.25, 12.53061224, 5.51785714]),
            )
            np.testing.assert_allclose(
                run.dff.values[20],
                np.array([-0.33383235, -0.7437746, -0.23345746]),
            )

            summaries = summarize_grouped_responses(run.grouped_responses)
            self.assertEqual(len(summaries), 9)
            self.assertAlmostEqual(summaries[0].mean_response, -0.18435099954739279)
            self.assertTrue(run.output_path.exists())
            self.assertTrue((root / "response_summary_trials.csv").exists())
            self.assertTrue((root / "response_summary_grouped.csv").exists())

            processed_run = analyze_recording_responses(
                FIXTURE_ROOT / "recording_data.h5",
                load_roi_set(FIXTURE_ROOT / "rois.h5"),
                output_path=root / "processed_analysis_outputs.h5",
                epoch_windows=windows,
                background_method="none",
                seconds_interleave_use=None,
                apply_motion_mask=True,
                fit_mode="robust",
                chunk_frames=11,
                response_processing_options=ResponseProcessingOptions(
                    smoothing=SmoothingOptions(
                        method="moving_average",
                        window_frames=3,
                    ),
                ),
            )
            self.assertFalse(np.allclose(processed_run.dff.values, run.dff.values))
            loaded = load_analysis_outputs(processed_run.output_path)
            self.assertIsNotNone(loaded.response_processing_options)
            if loaded.response_processing_options is not None:
                self.assertEqual(
                    loaded.response_processing_options.smoothing.method,
                    "moving_average",
                )


if __name__ == "__main__":
    unittest.main()
