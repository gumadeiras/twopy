"""Tests for native random-noise stimulus kernel fitting.

Inputs: synthetic stimulus streams, response traces, and converted recording
objects.
Outputs: assertions for filter orientation, epoch selection, and ipsi/contra
mapping.
"""

import json
import unittest
from pathlib import Path

import numpy as np
from tests.converted_files import write_converted_recording_files
from tests.tempdir import temporary_directory

from twopy.analysis import (
    AnalysisResponseComputation,
    BackgroundCorrectedRoiTraces,
    FrameWindow,
    GroupedRoiResponses,
    RecordingTiming,
    ResponseProcessingOptions,
    RoiDeltaFOverF,
    StimulusKernelOptions,
    fit_recording_stimulus_kernels,
    fit_stimulus_kernel,
)
from twopy.analysis.trials import EpochFrameWindow
from twopy.converted import load_converted_recording, recording_hemisphere
from twopy.roi import make_roi_set


class StimulusKernelFitTest(unittest.TestCase):
    """Tests for design-matrix kernel fitting."""

    def test_ols_recovers_known_future_to_past_kernel(self) -> None:
        """Confirm OLS returns coefficients on the documented lag axis."""
        rng = np.random.default_rng(5)
        stimulus = rng.normal(size=100).astype(np.float64)
        stimulus_times = np.arange(stimulus.size, dtype=np.float64) * 0.02
        expected = np.array([0.25, -0.5, 1.0, 0.75], dtype=np.float64)
        stimulus_indices = np.arange(2, stimulus.size - 1, dtype=np.int64)
        design = np.stack(
            [
                stimulus[stimulus_indices + 1],
                stimulus[stimulus_indices],
                stimulus[stimulus_indices - 1],
                stimulus[stimulus_indices - 2],
            ],
            axis=1,
        )
        response = design @ expected + 3.0
        response_times = stimulus_times[stimulus_indices]

        expected_fit = np.linalg.lstsq(
            design,
            response - float(np.mean(response)),
            rcond=None,
        )[0]

        kernel, time_seconds = fit_stimulus_kernel(
            stimulus_times,
            stimulus,
            response_times,
            response,
            num_stim_past=2,
            num_stim_future=1,
            method="ols",
        )

        np.testing.assert_allclose(kernel, expected_fit)
        np.testing.assert_allclose(time_seconds, [-0.02, 0.0, 0.02, 0.04])

    def test_recording_kernel_maps_right_hemisphere_to_ipsi(self) -> None:
        """Confirm recording-level fitting reads hemisphere metadata by default."""
        with temporary_directory() as temp_dir:
            recording_path = _write_olfactory_kernel_recording(temp_dir)
            computation = _kernel_computation(recording_path=recording_path)

            result = fit_recording_stimulus_kernels(
                computation,
                StimulusKernelOptions(
                    baseline_epoch_number=1,
                    discard_first_stimulus_epoch=False,
                    num_stim_past=1,
                    num_stim_future=0,
                    method="xcorr",
                ),
            )

            self.assertEqual(result.roi_labels, ("roi_0001",))
            self.assertEqual(result.hemisphere, "right")
            self.assertEqual(result.stimulus_modality, "olfaction")
            self.assertEqual(result.stimulus_column, "stimulus_specific_05")
            self.assertEqual(result.epoch_names, ("noise_a", "noise_b"))
            self.assertEqual(result.selected_epoch_numbers, (2, 3))
            self.assertEqual(result.selected_epoch_numbers_by_name, ((2,), (3,)))
            self.assertEqual(result.discarded_epoch_numbers, ())
            if (
                result.ipsilateral is None
                or result.contralateral is None
                or result.raw_right is None
                or result.raw_left is None
            ):
                self.fail("olfactory kernel result should include left/right kernels")
            np.testing.assert_array_equal(result.ipsilateral, result.raw_right)
            np.testing.assert_array_equal(result.contralateral, result.raw_left)
            self.assertEqual(result.response_sample_counts.tolist(), [[3], [3]])
            self.assertEqual(result.stimulus_sample_counts, (4, 4))

    def test_recording_kernel_limits_fit_to_selected_epochs(self) -> None:
        """Confirm explicit epoch selection limits kernel fitting."""
        with temporary_directory() as temp_dir:
            recording_path = _write_olfactory_kernel_recording(temp_dir)
            computation = _kernel_computation(recording_path=recording_path)

            result = fit_recording_stimulus_kernels(
                computation,
                StimulusKernelOptions(
                    baseline_epoch_number=1,
                    selected_epoch_numbers=(3,),
                    num_stim_past=1,
                    num_stim_future=0,
                    method="xcorr",
                ),
            )

            self.assertEqual(result.epoch_names, ("noise_b",))
            self.assertEqual(result.selected_epoch_numbers, (3,))
            self.assertEqual(result.selected_epoch_numbers_by_name, ((3,),))
            self.assertEqual(result.discarded_epoch_numbers, ())
            self.assertEqual(result.response_sample_counts.tolist(), [[3]])
            self.assertEqual(result.stimulus_sample_counts, (4,))

    def test_recording_kernel_fits_visual_contrast_without_hemisphere(self) -> None:
        """Confirm visual kernels use signed contrast without hemisphere metadata."""
        with temporary_directory() as temp_dir:
            recording_path = write_converted_recording_files(
                temp_dir,
                movie_values=np.ones((8, 2, 2), dtype=np.float64),
                stimulus_data=np.array(
                    [
                        [0.0, 1.0, 0.0],
                        [0.1, 1.0, 0.0],
                        [0.2, 2.0, -0.2],
                        [0.3, 2.0, 0.2],
                        [0.4, 2.0, -0.2],
                        [0.5, 2.0, 0.2],
                        [0.6, 3.0, -0.9],
                        [0.7, 3.0, 0.9],
                        [0.8, 3.0, -0.9],
                        [0.9, 3.0, 0.9],
                    ],
                    dtype=np.float64,
                ),
                stimulus_data_column_names=(
                    "time_seconds",
                    "epoch_number",
                    "stimulus_specific_05",
                ),
                stimulus_parameters_json=json.dumps(
                    [
                        {"epochName": "gray"},
                        {"epochName": "contrast 0.2"},
                        {"epochName": "contrast 0.9"},
                    ]
                ),
            )
            computation = _kernel_computation(recording_path=recording_path)

            result = fit_recording_stimulus_kernels(
                computation,
                StimulusKernelOptions(
                    stimulus_modality="vision",
                    baseline_epoch_number=1,
                    discard_first_stimulus_epoch=False,
                    num_stim_past=1,
                    num_stim_future=0,
                    method="xcorr",
                ),
            )

            self.assertEqual(result.stimulus_modality, "vision")
            self.assertEqual(result.stimulus_column, "stimulus_specific_05")
            self.assertIsNone(result.hemisphere)
            self.assertIsNone(result.ipsilateral)
            self.assertIsNone(result.contralateral)
            self.assertIsNone(result.raw_left)
            self.assertIsNone(result.raw_right)
            self.assertIsNotNone(result.contrast)
            if result.contrast is None:
                self.fail("visual kernel result should include contrast kernels")
            self.assertEqual(result.epoch_names, ("contrast 0.2", "contrast 0.9"))
            self.assertEqual(result.contrast.shape, (2, 1, 2))
            self.assertEqual(result.selected_epoch_numbers, (2, 3))
            self.assertEqual(result.response_sample_counts.tolist(), [[3], [3]])

    def test_recording_hemisphere_reads_converted_metadata(self) -> None:
        """Confirm the public helper exposes the recording hemisphere."""
        with temporary_directory() as temp_dir:
            recording_path = write_converted_recording_files(
                temp_dir,
                run_metadata={"hemisphere": "left"},
            )
            recording = load_converted_recording(recording_path)

            self.assertEqual(recording_hemisphere(recording), "left")

    def test_recording_hemisphere_requires_metadata(self) -> None:
        """Confirm missing hemisphere metadata fails loudly."""
        with temporary_directory() as temp_dir:
            recording_path = write_converted_recording_files(temp_dir)
            recording = load_converted_recording(recording_path)

            with self.assertRaisesRegex(ValueError, "does not include hemisphere"):
                recording_hemisphere(recording)


def _write_olfactory_kernel_recording(temp_dir: Path) -> Path:
    """Return a converted recording with two non-baseline stimulus epochs."""
    return write_converted_recording_files(
        temp_dir,
        movie_values=np.ones((8, 2, 2), dtype=np.float64),
        stimulus_data=np.array(
            [
                [0.0, 1.0, 2.0],
                [0.1, 1.0, 0.0],
                [0.2, 2.0, 2.0],
                [0.3, 2.0, 0.0],
                [0.4, 2.0, 2.0],
                [0.5, 2.0, 0.0],
                [0.6, 3.0, 0.0],
                [0.7, 3.0, 1.0],
                [0.8, 3.0, 3.0],
                [0.9, 3.0, 1.0],
            ],
            dtype=np.float64,
        ),
        stimulus_data_column_names=(
            "time_seconds",
            "epoch_number",
            "stimulus_specific_05",
        ),
        stimulus_parameters_json=json.dumps(
            [
                {"epochName": "gray"},
                {"epochName": "noise_a"},
                {"epochName": "noise_b"},
            ]
        ),
        run_metadata={"hemisphere": "right"},
    )


def _kernel_computation(recording_path: Path) -> AnalysisResponseComputation:
    """Return a minimal response computation for kernel tests."""
    recording = load_converted_recording(recording_path)
    roi_set = make_roi_set(
        np.array([[[True, False], [False, False]]], dtype=np.bool_),
        labels=("roi_0001",),
    )
    values = np.linspace(-1.0, 1.0, 8, dtype=np.float64)[:, None]
    traces = BackgroundCorrectedRoiTraces(
        raw_values=values,
        background_values=np.zeros_like(values),
        corrected_values=values,
        labels=roi_set.labels,
        start_frame=0,
        stop_frame=8,
        statistic="mean",
        method="none",
        metadata={"method": "none"},
    )
    dff = RoiDeltaFOverF(
        fluorescence=values,
        baseline=np.ones_like(values),
        values=values,
        labels=roi_set.labels,
        start_frame=0,
        stop_frame=8,
        tau=1.0,
        amplitudes=np.ones((1,), dtype=np.float64),
        baseline_frame_numbers=np.array([0.0], dtype=np.float64),
        baseline_fluorescence=np.ones((1, 1), dtype=np.float64),
        metadata={"data_rate_hz": 10.0},
    )
    epoch_windows = (
        EpochFrameWindow(FrameWindow(0, 0, 4, "epoch_2"), 2, "noise_a"),
        EpochFrameWindow(FrameWindow(1, 4, 8, "epoch_3"), 3, "noise_b"),
    )
    timing = RecordingTiming(
        frame_rate_hz=10.0,
        epoch_windows=epoch_windows,
        source="interpolated_epochs",
        metadata={"window_count": 2},
    )
    return AnalysisResponseComputation(
        recording=recording,
        roi_set=roi_set,
        traces=traces,
        dff=dff,
        timing=timing,
        epoch_windows=epoch_windows,
        baseline_windows=(),
        grouped_responses=GroupedRoiResponses(
            roi_labels=roi_set.labels,
            data_rate_hz=10.0,
            trials=(),
        ),
        response_processing_options=ResponseProcessingOptions(),
        normalization_factors=None,
        correlation_scores=None,
    )


if __name__ == "__main__":
    unittest.main()
