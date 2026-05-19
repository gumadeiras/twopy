"""Pure napari response plot-data tests.

Inputs: tiny grouped responses and saved analysis outputs.
Outputs: plot-ready data assertions without importing Qt.
"""

import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
from tests.napari_plot_data_helpers import (
    tiny_dff,
    tiny_grouped_responses,
    two_roi_response_plot_data_with_correlation_scores,
)
from tests.tempdir import temporary_directory

from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.persistence import save_analysis_outputs
from twopy.analysis.response_processing import RoiCorrelationScores
from twopy.analysis.response_window_options import ResponseWindowOptions
from twopy.analysis.responses import group_delta_f_over_f_by_epoch
from twopy.analysis.trials import EpochFrameWindow, FrameWindow
from twopy.napari.plotting.data import (
    ResponsePlotData,
    filter_response_plot_data_rois,
    load_response_plot_data,
    response_plot_data_from_grouped,
)


class NapariPlotDataTest(unittest.TestCase):
    """Napari response plot-data tests."""

    def test_filter_response_plot_data_rois_keeps_matching_rows(self) -> None:
        """Confirm plot data ROI filtering keeps labels and trace rows aligned.

        Inputs: two-ROI plot data and one row index to keep.
        Outputs: every epoch and correlation score contains only the requested
        ROI row.
        """
        filtered = filter_response_plot_data_rois(
            two_roi_response_plot_data_with_correlation_scores(),
            (1,),
        )
        epoch = filtered.epochs[0]

        self.assertEqual(epoch.roi_labels, ("roi_0002",))
        np.testing.assert_array_equal(
            epoch.mean_values,
            np.array([[2.0, 3.0]], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            epoch.sem_values,
            np.zeros((1, 2), dtype=np.float64),
        )
        correlation_scores = _require_correlation_scores(filtered.correlation_scores)
        self.assertEqual(correlation_scores.roi_labels, ("roi_0002",))
        np.testing.assert_array_equal(
            correlation_scores.included_mask,
            np.array([False]),
        )

    def test_filter_response_plot_data_rois_remaps_initial_visibility(self) -> None:
        """Confirm row filtering remaps explicit visible ROI indices."""
        filtered = filter_response_plot_data_rois(
            replace(
                two_roi_response_plot_data_with_correlation_scores(),
                visible_roi_indices=(1,),
            ),
            (1,),
        )

        self.assertEqual(filtered.visible_roi_indices, (0,))

    def test_response_plot_data_summarizes_epoch_roi_mean_and_sem(self) -> None:
        """Confirm plotting data has one mean and SEM trace per ROI.

        Inputs: two repeated trials for one epoch and two ROIs.
        Outputs: plot-ready arrays shaped as ``(roi, frame)``.
        """
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((4, 2), dtype=np.float64),
            baseline=np.ones((4, 2), dtype=np.float64),
            values=np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 8.0],
                    [7.0, 10.0],
                ],
                dtype=np.float64,
            ),
            labels=("roi_1", "roi_2"),
            start_frame=0,
            stop_frame=4,
            tau=0.0,
            amplitudes=np.ones(2, dtype=np.float64),
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 2), dtype=np.float64),
            metadata={"method": "test"},
        )
        grouped = group_delta_f_over_f_by_epoch(
            dff,
            (
                EpochFrameWindow(FrameWindow(0, 0, 2, "odor_1"), 2, "Odor"),
                EpochFrameWindow(FrameWindow(1, 2, 4, "odor_2"), 2, "Odor"),
            ),
            data_rate_hz=2.0,
        )

        plot_data = response_plot_data_from_grouped(grouped)

        self.assertEqual(len(plot_data.epochs), 1)
        epoch = plot_data.epochs[0]
        self.assertEqual(epoch.epoch_name, "Odor")
        np.testing.assert_allclose(epoch.time_seconds, np.array([0.0, 0.5]))
        np.testing.assert_allclose(
            epoch.mean_values,
            np.array([[3.0, 5.0], [5.0, 7.0]]),
        )
        np.testing.assert_allclose(
            epoch.sem_values,
            np.array([[2.0, 2.0], [3.0, 3.0]]),
        )

    def test_response_plot_data_avoids_sem_warning_for_single_trial(self) -> None:
        """Confirm one-trial epochs do not emit NumPy SEM warnings.

        Inputs: one trial for one epoch and one ROI.
        Outputs: plot-ready data has zero SEM and no runtime warning.
        """
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((2, 1), dtype=np.float64),
            baseline=np.ones((2, 1), dtype=np.float64),
            values=np.array([[1.0], [3.0]], dtype=np.float64),
            labels=("roi_1",),
            start_frame=0,
            stop_frame=2,
            tau=0.0,
            amplitudes=np.ones(1, dtype=np.float64),
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 1), dtype=np.float64),
            metadata={"method": "test"},
        )
        grouped = group_delta_f_over_f_by_epoch(
            dff,
            (EpochFrameWindow(FrameWindow(0, 0, 2, "odor"), 2, "Odor"),),
            data_rate_hz=2.0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            plot_data = response_plot_data_from_grouped(grouped)

        np.testing.assert_allclose(
            plot_data.epochs[0].sem_values,
            np.zeros((1, 2), dtype=np.float64),
        )

    def test_response_plot_data_uses_prestimulus_time_axis(self) -> None:
        """Confirm response plots preserve negative prestimulus times.

        Inputs: dF/F grouped with one second before stimulus onset.
        Outputs: plot-ready time axis starts at ``-1`` seconds.
        """
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((4, 1), dtype=np.float64),
            baseline=np.ones((4, 1), dtype=np.float64),
            values=np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64),
            labels=("roi_1",),
            start_frame=0,
            stop_frame=4,
            tau=0.0,
            amplitudes=np.ones(1, dtype=np.float64),
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 1), dtype=np.float64),
            metadata={"method": "test"},
        )
        grouped = group_delta_f_over_f_by_epoch(
            dff,
            (EpochFrameWindow(FrameWindow(0, 2, 4, "odor"), 2, "Odor"),),
            data_rate_hz=2.0,
            pre_window_seconds=1.0,
        )

        plot_data = response_plot_data_from_grouped(grouped)

        np.testing.assert_allclose(
            plot_data.epochs[0].time_seconds,
            np.array([-1.0, -0.5, 0.0, 0.5]),
        )
        np.testing.assert_allclose(
            plot_data.epochs[0].mean_values,
            np.array([[0.0, 1.0, 2.0, 3.0]]),
        )

    def test_response_plot_data_marks_epoch_span(self) -> None:
        """Confirm plot data carries the coarse epoch interval.

        Inputs: dF/F grouped with pre/post context and matching epoch windows.
        Outputs: plot data marks only the epoch window.
        """
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((6, 1), dtype=np.float64),
            baseline=np.ones((6, 1), dtype=np.float64),
            values=np.arange(6, dtype=np.float64).reshape(6, 1),
            labels=("roi_1",),
            start_frame=0,
            stop_frame=6,
            tau=0.0,
            amplitudes=np.ones(1, dtype=np.float64),
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 1), dtype=np.float64),
            metadata={"method": "test"},
        )
        epoch_windows = (EpochFrameWindow(FrameWindow(0, 2, 4, "odor"), 2, "Odor"),)
        grouped = group_delta_f_over_f_by_epoch(
            dff,
            epoch_windows,
            data_rate_hz=2.0,
            pre_window_seconds=1.0,
            post_window_seconds=1.0,
        )

        plot_data = response_plot_data_from_grouped(
            grouped,
            epoch_windows=epoch_windows,
        )

        np.testing.assert_allclose(
            plot_data.epochs[0].time_seconds,
            np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]),
        )
        self.assertEqual(plot_data.epochs[0].epoch_time_spans, ((0.0, 1.0),))

    def test_load_response_plot_data_restores_saved_auto_response_window(self) -> None:
        """Confirm saved analysis reloads the response-window Auto choice.

        Inputs: grouped responses persisted with ``response_window_auto`` true.
        Outputs: loaded plot data restores ``ResponseWindowOptions.auto`` true.
        """
        with temporary_directory() as temp_dir:
            path = Path(temp_dir) / "analysis_outputs.h5"
            grouped = replace(
                tiny_grouped_responses(),
                response_window_auto=True,
            )
            epoch_windows = (EpochFrameWindow(FrameWindow(0, 0, 2, "odor"), 1, "Odor"),)

            save_analysis_outputs(
                path,
                epoch_windows=epoch_windows,
                grouped_responses=grouped,
            )

            result = load_response_plot_data(path)

        result = _require_response_plot_data(result)
        self.assertEqual(
            result.response_window_options,
            ResponseWindowOptions(
                auto=True,
                pre_window_seconds=0.0,
                post_window_seconds=0.0,
            ),
        )
        self.assertEqual(result.epochs[0].epoch_time_spans, ((0.0, 2.0),))

    def test_load_response_plot_data_restores_saved_baseline_epoch_selection(
        self,
    ) -> None:
        """Confirm saved analysis reloads the baseline epoch selector.

        Inputs: persisted dF/F metadata with a non-default baseline epoch.
        Outputs: loaded plot data restores the dF/F Plot-tab selector.
        """
        with temporary_directory() as temp_dir:
            path = Path(temp_dir) / "analysis_outputs.h5"
            save_analysis_outputs(
                path,
                dff=tiny_dff(
                    {
                        "method": "test",
                        "baseline_mode": "no_baseline_epoch",
                        "baseline_epoch_number": 3,
                        "baseline_epoch_name": "Manual baseline",
                        "baseline_sample_seconds": "full",
                        "fit_mode": "log_linear",
                    },
                ),
                grouped_responses=tiny_grouped_responses(),
            )

            result = load_response_plot_data(path)

        result = _require_response_plot_data(result)
        self.assertEqual(
            result.delta_f_over_f_options,
            DeltaFOverFOptions(
                baseline_mode="no_baseline_epoch",
                baseline_epoch_number=3,
                baseline_epoch_name="Manual baseline",
                baseline_sample_seconds=None,
                fit_mode="log_linear",
                apply_motion_mask=False,
            ),
        )

    def test_response_plot_data_sorts_epochs_by_number(self) -> None:
        """Confirm epoch plots and option lists use epoch-number order.

        Inputs: grouped responses whose first trial is epoch 3 before epoch 1.
        Outputs: plot data ordered as epoch 1, then epoch 3.
        """
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((4, 1), dtype=np.float64),
            baseline=np.ones((4, 1), dtype=np.float64),
            values=np.arange(4, dtype=np.float64).reshape(4, 1),
            labels=("roi_1",),
            start_frame=0,
            stop_frame=4,
            tau=0.0,
            amplitudes=np.ones(1, dtype=np.float64),
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 1), dtype=np.float64),
            metadata={"method": "test"},
        )
        grouped = group_delta_f_over_f_by_epoch(
            dff,
            (
                EpochFrameWindow(FrameWindow(0, 0, 2, "epoch_3"), 3, "Third"),
                EpochFrameWindow(FrameWindow(1, 2, 4, "epoch_1"), 1, "First"),
            ),
            data_rate_hz=1.0,
        )

        plot_data = response_plot_data_from_grouped(grouped)

        self.assertEqual(
            tuple(epoch.epoch_number for epoch in plot_data.epochs),
            (1, 3),
        )


def _require_response_plot_data(result: ResponsePlotData | str) -> ResponsePlotData:
    """Return loaded plot data when a scenario requires analysis output."""
    if not isinstance(result, ResponsePlotData):
        raise AssertionError(f"expected response plot data; got {result!r}")
    return result


def _require_correlation_scores(
    scores: RoiCorrelationScores | None,
) -> RoiCorrelationScores:
    """Return correlation scores when a scenario requires QC output."""
    if scores is None:
        raise AssertionError("correlation scores should be present")
    return scores


if __name__ == "__main__":
    unittest.main()
