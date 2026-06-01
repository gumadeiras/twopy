"""Pure response-data fixtures shared by napari plot tests.

Inputs: tiny dF/F arrays and response-window metadata.
Outputs: grouped responses and plot-ready objects without importing Qt.
"""

import numpy as np

from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.response_plotting import (
    ResponsePlotData,
    response_plot_data_from_grouped,
)
from twopy.analysis.response_processing import RoiCorrelationScores
from twopy.analysis.responses import GroupedRoiResponses, group_delta_f_over_f_by_epoch
from twopy.analysis.trials import EpochFrameWindow, FrameWindow


def tiny_response_plot_data() -> ResponsePlotData:
    """Build one tiny response plot data object for napari plot tests.

    Args:
        None.

    Returns:
        Plot-ready response data with one ROI and one epoch.
    """
    return response_plot_data_from_grouped(tiny_grouped_responses())


def two_roi_response_plot_data() -> ResponsePlotData:
    """Build tiny response plot data with two ROI rows.

    Args:
        None.

    Returns:
        Plot data whose ROI labels map to Labels values 1 and 2.
    """
    return response_plot_data_from_grouped(_two_roi_grouped_responses())


def two_roi_response_plot_data_with_correlation_scores() -> ResponsePlotData:
    """Build tiny two-ROI plot data with ROI 2 excluded by correlation QC.

    Args:
        None.

    Returns:
        Plot data carrying correlation scores for ROI visibility tests.
    """
    return response_plot_data_from_grouped(
        _two_roi_grouped_responses(),
        correlation_scores=RoiCorrelationScores(
            roi_labels=("roi_0001", "roi_0002"),
            scores=np.array([0.9, 0.1], dtype=np.float64),
            included_mask=np.array([True, False], dtype=np.bool_),
            minimum_correlation=0.5,
            reference="epoch_mean",
            window_seconds=(0.0, None),
        ),
    )


def tiny_grouped_responses() -> GroupedRoiResponses:
    """Build one tiny grouped response object for napari plot tests.

    Args:
        None.

    Returns:
        Grouped response data with one ROI and one epoch.
    """
    return group_delta_f_over_f_by_epoch(
        tiny_dff(),
        (EpochFrameWindow(FrameWindow(0, 0, 2, "odor"), 1, "Odor"),),
        data_rate_hz=1.0,
    )


def tiny_dff(
    metadata: dict[str, str | int | float | bool] | None = None,
) -> RoiDeltaFOverF:
    """Build one tiny dF/F object for napari plot tests.

    Args:
        metadata: Optional dF/F metadata.

    Returns:
        dF/F data with one ROI and two frames.
    """
    return RoiDeltaFOverF(
        fluorescence=np.ones((2, 1), dtype=np.float64),
        baseline=np.ones((2, 1), dtype=np.float64),
        values=np.array([[0.0], [1.0]], dtype=np.float64),
        labels=("roi_1",),
        start_frame=0,
        stop_frame=2,
        tau=0.0,
        amplitudes=np.ones(1, dtype=np.float64),
        baseline_frame_numbers=np.array([0.0]),
        baseline_fluorescence=np.ones((1, 1), dtype=np.float64),
        metadata={"method": "test"} if metadata is None else metadata,
    )


def _two_roi_grouped_responses() -> GroupedRoiResponses:
    """Build one tiny grouped response object with two ROIs.

    Args:
        None.

    Returns:
        Grouped response data with one epoch and two ROI labels.
    """
    dff = RoiDeltaFOverF(
        fluorescence=np.ones((2, 2), dtype=np.float64),
        baseline=np.ones((2, 2), dtype=np.float64),
        values=np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float64),
        labels=("roi_0001", "roi_0002"),
        start_frame=0,
        stop_frame=2,
        tau=0.0,
        amplitudes=np.ones(2, dtype=np.float64),
        baseline_frame_numbers=np.array([0.0]),
        baseline_fluorescence=np.ones((1, 2), dtype=np.float64),
        metadata={"method": "test"},
    )
    return group_delta_f_over_f_by_epoch(
        dff,
        (EpochFrameWindow(FrameWindow(0, 0, 2, "odor"), 1, "Odor"),),
        data_rate_hz=1.0,
    )
