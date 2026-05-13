"""Response-computation helpers for the twopy napari adapter.

Inputs: a loaded converted recording and the active napari Labels ROI layer.
Outputs: plot-ready response data computed from the current labels.

This module is the small boundary between GUI ROI editing and core analysis.
It translates editable Labels pixels into a ``RoiSet`` and then calls the
GUI-independent in-memory response workflow.
"""

from pathlib import Path

import numpy as np

from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.response_processing import ResponseProcessingOptions
from twopy.analysis.response_window_options import ResponseWindowOptions
from twopy.analysis.workflow import compute_recording_responses
from twopy.converted import RecordingData
from twopy.napari.plotting.data import (
    ResponsePlotData,
    response_plot_data_from_grouped,
    response_plot_window_seconds_for_recording,
)
from twopy.napari.roi import roi_label_image_from_layer_for_recording
from twopy.roi import RoiSet, make_roi_set_from_label_image

__all__ = [
    "compute_response_plot_data_from_labels",
    "compute_response_plot_data_from_roi_set",
]


def compute_response_plot_data_from_labels(
    recording: RecordingData,
    roi_labels_layer: object,
    *,
    source_path: Path | None = None,
    delta_f_over_f_options: DeltaFOverFOptions | None = None,
    response_window_options: ResponseWindowOptions | None = None,
    response_processing_options: ResponseProcessingOptions | None = None,
) -> ResponsePlotData:
    """Compute plot-ready ROI responses from the current napari Labels layer.

    Args:
        recording: Loaded converted recording shown in napari.
        roi_labels_layer: Active napari Labels layer containing ROI pixels.
        source_path: Optional display path attached to the plot data.
        delta_f_over_f_options: Optional dF/F analysis settings.
        response_window_options: Optional response-window settings.
        response_processing_options: Optional smoothing, low-pass, and
            correlation-QC settings.

    Returns:
        Plot-ready response means and SEMs grouped by stimulus epoch.

    Raises:
        ValueError: If the layer shape is not the crop-native display shape, if
            no ROI labels are present, or if analysis inputs are invalid.

    The helper deliberately does not save analysis files. Live ROI editing can
    call it repeatedly without creating HDF5 churn; explicit save/export actions
    remain separate user decisions.
    """
    label_image = roi_label_image_from_layer_for_recording(roi_labels_layer, recording)
    if not np.any(label_image > 0):
        msg = "No ROI labels to analyze."
        raise ValueError(msg)

    roi_set = make_roi_set_from_label_image(label_image)
    return compute_response_plot_data_from_roi_set(
        recording,
        roi_set,
        source_path=source_path,
        delta_f_over_f_options=delta_f_over_f_options,
        response_window_options=response_window_options,
        response_processing_options=response_processing_options,
    )


def compute_response_plot_data_from_roi_set(
    recording: RecordingData,
    roi_set: RoiSet,
    *,
    source_path: Path | None = None,
    delta_f_over_f_options: DeltaFOverFOptions | None = None,
    response_window_options: ResponseWindowOptions | None = None,
    response_processing_options: ResponseProcessingOptions | None = None,
) -> ResponsePlotData:
    """Compute plot-ready ROI responses from a prepared ROI set.

    Args:
        recording: Loaded converted recording shown in napari.
        roi_set: ROI masks created from the current Labels layer.
        source_path: Optional display path attached to the plot data.
        delta_f_over_f_options: Optional dF/F analysis settings.
        response_window_options: Optional response-window settings.
        response_processing_options: Optional smoothing, low-pass, and
            correlation-QC settings.

    Returns:
        Plot-ready response means and SEMs grouped by stimulus epoch.

    This helper receives a prepared ROI set so live napari updates can copy
    label pixels on the GUI thread, then run the heavier movie streaming work in
    a worker thread.
    """
    dff_options = delta_f_over_f_options or DeltaFOverFOptions()
    pre_window_seconds, post_window_seconds = (
        response_plot_window_seconds_for_recording(
            recording,
            response_window_options or ResponseWindowOptions(),
        )
    )
    computation = compute_recording_responses(
        recording,
        roi_set,
        baseline_epoch_number=dff_options.baseline_epoch_number,
        baseline_epoch_name=dff_options.baseline_epoch_name,
        background_method=dff_options.background_method,
        baseline_sample_seconds=dff_options.baseline_sample_seconds,
        fit_mode=dff_options.fit_mode,
        apply_motion_mask=dff_options.apply_motion_mask,
        response_pre_window_seconds=pre_window_seconds,
        response_post_window_seconds=post_window_seconds,
        response_processing_options=response_processing_options,
    )
    return response_plot_data_from_grouped(
        computation.grouped_responses,
        source_path=source_path,
        delta_f_over_f_options=dff_options,
        response_window_options=response_window_options or ResponseWindowOptions(),
        response_processing_options=computation.response_processing_options,
        correlation_scores=computation.correlation_scores,
    )
