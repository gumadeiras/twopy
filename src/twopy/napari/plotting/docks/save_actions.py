"""Save Analysis action for response plotting docks.

Inputs: current recording, napari ROI Labels layer, Plot-tab analysis options,
and selected output paths.
Outputs: persisted ROI HDF5 plus normal twopy analysis response outputs.

This module keeps persistence side effects out of the response plot widget so
the widget only coordinates UI state and displays the result.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.response_processing import ResponseProcessingOptions
from twopy.analysis.response_window_options import ResponseWindowOptions
from twopy.analysis.workflow import analyze_recording_responses
from twopy.converted import RecordingData
from twopy.napari.display_paths import format_output_folder
from twopy.napari.plotting.data import response_plot_window_seconds_for_recording
from twopy.napari.plotting.docks.paths import (
    resolved_analysis_path,
    resolved_roi_save_file,
)
from twopy.napari.roi import (
    roi_label_image_from_layer_for_recording,
    save_napari_label_rois,
)
from twopy.napari.text import counted_noun


@dataclass(frozen=True)
class SaveAnalysisResult:
    """Result of the Save Analysis action.

    Args:
        roi_output_path: HDF5 path where ROIs were saved.
        analysis_output_path: HDF5 path where response analysis was saved.
        status_text: User-facing Update-tab status text.

    Returns:
        Immutable result values needed to refresh the dock after saving.
    """

    roi_output_path: Path
    analysis_output_path: Path
    status_text: str


def save_current_roi_analysis(
    *,
    recording: RecordingData | None,
    roi_labels_layer: object | None,
    roi_save_file: Path | None,
    analysis_path: Path | None,
    delta_f_over_f_options: DeltaFOverFOptions,
    response_window_options: ResponseWindowOptions,
    response_processing_options: ResponseProcessingOptions,
) -> SaveAnalysisResult:
    """Persist current Labels ROIs and run response analysis.

    Args:
        recording: Optional loaded converted recording.
        roi_labels_layer: Optional napari Labels layer containing ROIs.
        roi_save_file: Explicit ROI output path, if one is loaded.
        analysis_path: Explicit analysis output path, if one is loaded.
        delta_f_over_f_options: Plot-tab dF/F settings to apply.
        response_window_options: Plot-tab response-window settings to apply.
        response_processing_options: Plot-tab processing settings to apply.

    Returns:
        Saved output paths and status text for the Update tab.

    Raises:
        ValueError: If recording, ROI layer, ROI labels, or analysis settings are
            invalid for saving.
    """
    if recording is None:
        msg = "No recording loaded."
        raise ValueError(msg)
    if roi_labels_layer is None:
        msg = "No ROI Labels layer is available."
        raise ValueError(msg)

    label_image = roi_label_image_from_layer_for_recording(
        roi_labels_layer,
        recording,
    )
    if not np.any(label_image > 0):
        msg = "No ROI labels to save or analyze."
        raise ValueError(msg)

    roi_output_path = resolved_roi_save_file(roi_save_file, recording)
    analysis_output_path = resolved_analysis_path(analysis_path, recording)
    roi_set = save_napari_label_rois(label_image, roi_output_path)
    pre_window_seconds, post_window_seconds = (
        response_plot_window_seconds_for_recording(
            recording,
            response_window_options,
        )
    )
    run = analyze_recording_responses(
        recording.path,
        roi_set,
        output_path=analysis_output_path,
        interleave_epoch_number=delta_f_over_f_options.interleave_epoch_number,
        interleave_epoch_name=delta_f_over_f_options.interleave_epoch_name,
        background_method=delta_f_over_f_options.background_method,
        seconds_interleave_use=delta_f_over_f_options.seconds_interleave_use,
        fit_mode=delta_f_over_f_options.fit_mode,
        apply_motion_mask=delta_f_over_f_options.apply_motion_mask,
        response_pre_window_seconds=pre_window_seconds,
        response_post_window_seconds=post_window_seconds,
        response_processing_options=response_processing_options,
    )
    output_folder = format_output_folder(run.output_path, recording)
    status_text = (
        f"Saved {counted_noun(len(roi_set.labels), 'ROI', 'ROIs')} to {output_folder}"
    )
    return SaveAnalysisResult(
        roi_output_path=roi_output_path,
        analysis_output_path=run.output_path,
        status_text=status_text,
    )
