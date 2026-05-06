"""Analysis helpers for converted two-photon recordings.

Inputs: converted twopy recordings, ROI traces, photodiode-aligned windows, and
analysis parameters.
Outputs: background-corrected traces and trial/frame-window response objects.

This package is GUI-independent and operates only on twopy-owned converted
files or in-memory objects derived from them.
"""

from twopy.analysis.background_subtraction import (
    BackgroundCorrectedRoiTraces,
    BackgroundCorrectionMethod,
    extract_background_corrected_roi_traces,
)
from twopy.analysis.dff import (
    DeltaFOverFFitMode,
    RoiDeltaFOverF,
    compute_roi_delta_f_over_f,
)
from twopy.analysis.epoch_mapping import (
    InterpolatedEpochMapping,
    interpolate_stimulus_epochs_to_frame_windows,
)
from twopy.analysis.motion import apply_motion_artifact_mask_to_delta_f_over_f
from twopy.analysis.persistence import (
    ANALYSIS_OUTPUT_FILE_FORMAT,
    LoadedAnalysisOutputs,
    load_analysis_outputs,
    save_analysis_outputs,
    write_response_summary_csv,
)
from twopy.analysis.responses import (
    GroupedRoiResponses,
    RoiResponseSummary,
    RoiResponseTrial,
    group_delta_f_over_f_by_epoch,
    summarize_grouped_responses,
)
from twopy.analysis.trials import (
    EpochFrameWindow,
    FrameWindow,
    WindowedRoiResponse,
    frame_windows_from_photodiode_alignment,
    make_frame_windows,
    map_stimulus_epochs_to_frame_windows,
    select_epoch_frame_windows,
    split_traces_by_frame_windows,
)
from twopy.analysis.workflow import AnalysisResponseRun, analyze_recording_responses

__all__ = [
    "analyze_recording_responses",
    "AnalysisResponseRun",
    "apply_motion_artifact_mask_to_delta_f_over_f",
    "ANALYSIS_OUTPUT_FILE_FORMAT",
    "BackgroundCorrectedRoiTraces",
    "BackgroundCorrectionMethod",
    "compute_roi_delta_f_over_f",
    "DeltaFOverFFitMode",
    "EpochFrameWindow",
    "extract_background_corrected_roi_traces",
    "frame_windows_from_photodiode_alignment",
    "FrameWindow",
    "group_delta_f_over_f_by_epoch",
    "GroupedRoiResponses",
    "InterpolatedEpochMapping",
    "interpolate_stimulus_epochs_to_frame_windows",
    "load_analysis_outputs",
    "LoadedAnalysisOutputs",
    "make_frame_windows",
    "map_stimulus_epochs_to_frame_windows",
    "RoiDeltaFOverF",
    "RoiResponseSummary",
    "RoiResponseTrial",
    "save_analysis_outputs",
    "select_epoch_frame_windows",
    "split_traces_by_frame_windows",
    "summarize_grouped_responses",
    "WindowedRoiResponse",
    "write_response_summary_csv",
]
