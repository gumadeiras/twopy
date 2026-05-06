"""Top-level package for the twopy two-photon imaging analysis tool.

The package exposes project metadata and script-friendly public APIs.
Inputs: none.
Outputs: package constants and helper functions imported by users and scripts.
"""

from twopy.analysis import (
    AnalysisResponseRun,
    BackgroundCorrectedRoiTraces,
    BackgroundCorrectionMethod,
    DeltaFOverFFitMode,
    EpochFrameWindow,
    FrameWindow,
    GroupedRoiResponses,
    InterpolatedEpochMapping,
    LoadedAnalysisOutputs,
    RoiDeltaFOverF,
    RoiResponseSummary,
    RoiResponseTrial,
    WindowedRoiResponse,
    analyze_recording_responses,
    apply_motion_artifact_mask_to_delta_f_over_f,
    compute_roi_delta_f_over_f,
    extract_background_corrected_roi_traces,
    frame_windows_from_photodiode_alignment,
    group_delta_f_over_f_by_epoch,
    interpolate_stimulus_epochs_to_frame_windows,
    load_analysis_outputs,
    make_frame_windows,
    map_stimulus_epochs_to_frame_windows,
    save_analysis_outputs,
    select_epoch_frame_windows,
    split_traces_by_frame_windows,
    summarize_grouped_responses,
    write_response_summary_csv,
)
from twopy.api import find_recordings
from twopy.conversion import convert_recording_to_twopy, load_source_conversion_inputs
from twopy.converted import load_converted_recording
from twopy.photodiode_classification import (
    ClassifiedPhotodiodeEvent,
    ClassifiedStimulusTiming,
    ClassifiedStimulusWindow,
    PhotodiodeDurationClass,
    PhotodiodeEventType,
    classify_recording_photodiode_events,
)
from twopy.roi import (
    RoiSet,
    RoiTraces,
    extract_roi_traces,
    load_roi_set,
    make_roi_set,
    save_roi_set,
)
from twopy.spatial import SpatialCrop, SpatialDomain, full_frame_crop
from twopy.stimulus import (
    StimulusSpecificColumnMapping,
    map_stimulus_specific_column,
)
from twopy.synchronization import (
    AlignedPhotodiodeEvent,
    PhotodiodeAlignment,
    PhotodiodeEvent,
    PhotodiodeEventSet,
    detect_photodiode_events,
    detect_recording_photodiode_events,
    pair_photodiode_events_to_imaging_frames,
)

__all__ = [
    "__version__",
    "AlignedPhotodiodeEvent",
    "AnalysisResponseRun",
    "analyze_recording_responses",
    "apply_motion_artifact_mask_to_delta_f_over_f",
    "BackgroundCorrectedRoiTraces",
    "BackgroundCorrectionMethod",
    "ClassifiedPhotodiodeEvent",
    "ClassifiedStimulusTiming",
    "ClassifiedStimulusWindow",
    "classify_recording_photodiode_events",
    "compute_roi_delta_f_over_f",
    "convert_recording_to_twopy",
    "detect_photodiode_events",
    "detect_recording_photodiode_events",
    "DeltaFOverFFitMode",
    "EpochFrameWindow",
    "extract_background_corrected_roi_traces",
    "extract_roi_traces",
    "find_recordings",
    "frame_windows_from_photodiode_alignment",
    "FrameWindow",
    "full_frame_crop",
    "group_delta_f_over_f_by_epoch",
    "GroupedRoiResponses",
    "InterpolatedEpochMapping",
    "interpolate_stimulus_epochs_to_frame_windows",
    "load_analysis_outputs",
    "load_converted_recording",
    "load_roi_set",
    "load_source_conversion_inputs",
    "LoadedAnalysisOutputs",
    "make_frame_windows",
    "make_roi_set",
    "map_stimulus_epochs_to_frame_windows",
    "map_stimulus_specific_column",
    "pair_photodiode_events_to_imaging_frames",
    "PhotodiodeAlignment",
    "PhotodiodeDurationClass",
    "PhotodiodeEvent",
    "PhotodiodeEventType",
    "PhotodiodeEventSet",
    "RoiDeltaFOverF",
    "RoiResponseSummary",
    "RoiResponseTrial",
    "RoiSet",
    "RoiTraces",
    "save_analysis_outputs",
    "save_roi_set",
    "select_epoch_frame_windows",
    "split_traces_by_frame_windows",
    "SpatialCrop",
    "SpatialDomain",
    "StimulusSpecificColumnMapping",
    "summarize_grouped_responses",
    "WindowedRoiResponse",
    "write_response_summary_csv",
]

__version__ = "0.1.0"
