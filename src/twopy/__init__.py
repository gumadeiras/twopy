"""Top-level package for the twopy two-photon imaging analysis tool.

The package exposes project metadata and script-friendly public APIs.
Inputs: none.
Outputs: package constants and helper functions imported by users and scripts.
"""

# ruff: noqa: F401

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twopy._version import __version__
    from twopy.analysis import (
        DEFAULT_RESPONSE_POST_WINDOW_SECONDS,
        DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
        AnalysisResponseComputation,
        AnalysisResponseRun,
        BackgroundCorrectedRoiTraces,
        BackgroundCorrectionMethod,
        CorrelationFilterOptions,
        DeltaFOverFBaselineMode,
        DeltaFOverFFitMode,
        DeltaFOverFOptions,
        EpochFrameWindow,
        EpochResponseMap,
        FrameWindow,
        GroupedRoiResponses,
        GroupedRoiResponseSummary,
        Hemisphere,
        KernelFitMethod,
        KernelStimulusModality,
        LoadedAnalysisOutputs,
        LowPassFilterOptions,
        ManualFovGroupRow,
        ManualRoiMatchGroup,
        ManualRoiMatchRow,
        ManualRoiMatchStatus,
        RecordingKernelFit,
        RecordingTiming,
        ResponseMapData,
        ResponseMapMode,
        ResponseMapOptions,
        ResponseProcessingOptions,
        RoiCorrelationScores,
        RoiDeltaFOverF,
        RoiResponseSummary,
        RoiResponseTrial,
        SmoothingOptions,
        StimulusKernelOptions,
        TimingSource,
        WindowedRoiResponse,
        add_manual_roi_match_group,
        analyze_recording_responses,
        append_manual_roi_match_rows,
        apply_motion_artifact_mask_to_delta_f_over_f,
        compute_recording_response_maps,
        compute_recording_responses,
        compute_roi_delta_f_over_f,
        default_baseline_epoch_number,
        default_kernel_stimulus_column,
        default_recording_baseline_epoch_number,
        extract_background_corrected_roi_traces,
        finite_mean_and_sem,
        fit_recording_stimulus_kernels,
        fit_stimulus_kernel,
        group_delta_f_over_f_by_epoch,
        is_baseline_epoch_name,
        load_analysis_outputs,
        load_manual_fov_group_rows,
        load_manual_roi_match_rows,
        load_response_map_data,
        make_frame_windows,
        make_manual_fov_group_rows,
        make_manual_roi_match_rows,
        matched_manual_roi_groups,
        next_group_cell_id,
        no_baseline_epoch_frame_windows,
        remove_manual_roi_match_group,
        replace_manual_roi_match_group,
        resolve_baseline_frame_windows,
        resolve_recording_timing,
        save_analysis_outputs,
        save_manual_fov_group_rows,
        save_manual_roi_match_rows,
        save_response_map_data,
        select_baseline_frame_windows,
        split_traces_by_frame_windows,
        summarize_epoch_roi_responses,
        summarize_grouped_responses,
        validate_grouped_roi_responses,
        write_response_summary_grouped_csv,
        write_response_summary_trials_csv,
    )
    from twopy.api import find_recordings
    from twopy.conversion import (
        convert_recording_to_twopy,
        load_source_conversion_inputs,
    )
    from twopy.converted import (
        load_converted_recording,
        recording_frame_rate_hz,
        recording_hemisphere,
    )
    from twopy.photodiode_classification import (
        ClassifiedPhotodiodeEvent,
        ClassifiedStimulusTiming,
        ClassifiedStimulusWindow,
        PhotodiodeDurationClass,
        PhotodiodeEventType,
        classify_recording_photodiode_events,
    )
    from twopy.pixel_calibration import (
        DEFAULT_PIXEL_CALIBRATION_PATH,
        PixelCalibrationResolution,
        PixelCalibrationResolutionMethod,
        PixelCalibrationRow,
        load_pixel_calibrations,
        resolve_pixel_size_um,
    )
    from twopy.pixel_calibration_profiles import (
        DEFAULT_PIXEL_CALIBRATION_PROFILE_PATH,
        PixelCalibrationGroup,
        PixelCalibrationProfile,
        PixelCalibrationProfileMapping,
        PixelCalibrationProfileSource,
        load_pixel_calibration_profile_mappings,
        resolve_pixel_calibration_profile,
        select_pixel_calibration_group,
    )
    from twopy.response_roi_extraction import (
        ResponseNormalization,
        ResponsePolarity,
        ResponseStatistic,
        ResponseWatershedExtraction,
        ResponseWatershedScoreImages,
        extract_response_watershed_rois,
        response_watershed_roi_set,
    )
    from twopy.roi import (
        RoiSet,
        RoiTraces,
        extract_roi_traces,
        load_roi_set,
        make_roi_set,
        make_roi_set_from_label_image,
        roi_set_to_label_image,
        save_roi_set,
    )
    from twopy.roi_extraction import (
        RoiExtractionConfig,
        RoiExtractionMethod,
        extract_rois_from_image,
        grid_roi_set,
        grid_roi_set_microns,
        grid_size_pixels_from_microns,
        watershed_roi_set,
    )
    from twopy.spatial import SpatialCrop, SpatialDomain, full_frame_crop
    from twopy.stimulus import (
        StimulusSpecificColumnMapping,
        map_stimulus_specific_column,
        stimulus_epoch_names_by_number,
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

_LAZY_EXPORTS = {
    "__version__": ("twopy._version", "__version__"),
    "DEFAULT_RESPONSE_POST_WINDOW_SECONDS": (
        "twopy.analysis",
        "DEFAULT_RESPONSE_POST_WINDOW_SECONDS",
    ),
    "DEFAULT_RESPONSE_PRE_WINDOW_SECONDS": (
        "twopy.analysis",
        "DEFAULT_RESPONSE_PRE_WINDOW_SECONDS",
    ),
    "AnalysisResponseComputation": ("twopy.analysis", "AnalysisResponseComputation"),
    "AnalysisResponseRun": ("twopy.analysis", "AnalysisResponseRun"),
    "BackgroundCorrectedRoiTraces": ("twopy.analysis", "BackgroundCorrectedRoiTraces"),
    "BackgroundCorrectionMethod": ("twopy.analysis", "BackgroundCorrectionMethod"),
    "CorrelationFilterOptions": ("twopy.analysis", "CorrelationFilterOptions"),
    "DeltaFOverFBaselineMode": ("twopy.analysis", "DeltaFOverFBaselineMode"),
    "DeltaFOverFFitMode": ("twopy.analysis", "DeltaFOverFFitMode"),
    "DeltaFOverFOptions": ("twopy.analysis", "DeltaFOverFOptions"),
    "EpochFrameWindow": ("twopy.analysis", "EpochFrameWindow"),
    "EpochResponseMap": ("twopy.analysis", "EpochResponseMap"),
    "FrameWindow": ("twopy.analysis", "FrameWindow"),
    "GroupedRoiResponses": ("twopy.analysis", "GroupedRoiResponses"),
    "GroupedRoiResponseSummary": ("twopy.analysis", "GroupedRoiResponseSummary"),
    "Hemisphere": ("twopy.analysis", "Hemisphere"),
    "KernelFitMethod": ("twopy.analysis", "KernelFitMethod"),
    "KernelStimulusModality": ("twopy.analysis", "KernelStimulusModality"),
    "LoadedAnalysisOutputs": ("twopy.analysis", "LoadedAnalysisOutputs"),
    "LowPassFilterOptions": ("twopy.analysis", "LowPassFilterOptions"),
    "ManualFovGroupRow": ("twopy.analysis", "ManualFovGroupRow"),
    "ManualRoiMatchGroup": ("twopy.analysis", "ManualRoiMatchGroup"),
    "ManualRoiMatchRow": ("twopy.analysis", "ManualRoiMatchRow"),
    "ManualRoiMatchStatus": ("twopy.analysis", "ManualRoiMatchStatus"),
    "RecordingKernelFit": ("twopy.analysis", "RecordingKernelFit"),
    "RecordingTiming": ("twopy.analysis", "RecordingTiming"),
    "ResponseMapData": ("twopy.analysis", "ResponseMapData"),
    "ResponseMapMode": ("twopy.analysis", "ResponseMapMode"),
    "ResponseMapOptions": ("twopy.analysis", "ResponseMapOptions"),
    "ResponseProcessingOptions": ("twopy.analysis", "ResponseProcessingOptions"),
    "RoiCorrelationScores": ("twopy.analysis", "RoiCorrelationScores"),
    "RoiDeltaFOverF": ("twopy.analysis", "RoiDeltaFOverF"),
    "RoiResponseSummary": ("twopy.analysis", "RoiResponseSummary"),
    "RoiResponseTrial": ("twopy.analysis", "RoiResponseTrial"),
    "SmoothingOptions": ("twopy.analysis", "SmoothingOptions"),
    "StimulusKernelOptions": ("twopy.analysis", "StimulusKernelOptions"),
    "TimingSource": ("twopy.analysis", "TimingSource"),
    "WindowedRoiResponse": ("twopy.analysis", "WindowedRoiResponse"),
    "add_manual_roi_match_group": ("twopy.analysis", "add_manual_roi_match_group"),
    "analyze_recording_responses": ("twopy.analysis", "analyze_recording_responses"),
    "append_manual_roi_match_rows": ("twopy.analysis", "append_manual_roi_match_rows"),
    "apply_motion_artifact_mask_to_delta_f_over_f": (
        "twopy.analysis",
        "apply_motion_artifact_mask_to_delta_f_over_f",
    ),
    "compute_recording_response_maps": (
        "twopy.analysis",
        "compute_recording_response_maps",
    ),
    "compute_recording_responses": ("twopy.analysis", "compute_recording_responses"),
    "compute_roi_delta_f_over_f": ("twopy.analysis", "compute_roi_delta_f_over_f"),
    "default_baseline_epoch_number": (
        "twopy.analysis",
        "default_baseline_epoch_number",
    ),
    "default_kernel_stimulus_column": (
        "twopy.analysis",
        "default_kernel_stimulus_column",
    ),
    "default_recording_baseline_epoch_number": (
        "twopy.analysis",
        "default_recording_baseline_epoch_number",
    ),
    "extract_background_corrected_roi_traces": (
        "twopy.analysis",
        "extract_background_corrected_roi_traces",
    ),
    "finite_mean_and_sem": ("twopy.analysis", "finite_mean_and_sem"),
    "fit_recording_stimulus_kernels": (
        "twopy.analysis",
        "fit_recording_stimulus_kernels",
    ),
    "fit_stimulus_kernel": ("twopy.analysis", "fit_stimulus_kernel"),
    "group_delta_f_over_f_by_epoch": (
        "twopy.analysis",
        "group_delta_f_over_f_by_epoch",
    ),
    "is_baseline_epoch_name": ("twopy.analysis", "is_baseline_epoch_name"),
    "load_analysis_outputs": ("twopy.analysis", "load_analysis_outputs"),
    "load_manual_fov_group_rows": ("twopy.analysis", "load_manual_fov_group_rows"),
    "load_manual_roi_match_rows": ("twopy.analysis", "load_manual_roi_match_rows"),
    "load_response_map_data": ("twopy.analysis", "load_response_map_data"),
    "make_frame_windows": ("twopy.analysis", "make_frame_windows"),
    "make_manual_fov_group_rows": ("twopy.analysis", "make_manual_fov_group_rows"),
    "make_manual_roi_match_rows": ("twopy.analysis", "make_manual_roi_match_rows"),
    "matched_manual_roi_groups": ("twopy.analysis", "matched_manual_roi_groups"),
    "next_group_cell_id": ("twopy.analysis", "next_group_cell_id"),
    "no_baseline_epoch_frame_windows": (
        "twopy.analysis",
        "no_baseline_epoch_frame_windows",
    ),
    "remove_manual_roi_match_group": (
        "twopy.analysis",
        "remove_manual_roi_match_group",
    ),
    "replace_manual_roi_match_group": (
        "twopy.analysis",
        "replace_manual_roi_match_group",
    ),
    "resolve_baseline_frame_windows": (
        "twopy.analysis",
        "resolve_baseline_frame_windows",
    ),
    "resolve_recording_timing": ("twopy.analysis", "resolve_recording_timing"),
    "save_analysis_outputs": ("twopy.analysis", "save_analysis_outputs"),
    "save_manual_fov_group_rows": ("twopy.analysis", "save_manual_fov_group_rows"),
    "save_manual_roi_match_rows": ("twopy.analysis", "save_manual_roi_match_rows"),
    "save_response_map_data": ("twopy.analysis", "save_response_map_data"),
    "select_baseline_frame_windows": (
        "twopy.analysis",
        "select_baseline_frame_windows",
    ),
    "split_traces_by_frame_windows": (
        "twopy.analysis",
        "split_traces_by_frame_windows",
    ),
    "summarize_epoch_roi_responses": (
        "twopy.analysis",
        "summarize_epoch_roi_responses",
    ),
    "summarize_grouped_responses": ("twopy.analysis", "summarize_grouped_responses"),
    "validate_grouped_roi_responses": (
        "twopy.analysis",
        "validate_grouped_roi_responses",
    ),
    "write_response_summary_grouped_csv": (
        "twopy.analysis",
        "write_response_summary_grouped_csv",
    ),
    "write_response_summary_trials_csv": (
        "twopy.analysis",
        "write_response_summary_trials_csv",
    ),
    "find_recordings": ("twopy.api", "find_recordings"),
    "convert_recording_to_twopy": ("twopy.conversion", "convert_recording_to_twopy"),
    "load_source_conversion_inputs": (
        "twopy.conversion",
        "load_source_conversion_inputs",
    ),
    "load_converted_recording": ("twopy.converted", "load_converted_recording"),
    "recording_frame_rate_hz": ("twopy.converted", "recording_frame_rate_hz"),
    "recording_hemisphere": ("twopy.converted", "recording_hemisphere"),
    "ClassifiedPhotodiodeEvent": (
        "twopy.photodiode_classification",
        "ClassifiedPhotodiodeEvent",
    ),
    "ClassifiedStimulusTiming": (
        "twopy.photodiode_classification",
        "ClassifiedStimulusTiming",
    ),
    "ClassifiedStimulusWindow": (
        "twopy.photodiode_classification",
        "ClassifiedStimulusWindow",
    ),
    "PhotodiodeDurationClass": (
        "twopy.photodiode_classification",
        "PhotodiodeDurationClass",
    ),
    "PhotodiodeEventType": ("twopy.photodiode_classification", "PhotodiodeEventType"),
    "classify_recording_photodiode_events": (
        "twopy.photodiode_classification",
        "classify_recording_photodiode_events",
    ),
    "DEFAULT_PIXEL_CALIBRATION_PATH": (
        "twopy.pixel_calibration",
        "DEFAULT_PIXEL_CALIBRATION_PATH",
    ),
    "PixelCalibrationResolution": (
        "twopy.pixel_calibration",
        "PixelCalibrationResolution",
    ),
    "PixelCalibrationResolutionMethod": (
        "twopy.pixel_calibration",
        "PixelCalibrationResolutionMethod",
    ),
    "PixelCalibrationRow": ("twopy.pixel_calibration", "PixelCalibrationRow"),
    "load_pixel_calibrations": ("twopy.pixel_calibration", "load_pixel_calibrations"),
    "resolve_pixel_size_um": ("twopy.pixel_calibration", "resolve_pixel_size_um"),
    "DEFAULT_PIXEL_CALIBRATION_PROFILE_PATH": (
        "twopy.pixel_calibration_profiles",
        "DEFAULT_PIXEL_CALIBRATION_PROFILE_PATH",
    ),
    "PixelCalibrationGroup": (
        "twopy.pixel_calibration_profiles",
        "PixelCalibrationGroup",
    ),
    "PixelCalibrationProfile": (
        "twopy.pixel_calibration_profiles",
        "PixelCalibrationProfile",
    ),
    "PixelCalibrationProfileMapping": (
        "twopy.pixel_calibration_profiles",
        "PixelCalibrationProfileMapping",
    ),
    "PixelCalibrationProfileSource": (
        "twopy.pixel_calibration_profiles",
        "PixelCalibrationProfileSource",
    ),
    "load_pixel_calibration_profile_mappings": (
        "twopy.pixel_calibration_profiles",
        "load_pixel_calibration_profile_mappings",
    ),
    "resolve_pixel_calibration_profile": (
        "twopy.pixel_calibration_profiles",
        "resolve_pixel_calibration_profile",
    ),
    "select_pixel_calibration_group": (
        "twopy.pixel_calibration_profiles",
        "select_pixel_calibration_group",
    ),
    "ResponseNormalization": ("twopy.response_roi_extraction", "ResponseNormalization"),
    "ResponsePolarity": ("twopy.response_roi_extraction", "ResponsePolarity"),
    "ResponseStatistic": ("twopy.response_roi_extraction", "ResponseStatistic"),
    "ResponseWatershedExtraction": (
        "twopy.response_roi_extraction",
        "ResponseWatershedExtraction",
    ),
    "ResponseWatershedScoreImages": (
        "twopy.response_roi_extraction",
        "ResponseWatershedScoreImages",
    ),
    "extract_response_watershed_rois": (
        "twopy.response_roi_extraction",
        "extract_response_watershed_rois",
    ),
    "response_watershed_roi_set": (
        "twopy.response_roi_extraction",
        "response_watershed_roi_set",
    ),
    "RoiSet": ("twopy.roi", "RoiSet"),
    "RoiTraces": ("twopy.roi", "RoiTraces"),
    "extract_roi_traces": ("twopy.roi", "extract_roi_traces"),
    "load_roi_set": ("twopy.roi", "load_roi_set"),
    "make_roi_set": ("twopy.roi", "make_roi_set"),
    "make_roi_set_from_label_image": ("twopy.roi", "make_roi_set_from_label_image"),
    "roi_set_to_label_image": ("twopy.roi", "roi_set_to_label_image"),
    "save_roi_set": ("twopy.roi", "save_roi_set"),
    "RoiExtractionConfig": ("twopy.roi_extraction", "RoiExtractionConfig"),
    "RoiExtractionMethod": ("twopy.roi_extraction", "RoiExtractionMethod"),
    "extract_rois_from_image": ("twopy.roi_extraction", "extract_rois_from_image"),
    "grid_roi_set": ("twopy.roi_extraction", "grid_roi_set"),
    "grid_roi_set_microns": ("twopy.roi_extraction", "grid_roi_set_microns"),
    "grid_size_pixels_from_microns": (
        "twopy.roi_extraction",
        "grid_size_pixels_from_microns",
    ),
    "watershed_roi_set": ("twopy.roi_extraction", "watershed_roi_set"),
    "SpatialCrop": ("twopy.spatial", "SpatialCrop"),
    "SpatialDomain": ("twopy.spatial", "SpatialDomain"),
    "full_frame_crop": ("twopy.spatial", "full_frame_crop"),
    "StimulusSpecificColumnMapping": (
        "twopy.stimulus",
        "StimulusSpecificColumnMapping",
    ),
    "map_stimulus_specific_column": ("twopy.stimulus", "map_stimulus_specific_column"),
    "stimulus_epoch_names_by_number": (
        "twopy.stimulus",
        "stimulus_epoch_names_by_number",
    ),
    "AlignedPhotodiodeEvent": ("twopy.synchronization", "AlignedPhotodiodeEvent"),
    "PhotodiodeAlignment": ("twopy.synchronization", "PhotodiodeAlignment"),
    "PhotodiodeEvent": ("twopy.synchronization", "PhotodiodeEvent"),
    "PhotodiodeEventSet": ("twopy.synchronization", "PhotodiodeEventSet"),
    "detect_photodiode_events": ("twopy.synchronization", "detect_photodiode_events"),
    "detect_recording_photodiode_events": (
        "twopy.synchronization",
        "detect_recording_photodiode_events",
    ),
    "pair_photodiode_events_to_imaging_frames": (
        "twopy.synchronization",
        "pair_photodiode_events_to_imaging_frames",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> object:
    """Load public API names only when callers ask for them.

    Args:
        name: Public attribute requested from ``twopy``.

    Returns:
        The exported object from its owning module.

    The napari launcher imports this package before it can import
    ``twopy.napari``. Delaying the scientific API imports keeps application
    startup from paying for SciPy and HDF5 modules before the GUI needs them.
    """
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from error
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module attributes plus lazy public API names.

    Args:
        None.

    Returns:
        Sorted module names for notebooks, shells, and documentation tools.
    """
    return sorted(set(globals()) | set(__all__))
