"""Script-facing workflow for ROI response analysis.

Inputs: a converted recording, ROI masks, and analysis parameters.
Outputs: computed traces, dF/F, grouped responses, and saved analysis files.

This module wires together existing twopy analysis helpers. It does not read
source microscope files; callers must convert recordings before using it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
from twopy.analysis.persistence import save_analysis_outputs
from twopy.analysis.response_processing import (
    ResponseProcessingOptions,
    RoiCorrelationScores,
    apply_correlation_filter_to_grouped_roi_responses,
    process_roi_delta_f_over_f,
    validate_response_processing_options,
)
from twopy.analysis.response_window_options import (
    DEFAULT_RESPONSE_POST_WINDOW_SECONDS,
    DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
)
from twopy.analysis.responses import (
    GroupedRoiResponses,
    group_delta_f_over_f_by_epoch,
)
from twopy.analysis.trials import (
    EpochFrameWindow,
    FrameWindow,
    select_epoch_frame_windows,
)
from twopy.converted import (
    RecordingData,
    load_converted_recording,
    recording_frame_rate_hz,
)
from twopy.roi import RoiSet, load_roi_set
from twopy.spatial import SpatialDomain

__all__ = [
    "AnalysisResponseComputation",
    "AnalysisResponseRun",
    "DEFAULT_RESPONSE_POST_WINDOW_SECONDS",
    "DEFAULT_RESPONSE_PRE_WINDOW_SECONDS",
    "analyze_recording_responses",
    "compute_recording_responses",
]


@dataclass(frozen=True)
class AnalysisResponseComputation:
    """In-memory outputs from one response-analysis calculation.

    Inputs: one converted recording and one ROI set.
    Outputs: computed traces, dF/F, stimulus windows, and grouped responses.

    This object lets interactive tools update plots without writing analysis
    HDF5 files on every ROI edit. Persistence is a separate explicit step.
    """

    recording: RecordingData
    roi_set: RoiSet
    traces: BackgroundCorrectedRoiTraces
    dff: RoiDeltaFOverF
    epoch_mapping: InterpolatedEpochMapping | None
    epoch_windows: tuple[EpochFrameWindow, ...]
    interleave_windows: tuple[FrameWindow, ...]
    grouped_responses: GroupedRoiResponses
    response_processing_options: ResponseProcessingOptions
    correlation_scores: RoiCorrelationScores | None


@dataclass(frozen=True)
class AnalysisResponseRun(AnalysisResponseComputation):
    """Outputs from one script-facing response-analysis run.

    Inputs: one converted recording and one ROI set.
    Outputs: computed analysis objects plus file paths written by the run.

    Returning the objects lets scripts inspect results immediately while the
    HDF5/CSV files keep the same run auditable and reloadable later.
    """

    output_path: Path
    response_summary_trials_csv_path: Path | None
    response_summary_grouped_csv_path: Path | None


def analyze_recording_responses(
    recording_data_path: Path,
    roi_set: RoiSet | Path,
    *,
    output_path: Path | None = None,
    response_summary_trials_csv_path: Path | None = None,
    response_summary_grouped_csv_path: Path | None = None,
    write_summary_csv: bool = True,
    epoch_windows: Sequence[EpochFrameWindow] | None = None,
    interleave_epoch_number: int | None = 1,
    interleave_epoch_name: str | None = None,
    background_method: BackgroundCorrectionMethod = "movie_global_percentile",
    seconds_interleave_use: float | None = 1.0,
    fit_mode: DeltaFOverFFitMode = "robust",
    apply_motion_mask: bool = True,
    data_rate_hz: float | None = None,
    chunk_frames: int = 128,
    spatial_domain: SpatialDomain = "alignment_valid_crop",
    response_pre_window_seconds: float = DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
    response_post_window_seconds: float = 0.0,
    response_processing_options: ResponseProcessingOptions | None = None,
) -> AnalysisResponseRun:
    """Run the standard ROI response analysis workflow.

    Args:
        recording_data_path: Path to converted ``recording_data.h5``.
        roi_set: ``RoiSet`` object or path to a saved ROI HDF5 file.
        output_path: Optional destination HDF5 path. When omitted, output is
            written beside ``recording_data.h5`` as ``analysis_outputs.h5``.
        response_summary_trials_csv_path: Optional trial-level CSV path. When
            omitted and ``write_summary_csv`` is true, output is written under
            ``exports/csvs/response_summary_trials.csv`` beside the HDF5 file.
        response_summary_grouped_csv_path: Optional grouped CSV path. When
            omitted and ``write_summary_csv`` is true, output is written under
            ``exports/csvs/response_summary_grouped.csv`` beside the HDF5 file.
        write_summary_csv: Whether to write response time-series CSV files.
        epoch_windows: Optional explicit epoch windows. When omitted, twopy
            interpolates epochs from converted stimulus data.
        interleave_epoch_number: Optional baseline epoch number selector.
        interleave_epoch_name: Optional baseline epoch name selector.
        background_method: Background correction method before dF/F.
        seconds_interleave_use: Seconds from the end of each interleave window.
        fit_mode: dF/F exponential fit mode.
        apply_motion_mask: Whether to mark converted high-motion frames as NaN
            after dF/F.
        data_rate_hz: Optional frame rate. When omitted, twopy reads
            ``acq.frameRate`` from converted metadata.
        chunk_frames: Number of movie frames to read per HDF5 chunk.
        spatial_domain: Spatial domain used for trace extraction.
        response_pre_window_seconds: Seconds before each stimulus window to
            include in grouped responses. The default shows the transition from
            gray interleave baseline into the stimulus.
        response_post_window_seconds: Seconds after each stimulus window to
            include in grouped responses. The default is zero so persisted
            summaries keep their epoch-window meaning; plotting code can add
            visual context separately from saved dF/F and epoch windows.
        response_processing_options: Optional smoothing, low-pass, and
            correlation-QC settings to apply after dF/F. Signal filters are
            applied to continuous dF/F before grouping; correlation QC is
            applied to grouped responses.

    Returns:
        ``AnalysisResponseRun`` with computed objects and output paths.

    Raises:
        ValueError: If no interleave windows match the selector.

    This is the main script API for a converted recording: load, extract traces,
    compute dF/F, group responses, and persist the results.
    """
    recording = load_converted_recording(recording_data_path)
    loaded_roi_set = _resolve_roi_set(roi_set)
    computation = compute_recording_responses(
        recording,
        loaded_roi_set,
        epoch_windows=epoch_windows,
        interleave_epoch_number=interleave_epoch_number,
        interleave_epoch_name=interleave_epoch_name,
        background_method=background_method,
        seconds_interleave_use=seconds_interleave_use,
        fit_mode=fit_mode,
        apply_motion_mask=apply_motion_mask,
        data_rate_hz=data_rate_hz,
        chunk_frames=chunk_frames,
        spatial_domain=spatial_domain,
        response_pre_window_seconds=response_pre_window_seconds,
        response_post_window_seconds=response_post_window_seconds,
        response_processing_options=response_processing_options,
    )
    resolved_output_path = _resolve_output_path(recording.path, output_path)
    resolved_trials_summary_path = _resolve_summary_csv_path(
        output_path=resolved_output_path,
        requested_path=response_summary_trials_csv_path,
        write_summary_csv=write_summary_csv,
        filename="response_summary_trials.csv",
    )
    resolved_grouped_summary_path = _resolve_summary_csv_path(
        output_path=resolved_output_path,
        requested_path=response_summary_grouped_csv_path,
        write_summary_csv=write_summary_csv,
        filename="response_summary_grouped.csv",
    )
    save_analysis_outputs(
        resolved_output_path,
        roi_set=computation.roi_set,
        traces=computation.traces,
        dff=computation.dff,
        epoch_windows=computation.epoch_windows,
        grouped_responses=computation.grouped_responses,
        response_processing_options=computation.response_processing_options,
        correlation_scores=computation.correlation_scores,
        response_summary_trials_csv=resolved_trials_summary_path,
        response_summary_grouped_csv=resolved_grouped_summary_path,
    )

    return AnalysisResponseRun(
        recording=computation.recording,
        roi_set=computation.roi_set,
        traces=computation.traces,
        dff=computation.dff,
        epoch_mapping=computation.epoch_mapping,
        epoch_windows=computation.epoch_windows,
        interleave_windows=computation.interleave_windows,
        grouped_responses=computation.grouped_responses,
        response_processing_options=computation.response_processing_options,
        correlation_scores=computation.correlation_scores,
        output_path=resolved_output_path,
        response_summary_trials_csv_path=resolved_trials_summary_path,
        response_summary_grouped_csv_path=resolved_grouped_summary_path,
    )


def compute_recording_responses(
    recording: RecordingData,
    roi_set: RoiSet,
    *,
    epoch_windows: Sequence[EpochFrameWindow] | None = None,
    interleave_epoch_number: int | None = 1,
    interleave_epoch_name: str | None = None,
    background_method: BackgroundCorrectionMethod = "movie_global_percentile",
    seconds_interleave_use: float | None = 1.0,
    fit_mode: DeltaFOverFFitMode = "robust",
    apply_motion_mask: bool = True,
    data_rate_hz: float | None = None,
    chunk_frames: int = 128,
    spatial_domain: SpatialDomain = "alignment_valid_crop",
    response_pre_window_seconds: float = DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
    response_post_window_seconds: float = 0.0,
    response_processing_options: ResponseProcessingOptions | None = None,
) -> AnalysisResponseComputation:
    """Compute ROI responses from converted data without writing files.

    Args:
        recording: Loaded converted recording.
        roi_set: ROI masks to extract from the recording.
        epoch_windows: Optional explicit epoch windows. When omitted, twopy
            interpolates epochs from converted stimulus data.
        interleave_epoch_number: Optional baseline epoch number selector.
        interleave_epoch_name: Optional baseline epoch name selector.
        background_method: Background correction method before dF/F.
        seconds_interleave_use: Seconds from the end of each interleave window.
        fit_mode: dF/F exponential fit mode.
        apply_motion_mask: Whether to mark converted high-motion frames as NaN
            after dF/F.
        data_rate_hz: Optional frame rate. When omitted, twopy reads
            ``acq.frameRate`` from converted metadata.
        chunk_frames: Number of movie frames to read per HDF5 chunk.
        spatial_domain: Spatial domain used for trace extraction.
        response_pre_window_seconds: Seconds before each stimulus window to
            include in grouped responses.
        response_post_window_seconds: Seconds after each stimulus window to
            include in grouped responses.
        response_processing_options: Optional smoothing, low-pass, and
            correlation-QC settings to apply after dF/F.

    Returns:
        In-memory response-analysis objects.

    Raises:
        ValueError: If no interleave windows match the selector.

    This is the shared calculation used by scripts, tests, and the napari live
    plotter. It streams movie frames in chunks and leaves file writing to a
    separate explicit call.
    """
    frame_rate_hz = (
        recording_frame_rate_hz(recording)
        if data_rate_hz is None
        else float(data_rate_hz)
    )
    processing_options = _resolve_response_processing_options(
        response_processing_options,
        data_rate_hz=frame_rate_hz,
    )
    mapping, resolved_epoch_windows = _resolve_epoch_windows(recording, epoch_windows)
    trace_start, trace_stop = _frame_range_for_epoch_windows(
        resolved_epoch_windows,
        frame_count=recording.movie.shape[0],
        data_rate_hz=frame_rate_hz,
        pre_window_seconds=response_pre_window_seconds,
        post_window_seconds=response_post_window_seconds,
    )
    interleave_windows = select_epoch_frame_windows(
        resolved_epoch_windows,
        epoch_name=interleave_epoch_name,
        epoch_number=interleave_epoch_number,
    )
    if len(interleave_windows) == 0:
        msg = "No interleave windows matched the requested selector"
        raise ValueError(msg)

    traces = extract_background_corrected_roi_traces(
        recording,
        roi_set,
        method=background_method,
        start_frame=trace_start,
        stop_frame=trace_stop,
        chunk_frames=chunk_frames,
        spatial_domain=spatial_domain,
    )
    dff = compute_roi_delta_f_over_f(
        traces,
        interleave_windows,
        data_rate_hz=frame_rate_hz,
        seconds_interleave_use=seconds_interleave_use,
        fit_mode=fit_mode,
    )
    if apply_motion_mask:
        dff = apply_motion_artifact_mask_to_delta_f_over_f(dff, recording)
    if _has_continuous_response_processing(processing_options):
        dff = process_roi_delta_f_over_f(
            dff,
            options=processing_options,
            data_rate_hz=frame_rate_hz,
        )

    grouped_responses = group_delta_f_over_f_by_epoch(
        dff,
        resolved_epoch_windows,
        data_rate_hz=frame_rate_hz,
        pre_window_seconds=response_pre_window_seconds,
        post_window_seconds=response_post_window_seconds,
    )
    grouped_responses, correlation_scores = (
        apply_correlation_filter_to_grouped_roi_responses(
            grouped_responses,
            options=processing_options,
        )
    )

    return AnalysisResponseComputation(
        recording=recording,
        roi_set=roi_set,
        traces=traces,
        dff=dff,
        epoch_mapping=mapping,
        epoch_windows=resolved_epoch_windows,
        interleave_windows=interleave_windows,
        grouped_responses=grouped_responses,
        response_processing_options=processing_options,
        correlation_scores=correlation_scores,
    )


def _resolve_response_processing_options(
    options: ResponseProcessingOptions | None,
    *,
    data_rate_hz: float,
) -> ResponseProcessingOptions:
    """Return validated response-processing options for one run.

    Args:
        options: Caller-provided options, or ``None`` for defaults.
        data_rate_hz: Imaging frame rate used for low-pass validation.

    Returns:
        Validated ``ResponseProcessingOptions``.
    """
    resolved = ResponseProcessingOptions() if options is None else options
    validate_response_processing_options(resolved, data_rate_hz=data_rate_hz)
    return resolved


def _has_continuous_response_processing(options: ResponseProcessingOptions) -> bool:
    """Return whether dF/F values need continuous signal processing."""
    return options.smoothing.method != "none" or options.low_pass.method != "none"


def _resolve_roi_set(roi_set: RoiSet | Path) -> RoiSet:
    """Return a ROI set from an object or HDF5 path.

    Args:
        roi_set: Existing ``RoiSet`` or path to one saved by ``save_roi_set``.

    Returns:
        ``RoiSet`` ready for trace extraction.
    """
    if isinstance(roi_set, Path):
        return load_roi_set(roi_set)
    return roi_set


def _resolve_epoch_windows(
    recording: RecordingData,
    epoch_windows: Sequence[EpochFrameWindow] | None,
) -> tuple[InterpolatedEpochMapping | None, tuple[EpochFrameWindow, ...]]:
    """Resolve epoch windows from caller input or converted stimulus data.

    Args:
        recording: Loaded converted recording.
        epoch_windows: Optional explicit epoch windows.

    Returns:
        ``(mapping, windows)``. ``mapping`` is ``None`` when windows were passed
        explicitly.
    """
    if epoch_windows is not None:
        return None, tuple(epoch_windows)
    mapping = interpolate_stimulus_epochs_to_frame_windows(recording)
    return mapping, mapping.windows


def _frame_range_for_epoch_windows(
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    frame_count: int,
    data_rate_hz: float,
    pre_window_seconds: float,
    post_window_seconds: float,
) -> tuple[int, int]:
    """Return the smallest frame range covering all epoch windows.

    Args:
        epoch_windows: Windows used for response analysis.
        frame_count: Total aligned-movie frame count.
        data_rate_hz: Imaging frame rate in hertz.
        pre_window_seconds: Seconds before each epoch to include.
        post_window_seconds: Seconds after each epoch to include.

    Returns:
        ``(start, stop)`` absolute movie-frame range.
    """
    if len(epoch_windows) == 0:
        msg = "At least one epoch window is required for response analysis"
        raise ValueError(msg)
    pre_frames = int(round(pre_window_seconds * data_rate_hz))
    post_frames = int(round(post_window_seconds * data_rate_hz))
    starts = [
        max(epoch_window.window.start_frame - pre_frames, 0)
        for epoch_window in epoch_windows
    ]
    stops = [
        min(epoch_window.window.stop_frame + post_frames, frame_count)
        for epoch_window in epoch_windows
    ]
    return min(starts), max(stops)


def _resolve_output_path(recording_data_path: Path, output_path: Path | None) -> Path:
    """Resolve the analysis HDF5 output path.

    Args:
        recording_data_path: Converted recording manifest path.
        output_path: Optional caller path.

    Returns:
        Expanded output path.
    """
    if output_path is not None:
        return output_path.expanduser()
    return recording_data_path.expanduser().parent / "analysis_outputs.h5"


def _resolve_summary_csv_path(
    *,
    output_path: Path,
    requested_path: Path | None,
    write_summary_csv: bool,
    filename: str,
) -> Path | None:
    """Resolve the optional response time-series CSV path.

    Args:
        output_path: Analysis HDF5 output path.
        requested_path: Optional caller path.
        write_summary_csv: Whether a CSV should be written.
        filename: Default CSV filename when the caller does not provide one.

    Returns:
        CSV path, or ``None`` when disabled.
    """
    if not write_summary_csv:
        return None
    if requested_path is not None:
        return requested_path.expanduser()
    return output_path.parent / "exports" / "csvs" / filename
