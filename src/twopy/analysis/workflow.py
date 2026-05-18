"""Script-facing workflow for ROI response analysis.

Inputs: a converted recording, ROI masks, and analysis parameters.
Outputs: computed traces, dF/F, grouped responses, and saved analysis files.

This module wires together existing twopy analysis helpers. It does not read
source microscope files; callers must convert recordings before using it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
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
from twopy.analysis.dff_options import DeltaFOverFBaselineMode
from twopy.analysis.epoch_mapping import InterpolatedEpochMapping
from twopy.analysis.motion import apply_motion_artifact_mask_to_delta_f_over_f
from twopy.analysis.persistence import save_analysis_outputs
from twopy.analysis.response_context import (
    ResponseAnalysisContext,
    resolve_response_analysis_context,
)
from twopy.analysis.response_processing import (
    ResponseProcessingOptions,
    RoiCorrelationScores,
    RoiNormalizationFactors,
    apply_correlation_filter_to_grouped_roi_responses,
    normalize_grouped_roi_responses_by_epoch_peak,
    process_roi_delta_f_over_f,
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
)
from twopy.converted import (
    RecordingData,
    load_converted_recording,
)
from twopy.filenames import (
    ANALYSIS_OUTPUT_FILENAME,
    CSV_EXPORTS_DIRNAME,
    EXPORTS_DIRNAME,
    RESPONSE_SUMMARY_GROUPED_FILENAME,
    RESPONSE_SUMMARY_TRIALS_FILENAME,
)
from twopy.roi import RoiSet, load_roi_set
from twopy.spatial import SpatialDomain

__all__ = [
    "AnalysisResponseComputation",
    "AnalysisResponseRun",
    "DEFAULT_RESPONSE_POST_WINDOW_SECONDS",
    "DEFAULT_RESPONSE_PRE_WINDOW_SECONDS",
    "analyze_recording_responses",
    "compute_recording_responses_from_traces",
    "compute_recording_responses",
    "ResponseAnalysisContext",
    "resolve_response_analysis_context",
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
    baseline_windows: tuple[FrameWindow, ...]
    grouped_responses: GroupedRoiResponses
    response_processing_options: ResponseProcessingOptions
    normalization_factors: RoiNormalizationFactors | None
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
    baseline_windows: Sequence[FrameWindow] | None = None,
    baseline_mode: DeltaFOverFBaselineMode = "epoch",
    baseline_epoch_number: int | None = None,
    baseline_epoch_name: str | None = None,
    background_method: BackgroundCorrectionMethod = "movie_global_percentile",
    baseline_sample_seconds: float | None = 1.0,
    fit_mode: DeltaFOverFFitMode = "direct_bounded_tau",
    apply_motion_mask: bool = True,
    data_rate_hz: float | None = None,
    chunk_frames: int = 128,
    spatial_domain: SpatialDomain = "alignment_valid_crop",
    response_pre_window_seconds: float = DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
    response_post_window_seconds: float = 0.0,
    response_window_auto: bool | None = None,
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
        baseline_windows: Optional explicit baseline windows. When omitted,
            twopy selects baseline windows from epoch name or number.
        baseline_mode: Baseline selection mode. ``epoch`` uses selected epoch
            windows, while ``no_baseline_epoch`` uses one continuous span from
            the selected epoch number onward.
        baseline_epoch_number: Optional baseline epoch number selector. ``None``
            uses the shared recording-level default.
        baseline_epoch_name: Optional baseline epoch name selector.
        background_method: Background correction method before dF/F.
        baseline_sample_seconds: Seconds from the end of each baseline window.
        fit_mode: dF/F exponential fit mode.
        apply_motion_mask: Whether to mark converted high-motion frames as NaN
            after dF/F.
        data_rate_hz: Optional frame rate. When omitted, twopy reads
            ``acq.frameRate`` from converted metadata.
        chunk_frames: Number of movie frames to read per HDF5 chunk.
        spatial_domain: Spatial domain used for trace extraction.
        response_pre_window_seconds: Seconds before each stimulus window to
            include in grouped responses. The default shows the transition from
            baseline context into the stimulus.
        response_post_window_seconds: Seconds after each stimulus window to
            include in grouped responses. The default is zero so persisted
            summaries keep their epoch-window meaning; plotting code can add
            visual context separately from saved dF/F and epoch windows.
        response_window_auto: Optional provenance flag recording whether the
            response window was selected by automatic GUI settings.
        response_processing_options: Optional smoothing, low-pass, and
            correlation-QC settings to apply after dF/F. Signal filters are
            applied to continuous dF/F before grouping; correlation QC is
            applied to grouped responses.

    Returns:
        ``AnalysisResponseRun`` with computed objects and output paths.

    Raises:
        ValueError: If no baseline windows match the selector.

    This is the main script API for a converted recording: load, extract traces,
    compute dF/F, group responses, and persist the results.
    """
    recording = load_converted_recording(recording_data_path)
    loaded_roi_set = _resolve_roi_set(roi_set)
    computation = compute_recording_responses(
        recording,
        loaded_roi_set,
        epoch_windows=epoch_windows,
        baseline_windows=baseline_windows,
        baseline_mode=baseline_mode,
        baseline_epoch_number=baseline_epoch_number,
        baseline_epoch_name=baseline_epoch_name,
        background_method=background_method,
        baseline_sample_seconds=baseline_sample_seconds,
        fit_mode=fit_mode,
        apply_motion_mask=apply_motion_mask,
        data_rate_hz=data_rate_hz,
        chunk_frames=chunk_frames,
        spatial_domain=spatial_domain,
        response_pre_window_seconds=response_pre_window_seconds,
        response_post_window_seconds=response_post_window_seconds,
        response_processing_options=response_processing_options,
    )
    if response_window_auto is not None:
        computation = replace(
            computation,
            grouped_responses=replace(
                computation.grouped_responses,
                response_window_auto=bool(response_window_auto),
            ),
        )
    computation = replace(
        computation,
        dff=_dff_with_analysis_options(
            computation.dff,
            baseline_mode=baseline_mode,
            baseline_epoch_number=baseline_epoch_number,
            baseline_epoch_name=baseline_epoch_name,
        ),
    )
    return _save_response_analysis_run(
        computation,
        output_path=output_path,
        response_summary_trials_csv_path=response_summary_trials_csv_path,
        response_summary_grouped_csv_path=response_summary_grouped_csv_path,
        write_summary_csv=write_summary_csv,
    )


def _save_response_analysis_run(
    computation: AnalysisResponseComputation,
    *,
    output_path: Path | None,
    response_summary_trials_csv_path: Path | None,
    response_summary_grouped_csv_path: Path | None,
    write_summary_csv: bool,
) -> AnalysisResponseRun:
    """Persist one in-memory response computation.

    Args:
        computation: Computed response-analysis objects.
        output_path: Optional destination HDF5 path.
        response_summary_trials_csv_path: Optional trial-level CSV path.
        response_summary_grouped_csv_path: Optional grouped CSV path.
        write_summary_csv: Whether to write response CSV files.

    Returns:
        ``AnalysisResponseRun`` with saved output paths.
    """
    resolved_output_path = _resolve_output_path(computation.recording.path, output_path)
    resolved_trials_summary_path = _resolve_summary_csv_path(
        output_path=resolved_output_path,
        requested_path=response_summary_trials_csv_path,
        write_summary_csv=write_summary_csv,
        filename=RESPONSE_SUMMARY_TRIALS_FILENAME,
    )
    resolved_grouped_summary_path = _resolve_summary_csv_path(
        output_path=resolved_output_path,
        requested_path=response_summary_grouped_csv_path,
        write_summary_csv=write_summary_csv,
        filename=RESPONSE_SUMMARY_GROUPED_FILENAME,
    )
    save_analysis_outputs(
        resolved_output_path,
        roi_set=computation.roi_set,
        traces=computation.traces,
        dff=computation.dff,
        epoch_windows=computation.epoch_windows,
        baseline_windows=computation.baseline_windows,
        grouped_responses=computation.grouped_responses,
        response_processing_options=computation.response_processing_options,
        normalization_factors=computation.normalization_factors,
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
        baseline_windows=computation.baseline_windows,
        grouped_responses=computation.grouped_responses,
        response_processing_options=computation.response_processing_options,
        normalization_factors=computation.normalization_factors,
        correlation_scores=computation.correlation_scores,
        output_path=resolved_output_path,
        response_summary_trials_csv_path=resolved_trials_summary_path,
        response_summary_grouped_csv_path=resolved_grouped_summary_path,
    )


def _dff_with_analysis_options(
    dff: RoiDeltaFOverF,
    *,
    baseline_mode: DeltaFOverFBaselineMode,
    baseline_epoch_number: int | None,
    baseline_epoch_name: str | None,
) -> RoiDeltaFOverF:
    """Return dF/F with the selected baseline options saved in metadata.

    Args:
        dff: Computed dF/F output.
        baseline_mode: Baseline selection mode used by the workflow.
        baseline_epoch_number: Baseline epoch selector used by the workflow.
        baseline_epoch_name: Optional baseline epoch name selector.

    Returns:
        dF/F output with metadata used to restore saved Plot-tab controls.
    """
    metadata = dict(dff.metadata)
    metadata["baseline_mode"] = baseline_mode
    if baseline_epoch_number is not None:
        metadata["baseline_epoch_number"] = int(baseline_epoch_number)
    if baseline_epoch_name is not None:
        metadata["baseline_epoch_name"] = baseline_epoch_name
    return replace(dff, metadata=metadata)


def compute_recording_responses(
    recording: RecordingData,
    roi_set: RoiSet,
    *,
    epoch_windows: Sequence[EpochFrameWindow] | None = None,
    baseline_windows: Sequence[FrameWindow] | None = None,
    baseline_mode: DeltaFOverFBaselineMode = "epoch",
    baseline_epoch_number: int | None = None,
    baseline_epoch_name: str | None = None,
    background_method: BackgroundCorrectionMethod = "movie_global_percentile",
    baseline_sample_seconds: float | None = 1.0,
    fit_mode: DeltaFOverFFitMode = "direct_bounded_tau",
    apply_motion_mask: bool = True,
    data_rate_hz: float | None = None,
    chunk_frames: int = 128,
    spatial_domain: SpatialDomain = "alignment_valid_crop",
    response_pre_window_seconds: float = DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
    response_post_window_seconds: float = 0.0,
    response_processing_options: ResponseProcessingOptions | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> AnalysisResponseComputation:
    """Compute ROI responses from converted data without writing files.

    Args:
        recording: Loaded converted recording.
        roi_set: ROI masks to extract from the recording.
        epoch_windows: Optional explicit epoch windows. When omitted, twopy
            interpolates epochs from converted stimulus data.
        baseline_windows: Optional explicit baseline windows. When omitted,
            twopy selects baseline windows from epoch name or number.
        baseline_mode: Baseline selection mode. ``epoch`` uses selected epoch
            windows, while ``no_baseline_epoch`` uses one continuous span from
            the selected epoch number onward.
        baseline_epoch_number: Optional baseline epoch number selector. ``None``
            uses the shared recording-level default.
        baseline_epoch_name: Optional baseline epoch name selector.
        background_method: Background correction method before dF/F.
        baseline_sample_seconds: Seconds from the end of each baseline window.
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
        check_cancelled: Optional callback that raises when an interactive
            caller has superseded this computation.

    Returns:
        In-memory response-analysis objects.

    Raises:
        ValueError: If no baseline windows match the selector.

    This is the shared calculation used by scripts, tests, and the napari live
    plotter. It streams movie frames in chunks and leaves file writing to a
    separate explicit call.
    """
    context = resolve_response_analysis_context(
        recording,
        epoch_windows=epoch_windows,
        baseline_windows=baseline_windows,
        baseline_mode=baseline_mode,
        baseline_epoch_number=baseline_epoch_number,
        baseline_epoch_name=baseline_epoch_name,
        data_rate_hz=data_rate_hz,
        response_pre_window_seconds=response_pre_window_seconds,
        response_post_window_seconds=response_post_window_seconds,
        response_processing_options=response_processing_options,
    )
    _check_cancelled(check_cancelled)
    traces = extract_background_corrected_roi_traces(
        recording,
        roi_set,
        method=background_method,
        start_frame=context.trace_start,
        stop_frame=context.trace_stop,
        chunk_frames=chunk_frames,
        spatial_domain=spatial_domain,
        check_cancelled=check_cancelled,
    )
    _check_cancelled(check_cancelled)
    return compute_recording_responses_from_traces(
        recording,
        roi_set,
        traces,
        context=context,
        baseline_sample_seconds=baseline_sample_seconds,
        fit_mode=fit_mode,
        apply_motion_mask=apply_motion_mask,
        check_cancelled=check_cancelled,
    )


def compute_recording_responses_from_traces(
    recording: RecordingData,
    roi_set: RoiSet,
    traces: BackgroundCorrectedRoiTraces,
    *,
    context: ResponseAnalysisContext,
    baseline_sample_seconds: float | None = 1.0,
    fit_mode: DeltaFOverFFitMode = "direct_bounded_tau",
    apply_motion_mask: bool = True,
    check_cancelled: Callable[[], None] | None = None,
) -> AnalysisResponseComputation:
    """Compute grouped responses from already-extracted ROI traces.

    Args:
        recording: Loaded converted recording.
        roi_set: ROI masks represented by ``traces``.
        traces: Background-corrected traces for the current ROI set.
        context: Resolved timing and processing context for this recording.
        baseline_sample_seconds: Optional seconds from each baseline end.
        fit_mode: dF/F exponential fit mode.
        apply_motion_mask: Whether to mark converted high-motion frames as NaN.
        check_cancelled: Optional callback that raises when work is obsolete.

    Returns:
        In-memory response-analysis objects.

    This helper keeps the scientific math identical to
    ``compute_recording_responses`` while allowing interactive callers to reuse
    movie-derived traces when ROI masks have not changed.
    """
    _validate_traces_for_context(traces, roi_set, context)
    _check_cancelled(check_cancelled)
    computation = _compute_recording_responses_from_baseline(
        recording,
        roi_set,
        traces=traces,
        mapping=context.epoch_mapping,
        resolved_epoch_windows=context.epoch_windows,
        baseline_windows=context.baseline_windows,
        frame_rate_hz=context.frame_rate_hz,
        processing_options=context.processing_options,
        baseline_sample_seconds=baseline_sample_seconds,
        fit_mode=fit_mode,
        apply_motion_mask=apply_motion_mask,
        response_pre_window_seconds=context.response_pre_window_seconds,
        response_post_window_seconds=context.response_post_window_seconds,
        check_cancelled=check_cancelled,
    )
    return replace(
        computation,
        dff=_dff_with_analysis_options(
            computation.dff,
            baseline_mode=context.baseline_mode,
            baseline_epoch_number=context.baseline_epoch_number,
            baseline_epoch_name=context.baseline_epoch_name,
        ),
    )


def _compute_recording_responses_from_baseline(
    recording: RecordingData,
    roi_set: RoiSet,
    *,
    traces: BackgroundCorrectedRoiTraces,
    mapping: InterpolatedEpochMapping | None,
    resolved_epoch_windows: Sequence[EpochFrameWindow],
    baseline_windows: tuple[FrameWindow, ...],
    frame_rate_hz: float,
    processing_options: ResponseProcessingOptions,
    baseline_sample_seconds: float | None,
    fit_mode: DeltaFOverFFitMode,
    apply_motion_mask: bool,
    response_pre_window_seconds: float,
    response_post_window_seconds: float,
    check_cancelled: Callable[[], None] | None = None,
) -> AnalysisResponseComputation:
    """Run the shared response computation after baseline windows are selected.

    Args:
        recording: Loaded converted recording.
        roi_set: ROI masks to extract from the recording.
        traces: Background-corrected ROI traces.
        mapping: Optional interpolated epoch mapping used to audit timing.
        resolved_epoch_windows: Stimulus windows used for grouping.
        baseline_windows: Baseline windows used for dF/F fitting.
        frame_rate_hz: Effective analysis frame rate.
        processing_options: Post-dF/F response processing settings.
        baseline_sample_seconds: Optional seconds from each baseline end.
        fit_mode: dF/F exponential fit mode.
        apply_motion_mask: Whether to mark converted high-motion frames as NaN.
        response_pre_window_seconds: Seconds before each stimulus window.
        response_post_window_seconds: Seconds after each stimulus window.
        check_cancelled: Optional callback that raises when work is obsolete.

    Returns:
        In-memory response-analysis objects.
    """
    dff = compute_roi_delta_f_over_f(
        traces,
        baseline_windows,
        data_rate_hz=frame_rate_hz,
        baseline_sample_seconds=baseline_sample_seconds,
        fit_mode=fit_mode,
    )
    _check_cancelled(check_cancelled)
    if apply_motion_mask:
        dff = apply_motion_artifact_mask_to_delta_f_over_f(dff, recording)
    if _has_continuous_response_processing(processing_options):
        dff = process_roi_delta_f_over_f(
            dff,
            options=processing_options,
            data_rate_hz=frame_rate_hz,
        )
    _check_cancelled(check_cancelled)

    grouped_responses = group_delta_f_over_f_by_epoch(
        dff,
        resolved_epoch_windows,
        data_rate_hz=frame_rate_hz,
        pre_window_seconds=response_pre_window_seconds,
        post_window_seconds=response_post_window_seconds,
    )
    _check_cancelled(check_cancelled)
    grouped_responses, normalization_factors = (
        normalize_grouped_roi_responses_by_epoch_peak(
            grouped_responses,
            options=processing_options.normalization,
        )
    )
    grouped_responses, correlation_scores = (
        apply_correlation_filter_to_grouped_roi_responses(
            grouped_responses,
            options=processing_options,
        )
    )
    _check_cancelled(check_cancelled)

    return AnalysisResponseComputation(
        recording=recording,
        roi_set=roi_set,
        traces=traces,
        dff=dff,
        epoch_mapping=mapping,
        epoch_windows=tuple(resolved_epoch_windows),
        baseline_windows=baseline_windows,
        grouped_responses=grouped_responses,
        response_processing_options=processing_options,
        normalization_factors=normalization_factors,
        correlation_scores=correlation_scores,
    )


def _validate_traces_for_context(
    traces: BackgroundCorrectedRoiTraces,
    roi_set: RoiSet,
    context: ResponseAnalysisContext,
) -> None:
    """Confirm cached traces match the ROI set and resolved frame range."""
    if traces.labels != roi_set.labels:
        msg = "Cached trace labels do not match ROI labels."
        raise ValueError(msg)
    if traces.corrected_values.ndim != 2:
        msg = "Cached corrected trace values must have shape (frames, rois)."
        raise ValueError(msg)
    expected_frame_count = context.trace_stop - context.trace_start
    if traces.corrected_values.shape != (expected_frame_count, len(roi_set.labels)):
        msg = (
            "Cached trace shape does not match resolved analysis context: "
            f"{traces.corrected_values.shape} for {expected_frame_count} frames "
            f"and {len(roi_set.labels)} ROIs."
        )
        raise ValueError(msg)
    if (
        traces.raw_values.shape != traces.corrected_values.shape
        or traces.background_values.shape != traces.corrected_values.shape
    ):
        msg = "Cached raw, background, and corrected traces must share one shape."
        raise ValueError(msg)
    if (
        traces.start_frame != context.trace_start
        or traces.stop_frame != context.trace_stop
    ):
        msg = (
            "Cached trace frame range does not match resolved analysis context: "
            f"[{traces.start_frame}, {traces.stop_frame}) versus "
            f"[{context.trace_start}, {context.trace_stop})."
        )
        raise ValueError(msg)


def _check_cancelled(check_cancelled: Callable[[], None] | None) -> None:
    """Raise through the caller callback when a computation is obsolete."""
    if check_cancelled is not None:
        check_cancelled()


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
    return recording_data_path.expanduser().parent / ANALYSIS_OUTPUT_FILENAME


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
    return output_path.parent / EXPORTS_DIRNAME / CSV_EXPORTS_DIRNAME / filename
