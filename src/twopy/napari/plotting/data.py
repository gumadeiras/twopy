"""Build plot-ready response data for the twopy napari adapter.

This module has no Qt imports. It keeps response-data preparation testable and
separate from the napari widgets that draw and control the plots.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from twopy.analysis.background_subtraction import BackgroundCorrectedRoiTraces
from twopy.analysis.dff import DeltaFOverFFitMode, RoiDeltaFOverF
from twopy.analysis.dff_options import DeltaFOverFBaselineMode, DeltaFOverFOptions
from twopy.analysis.persistence import LoadedAnalysisOutputs, load_analysis_outputs
from twopy.analysis.response_processing import (
    ResponseProcessingOptions,
    RoiCorrelationScores,
    mask_grouped_roi_responses_by_included_rois,
)
from twopy.analysis.response_window_options import (
    DEFAULT_RESPONSE_POST_WINDOW_SECONDS,
    DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
    ResponseWindowOptions,
    resolve_response_window_seconds,
)
from twopy.analysis.responses import (
    GroupedRoiResponses,
    RoiResponseTrial,
    group_delta_f_over_f_by_epoch,
    summarize_roi_response_trials,
)
from twopy.analysis.timing import resolve_recording_timing
from twopy.analysis.trials import EpochFrameWindow, is_baseline_epoch_name
from twopy.converted import RecordingData
from twopy.filenames import ANALYSIS_OUTPUT_FILENAME
from twopy.stimulus import stimulus_epoch_names_by_number
from twopy.typing_guards import require_string_choice

__all__ = [
    "EpochResponsePlotData",
    "ResponsePlotData",
    "default_analysis_output_path",
    "filter_response_plot_data_rois",
    "load_response_plot_data",
    "response_plot_baseline_window_limit_for_recording",
    "response_plot_min_epoch_duration_for_recording",
    "response_plot_min_epoch_duration_seconds",
    "response_plot_post_window_seconds_for_recording",
    "response_plot_post_window_seconds",
    "response_plot_window_seconds_for_recording",
    "response_plot_data_from_grouped",
]


@dataclass(frozen=True)
class EpochResponsePlotData:
    """Mean and SEM traces for one stimulus epoch type.

    Each ROI gets one mean trace and one SEM trace on a shared time axis. Epoch
    time spans are optional markers for the plot.
    """

    epoch_name: str
    epoch_number: int
    roi_labels: tuple[str, ...]
    time_seconds: npt.NDArray[np.float64]
    mean_values: npt.NDArray[np.float64]
    sem_values: npt.NDArray[np.float64]
    epoch_time_spans: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class ResponsePlotData:
    """Plot-ready ROI responses grouped by stimulus epoch.

    Each epoch carries its own ``time_seconds`` vector, so plots do not infer
    time from array position. ``visible_roi_indices`` can preselect rows without
    removing hidden ROIs from the ROIs tab.
    """

    source_path: Path | None
    epochs: tuple[EpochResponsePlotData, ...]
    delta_f_over_f_options: DeltaFOverFOptions | None = None
    response_window_options: ResponseWindowOptions | None = None
    response_processing_options: ResponseProcessingOptions | None = None
    correlation_scores: RoiCorrelationScores | None = None
    correlation_window_stop_default_seconds: float | None = None
    visible_roi_indices: tuple[int, ...] | None = None


def response_plot_data_from_grouped(
    grouped: GroupedRoiResponses,
    *,
    source_path: Path | None = None,
    epoch_windows: Sequence[EpochFrameWindow] = (),
    delta_f_over_f_options: DeltaFOverFOptions | None = None,
    response_window_options: ResponseWindowOptions | None = None,
    response_processing_options: ResponseProcessingOptions | None = None,
    correlation_scores: RoiCorrelationScores | None = None,
    correlation_window_stop_default_seconds: float | None = None,
) -> ResponsePlotData:
    """Summarize grouped responses into mean and SEM traces for plotting.

    Args:
        grouped: Grouped response object from analysis.
        source_path: Optional analysis output path used for display.
        epoch_windows: Optional stimulus windows used to compute the grouped
            responses. When supplied, plots mark the epoch span.
        delta_f_over_f_options: Optional dF/F settings used to produce the
            grouped responses.
        response_window_options: Optional response-window settings used to
            produce the grouped responses.
        response_processing_options: Optional processing settings used to
            produce the grouped responses.
        correlation_scores: Optional ROI-level correlation QC scores used to
            hide excluded ROIs in the napari ROI controls.
        correlation_window_stop_default_seconds: Optional default stop time
            for enabling Plot-tab correlation QC.

    Returns:
        Plot-ready data with one item per stimulus epoch type.
    """
    epoch_keys: list[tuple[int, str]] = []
    trials_by_epoch: dict[tuple[int, str], list[RoiResponseTrial]] = {}
    for trial in grouped.trials:
        key = (trial.epoch_number, trial.epoch_name)
        if key not in trials_by_epoch:
            epoch_keys.append(key)
            trials_by_epoch[key] = []
        trials_by_epoch[key].append(trial)

    epoch_spans_by_epoch = _epoch_time_spans_by_epoch(
        epoch_windows,
        data_rate_hz=grouped.data_rate_hz,
    )
    epochs = tuple(
        _epoch_plot_data(
            epoch_number=epoch_number,
            epoch_name=epoch_name,
            trials=tuple(trials_by_epoch[(epoch_number, epoch_name)]),
            roi_labels=grouped.roi_labels,
            data_rate_hz=grouped.data_rate_hz,
            epoch_time_spans=epoch_spans_by_epoch.get((epoch_number, epoch_name), ()),
        )
        for epoch_number, epoch_name in sorted(epoch_keys)
    )
    return ResponsePlotData(
        source_path=source_path,
        epochs=epochs,
        delta_f_over_f_options=delta_f_over_f_options,
        response_window_options=(
            response_window_options
            if response_window_options is not None
            else _response_window_options_from_grouped(grouped)
        ),
        response_processing_options=response_processing_options,
        correlation_scores=correlation_scores,
        correlation_window_stop_default_seconds=correlation_window_stop_default_seconds,
    )


def filter_response_plot_data_rois(
    plot_data: ResponsePlotData,
    roi_indices: Sequence[int],
) -> ResponsePlotData:
    """Return plot data with only the requested ROI rows.

    Args:
        plot_data: Current plot data shown in the response dock.
        roi_indices: Zero-based ROI row indices to keep, in display order.

    Returns:
        Plot data with ROI labels, means, and SEMs filtered consistently for
        every epoch.

    This helper lets the GUI update ROI option rows immediately after users
    delete Labels pixels, while the heavier response recomputation still runs
    through the normal analysis path.
    """
    if len(plot_data.epochs) == 0:
        return plot_data
    roi_count = len(plot_data.epochs[0].roi_labels)
    keep_indices = tuple(index for index in roi_indices if 0 <= index < roi_count)
    keep_rows = list(keep_indices)
    return ResponsePlotData(
        source_path=plot_data.source_path,
        epochs=tuple(
            EpochResponsePlotData(
                epoch_name=epoch.epoch_name,
                epoch_number=epoch.epoch_number,
                roi_labels=tuple(epoch.roi_labels[index] for index in keep_indices),
                time_seconds=epoch.time_seconds,
                mean_values=epoch.mean_values[keep_rows, :],
                sem_values=epoch.sem_values[keep_rows, :],
                epoch_time_spans=epoch.epoch_time_spans,
            )
            for epoch in plot_data.epochs
        ),
        delta_f_over_f_options=plot_data.delta_f_over_f_options,
        response_window_options=plot_data.response_window_options,
        response_processing_options=plot_data.response_processing_options,
        correlation_scores=_filter_correlation_scores(
            plot_data.correlation_scores,
            keep_indices,
        ),
        correlation_window_stop_default_seconds=(
            plot_data.correlation_window_stop_default_seconds
        ),
        visible_roi_indices=_filter_visible_roi_indices(
            plot_data.visible_roi_indices,
            keep_indices,
        ),
    )


def _filter_visible_roi_indices(
    visible_indices: tuple[int, ...] | None,
    keep_indices: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Return visible ROI rows remapped after row filtering."""
    if visible_indices is None:
        return None
    visible_set = set(visible_indices)
    return tuple(
        new_index
        for new_index, old_index in enumerate(keep_indices)
        if old_index in visible_set
    )


def _filter_correlation_scores(
    scores: RoiCorrelationScores | None,
    keep_indices: tuple[int, ...],
) -> RoiCorrelationScores | None:
    """Return correlation scores filtered to the kept ROI rows."""
    if scores is None:
        return None
    keep_rows = list(keep_indices)
    return RoiCorrelationScores(
        roi_labels=tuple(scores.roi_labels[index] for index in keep_indices),
        scores=scores.scores[keep_rows],
        included_mask=scores.included_mask[keep_rows],
        minimum_correlation=scores.minimum_correlation,
        reference=scores.reference,
        window_seconds=scores.window_seconds,
    )


def default_analysis_output_path(recording: RecordingData) -> Path:
    """Return the default analysis output path for one recording.

    Args:
        recording: Loaded converted recording.

    Returns:
        Path to ``analysis_outputs.h5`` beside ``recording_data.h5``.
    """
    return recording.path.expanduser().parent / ANALYSIS_OUTPUT_FILENAME


def response_plot_post_window_seconds(epoch_names: Iterable[str]) -> float:
    """Return post-epoch context for response plots from epoch names.

    Args:
        epoch_names: Stimulus epoch names available for one recording or saved
            analysis output.

    Returns:
        Two seconds when a baseline-like gray/interleave epoch exists, otherwise zero.

    Response plots show the return into baseline context when the recording has
    such a baseline epoch. Keeping this rule shared prevents saved-output plots
    and live recompute previews from showing different x-axis ranges.
    """
    for epoch_name in epoch_names:
        if is_baseline_epoch_name(epoch_name):
            return DEFAULT_RESPONSE_POST_WINDOW_SECONDS
    return 0.0


def response_plot_post_window_seconds_for_recording(recording: RecordingData) -> float:
    """Return post-epoch plot context for one loaded recording.

    Args:
        recording: Loaded converted recording with stimulus epoch metadata.

    Returns:
        Two seconds when a baseline-like gray/interleave epoch exists, otherwise zero.
    """
    return response_plot_post_window_seconds(
        stimulus_epoch_names_by_number(recording).values(),
    )


def _response_plot_baseline_window_seconds_for_recording(
    recording: RecordingData,
) -> float | None:
    """Return the shortest named baseline epoch duration for one recording.

    Args:
        recording: Loaded converted recording with stimulus epoch metadata.

    Returns:
        Duration in seconds for the shortest gray/grey/interleave epoch window,
        or ``None`` when no such named window is available.

    Manual Plot-tab response windows are capped by this value so pre/post
    context cannot exceed the available baseline epoch duration.
    """
    timing = resolve_recording_timing(recording)
    durations = tuple(
        (epoch_window.window.stop_frame - epoch_window.window.start_frame)
        / timing.frame_rate_hz
        for epoch_window in timing.epoch_windows
        if is_baseline_epoch_name(epoch_window.epoch_name)
    )
    if len(durations) == 0:
        return None
    return max(0.0, min(durations))


def response_plot_baseline_window_limit_for_recording(
    recording: RecordingData,
) -> float | None:
    """Return the manual response-window cap for one recording.

    Args:
        recording: Loaded converted recording with stimulus timing metadata.

    Returns:
        Shortest named gray/interleave epoch duration in seconds, or ``None``
        when no reliable cap is available.
    """
    try:
        return _response_plot_baseline_window_seconds_for_recording(recording)
    except ValueError:
        return None


def response_plot_min_epoch_duration_for_recording(
    recording: RecordingData,
) -> float | None:
    """Return the shortest stimulus epoch duration for one recording.

    Args:
        recording: Loaded converted recording with stimulus timing metadata.

    Returns:
        Shortest positive photodiode-aligned epoch duration in seconds, or
        ``None`` when timing evidence is unavailable.
    """
    try:
        timing = resolve_recording_timing(recording)
    except ValueError:
        return None
    durations = tuple(
        (epoch_window.window.stop_frame - epoch_window.window.start_frame)
        / timing.frame_rate_hz
        for epoch_window in timing.epoch_windows
        if epoch_window.window.stop_frame > epoch_window.window.start_frame
    )
    return _minimum_positive_duration(durations)


def response_plot_min_epoch_duration_seconds(
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    data_rate_hz: float,
) -> float | None:
    """Return the shortest stimulus epoch duration from exact frame windows.

    Args:
        epoch_windows: Photodiode-aligned stimulus epoch windows.
        data_rate_hz: Imaging frame rate in hertz.

    Returns:
        Shortest positive epoch duration in seconds, or ``None`` when no
        positive duration can be computed.
    """
    if not np.isfinite(data_rate_hz) or data_rate_hz <= 0.0:
        return None
    return _minimum_positive_duration(
        (epoch.window.stop_frame - epoch.window.start_frame) / data_rate_hz
        for epoch in epoch_windows
        if epoch.window.stop_frame > epoch.window.start_frame
    )


def response_plot_window_seconds_for_recording(
    recording: RecordingData,
    options: ResponseWindowOptions,
) -> tuple[float, float]:
    """Resolve Plot-tab response-window options for one recording.

    Args:
        recording: Loaded converted recording.
        options: User-selected automatic/manual response-window options.

    Returns:
        ``(pre_window_seconds, post_window_seconds)`` for response grouping.
    """
    return resolve_response_window_seconds(
        options,
        automatic_pre_window_seconds=DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
        automatic_post_window_seconds=response_plot_post_window_seconds_for_recording(
            recording,
        ),
        max_window_seconds=response_plot_baseline_window_limit_for_recording(
            recording,
        ),
    )


def load_response_plot_data(path: Path) -> ResponsePlotData | str:
    """Load response plot data from an analysis output file.

    Args:
        path: Candidate ``analysis_outputs.h5`` path.

    Returns:
        Plot data, or a status message for the user.
    """
    if not path.is_file():
        return f"No analysis output found: {path}"
    outputs = load_analysis_outputs(path)
    if outputs.grouped_responses is not None:
        saved_window_options = _response_window_options_from_grouped(
            outputs.grouped_responses,
        )
    else:
        saved_window_options = None
    if outputs.grouped_responses is not None and saved_window_options is not None:
        return response_plot_data_from_grouped(
            outputs.grouped_responses,
            source_path=outputs.path,
            epoch_windows=outputs.epoch_windows,
            delta_f_over_f_options=_delta_f_over_f_options_from_outputs(
                traces=outputs.traces,
                dff=outputs.dff,
            ),
            response_window_options=saved_window_options,
            response_processing_options=outputs.response_processing_options,
            correlation_scores=outputs.correlation_scores,
            correlation_window_stop_default_seconds=(
                _correlation_stop_default_from_outputs(outputs)
            ),
        )
    if outputs.dff is not None and len(outputs.epoch_windows) > 0:
        data_rate_hz = _analysis_output_data_rate_hz(
            grouped=outputs.grouped_responses,
            dff=outputs.dff,
        )
        if data_rate_hz is not None:
            post_window_seconds = _response_post_window_seconds(outputs.epoch_windows)
            grouped = group_delta_f_over_f_by_epoch(
                outputs.dff,
                outputs.epoch_windows,
                data_rate_hz=data_rate_hz,
                pre_window_seconds=DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
                post_window_seconds=post_window_seconds,
            )
            if outputs.correlation_scores is not None:
                grouped = mask_grouped_roi_responses_by_included_rois(
                    grouped,
                    included_mask=outputs.correlation_scores.included_mask,
                )
            return response_plot_data_from_grouped(
                grouped,
                source_path=outputs.path,
                epoch_windows=outputs.epoch_windows,
                delta_f_over_f_options=_delta_f_over_f_options_from_outputs(
                    traces=outputs.traces,
                    dff=outputs.dff,
                ),
                response_window_options=_response_window_options_from_grouped(grouped),
                response_processing_options=outputs.response_processing_options,
                correlation_scores=outputs.correlation_scores,
                correlation_window_stop_default_seconds=(
                    _correlation_stop_default_from_outputs(outputs)
                ),
            )
    if outputs.grouped_responses is None:
        return f"No grouped responses in: {path}"
    return response_plot_data_from_grouped(
        outputs.grouped_responses,
        source_path=outputs.path,
        epoch_windows=outputs.epoch_windows,
        delta_f_over_f_options=_delta_f_over_f_options_from_outputs(
            traces=outputs.traces,
            dff=outputs.dff,
        ),
        response_window_options=_response_window_options_from_grouped(
            outputs.grouped_responses,
        ),
        response_processing_options=outputs.response_processing_options,
        correlation_scores=outputs.correlation_scores,
        correlation_window_stop_default_seconds=_correlation_stop_default_from_outputs(
            outputs,
        ),
    )


def _response_window_options_from_grouped(
    grouped: GroupedRoiResponses,
) -> ResponseWindowOptions | None:
    """Return saved response-window options from grouped metadata."""
    pre_window_seconds = grouped.pre_window_seconds
    post_window_seconds = grouped.post_window_seconds
    if pre_window_seconds is None or post_window_seconds is None:
        return None
    return ResponseWindowOptions(
        auto=bool(grouped.response_window_auto)
        if grouped.response_window_auto is not None
        else False,
        pre_window_seconds=float(pre_window_seconds),
        post_window_seconds=float(post_window_seconds),
    )


def _delta_f_over_f_options_from_outputs(
    *,
    traces: BackgroundCorrectedRoiTraces | None,
    dff: RoiDeltaFOverF | None,
) -> DeltaFOverFOptions | None:
    """Return inspectable dF/F options from saved analysis metadata.

    Args:
        traces: Optional saved trace outputs carrying the background method.
        dff: Optional saved dF/F outputs carrying baseline and fit metadata.

    Returns:
        ``DeltaFOverFOptions`` when saved outputs expose dF/F settings,
        otherwise ``None``.
    """
    if traces is None and dff is None:
        return None

    defaults = DeltaFOverFOptions()
    background_method = (
        traces.method if traces is not None else defaults.background_method
    )
    if dff is None:
        return DeltaFOverFOptions(background_method=background_method)

    return DeltaFOverFOptions(
        baseline_mode=_saved_baseline_mode(dff),
        baseline_epoch_number=_saved_baseline_epoch_number(dff),
        baseline_epoch_name=_saved_baseline_epoch_name(dff),
        background_method=background_method,
        baseline_sample_seconds=_saved_baseline_sample_seconds(dff),
        fit_mode=_saved_fit_mode(dff),
        apply_motion_mask="motion_artifact_masked_frame_count" in dff.metadata,
    )


def _saved_baseline_mode(dff: RoiDeltaFOverF) -> DeltaFOverFBaselineMode:
    """Return the saved baseline-selection mode from dF/F metadata."""
    value = dff.metadata.get("baseline_mode")
    if isinstance(value, str) and value in {"epoch", "no_baseline_epoch"}:
        return require_string_choice(
            value,
            name="dF/F baseline mode",
            allowed=("epoch", "no_baseline_epoch"),
        )
    return DeltaFOverFOptions().baseline_mode


def _saved_baseline_sample_seconds(dff: RoiDeltaFOverF) -> float | None:
    """Return the saved baseline sampling window from dF/F metadata.

    Args:
        dff: Saved dF/F object with audit metadata.

    Returns:
        Seconds from each baseline window end, or ``None`` for full windows.
    """
    value = dff.metadata.get("baseline_sample_seconds")
    if value == "full":
        return None
    if isinstance(value, bool):
        return DeltaFOverFOptions().baseline_sample_seconds
    if isinstance(value, int | float):
        return float(value)
    return DeltaFOverFOptions().baseline_sample_seconds


def _saved_baseline_epoch_number(dff: RoiDeltaFOverF) -> int | None:
    """Return the saved baseline epoch number selector from dF/F metadata."""
    value = dff.metadata.get("baseline_epoch_number")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return DeltaFOverFOptions().baseline_epoch_number


def _saved_baseline_epoch_name(dff: RoiDeltaFOverF) -> str | None:
    """Return the saved baseline epoch name selector from dF/F metadata."""
    value = dff.metadata.get("baseline_epoch_name")
    if isinstance(value, str):
        return value
    return DeltaFOverFOptions().baseline_epoch_name


def _saved_fit_mode(dff: RoiDeltaFOverF) -> DeltaFOverFFitMode:
    """Return the saved dF/F fit mode from metadata when recognized.

    Args:
        dff: Saved dF/F object with audit metadata.

    Returns:
        Recognized fit mode, otherwise the current default.
    """
    value = dff.metadata.get("fit_mode")
    if isinstance(value, str) and value in {
        "direct_bounded_tau",
        "direct_bounded_tau_and_log_amplitude",
        "log_linear",
    }:
        return require_string_choice(
            value,
            name="dF/F fit mode",
            allowed=(
                "direct_bounded_tau",
                "direct_bounded_tau_and_log_amplitude",
                "log_linear",
            ),
        )
    return DeltaFOverFOptions().fit_mode


def _epoch_plot_data(
    *,
    epoch_number: int,
    epoch_name: str,
    trials: tuple[RoiResponseTrial, ...],
    roi_labels: tuple[str, ...],
    data_rate_hz: float,
    epoch_time_spans: tuple[tuple[float, float], ...],
) -> EpochResponsePlotData:
    """Build plot data for one epoch type.

    Args:
        epoch_number: Stimulus epoch number.
        epoch_name: Stimulus epoch name.
        trials: Trial responses for this epoch type.
        roi_labels: ROI labels in column order.
        data_rate_hz: Imaging frame rate in hertz.
        epoch_time_spans: Epoch intervals in epoch-relative seconds.

    Returns:
        Mean and SEM traces with shape ``(rois, frames)``.
    """
    summary = summarize_roi_response_trials(
        trials,
        roi_labels=roi_labels,
        data_rate_hz=data_rate_hz,
    )
    return EpochResponsePlotData(
        epoch_name=epoch_name,
        epoch_number=epoch_number,
        roi_labels=roi_labels,
        time_seconds=summary.time_seconds,
        mean_values=summary.mean_values,
        sem_values=summary.sem_values,
        epoch_time_spans=epoch_time_spans,
    )


def _epoch_time_spans_by_epoch(
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    data_rate_hz: float,
) -> dict[tuple[int, str], tuple[tuple[float, float], ...]]:
    """Return coarse epoch spans for each plotted epoch type."""
    if not np.isfinite(data_rate_hz) or data_rate_hz <= 0.0:
        return {}
    duration_by_epoch: dict[tuple[int, str], float] = {}
    for epoch_window in epoch_windows:
        duration = (
            epoch_window.window.stop_frame - epoch_window.window.start_frame
        ) / data_rate_hz
        if not np.isfinite(duration) or duration <= 0.0:
            continue
        key = (epoch_window.epoch_number, epoch_window.epoch_name)
        duration_by_epoch[key] = max(duration_by_epoch.get(key, 0.0), float(duration))
    return {key: ((0.0, duration),) for key, duration in duration_by_epoch.items()}


def _correlation_stop_default_from_outputs(
    outputs: LoadedAnalysisOutputs,
) -> float | None:
    """Return the exact saved-analysis correlation stop default."""
    if outputs.grouped_responses is None or len(outputs.epoch_windows) == 0:
        return None
    return response_plot_min_epoch_duration_seconds(
        outputs.epoch_windows,
        data_rate_hz=outputs.grouped_responses.data_rate_hz,
    )


def _minimum_positive_duration(durations: Iterable[float]) -> float | None:
    """Return the smallest finite positive duration from candidates."""
    positive = tuple(
        float(duration)
        for duration in durations
        if np.isfinite(duration) and duration > 0.0
    )
    if len(positive) == 0:
        return None
    return min(positive)


def _analysis_output_data_rate_hz(
    *,
    grouped: GroupedRoiResponses | None,
    dff: RoiDeltaFOverF,
) -> float | None:
    """Return the saved response frame rate when available.

    Args:
        grouped: Optional grouped responses loaded from analysis outputs.
        dff: dF/F result with audit metadata.

    Returns:
        Data rate in hertz, or ``None`` when unavailable.
    """
    if grouped is not None:
        return grouped.data_rate_hz
    value = dff.metadata.get("data_rate_hz")
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _response_post_window_seconds(
    epoch_windows: Sequence[EpochFrameWindow],
) -> float:
    """Return the default post-epoch plotting context from available windows.

    Args:
        epoch_windows: Iterable of epoch-window objects with ``epoch_name``.

    Returns:
        Two seconds when a gray interleave epoch exists, otherwise zero.
    """
    return response_plot_post_window_seconds(
        epoch_window.epoch_name for epoch_window in epoch_windows
    )
