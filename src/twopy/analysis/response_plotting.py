"""Build GUI-neutral response data for plotting ROI responses.

Inputs: grouped ROI responses, optional epoch windows, and analysis option
metadata.
Outputs: plain plot-ready response objects with ROI labels, time axes, and
mean/SEM arrays.

The objects in this module do not depend on napari or Qt. GUI code, scripts, and
custom workflows can share the same response-plot data contract.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.response_processing import (
    ResponseProcessingOptions,
    RoiCorrelationScores,
)
from twopy.analysis.response_window_options import ResponseWindowOptions
from twopy.analysis.responses import (
    GroupedRoiResponses,
    RoiResponseTrial,
    summarize_roi_response_trials,
)
from twopy.analysis.trials import EpochFrameWindow

__all__ = [
    "EpochResponsePlotData",
    "ResponsePlotData",
    "filter_response_plot_data_rois",
    "response_plot_data_from_grouped",
    "response_plot_window_options_from_grouped",
]


@dataclass(frozen=True)
class EpochResponsePlotData:
    """Mean and SEM traces for one stimulus epoch type.

    Args:
        epoch_name: Display name for the stimulus epoch.
        epoch_number: One-based stimulus epoch number.
        roi_labels: ROI labels in trace-row order.
        time_seconds: Time axis relative to epoch onset.
        mean_values: Mean response values with shape ``(rois, samples)``.
        sem_values: SEM values with shape ``(rois, samples)``.
        epoch_time_spans: Optional shaded epoch spans in relative seconds.

    Returns:
        Immutable plot-data row for one epoch type.
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

    Args:
        source_path: Optional file path used to display where responses came from.
        epochs: Plot data for each stimulus epoch type.
        delta_f_over_f_options: Optional dF/F settings used to create responses.
        response_window_options: Optional response-window settings used to group
            responses.
        response_processing_options: Optional processing settings applied before
            plotting.
        correlation_scores: Optional ROI correlation-QC scores.
        correlation_window_stop_default_seconds: Optional default QC stop time.
        visible_roi_indices: Optional ROI rows to show initially while keeping all
            ROI data available.

    Returns:
        Immutable response plot view model with no GUI object references.
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
        correlation_scores: Optional ROI-level correlation QC scores.
        correlation_window_stop_default_seconds: Optional default stop time for
            enabling Plot-tab correlation QC.

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
            else response_plot_window_options_from_grouped(grouped)
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
        plot_data: Current response plot data.
        roi_indices: Zero-based ROI row indices to keep, in display order.

    Returns:
        Plot data with ROI labels, means, and SEMs filtered consistently for
        every epoch.

    Filtering is a presentation operation. It does not recompute responses or
    change the underlying scientific calculation.
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


def response_plot_window_options_from_grouped(
    grouped: GroupedRoiResponses,
) -> ResponseWindowOptions | None:
    """Return response-window options stored with grouped responses.

    Args:
        grouped: Grouped response object with optional response-window metadata.

    Returns:
        Response-window settings, or ``None`` when the grouped response object
        does not carry enough metadata.
    """
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
