"""Apply response post-processing options to core analysis objects.

Inputs: ROI dF/F traces, grouped trial responses, and typed processing options.
Outputs: new analysis objects with processed values plus optional QC scores.

This module coordinates the pure signal kernels and QC scoring helpers. It does
not know how options were selected, where outputs are saved, or how plots are
drawn.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.response_processing.options import (
    ResponseProcessingOptions,
    validate_response_processing_options,
)
from twopy.analysis.response_processing.qc import (
    RoiCorrelationScores,
    score_roi_correlations,
)
from twopy.analysis.response_processing.signals import (
    butterworth_low_pass,
    nan_aware_moving_average,
    nan_aware_savgol_filter,
)
from twopy.analysis.responses import GroupedRoiResponses, RoiResponseTrial

__all__ = [
    "apply_correlation_filter_to_grouped_roi_responses",
    "mask_grouped_roi_responses_by_included_rois",
    "ResponseProcessingResult",
    "process_grouped_roi_responses",
    "process_roi_delta_f_over_f",
]


@dataclass(frozen=True)
class ResponseProcessingResult:
    """Processed grouped responses plus optional QC scores.

    Inputs: grouped trial responses and one processing option object.
    Outputs: grouped responses with derived values, the options used to create
    them, and optional ROI correlation scores.
    """

    grouped_responses: GroupedRoiResponses
    options: ResponseProcessingOptions
    correlation_scores: RoiCorrelationScores | None


def process_roi_delta_f_over_f(
    dff: RoiDeltaFOverF,
    *,
    options: ResponseProcessingOptions,
    data_rate_hz: float,
) -> RoiDeltaFOverF:
    """Apply smoothing and low-pass filtering to continuous ROI dF/F.

    Args:
        dff: Continuous ROI dF/F result.
        options: Processing options. Correlation filtering is ignored here
            because it needs grouped repeated-trial responses.
        data_rate_hz: Sampling rate in hertz.

    Returns:
        New ``RoiDeltaFOverF`` with processed ``values`` and original raw audit
        arrays preserved.

    Raises:
        ValueError: If options or dF/F shapes are invalid.

    This is the preferred stage for smoothing and low-pass filtering because it
    operates on continuous traces before trial slicing, avoiding per-trial edge
    artifacts.
    """
    _validate_dff_for_processing(dff)
    validate_response_processing_options(options, data_rate_hz=data_rate_hz)
    processed_values = _process_values(
        dff.values,
        options=options,
        data_rate_hz=data_rate_hz,
    )
    metadata = dict(dff.metadata)
    metadata.update(_processing_metadata(options))
    return RoiDeltaFOverF(
        fluorescence=dff.fluorescence,
        baseline=dff.baseline,
        values=processed_values,
        labels=dff.labels,
        start_frame=dff.start_frame,
        stop_frame=dff.stop_frame,
        tau=dff.tau,
        amplitudes=dff.amplitudes,
        baseline_frame_numbers=dff.baseline_frame_numbers,
        baseline_fluorescence=dff.baseline_fluorescence,
        metadata=metadata,
    )


def process_grouped_roi_responses(
    grouped: GroupedRoiResponses,
    *,
    options: ResponseProcessingOptions,
) -> ResponseProcessingResult:
    """Apply response post-processing to grouped trial responses.

    Args:
        grouped: Grouped ROI responses.
        options: Processing options.

    Returns:
        Processed grouped responses plus optional correlation scores.

    Raises:
        ValueError: If options or grouped response shapes are invalid.

    This grouped-response path is useful for saved responses or plot-only
    previews. New full workflow wiring should prefer continuous dF/F processing
    before grouping when smoothing or low-pass filtering are enabled.
    """
    validate_response_processing_options(options, data_rate_hz=grouped.data_rate_hz)
    _validate_grouped_for_processing(grouped)
    processed = _copy_grouped_with_processed_values(grouped, options=options)
    processed, correlation_scores = apply_correlation_filter_to_grouped_roi_responses(
        processed,
        options=options,
    )
    return ResponseProcessingResult(
        grouped_responses=processed,
        options=options,
        correlation_scores=correlation_scores,
    )


def apply_correlation_filter_to_grouped_roi_responses(
    grouped: GroupedRoiResponses,
    *,
    options: ResponseProcessingOptions,
) -> tuple[GroupedRoiResponses, RoiCorrelationScores | None]:
    """Apply only the correlation filter from a processing option object.

    Args:
        grouped: Grouped ROI responses to score and optionally mask.
        options: Complete processing options. Smoothing and low-pass settings
            are validated but not re-applied here.

    Returns:
        ``(grouped_responses, scores)``. ``scores`` is ``None`` when the
        correlation filter is disabled.

    This helper lets the standard workflow process continuous dF/F first, then
    perform trial-level quality control after grouping without accidentally
    smoothing or filtering the same traces a second time.
    """
    validate_response_processing_options(options, data_rate_hz=grouped.data_rate_hz)
    _validate_grouped_for_processing(grouped)
    if options.correlation_filter.reference not in {"epoch_mean", "epoch_peak"}:
        return grouped, None
    correlation_scores = score_roi_correlations(
        grouped,
        minimum_correlation=options.correlation_filter.minimum_correlation,
        reference=options.correlation_filter.reference,
        window_seconds=options.correlation_filter.window_seconds,
    )
    return (
        mask_grouped_roi_responses_by_included_rois(
            grouped,
            included_mask=correlation_scores.included_mask,
        ),
        correlation_scores,
    )


def mask_grouped_roi_responses_by_included_rois(
    grouped: GroupedRoiResponses,
    *,
    included_mask: npt.NDArray[np.bool_],
) -> GroupedRoiResponses:
    """Return grouped responses with excluded ROI columns set to NaN.

    Args:
        grouped: Grouped ROI responses to mask.
        included_mask: Boolean vector matching ``grouped.roi_labels``.

    Returns:
        New grouped responses with the same shape and labels.

    The mask is public because persisted QC scores should be able to hide ROIs
    during plot reloads without recomputing the scientific QC decision.
    """
    return _mask_excluded_rois(grouped, included_mask=included_mask)


def _process_values(
    values: npt.NDArray[np.float64],
    *,
    options: ResponseProcessingOptions,
    data_rate_hz: float,
) -> npt.NDArray[np.float64]:
    """Apply enabled signal transforms to frame-by-ROI values."""
    processed = values.astype(np.float64, copy=True)
    if options.smoothing.method == "moving_average":
        processed = nan_aware_moving_average(
            processed,
            window_frames=options.smoothing.window_frames,
        )
    if options.smoothing.method == "savgol":
        processed = nan_aware_savgol_filter(
            processed,
            window_frames=options.smoothing.window_frames,
            polynomial_order=options.smoothing.polynomial_order,
        )
    if options.low_pass.method == "butterworth":
        if options.low_pass.cutoff_hz is None:
            msg = "enabled low-pass filtering requires cutoff_hz"
            raise ValueError(msg)
        processed = butterworth_low_pass(
            processed,
            sample_rate_hz=data_rate_hz,
            cutoff_hz=options.low_pass.cutoff_hz,
            order=options.low_pass.order,
        )
    return processed


def _copy_grouped_with_processed_values(
    grouped: GroupedRoiResponses,
    *,
    options: ResponseProcessingOptions,
) -> GroupedRoiResponses:
    """Return grouped responses with each trial value matrix processed."""
    trials = tuple(
        _copy_trial_with_values(
            trial,
            _process_values(
                trial.values,
                options=options,
                data_rate_hz=grouped.data_rate_hz,
            ),
        )
        for trial in grouped.trials
    )
    return GroupedRoiResponses(
        roi_labels=grouped.roi_labels,
        data_rate_hz=grouped.data_rate_hz,
        trials=trials,
        pre_window_seconds=grouped.pre_window_seconds,
        post_window_seconds=grouped.post_window_seconds,
    )


def _mask_excluded_rois(
    grouped: GroupedRoiResponses,
    *,
    included_mask: npt.NDArray[np.bool_],
) -> GroupedRoiResponses:
    """Return grouped responses with excluded ROI columns set to NaN."""
    if included_mask.shape != (len(grouped.roi_labels),):
        msg = (
            "included_mask must match ROI label count; "
            f"got {included_mask.shape} for {len(grouped.roi_labels)} labels"
        )
        raise ValueError(msg)
    trials: list[RoiResponseTrial] = []
    for trial in grouped.trials:
        values = trial.values.copy()
        values[:, ~included_mask] = np.nan
        trials.append(_copy_trial_with_values(trial, values))
    return GroupedRoiResponses(
        roi_labels=grouped.roi_labels,
        data_rate_hz=grouped.data_rate_hz,
        trials=tuple(trials),
        pre_window_seconds=grouped.pre_window_seconds,
        post_window_seconds=grouped.post_window_seconds,
    )


def _copy_trial_with_values(
    trial: RoiResponseTrial,
    values: npt.NDArray[np.float64],
) -> RoiResponseTrial:
    """Return one trial response with replacement values."""
    return RoiResponseTrial(
        epoch_number=trial.epoch_number,
        epoch_name=trial.epoch_name,
        trial_index=trial.trial_index,
        window_index=trial.window_index,
        start_frame=trial.start_frame,
        stop_frame=trial.stop_frame,
        frame_numbers=trial.frame_numbers,
        time_seconds=trial.time_seconds,
        values=values,
    )


def _validate_dff_for_processing(dff: RoiDeltaFOverF) -> None:
    """Validate a dF/F object before value-only processing."""
    if dff.values.ndim != 2:
        msg = f"dF/F values must have shape (frames, rois); got {dff.values.shape}"
        raise ValueError(msg)
    if dff.values.shape[1] != len(dff.labels):
        msg = (
            "dF/F values width does not match labels: "
            f"{dff.values.shape[1]} values, {len(dff.labels)} labels"
        )
        raise ValueError(msg)


def _validate_grouped_for_processing(grouped: GroupedRoiResponses) -> None:
    """Validate grouped responses before value-only processing."""
    if grouped.data_rate_hz <= 0:
        msg = f"data_rate_hz must be positive; got {grouped.data_rate_hz}"
        raise ValueError(msg)
    for trial in grouped.trials:
        if trial.values.ndim != 2:
            msg = (
                f"trial values must have shape (frames, rois); got {trial.values.shape}"
            )
            raise ValueError(msg)
        if trial.values.shape[1] != len(grouped.roi_labels):
            msg = (
                "trial response width does not match ROI labels: "
                f"{trial.values.shape[1]} values, {len(grouped.roi_labels)} labels"
            )
            raise ValueError(msg)


def _processing_metadata(
    options: ResponseProcessingOptions,
) -> dict[str, str | int | float | bool]:
    """Return flat metadata for a processed dF/F result."""
    return {
        "response_processing_smoothing_method": options.smoothing.method,
        "response_processing_smoothing_window_frames": options.smoothing.window_frames,
        "response_processing_smoothing_polynomial_order": (
            options.smoothing.polynomial_order
        ),
        "response_processing_low_pass_method": options.low_pass.method,
        "response_processing_low_pass_cutoff_hz": (
            "none"
            if options.low_pass.cutoff_hz is None
            else float(options.low_pass.cutoff_hz)
        ),
        "response_processing_low_pass_order": options.low_pass.order,
        "response_processing_correlation_reference": (
            options.correlation_filter.reference
        ),
        "response_processing_minimum_correlation": (
            options.correlation_filter.minimum_correlation
        ),
        "response_processing_correlation_window_start_seconds": (
            "none"
            if options.correlation_filter.window_seconds[0] is None
            else float(options.correlation_filter.window_seconds[0])
        ),
        "response_processing_correlation_window_stop_seconds": (
            "none"
            if options.correlation_filter.window_seconds[1] is None
            else float(options.correlation_filter.window_seconds[1])
        ),
    }
