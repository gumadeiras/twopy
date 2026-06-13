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
    NormalizationOptions,
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
from twopy.analysis.responses import (
    GroupedRoiResponses,
    RoiResponseTrial,
    summarize_roi_response_trials,
    validate_grouped_roi_responses,
)

__all__ = [
    "apply_correlation_filter_to_grouped_roi_responses",
    "normalize_grouped_roi_responses_by_epoch_abs_peak",
    "mask_grouped_roi_responses_by_included_rois",
    "RoiNormalizationFactors",
    "ResponseProcessingResult",
    "process_grouped_roi_responses",
    "process_roi_delta_f_over_f",
]

_NORMALIZATION_MIN_PEAK = 1e-12


@dataclass(frozen=True)
class RoiNormalizationFactors:
    """Per-ROI scale factors used for response-size normalization.

    Inputs: grouped response trials and one enabled normalization option.
    Outputs: the positive response size used to scale each ROI.
    """

    roi_labels: tuple[str, ...]
    factors: npt.NDArray[np.float64]
    epoch_number: int
    epoch_name: str | None


@dataclass(frozen=True)
class ResponseProcessingResult:
    """Processed grouped responses plus optional QC scores.

    Inputs: grouped trial responses and one processing option object.
    Outputs: grouped responses with derived values, the options used to create
    them, and optional ROI correlation scores.
    """

    grouped_responses: GroupedRoiResponses
    options: ResponseProcessingOptions
    normalization_factors: RoiNormalizationFactors | None
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
        options: Processing options. Normalization and correlation filtering
            are ignored here because they need grouped repeated-trial responses.
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
    validate_grouped_roi_responses(grouped)
    processed = _copy_grouped_with_processed_values(grouped, options=options)
    processed, normalization_factors = (
        normalize_grouped_roi_responses_by_epoch_abs_peak(
            processed,
            options=options.normalization,
        )
    )
    processed, correlation_scores = apply_correlation_filter_to_grouped_roi_responses(
        processed,
        options=options,
    )
    return ResponseProcessingResult(
        grouped_responses=processed,
        options=options,
        normalization_factors=normalization_factors,
        correlation_scores=correlation_scores,
    )


def normalize_grouped_roi_responses_by_epoch_abs_peak(
    grouped: GroupedRoiResponses,
    *,
    options: NormalizationOptions,
) -> tuple[GroupedRoiResponses, RoiNormalizationFactors | None]:
    """Scale each ROI by its strongest average response in the selected epoch.

    Args:
        grouped: Grouped ROI responses to normalize.
        options: Normalization settings.

    Returns:
        ``(grouped_responses, factors)``. ``factors`` is ``None`` when
        normalization is disabled.

    Raises:
        ValueError: If the selected epoch is absent or any ROI lacks a finite
        nonzero response in that epoch.
    """
    validate_grouped_roi_responses(grouped)
    if options.method not in {"none", "epoch_abs_peak"}:
        msg = f"Unknown normalization method {options.method!r}"
        raise ValueError(msg)
    if options.method == "none":
        return grouped, None
    if options.epoch_number is None:
        msg = "response-size normalization requires an epoch_number"
        raise ValueError(msg)
    factors = _normalization_factors(grouped, options=options)
    trials = tuple(
        _copy_trial_with_values(trial, trial.values / factors.factors)
        for trial in grouped.trials
    )
    return (
        GroupedRoiResponses(
            roi_labels=grouped.roi_labels,
            data_rate_hz=grouped.data_rate_hz,
            trials=trials,
            pre_window_seconds=grouped.pre_window_seconds,
            post_window_seconds=grouped.post_window_seconds,
            response_window_auto=grouped.response_window_auto,
        ),
        factors,
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
    validate_grouped_roi_responses(grouped)
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
    validate_grouped_roi_responses(grouped)
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
        response_window_auto=grouped.response_window_auto,
    )


def _normalization_factors(
    grouped: GroupedRoiResponses,
    *,
    options: NormalizationOptions,
) -> RoiNormalizationFactors:
    """Return each ROI's selected-epoch response size."""
    selected_trials = tuple(
        trial
        for trial in grouped.trials
        if trial.epoch_number == options.epoch_number
        and (options.epoch_name is None or trial.epoch_name == options.epoch_name)
    )
    if len(selected_trials) == 0:
        label = (
            f"{options.epoch_number}: {options.epoch_name}"
            if options.epoch_name is not None
            else str(options.epoch_number)
        )
        msg = f"No trials matched normalization epoch {label}"
        raise ValueError(msg)
    summary = summarize_roi_response_trials(
        selected_trials,
        roi_labels=grouped.roi_labels,
        data_rate_hz=grouped.data_rate_hz,
    )
    factors = np.nanmax(np.abs(summary.mean_values), axis=1)
    bad_indices = tuple(
        index
        for index, factor in enumerate(factors)
        if not np.isfinite(factor) or factor <= _NORMALIZATION_MIN_PEAK
    )
    if len(bad_indices) > 0:
        bad_labels = ", ".join(grouped.roi_labels[index] for index in bad_indices)
        msg = (
            "Normalization epoch must have a nonzero response for every ROI; "
            f"invalid ROI labels: {bad_labels}"
        )
        raise ValueError(msg)
    return RoiNormalizationFactors(
        roi_labels=grouped.roi_labels,
        factors=factors.astype(np.float64, copy=False),
        epoch_number=int(options.epoch_number or 0),
        epoch_name=options.epoch_name,
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
        response_window_auto=grouped.response_window_auto,
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
        "response_processing_normalization_method": options.normalization.method,
        "response_processing_normalization_epoch_number": (
            "none"
            if options.normalization.epoch_number is None
            else int(options.normalization.epoch_number)
        ),
        "response_processing_normalization_epoch_name": (
            "none"
            if options.normalization.epoch_name is None
            else options.normalization.epoch_name
        ),
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
