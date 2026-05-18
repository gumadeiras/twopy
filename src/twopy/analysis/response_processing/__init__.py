"""Core response post-processing helpers.

Inputs: ROI dF/F traces, grouped trial responses, and explicit processing
options.
Outputs: processed response objects plus optional quality-control scores.

This package is GUI-independent. It owns the math and validates option values
for response smoothing, low-pass filtering, and correlation-based response
quality control. Workflow code decides when to call it; persistence code decides
how to save the resulting audit trail.
"""

from twopy.analysis.response_processing.apply import (
    ResponseProcessingResult,
    RoiNormalizationFactors,
    apply_correlation_filter_to_grouped_roi_responses,
    mask_grouped_roi_responses_by_included_rois,
    normalize_grouped_roi_responses_by_epoch_peak,
    process_grouped_roi_responses,
    process_roi_delta_f_over_f,
)
from twopy.analysis.response_processing.options import (
    CorrelationFilterOptions,
    CorrelationFilterReference,
    CorrelationWindowSeconds,
    LowPassFilterMethod,
    LowPassFilterOptions,
    NormalizationMethod,
    NormalizationOptions,
    ResponseProcessingOptions,
    SmoothingMethod,
    SmoothingOptions,
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

__all__ = [
    "butterworth_low_pass",
    "CorrelationFilterOptions",
    "CorrelationFilterReference",
    "CorrelationWindowSeconds",
    "LowPassFilterMethod",
    "LowPassFilterOptions",
    "NormalizationMethod",
    "NormalizationOptions",
    "apply_correlation_filter_to_grouped_roi_responses",
    "mask_grouped_roi_responses_by_included_rois",
    "nan_aware_moving_average",
    "nan_aware_savgol_filter",
    "normalize_grouped_roi_responses_by_epoch_peak",
    "process_grouped_roi_responses",
    "process_roi_delta_f_over_f",
    "ResponseProcessingOptions",
    "ResponseProcessingResult",
    "RoiCorrelationScores",
    "RoiNormalizationFactors",
    "score_roi_correlations",
    "SmoothingMethod",
    "SmoothingOptions",
    "validate_response_processing_options",
]
