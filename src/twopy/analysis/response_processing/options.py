"""Typed options for response post-processing.

Inputs: user or script-selected processing parameters.
Outputs: immutable option objects validated before math is run.

These objects are plain dataclasses so GUI controls, scripts, and saved files
can all pass the same processing choices without importing Qt or HDF5 code.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "CorrelationFilterOptions",
    "CorrelationFilterReference",
    "CorrelationWindowSeconds",
    "LowPassFilterMethod",
    "LowPassFilterOptions",
    "NormalizationMethod",
    "NormalizationOptions",
    "ResponseProcessingOptions",
    "SmoothingMethod",
    "SmoothingOptions",
    "validate_response_processing_options",
]

SmoothingMethod = Literal["none", "moving_average", "savgol"]
LowPassFilterMethod = Literal["none", "butterworth"]
NormalizationMethod = Literal["none", "epoch_peak"]
CorrelationFilterReference = Literal["none", "epoch_mean", "epoch_peak"]
CorrelationWindowSeconds = tuple[float | None, float | None]


@dataclass(frozen=True)
class SmoothingOptions:
    """Response-trace smoothing settings.

    Inputs: a smoothing method and window length in imaging frames.
    Outputs: smoothing values passed to signal-processing code.

    ``method="none"`` leaves values unchanged. ``window_frames`` and
    ``polynomial_order`` are still kept explicit so saved options can report
    exactly what the user selected. ``method="savgol"`` defaults to a
    seven-frame, second-order Savitzky-Golay smoother because that preserves
    local peak shape better than a moving average.
    """

    method: SmoothingMethod = "none"
    window_frames: int = 7
    polynomial_order: int = 2


@dataclass(frozen=True)
class LowPassFilterOptions:
    """Low-pass filter settings for ROI response traces.

    Inputs: filter method, cutoff frequency, and Butterworth order.
    Outputs: low-pass values passed to filtering code.

    ``cutoff_hz`` is optional because disabled filters do not need a cutoff.
    Enabled Butterworth filters must provide a positive cutoff below Nyquist.
    """

    method: LowPassFilterMethod = "none"
    cutoff_hz: float | None = None
    order: int = 2


@dataclass(frozen=True)
class NormalizationOptions:
    """Epoch-peak normalization settings for grouped ROI responses.

    Inputs: a normalization method and the stimulus epoch used as the scale
    reference.
    Outputs: normalization values applied after trial grouping.

    ``method="none"`` leaves grouped responses unchanged. ``method="epoch_peak"``
    divides every response for each ROI by that ROI's mean-response peak in the
    selected epoch, so the selected epoch's plotted peak becomes one.
    """

    method: NormalizationMethod = "none"
    epoch_number: int | None = None
    epoch_name: str | None = None


@dataclass(frozen=True)
class CorrelationFilterOptions:
    """Correlation-based response quality-control settings.

    Inputs: reference strategy, accepted correlation, and relative time window.
    Outputs: ROI-level scores and inclusion masks when enabled.

    Correlation filtering should produce auditable scores and masks. Downstream
    callers can hide, mark, or exclude responses, but raw data should remain
    available elsewhere in the analysis record.

    ``window_seconds`` is a half-open ``(start, stop)`` interval on each trial's
    epoch-relative time axis. ``None`` on the stop side leaves it unbounded, so
    the default starts at epoch onset and uses the full stored epoch response.
    """

    reference: CorrelationFilterReference = "none"
    minimum_correlation: float = 0.0
    window_seconds: CorrelationWindowSeconds = (0.0, None)


@dataclass(frozen=True)
class ResponseProcessingOptions:
    """Complete response post-processing configuration.

    Inputs: optional smoothing, low-pass filtering, epoch-peak normalization,
    and correlation QC settings.
    Outputs: one immutable object used by workflow, plotting, and saved analysis
    metadata.
    """

    smoothing: SmoothingOptions = field(default_factory=SmoothingOptions)
    low_pass: LowPassFilterOptions = field(default_factory=LowPassFilterOptions)
    normalization: NormalizationOptions = field(default_factory=NormalizationOptions)
    correlation_filter: CorrelationFilterOptions = field(
        default_factory=CorrelationFilterOptions
    )


def validate_response_processing_options(
    options: ResponseProcessingOptions,
    *,
    data_rate_hz: float | None = None,
) -> None:
    """Validate one response-processing option object.

    Args:
        options: Candidate processing configuration.
        data_rate_hz: Optional sampling rate used to validate low-pass Nyquist
            bounds when the filter is enabled.

    Returns:
        None.

    Raises:
        ValueError: If any enabled option is invalid.
    """
    _validate_smoothing_options(options.smoothing)
    _validate_low_pass_options(options.low_pass, data_rate_hz=data_rate_hz)
    _validate_normalization_options(options.normalization)
    _validate_correlation_options(options.correlation_filter)


def _validate_smoothing_options(options: SmoothingOptions) -> None:
    """Validate smoothing options."""
    if options.method not in {"none", "moving_average", "savgol"}:
        msg = f"Unknown smoothing method {options.method!r}"
        raise ValueError(msg)
    if options.window_frames < 1:
        msg = f"smoothing window_frames must be at least 1; got {options.window_frames}"
        raise ValueError(msg)
    if options.polynomial_order < 0:
        msg = (
            "smoothing polynomial_order must be non-negative; "
            f"got {options.polynomial_order}"
        )
        raise ValueError(msg)
    if options.method == "savgol":
        if options.window_frames % 2 == 0:
            msg = (
                "Savitzky-Golay smoothing window_frames must be odd; "
                f"got {options.window_frames}"
            )
            raise ValueError(msg)
        if options.polynomial_order >= options.window_frames:
            msg = (
                "Savitzky-Golay polynomial_order must be below window_frames; "
                f"got order {options.polynomial_order} and window "
                f"{options.window_frames}"
            )
            raise ValueError(msg)


def _validate_low_pass_options(
    options: LowPassFilterOptions,
    *,
    data_rate_hz: float | None,
) -> None:
    """Validate low-pass filter options."""
    if options.method not in {"none", "butterworth"}:
        msg = f"Unknown low-pass method {options.method!r}"
        raise ValueError(msg)
    if options.order < 1:
        msg = f"low-pass order must be at least 1; got {options.order}"
        raise ValueError(msg)
    if options.method == "none":
        return
    if options.cutoff_hz is None or options.cutoff_hz <= 0:
        msg = "enabled low-pass filtering requires a positive cutoff_hz"
        raise ValueError(msg)
    if data_rate_hz is not None:
        if data_rate_hz <= 0:
            msg = f"data_rate_hz must be positive; got {data_rate_hz}"
            raise ValueError(msg)
        nyquist_hz = data_rate_hz / 2.0
        if options.cutoff_hz >= nyquist_hz:
            msg = (
                "low-pass cutoff_hz must be below Nyquist; "
                f"got {options.cutoff_hz} Hz for {data_rate_hz} Hz data"
            )
            raise ValueError(msg)


def _validate_normalization_options(options: NormalizationOptions) -> None:
    """Validate epoch-peak normalization options."""
    if options.method not in {"none", "epoch_peak"}:
        msg = f"Unknown normalization method {options.method!r}"
        raise ValueError(msg)
    if options.epoch_number is not None and options.epoch_number < 1:
        msg = f"normalization epoch_number must be positive; got {options.epoch_number}"
        raise ValueError(msg)
    if options.method == "epoch_peak" and options.epoch_number is None:
        msg = "epoch-peak normalization requires an epoch_number"
        raise ValueError(msg)


def _validate_correlation_options(options: CorrelationFilterOptions) -> None:
    """Validate correlation-filter options."""
    if options.reference not in {"none", "epoch_mean", "epoch_peak"}:
        msg = f"Unknown correlation reference {options.reference!r}"
        raise ValueError(msg)
    if not -1.0 <= options.minimum_correlation <= 1.0:
        msg = (
            "minimum_correlation must be between -1 and 1; "
            f"got {options.minimum_correlation}"
        )
        raise ValueError(msg)
    start_seconds, stop_seconds = options.window_seconds
    if start_seconds is not None and not np.isfinite(start_seconds):
        msg = f"correlation window start must be finite or None; got {start_seconds}"
        raise ValueError(msg)
    if stop_seconds is not None and not np.isfinite(stop_seconds):
        msg = f"correlation window stop must be finite or None; got {stop_seconds}"
        raise ValueError(msg)
    if (
        start_seconds is not None
        and stop_seconds is not None
        and start_seconds >= stop_seconds
    ):
        msg = (
            "correlation window start must be before stop; "
            f"got ({start_seconds}, {stop_seconds})"
        )
        raise ValueError(msg)
