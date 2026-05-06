"""Persist response-processing settings and QC audit data.

Inputs: typed response-processing options and optional ROI correlation scores.
Outputs: one inspectable HDF5 group that analysis persistence can embed.

This module keeps the response-processing audit contract near the processing
types while the top-level analysis persistence module decides when to save it.
"""

from dataclasses import dataclass

import h5py
import numpy as np

from twopy.analysis.response_processing.options import (
    CorrelationFilterOptions,
    CorrelationFilterReference,
    CorrelationWindowSeconds,
    LowPassFilterMethod,
    LowPassFilterOptions,
    ResponseProcessingOptions,
    SmoothingMethod,
    SmoothingOptions,
)
from twopy.analysis.response_processing.qc import RoiCorrelationScores
from twopy.typing_guards import (
    require_bool_array,
    require_float64_array,
    require_string_choice,
)

__all__ = [
    "PersistedResponseProcessing",
    "read_response_processing_group",
    "write_response_processing_group",
]

_SMOOTHING_METHODS: tuple[SmoothingMethod, ...] = (
    "none",
    "moving_average",
    "savgol",
)
_LOW_PASS_METHODS: tuple[LowPassFilterMethod, ...] = ("none", "butterworth")
_CORRELATION_REFERENCES: tuple[CorrelationFilterReference, ...] = (
    "none",
    "epoch_mean",
    "epoch_peak",
)


@dataclass(frozen=True)
class PersistedResponseProcessing:
    """Response-processing audit data loaded from HDF5.

    Inputs: one ``response_processing`` HDF5 group.
    Outputs: typed options plus optional ROI correlation scores.
    """

    options: ResponseProcessingOptions
    correlation_scores: RoiCorrelationScores | None


def write_response_processing_group(
    group: h5py.Group,
    *,
    options: ResponseProcessingOptions,
    correlation_scores: RoiCorrelationScores | None = None,
) -> None:
    """Write response-processing settings and optional QC scores.

    Args:
        group: Destination ``response_processing`` HDF5 group.
        options: Processing options used for the saved analysis output.
        correlation_scores: Optional ROI-level correlation QC result.

    Returns:
        None.
    """
    smoothing = group.create_group("smoothing")
    smoothing.attrs["method"] = options.smoothing.method
    smoothing.attrs["window_frames"] = options.smoothing.window_frames
    smoothing.attrs["polynomial_order"] = options.smoothing.polynomial_order

    low_pass = group.create_group("low_pass")
    low_pass.attrs["method"] = options.low_pass.method
    low_pass.attrs["cutoff_hz"] = _optional_float_attr(options.low_pass.cutoff_hz)
    low_pass.attrs["order"] = options.low_pass.order

    correlation = group.create_group("correlation_filter")
    correlation.attrs["reference"] = options.correlation_filter.reference
    correlation.attrs["minimum_correlation"] = (
        options.correlation_filter.minimum_correlation
    )
    correlation.attrs["window_start_seconds"] = _optional_float_attr(
        options.correlation_filter.window_seconds[0],
    )
    correlation.attrs["window_stop_seconds"] = _optional_float_attr(
        options.correlation_filter.window_seconds[1],
    )

    if correlation_scores is not None:
        scores_group = group.create_group("correlation_scores")
        scores_group.attrs["minimum_correlation"] = (
            correlation_scores.minimum_correlation
        )
        scores_group.attrs["reference"] = correlation_scores.reference
        scores_group.attrs["window_start_seconds"] = _optional_float_attr(
            correlation_scores.window_seconds[0],
        )
        scores_group.attrs["window_stop_seconds"] = _optional_float_attr(
            correlation_scores.window_seconds[1],
        )
        scores_group.create_dataset(
            "roi_labels",
            data=np.asarray(
                correlation_scores.roi_labels,
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        scores_group.create_dataset("scores", data=correlation_scores.scores)
        scores_group.create_dataset(
            "included_mask",
            data=correlation_scores.included_mask,
        )


def read_response_processing_group(group: h5py.Group) -> PersistedResponseProcessing:
    """Read persisted response-processing settings and QC scores.

    Args:
        group: HDF5 ``response_processing`` group.

    Returns:
        Typed processing options plus optional correlation scores.
    """
    options = ResponseProcessingOptions(
        smoothing=_read_smoothing_options(group["smoothing"]),
        low_pass=_read_low_pass_options(group["low_pass"]),
        correlation_filter=_read_correlation_filter_options(
            group["correlation_filter"],
        ),
    )
    return PersistedResponseProcessing(
        options=options,
        correlation_scores=(
            _read_correlation_scores(group["correlation_scores"])
            if "correlation_scores" in group
            else None
        ),
    )


def _read_smoothing_options(group: h5py.Group) -> SmoothingOptions:
    """Read smoothing settings from one HDF5 group."""
    return SmoothingOptions(
        method=require_string_choice(
            _plain_attr_value(group.attrs["method"]),
            name="response_processing smoothing method",
            allowed=_SMOOTHING_METHODS,
        ),
        window_frames=int(group.attrs["window_frames"]),
        polynomial_order=int(group.attrs["polynomial_order"]),
    )


def _read_low_pass_options(group: h5py.Group) -> LowPassFilterOptions:
    """Read low-pass settings from one HDF5 group."""
    return LowPassFilterOptions(
        method=require_string_choice(
            _plain_attr_value(group.attrs["method"]),
            name="response_processing low-pass method",
            allowed=_LOW_PASS_METHODS,
        ),
        cutoff_hz=_optional_float_from_attr(group.attrs["cutoff_hz"]),
        order=int(group.attrs["order"]),
    )


def _read_correlation_filter_options(group: h5py.Group) -> CorrelationFilterOptions:
    """Read correlation-filter settings from one HDF5 group."""
    return CorrelationFilterOptions(
        reference=require_string_choice(
            _plain_attr_value(group.attrs["reference"]),
            name="response_processing correlation reference",
            allowed=_CORRELATION_REFERENCES,
        ),
        minimum_correlation=float(group.attrs["minimum_correlation"]),
        window_seconds=_window_seconds_from_attrs(group),
    )


def _read_correlation_scores(group: h5py.Group) -> RoiCorrelationScores:
    """Read persisted ROI correlation QC scores."""
    return RoiCorrelationScores(
        roi_labels=_read_string_dataset(group, "roi_labels"),
        scores=require_float64_array(
            group["scores"][()],
            name="response_processing/correlation_scores/scores",
            ndim=1,
        ),
        included_mask=require_bool_array(
            group["included_mask"][()],
            name="response_processing/correlation_scores/included_mask",
            ndim=1,
        ),
        minimum_correlation=float(group.attrs["minimum_correlation"]),
        reference=require_string_choice(
            _plain_attr_value(group.attrs["reference"]),
            name="response_processing correlation scores reference",
            allowed=_CORRELATION_REFERENCES,
        ),
        window_seconds=_window_seconds_from_attrs(group),
    )


def _window_seconds_from_attrs(group: h5py.Group) -> CorrelationWindowSeconds:
    """Read an optional half-open correlation window from attributes."""
    return (
        _optional_float_from_attr(group.attrs["window_start_seconds"]),
        _optional_float_from_attr(group.attrs["window_stop_seconds"]),
    )


def _optional_float_attr(value: float | None) -> str | float:
    """Return an HDF5-friendly optional float attribute value."""
    return "none" if value is None else float(value)


def _optional_float_from_attr(value: object) -> float | None:
    """Read an optional float value written by ``_optional_float_attr``."""
    plain_value = _plain_attr_value(value)
    if plain_value == "none":
        return None
    return float(plain_value)


def _read_string_dataset(group: h5py.Group, name: str) -> tuple[str, ...]:
    """Read a UTF-8 string dataset from HDF5."""
    return tuple(_decode_string(value) for value in group[name][()])


def _decode_string(value: object) -> str:
    """Decode one HDF5 string scalar into Python text."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _plain_attr_value(value: object) -> str | int | float | bool:
    """Convert common HDF5 attribute scalars to plain Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        if isinstance(item, str | int | float | bool):
            return item
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)
