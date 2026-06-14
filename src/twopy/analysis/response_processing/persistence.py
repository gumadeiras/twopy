"""Persist response-processing settings and QC audit data.

Inputs: typed response-processing options plus optional normalization and ROI
correlation audit data.
Outputs: one inspectable HDF5 group that analysis persistence can embed.

This module keeps the response-processing audit contract near the processing
types while the top-level analysis persistence module decides when to save it.
"""

from dataclasses import dataclass

import h5py

from twopy.analysis.response_processing.apply import RoiNormalizationFactors
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
)
from twopy.analysis.response_processing.qc import RoiCorrelationScores
from twopy.hdf5_utils import (
    plain_hdf5_attr_value,
    write_string_dataset,
)
from twopy.hdf5_utils import (
    read_string_dataset as _read_string_dataset,
)
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
_NORMALIZATION_METHODS: tuple[NormalizationMethod, ...] = ("none", "epoch_abs_peak")
_CORRELATION_REFERENCES: tuple[CorrelationFilterReference, ...] = (
    "none",
    "epoch_mean",
    "epoch_peak",
)


@dataclass(frozen=True)
class PersistedResponseProcessing:
    """Response-processing audit data loaded from HDF5.

    Inputs: one ``response_processing`` HDF5 group.
    Outputs: typed options plus optional factors, ROI correlation scores, and a
        warning when an old file uses a setting twopy no longer uses.
    """

    options: ResponseProcessingOptions
    normalization_factors: RoiNormalizationFactors | None
    correlation_scores: RoiCorrelationScores | None
    load_warning: str | None = None


def write_response_processing_group(
    group: h5py.Group,
    *,
    options: ResponseProcessingOptions,
    normalization_factors: RoiNormalizationFactors | None = None,
    correlation_scores: RoiCorrelationScores | None = None,
) -> None:
    """Write response-processing settings and optional QC scores.

    Args:
        group: Destination ``response_processing`` HDF5 group.
        options: Processing options used for the saved analysis output.
        normalization_factors: Optional per-ROI normalization factors used for
            response-size normalization.
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

    normalization = group.create_group("normalization")
    normalization.attrs["method"] = options.normalization.method
    normalization.attrs["epoch_number"] = _optional_int_attr(
        options.normalization.epoch_number,
    )
    normalization.attrs["epoch_name"] = _optional_text_attr(
        options.normalization.epoch_name,
    )
    if normalization_factors is not None:
        write_string_dataset(
            normalization,
            "roi_labels",
            normalization_factors.roi_labels,
        )
        normalization.create_dataset("factors", data=normalization_factors.factors)

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
        write_string_dataset(scores_group, "roi_labels", correlation_scores.roi_labels)
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
    normalization_options, normalization_load_warning = (
        _read_normalization_options(
            group["normalization"],
        )
        if "normalization" in group
        else (NormalizationOptions(), None)
    )
    options = ResponseProcessingOptions(
        smoothing=_read_smoothing_options(group["smoothing"]),
        low_pass=_read_low_pass_options(group["low_pass"]),
        normalization=normalization_options,
        correlation_filter=_read_correlation_filter_options(
            group["correlation_filter"],
        ),
    )
    return PersistedResponseProcessing(
        options=options,
        normalization_factors=(
            _read_normalization_factors(
                group["normalization"],
                options=options.normalization,
            )
            if (
                "normalization" in group
                and "factors" in group["normalization"]
                and options.normalization.method != "none"
            )
            else None
        ),
        correlation_scores=(
            _read_correlation_scores(group["correlation_scores"])
            if "correlation_scores" in group
            else None
        ),
        load_warning=normalization_load_warning,
    )


def _read_smoothing_options(group: h5py.Group) -> SmoothingOptions:
    """Read smoothing settings from one HDF5 group."""
    return SmoothingOptions(
        method=require_string_choice(
            plain_hdf5_attr_value(group.attrs["method"], scalar_only=True),
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
            plain_hdf5_attr_value(group.attrs["method"], scalar_only=True),
            name="response_processing low-pass method",
            allowed=_LOW_PASS_METHODS,
        ),
        cutoff_hz=_optional_float_from_attr(group.attrs["cutoff_hz"]),
        order=int(group.attrs["order"]),
    )


def _read_normalization_options(
    group: h5py.Group,
) -> tuple[NormalizationOptions, str | None]:
    """Read normalization settings and report old settings twopy no longer uses."""
    raw_method = plain_hdf5_attr_value(group.attrs["method"], scalar_only=True)
    if raw_method == "epoch_peak":
        return (
            NormalizationOptions(
                method="none",
                epoch_number=_optional_int_from_attr(group.attrs["epoch_number"]),
                epoch_name=_optional_text_from_attr(group.attrs["epoch_name"]),
            ),
            (
                "This saved analysis used old positive-peak normalization. "
                "Twopy is showing the saved responses as stored. Recompute to use "
                "strongest-response normalization."
            ),
        )
    return (
        NormalizationOptions(
            method=require_string_choice(
                raw_method,
                name="response_processing normalization method",
                allowed=_NORMALIZATION_METHODS,
            ),
            epoch_number=_optional_int_from_attr(group.attrs["epoch_number"]),
            epoch_name=_optional_text_from_attr(group.attrs["epoch_name"]),
        ),
        None,
    )


def _read_correlation_filter_options(group: h5py.Group) -> CorrelationFilterOptions:
    """Read correlation-filter settings from one HDF5 group."""
    return CorrelationFilterOptions(
        reference=require_string_choice(
            plain_hdf5_attr_value(group.attrs["reference"], scalar_only=True),
            name="response_processing correlation reference",
            allowed=_CORRELATION_REFERENCES,
        ),
        minimum_correlation=float(group.attrs["minimum_correlation"]),
        window_seconds=_window_seconds_from_attrs(group),
    )


def _read_normalization_factors(
    group: h5py.Group,
    *,
    options: NormalizationOptions,
) -> RoiNormalizationFactors:
    """Read persisted response-size normalization factors."""
    return RoiNormalizationFactors(
        roi_labels=_read_string_dataset(group, "roi_labels"),
        factors=require_float64_array(
            group["factors"][()],
            name="response_processing/normalization/factors",
            ndim=1,
        ),
        epoch_number=int(options.epoch_number or 0),
        epoch_name=options.epoch_name,
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
            plain_hdf5_attr_value(group.attrs["reference"], scalar_only=True),
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


def _optional_int_attr(value: int | None) -> str | int:
    """Return an HDF5-friendly optional integer attribute value."""
    return "none" if value is None else int(value)


def _optional_text_attr(value: str | None) -> str:
    """Return an HDF5-friendly optional text attribute value."""
    return "none" if value is None else value


def _optional_float_from_attr(value: object) -> float | None:
    """Read an optional float value written by ``_optional_float_attr``."""
    plain_value = plain_hdf5_attr_value(value, scalar_only=True)
    if plain_value == "none":
        return None
    return float(plain_value)


def _optional_int_from_attr(value: object) -> int | None:
    """Read an optional integer value written by ``_optional_int_attr``."""
    plain_value = plain_hdf5_attr_value(value, scalar_only=True)
    if plain_value == "none":
        return None
    return int(plain_value)


def _optional_text_from_attr(value: object) -> str | None:
    """Read an optional text value written by ``_optional_text_attr``."""
    plain_value = plain_hdf5_attr_value(value, scalar_only=True)
    if plain_value == "none":
        return None
    return str(plain_value)
