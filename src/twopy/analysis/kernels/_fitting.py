"""Primitive temporal-kernel fitting helpers.

Inputs: regular stimulus samples plus sparse ROI response samples.
Outputs: temporal kernel weights, lag times, and actual fitted sample counts.

This module owns only numeric fitting. Recording and stimulus-table assembly
stay in the package-level workflow and stimulus helper module.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

KernelFitMethod = Literal["ols", "xcorr"]


@dataclass(frozen=True)
class KernelFit:
    """One temporal kernel plus the number of response samples used."""

    kernel: npt.NDArray[np.float64]
    time_seconds: npt.NDArray[np.float64]
    response_sample_count: int


@dataclass(frozen=True)
class KernelFitsByRoi:
    """Temporal kernels for multiple ROIs fitted to the same stimulus stream."""

    kernels: npt.NDArray[np.float64]
    time_seconds: npt.NDArray[np.float64]
    response_sample_counts: npt.NDArray[np.int64]


def fit_stimulus_kernel(
    stimulus_times: npt.NDArray[np.float64],
    stimulus: npt.NDArray[np.float64],
    response_times: npt.NDArray[np.float64],
    response: npt.NDArray[np.float64],
    *,
    num_stim_past: int,
    num_stim_future: int,
    method: KernelFitMethod = "ols",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit one temporal kernel from a regular stimulus and sparse responses.

    Args:
        stimulus_times: Strictly increasing regular stimulus sample times.
        stimulus: Stimulus values at ``stimulus_times``.
        response_times: Response sample times in the same time base.
        response: Response values at ``response_times``.
        num_stim_past: Number of samples before each response included.
        num_stim_future: Number of samples after each response included.
        method: ``"ols"`` or ``"xcorr"``.

    Returns:
        ``(kernel, time_seconds)``. The kernel time axis runs from
        ``-num_stim_future * dt`` through ``num_stim_past * dt``.

    This mirrors the MATLAB filter convention: the left side of the kernel is
    future stimulus samples and should usually stay near zero.
    """
    result = fit_stimulus_kernel_with_count(
        stimulus_times,
        stimulus,
        response_times,
        response,
        num_stim_past=num_stim_past,
        num_stim_future=num_stim_future,
        method=method,
    )
    return result.kernel, result.time_seconds


def fit_stimulus_kernel_with_count(
    stimulus_times: npt.NDArray[np.float64],
    stimulus: npt.NDArray[np.float64],
    response_times: npt.NDArray[np.float64],
    response: npt.NDArray[np.float64],
    *,
    num_stim_past: int,
    num_stim_future: int,
    method: KernelFitMethod = "ols",
) -> KernelFit:
    """Fit one temporal kernel and report actual response samples used."""
    _validate_fit_inputs(
        stimulus_times=stimulus_times,
        stimulus=stimulus,
        response_times=response_times,
        response=response,
        num_stim_past=num_stim_past,
        num_stim_future=num_stim_future,
        method=method,
    )
    stimulus_indices = _nearest_stimulus_indices(
        stimulus_times=stimulus_times,
        response_times=response_times,
    )
    valid = _valid_response_sample_mask(
        stimulus_indices=stimulus_indices,
        stimulus_size=stimulus.size,
        response=response,
        num_stim_past=num_stim_past,
        num_stim_future=num_stim_future,
    )
    if not np.any(valid):
        msg = "No response samples have a complete stimulus history/future window."
        raise ValueError(msg)
    valid_indices = stimulus_indices[valid]
    centered_response = response[valid] - float(np.nanmean(response[valid]))
    design = _stimulus_design_matrix(
        stimulus,
        valid_indices,
        num_stim_past=num_stim_past,
        num_stim_future=num_stim_future,
    )
    if method == "ols":
        kernel = np.linalg.lstsq(design, centered_response, rcond=None)[0]
    elif method == "xcorr":
        kernel = centered_response @ design / centered_response.size
    else:
        msg = f"Unknown kernel fit method {method!r}."
        raise ValueError(msg)
    time_seconds = _kernel_time_axis(
        stimulus_times,
        num_stim_past=num_stim_past,
        num_stim_future=num_stim_future,
    )
    return KernelFit(
        kernel=kernel.astype(np.float64, copy=False),
        time_seconds=time_seconds.astype(np.float64, copy=False),
        response_sample_count=int(np.count_nonzero(valid)),
    )


def fit_stimulus_kernel_by_roi(
    stimulus_times: npt.NDArray[np.float64],
    stimulus: npt.NDArray[np.float64],
    response_times_by_roi: Sequence[npt.NDArray[np.float64]],
    responses_by_roi: Sequence[npt.NDArray[np.float64]],
    *,
    num_stim_past: int,
    num_stim_future: int,
    method: KernelFitMethod,
) -> KernelFitsByRoi:
    """Fit the same stimulus stream against every ROI response."""
    kernels: list[npt.NDArray[np.float64]] = []
    counts: list[int] = []
    time_seconds: npt.NDArray[np.float64] | None = None
    for response_times, responses in zip(
        response_times_by_roi,
        responses_by_roi,
        strict=True,
    ):
        fit = fit_stimulus_kernel_with_count(
            stimulus_times,
            stimulus,
            response_times,
            responses,
            num_stim_past=num_stim_past,
            num_stim_future=num_stim_future,
            method=method,
        )
        if time_seconds is None:
            time_seconds = fit.time_seconds
        elif not np.array_equal(time_seconds, fit.time_seconds):
            msg = "ROI kernel time axes differ."
            raise ValueError(msg)
        kernels.append(fit.kernel)
        counts.append(fit.response_sample_count)
    if time_seconds is None:
        msg = "At least one ROI is required for kernel fitting."
        raise ValueError(msg)
    return KernelFitsByRoi(
        kernels=np.stack(kernels, axis=0),
        time_seconds=time_seconds,
        response_sample_counts=np.asarray(counts, dtype=np.int64),
    )


def _validate_fit_inputs(
    *,
    stimulus_times: npt.NDArray[np.float64],
    stimulus: npt.NDArray[np.float64],
    response_times: npt.NDArray[np.float64],
    response: npt.NDArray[np.float64],
    num_stim_past: int,
    num_stim_future: int,
    method: KernelFitMethod,
) -> None:
    """Validate the primitive filter inputs."""
    if stimulus_times.ndim != 1 or stimulus.ndim != 1:
        msg = "stimulus_times and stimulus must be one-dimensional."
        raise ValueError(msg)
    if response_times.ndim != 1 or response.ndim != 1:
        msg = "response_times and response must be one-dimensional."
        raise ValueError(msg)
    if stimulus_times.shape != stimulus.shape:
        msg = "stimulus_times and stimulus must have the same length."
        raise ValueError(msg)
    if response_times.shape != response.shape:
        msg = "response_times and response must have the same length."
        raise ValueError(msg)
    if stimulus_times.size < 2:
        msg = "At least two stimulus samples are required."
        raise ValueError(msg)
    if response_times.size == 0:
        msg = "At least one response sample is required."
        raise ValueError(msg)
    if num_stim_past < 0 or num_stim_future < 0:
        msg = "num_stim_past and num_stim_future must be nonnegative."
        raise ValueError(msg)
    if method not in {"ols", "xcorr"}:
        msg = f"Unknown kernel fit method {method!r}."
        raise ValueError(msg)
    validate_regular_stimulus_times(stimulus_times, context="kernel stimulus")
    if not np.all(np.diff(response_times) >= 0):
        msg = "response_times must be sorted."
        raise ValueError(msg)


def _nearest_stimulus_indices(
    *,
    stimulus_times: npt.NDArray[np.float64],
    response_times: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """Return zero-based stimulus indices nearest to each response time."""
    stimulus_clock = np.polyfit(
        np.arange(1, stimulus_times.size + 1, dtype=np.float64),
        stimulus_times,
        deg=1,
    )
    stim_dt = float(stimulus_clock[0])
    time_offset = float(stimulus_clock[1])
    return np.rint((response_times - time_offset) / stim_dt).astype(np.int64) - 1


def _valid_response_sample_mask(
    *,
    stimulus_indices: npt.NDArray[np.int64],
    stimulus_size: int,
    response: npt.NDArray[np.float64],
    num_stim_past: int,
    num_stim_future: int,
) -> npt.NDArray[np.bool_]:
    """Return response samples that have complete stimulus context."""
    return (
        (stimulus_indices - num_stim_past >= 0)
        & (stimulus_indices + num_stim_future < stimulus_size)
        & np.isfinite(response)
    )


def _stimulus_design_matrix(
    stimulus: npt.NDArray[np.float64],
    stimulus_indices: npt.NDArray[np.int64],
    *,
    num_stim_past: int,
    num_stim_future: int,
) -> npt.NDArray[np.float64]:
    """Return a design matrix ordered from future samples to past samples."""
    offsets = np.arange(
        num_stim_future,
        -num_stim_past - 1,
        -1,
        dtype=np.int64,
    )
    return stimulus[stimulus_indices[:, None] + offsets[None, :]]


def _kernel_time_axis(
    stimulus_times: npt.NDArray[np.float64],
    *,
    num_stim_past: int,
    num_stim_future: int,
) -> npt.NDArray[np.float64]:
    """Return the kernel lag axis in seconds."""
    stim_dt = stimulus_dt(stimulus_times)
    return stim_dt * np.arange(
        -num_stim_future,
        num_stim_past + 1,
        dtype=np.float64,
    )


def validate_regular_stimulus_times(
    times: npt.NDArray[np.float64],
    *,
    context: str,
) -> None:
    """Check that stimulus sample times are strictly increasing and regular."""
    if times.ndim != 1:
        msg = f"{context} times must be one-dimensional."
        raise ValueError(msg)
    diffs = np.diff(times)
    if not np.all(diffs > 0):
        msg = f"{context} times must be strictly increasing."
        raise ValueError(msg)
    mean_dt = float(np.mean(diffs))
    if mean_dt <= 0:
        msg = f"{context} sample interval must be positive."
        raise ValueError(msg)
    if float(np.std(diffs)) / mean_dt >= 0.1:
        msg = f"{context} times are not regular enough for kernel fitting."
        raise ValueError(msg)


def stimulus_dt(times: npt.NDArray[np.float64]) -> float:
    """Return the mean stimulus sample interval."""
    return float(np.mean(np.diff(times)))
