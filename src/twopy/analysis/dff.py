"""Compute ROI-level delta F over F traces.

Inputs: background-corrected ROI fluorescence traces and gray-interleave frame
windows.
Outputs: ROI dF/F traces, fitted baselines, shared exponential tau, and
per-ROI amplitudes.

This module owns ROI-level dF/F calculation from twopy Python objects. It does
not decide which stimulus events are interleaves; callers pass explicit
interleave windows derived from converted data and photodiode alignment.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from twopy.analysis.background_subtraction import BackgroundCorrectedRoiTraces
from twopy.analysis.trials import FrameWindow

__all__ = [
    "DeltaFOverFFitMode",
    "RoiDeltaFOverF",
    "compute_roi_delta_f_over_f",
]

DeltaFOverFFitMode = Literal["robust", "source_bounds"]
DffMetadataValue = str | int | float | bool


@dataclass(frozen=True)
class RoiDeltaFOverF:
    """ROI-level dF/F and the fitted baseline used to compute it.

    Inputs: background-corrected ROI fluorescence and gray-interleave windows.
    Outputs: fluorescence, baseline, dF/F, ROI labels, shared tau, per-ROI
    amplitudes, and fit metadata.

    Keeping fluorescence and baseline beside dF/F makes the calculation easy to
    audit. The final trace is ``(fluorescence - baseline) / baseline``.
    """

    fluorescence: npt.NDArray[np.float64]
    baseline: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    labels: tuple[str, ...]
    start_frame: int
    stop_frame: int
    tau: float
    amplitudes: npt.NDArray[np.float64]
    interleave_frame_numbers: npt.NDArray[np.float64]
    interleave_fluorescence: npt.NDArray[np.float64]
    metadata: dict[str, DffMetadataValue]


def compute_roi_delta_f_over_f(
    traces: BackgroundCorrectedRoiTraces,
    interleave_windows: Sequence[FrameWindow],
    *,
    data_rate_hz: float,
    seconds_interleave_use: float | None = 1.0,
    fit_mode: DeltaFOverFFitMode = "robust",
) -> RoiDeltaFOverF:
    """Compute ROI-level dF/F from corrected fluorescence.

    Args:
        traces: Background-corrected ROI traces. ``corrected_values`` is treated
            as final ROI fluorescence ``F_roi(t)``.
        interleave_windows: Gray-interleave frame windows in the same absolute
            frame coordinates as ``traces``.
        data_rate_hz: Imaging frame rate in frames per second.
        seconds_interleave_use: Number of seconds to keep from the end of each
            interleave window. ``None`` uses the full window.
        fit_mode: Exponential fit bounds. ``"robust"`` keeps the shared tau
            bounded but leaves log-amplitude unbounded and ignores nonpositive
            fit samples. ``"source_bounds"`` uses bounded log-amplitude from the
            original analysis source for audit comparisons.

    Returns:
        ``RoiDeltaFOverF`` with dF/F values shaped ``(frames, rois)``.

    Raises:
        ValueError: If no interleave windows are supplied, windows fall outside
            the traces, or the fit has no positive finite baseline samples.

    The calculation uses the current twopy analysis contract: first use the
    background-corrected mean fluorescence inside each ROI, then fit one
    exponential decay constant shared by all ROIs from gray interleaves, then
    estimate one amplitude per ROI.
    """
    _validate_dff_inputs(
        traces=traces,
        interleave_windows=interleave_windows,
        data_rate_hz=data_rate_hz,
        seconds_interleave_use=seconds_interleave_use,
        fit_mode=fit_mode,
    )

    fluorescence = traces.corrected_values.astype(np.float64, copy=False)
    interleave_samples = _interleave_samples(
        fluorescence=fluorescence,
        trace_start_frame=traces.start_frame,
        interleave_windows=interleave_windows,
        data_rate_hz=data_rate_hz,
        seconds_interleave_use=seconds_interleave_use,
    )
    tau = _fit_shared_tau(
        frame_numbers=interleave_samples.frame_numbers,
        mean_fluorescence=interleave_samples.values.mean(axis=1),
        fit_mode=fit_mode,
    )
    amplitudes = _compute_roi_amplitudes(
        interleave_values=interleave_samples.values,
        interleave_frame_numbers=interleave_samples.frame_numbers,
        tau=tau,
    )
    baseline = _evaluate_baseline(
        frame_count=fluorescence.shape[0],
        amplitudes=amplitudes,
        tau=tau,
    )
    values = (fluorescence - baseline) / baseline

    return RoiDeltaFOverF(
        fluorescence=fluorescence,
        baseline=baseline,
        values=values,
        labels=traces.labels,
        start_frame=traces.start_frame,
        stop_frame=traces.stop_frame,
        tau=tau,
        amplitudes=amplitudes,
        interleave_frame_numbers=interleave_samples.frame_numbers,
        interleave_fluorescence=interleave_samples.values,
        metadata={
            "method": "shared_tau_roi_amplitude",
            "data_rate_hz": float(data_rate_hz),
            "seconds_interleave_use": (
                "full"
                if seconds_interleave_use is None
                else float(seconds_interleave_use)
            ),
            "interleave_window_count": len(interleave_windows),
            "fit_mode": fit_mode,
        },
    )


@dataclass(frozen=True)
class _InterleaveSamples:
    """Mean fluorescence samples from interleave windows.

    Inputs: gray-interleave frame windows and ROI fluorescence traces.
    Outputs: one mean fluorescence value per interleave window and ROI, plus
    the frame number used for exponential fitting.
    """

    frame_numbers: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]


def _validate_dff_inputs(
    *,
    traces: BackgroundCorrectedRoiTraces,
    interleave_windows: Sequence[FrameWindow],
    data_rate_hz: float,
    seconds_interleave_use: float | None,
    fit_mode: DeltaFOverFFitMode,
) -> None:
    """Validate the public dF/F inputs.

    Args:
        traces: Candidate background-corrected ROI traces.
        interleave_windows: Candidate baseline windows.
        data_rate_hz: Imaging frame rate.
        seconds_interleave_use: Optional seconds from each interleave end.
        fit_mode: Requested exponential fit mode.

    Returns:
        None.
    """
    if traces.corrected_values.ndim != 2:
        msg = "corrected_values must have shape (frames, rois)"
        raise ValueError(msg)
    if traces.corrected_values.shape[1] != len(traces.labels):
        msg = (
            "ROI labels must match corrected_values width; "
            f"got {len(traces.labels)} labels for "
            f"{traces.corrected_values.shape[1]} ROIs"
        )
        raise ValueError(msg)
    if len(interleave_windows) == 0:
        msg = "At least one interleave window is required for dF/F"
        raise ValueError(msg)
    if data_rate_hz <= 0:
        msg = f"data_rate_hz must be positive; got {data_rate_hz}"
        raise ValueError(msg)
    if seconds_interleave_use is not None and seconds_interleave_use <= 0:
        msg = (
            "seconds_interleave_use must be positive or None; "
            f"got {seconds_interleave_use}"
        )
        raise ValueError(msg)
    if fit_mode not in {"robust", "source_bounds"}:
        msg = f"Unknown dF/F fit_mode {fit_mode!r}"
        raise ValueError(msg)


def _interleave_samples(
    *,
    fluorescence: npt.NDArray[np.float64],
    trace_start_frame: int,
    interleave_windows: Sequence[FrameWindow],
    data_rate_hz: float,
    seconds_interleave_use: float | None,
) -> _InterleaveSamples:
    """Average the chosen part of each interleave window.

    Args:
        fluorescence: ROI fluorescence shaped ``(frames, rois)``.
        trace_start_frame: Absolute frame index for ``fluorescence[0]``.
        interleave_windows: Gray-interleave windows in absolute frame indices.
        data_rate_hz: Imaging frame rate.
        seconds_interleave_use: Optional seconds from each interleave end.

    Returns:
        Mean interleave fluorescence and one-based fit frame numbers.

    Each interleave is first narrowed to the last requested second, then the
    selected frames are averaged into one point per interleave. That keeps
    longer interleaves from dominating the exponential fit.
    """
    frame_numbers: list[float] = []
    values: list[npt.NDArray[np.float64]] = []
    for window in interleave_windows:
        local_start = window.start_frame - trace_start_frame
        local_stop = window.stop_frame - trace_start_frame
        _validate_interleave_local_range(
            window=window,
            local_start=local_start,
            local_stop=local_stop,
            frame_count=fluorescence.shape[0],
        )
        sample_start, sample_stop = _last_interleave_sample_range(
            local_start=local_start,
            local_stop=local_stop,
            data_rate_hz=data_rate_hz,
            seconds_interleave_use=seconds_interleave_use,
        )

        duration = sample_stop - sample_start
        # The fit uses one-based frame numbers centered on the averaged
        # half-open sample range: sample_start + 1 + N / 2.
        frame_numbers.append(float(sample_start + 1) + (duration / 2.0))
        values.append(fluorescence[sample_start:sample_stop, :].mean(axis=0))

    return _InterleaveSamples(
        frame_numbers=np.asarray(frame_numbers, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
    )


def _validate_interleave_local_range(
    *,
    window: FrameWindow,
    local_start: int,
    local_stop: int,
    frame_count: int,
) -> None:
    """Check that one interleave window can slice the trace array.

    Args:
        window: Window in absolute frame coordinates.
        local_start: Window start after subtracting trace start.
        local_stop: Window stop after subtracting trace start.
        frame_count: Number of trace frames available.

    Returns:
        None.
    """
    if local_start < 0 or local_stop > frame_count or local_start >= local_stop:
        msg = (
            f"Interleave window [{window.start_frame}, {window.stop_frame}) is "
            f"outside trace range with {frame_count} frames"
        )
        raise ValueError(msg)


def _last_interleave_sample_range(
    *,
    local_start: int,
    local_stop: int,
    data_rate_hz: float,
    seconds_interleave_use: float | None,
) -> tuple[int, int]:
    """Choose the final part of one interleave window.

    Args:
        local_start: Window start in local trace coordinates.
        local_stop: Window stop in local trace coordinates.
        data_rate_hz: Imaging frame rate.
        seconds_interleave_use: Optional seconds from the window end.

    Returns:
        Half-open local frame range to average.
    """
    if seconds_interleave_use is None:
        return local_start, local_stop

    frame_count = int(round(seconds_interleave_use * data_rate_hz))
    frame_count = max(frame_count, 1)
    duration = local_stop - local_start
    if frame_count > duration:
        msg = (
            "Requested interleave baseline is longer than a window: "
            f"requested_frames={frame_count}, window_frames={duration}"
        )
        raise ValueError(msg)
    return local_stop - frame_count, local_stop


def _fit_shared_tau(
    *,
    frame_numbers: npt.NDArray[np.float64],
    mean_fluorescence: npt.NDArray[np.float64],
    fit_mode: DeltaFOverFFitMode,
) -> float:
    """Fit one exponential decay constant shared across ROIs.

    Args:
        frame_numbers: One-based frame number for each interleave sample.
        mean_fluorescence: Mean fluorescence across ROIs for each sample.
        fit_mode: Fit bounds to use.

    Returns:
        Shared exponential ``tau``.

    The fit model is ``exp(tau * t + b)`` on the ROI-averaged interleave trace.
    Only ``tau`` is shared; per-ROI amplitudes are computed in a separate
    transparent step. ``robust`` mode handles nonpositive outliers defensively;
    ``source_bounds`` preserves the original bounded log-amplitude fit for
    audit comparisons.
    """
    if fit_mode == "source_bounds":
        return _fit_shared_tau_with_source_bounds(
            frame_numbers=frame_numbers,
            mean_fluorescence=mean_fluorescence,
        )
    return _fit_shared_tau_robust(
        frame_numbers=frame_numbers,
        mean_fluorescence=mean_fluorescence,
    )


def _fit_shared_tau_robust(
    *,
    frame_numbers: npt.NDArray[np.float64],
    mean_fluorescence: npt.NDArray[np.float64],
) -> float:
    """Fit shared tau with defensive handling for nonpositive samples.

    Args:
        frame_numbers: One-based frame number for each interleave sample.
        mean_fluorescence: Mean fluorescence across ROIs for each sample.

    Returns:
        Shared exponential ``tau``.

    This is the default fit path because background correction can create
    nonpositive values. It preserves the source tau bounds but avoids failing
    the whole trace because one interleave average is nonpositive.
    """
    finite_positive = np.isfinite(mean_fluorescence) & (mean_fluorescence > 0)
    if not np.any(finite_positive):
        msg = "Cannot fit dF/F baseline: interleave fluorescence has no positive values"
        raise ValueError(msg)
    if np.count_nonzero(finite_positive) == 1:
        return 0.0

    x = frame_numbers[finite_positive]
    y = mean_fluorescence[finite_positive]
    initial_log_amplitude = float(np.log(y[0]))

    def residual(parameters: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return direct exponential residuals for SciPy least squares."""
        tau = parameters[0]
        log_amplitude = parameters[1]
        return np.exp((tau * x) + log_amplitude) - y

    fit = least_squares(
        residual,
        x0=np.asarray([0.0, initial_log_amplitude], dtype=np.float64),
        bounds=(
            np.asarray([-0.01, -np.inf], dtype=np.float64),
            np.asarray([0.01, np.inf], dtype=np.float64),
        ),
    )
    return float(fit.x[0])


def _fit_shared_tau_with_source_bounds(
    *,
    frame_numbers: npt.NDArray[np.float64],
    mean_fluorescence: npt.NDArray[np.float64],
) -> float:
    """Fit shared tau using the original bounded log-amplitude contract.

    Args:
        frame_numbers: One-based frame number for each interleave sample.
        mean_fluorescence: Mean fluorescence across ROIs for each sample.

    Returns:
        Shared exponential ``tau``.

    Raises:
        ValueError: If any fit sample is nonfinite or the first sample cannot
            define a positive log-amplitude bound.

    This mode exists for audit comparisons. It intentionally does not filter
    nonpositive samples because doing so would change the source-bounds
    contract.
    """
    if np.any(~np.isfinite(mean_fluorescence)):
        msg = "source_bounds dF/F fit requires finite interleave fluorescence"
        raise ValueError(msg)
    if mean_fluorescence.size == 0:
        msg = "source_bounds dF/F fit requires at least one interleave sample"
        raise ValueError(msg)
    first_value = float(mean_fluorescence[0])
    if first_value <= 1.0:
        msg = (
            "source_bounds dF/F fit requires first interleave fluorescence "
            "greater than 1.0 to define an ordered log-amplitude bound; "
            f"got {first_value}"
        )
        raise ValueError(msg)
    if np.any(mean_fluorescence <= 0):
        msg = "source_bounds dF/F fit requires positive interleave fluorescence"
        raise ValueError(msg)
    if mean_fluorescence.size == 1:
        return 0.0

    initial_log_amplitude = float(np.log(first_value))
    upper_log_amplitude = 10.0 * initial_log_amplitude

    def residual(parameters: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return direct exponential residuals for source-bound least squares."""
        tau = parameters[0]
        log_amplitude = parameters[1]
        return np.exp((tau * frame_numbers) + log_amplitude) - mean_fluorescence

    fit = least_squares(
        residual,
        x0=np.asarray([0.0, initial_log_amplitude], dtype=np.float64),
        bounds=(
            np.asarray([-0.01, 0.0], dtype=np.float64),
            np.asarray([0.01, upper_log_amplitude], dtype=np.float64),
        ),
    )
    return float(fit.x[0])


def _compute_roi_amplitudes(
    *,
    interleave_values: npt.NDArray[np.float64],
    interleave_frame_numbers: npt.NDArray[np.float64],
    tau: float,
) -> npt.NDArray[np.float64]:
    """Estimate one exponential amplitude per ROI.

    Args:
        interleave_values: Mean interleave fluorescence shaped
            ``(interleaves, rois)``.
        interleave_frame_numbers: One-based fit frame number per interleave.
        tau: Shared exponential decay constant.

    Returns:
        One amplitude per ROI.
    """
    decay = np.exp(tau * interleave_frame_numbers)[:, None]
    if tau == 0:
        return interleave_values.mean(axis=0)
    return (interleave_values / decay).mean(axis=0)


def _evaluate_baseline(
    *,
    frame_count: int,
    amplitudes: npt.NDArray[np.float64],
    tau: float,
) -> npt.NDArray[np.float64]:
    """Evaluate the fitted baseline for every frame and ROI.

    Args:
        frame_count: Number of fluorescence frames.
        amplitudes: Per-ROI amplitudes.
        tau: Shared exponential decay constant.

    Returns:
        Baseline shaped ``(frames, rois)``.
    """
    frame_numbers = np.arange(1, frame_count + 1, dtype=np.float64)
    return amplitudes[None, :] * np.exp(tau * frame_numbers[:, None])
