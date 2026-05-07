"""Compute ROI-level delta F over F traces.

Inputs: background-corrected ROI fluorescence traces and baseline frame
windows.
Outputs: ROI dF/F traces, fitted baselines, shared exponential tau, and
per-ROI amplitudes.

This module owns ROI-level dF/F calculation from twopy Python objects. It does
not decide which stimulus events are baseline epochs; callers pass explicit
baseline windows derived from converted data and photodiode alignment.
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

DeltaFOverFFitMode = Literal[
    "direct_bounded_tau",
    "direct_bounded_tau_and_log_amplitude",
    "log_linear",
]
_FIT_MODES: frozenset[DeltaFOverFFitMode] = frozenset(
    ("direct_bounded_tau", "direct_bounded_tau_and_log_amplitude", "log_linear"),
)
DffMetadataValue = str | int | float | bool


@dataclass(frozen=True)
class RoiDeltaFOverF:
    """ROI-level dF/F and the fitted baseline used to compute it.

    Inputs: background-corrected ROI fluorescence and baseline windows.
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
    baseline_frame_numbers: npt.NDArray[np.float64]
    baseline_fluorescence: npt.NDArray[np.float64]
    metadata: dict[str, DffMetadataValue]


def compute_roi_delta_f_over_f(
    traces: BackgroundCorrectedRoiTraces,
    baseline_windows: Sequence[FrameWindow],
    *,
    data_rate_hz: float,
    baseline_sample_seconds: float | None = 1.0,
    fit_mode: DeltaFOverFFitMode = "direct_bounded_tau",
) -> RoiDeltaFOverF:
    """Compute ROI-level dF/F from corrected fluorescence.

    Args:
        traces: Background-corrected ROI traces. ``corrected_values`` is treated
            as final ROI fluorescence ``F_roi(t)``.
        baseline_windows: Baseline frame windows in the same absolute frame
            coordinates as ``traces``.
        data_rate_hz: Imaging frame rate in frames per second.
        baseline_sample_seconds: Number of seconds to keep from the end of each
            baseline window. ``None`` uses the full window.
        fit_mode: Exponential fit mode. ``"direct_bounded_tau"`` fits the
            exponential directly with bounded tau, unbounded log-amplitude, and
            only positive finite samples. ``"direct_bounded_tau_and_log_amplitude"``
            also bounds log-amplitude. ``"log_linear"`` fits a line to
            ``log(F)`` with ``polyfit``.

    Returns:
        ``RoiDeltaFOverF`` with dF/F values shaped ``(frames, rois)``.

    Raises:
        ValueError: If no baseline windows are supplied, windows fall outside
            the traces, or the fit has no positive finite baseline samples.

    The calculation uses the current twopy analysis contract: first use the
    background-corrected mean fluorescence inside each ROI, then fit one
    exponential decay constant shared by all ROIs from baseline windows, then
    estimate one amplitude per ROI.
    """
    _validate_dff_inputs(
        traces=traces,
        baseline_windows=baseline_windows,
        data_rate_hz=data_rate_hz,
        baseline_sample_seconds=baseline_sample_seconds,
        fit_mode=fit_mode,
    )

    fluorescence = traces.corrected_values.astype(np.float64, copy=False)
    baseline_samples = _baseline_samples(
        fluorescence=fluorescence,
        trace_start_frame=traces.start_frame,
        baseline_windows=baseline_windows,
        data_rate_hz=data_rate_hz,
        baseline_sample_seconds=baseline_sample_seconds,
    )
    tau = _fit_shared_tau(
        frame_numbers=baseline_samples.frame_numbers,
        mean_fluorescence=baseline_samples.values.mean(axis=1),
        fit_mode=fit_mode,
    )
    amplitudes = _compute_roi_amplitudes(
        baseline_values=baseline_samples.values,
        baseline_frame_numbers=baseline_samples.frame_numbers,
        tau=tau,
    )
    baseline = _evaluate_baseline(
        frame_count=fluorescence.shape[0],
        start_frame=traces.start_frame,
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
        baseline_frame_numbers=baseline_samples.frame_numbers,
        baseline_fluorescence=baseline_samples.values,
        metadata={
            "method": "shared_tau_roi_amplitude",
            "data_rate_hz": float(data_rate_hz),
            "baseline_sample_seconds": (
                "full"
                if baseline_sample_seconds is None
                else float(baseline_sample_seconds)
            ),
            "baseline_window_count": len(baseline_windows),
            "fit_mode": fit_mode,
        },
    )


@dataclass(frozen=True)
class _BaselineSamples:
    """Mean fluorescence samples from baseline windows.

    Inputs: baseline frame windows and ROI fluorescence traces.
    Outputs: one mean fluorescence value per baseline window and ROI, plus
    the frame number used for exponential fitting.
    """

    frame_numbers: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]


def _validate_dff_inputs(
    *,
    traces: BackgroundCorrectedRoiTraces,
    baseline_windows: Sequence[FrameWindow],
    data_rate_hz: float,
    baseline_sample_seconds: float | None,
    fit_mode: DeltaFOverFFitMode,
) -> None:
    """Validate the public dF/F inputs.

    Args:
        traces: Candidate background-corrected ROI traces.
        baseline_windows: Candidate baseline windows.
        data_rate_hz: Imaging frame rate.
        baseline_sample_seconds: Optional seconds from each baseline end.
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
    if len(baseline_windows) == 0:
        msg = "At least one baseline window is required for dF/F"
        raise ValueError(msg)
    if data_rate_hz <= 0:
        msg = f"data_rate_hz must be positive; got {data_rate_hz}"
        raise ValueError(msg)
    if baseline_sample_seconds is not None and baseline_sample_seconds <= 0:
        msg = (
            "baseline_sample_seconds must be positive or None; "
            f"got {baseline_sample_seconds}"
        )
        raise ValueError(msg)
    if fit_mode not in _FIT_MODES:
        msg = f"Unknown dF/F fit_mode {fit_mode!r}"
        raise ValueError(msg)


def _baseline_samples(
    *,
    fluorescence: npt.NDArray[np.float64],
    trace_start_frame: int,
    baseline_windows: Sequence[FrameWindow],
    data_rate_hz: float,
    baseline_sample_seconds: float | None,
) -> _BaselineSamples:
    """Average the chosen part of each baseline window.

    Args:
        fluorescence: ROI fluorescence shaped ``(frames, rois)``.
        trace_start_frame: Absolute frame index for ``fluorescence[0]``.
        baseline_windows: Baseline windows in absolute frame indices.
        data_rate_hz: Imaging frame rate.
        baseline_sample_seconds: Optional seconds from each baseline end.

    Returns:
        Mean baseline fluorescence and one-based fit frame numbers.

    Each baseline window is first narrowed to the last requested second, then
    selected frames are averaged into one point per window. That keeps longer
    baseline windows from dominating the exponential fit.
    """
    frame_numbers: list[float] = []
    values: list[npt.NDArray[np.float64]] = []
    for window in baseline_windows:
        local_start = window.start_frame - trace_start_frame
        local_stop = window.stop_frame - trace_start_frame
        _validate_baseline_local_range(
            window=window,
            local_start=local_start,
            local_stop=local_stop,
            frame_count=fluorescence.shape[0],
        )
        sample_start, sample_stop = _last_baseline_sample_range(
            local_start=local_start,
            local_stop=local_stop,
            data_rate_hz=data_rate_hz,
            baseline_sample_seconds=baseline_sample_seconds,
        )

        duration = sample_stop - sample_start
        absolute_sample_start = trace_start_frame + sample_start
        # The fit uses one-based movie frame numbers centered on the averaged
        # half-open sample range. Local indices slice the trace array; absolute
        # frame numbers preserve the recording-wide time coordinate.
        frame_numbers.append(float(absolute_sample_start + 1) + (duration / 2.0))
        values.append(fluorescence[sample_start:sample_stop, :].mean(axis=0))

    return _BaselineSamples(
        frame_numbers=np.asarray(frame_numbers, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
    )


def _validate_baseline_local_range(
    *,
    window: FrameWindow,
    local_start: int,
    local_stop: int,
    frame_count: int,
) -> None:
    """Check that one baseline window can slice the trace array.

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
            f"Baseline window [{window.start_frame}, {window.stop_frame}) is "
            f"outside trace range with {frame_count} frames"
        )
        raise ValueError(msg)


def _last_baseline_sample_range(
    *,
    local_start: int,
    local_stop: int,
    data_rate_hz: float,
    baseline_sample_seconds: float | None,
) -> tuple[int, int]:
    """Choose the final part of one baseline window.

    Args:
        local_start: Window start in local trace coordinates.
        local_stop: Window stop in local trace coordinates.
        data_rate_hz: Imaging frame rate.
        baseline_sample_seconds: Optional seconds from the window end.

    Returns:
        Half-open local frame range to average.
    """
    if baseline_sample_seconds is None:
        return local_start, local_stop

    frame_count = int(round(baseline_sample_seconds * data_rate_hz))
    frame_count = max(frame_count, 1)
    duration = local_stop - local_start
    if frame_count > duration:
        msg = (
            "Requested baseline sample is longer than a window: "
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
        frame_numbers: One-based frame number for each baseline sample.
        mean_fluorescence: Mean fluorescence across ROIs for each sample.
        fit_mode: Exponential fit mode to use.

    Returns:
        Shared exponential ``tau``.

    The fit model is ``exp(tau * t + b)`` on the ROI-averaged baseline trace.
    Only ``tau`` is shared; per-ROI amplitudes are computed in a separate
    transparent step. ``direct_bounded_tau`` handles nonpositive outliers
    defensively; ``direct_bounded_tau_and_log_amplitude`` preserves a bounded
    log-amplitude contract; and ``log_linear`` fits in log space.
    """
    if fit_mode == "log_linear":
        return _fit_shared_tau_log_linear(
            frame_numbers=frame_numbers,
            mean_fluorescence=mean_fluorescence,
        )
    if fit_mode == "direct_bounded_tau_and_log_amplitude":
        return _fit_shared_tau_with_bounded_log_amplitude(
            frame_numbers=frame_numbers,
            mean_fluorescence=mean_fluorescence,
        )
    return _fit_shared_tau_direct_bounded(
        frame_numbers=frame_numbers,
        mean_fluorescence=mean_fluorescence,
    )


def _fit_shared_tau_direct_bounded(
    *,
    frame_numbers: npt.NDArray[np.float64],
    mean_fluorescence: npt.NDArray[np.float64],
) -> float:
    """Fit shared tau with defensive handling for nonpositive samples.

    Args:
        frame_numbers: One-based frame number for each baseline sample.
        mean_fluorescence: Mean fluorescence across ROIs for each sample.

    Returns:
        Shared exponential ``tau``.

    This is the default fit path because background correction can create
    nonpositive values and the first baseline sample can be noisy. Keeping
    log-amplitude unbounded avoids making one baseline sample define the
    amplitude range while still bounding the decay rate.
    """
    finite_positive = np.isfinite(mean_fluorescence) & (mean_fluorescence > 0)
    if not np.any(finite_positive):
        msg = "Cannot fit dF/F baseline: baseline fluorescence has no positive values"
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


def _fit_shared_tau_log_linear(
    *,
    frame_numbers: npt.NDArray[np.float64],
    mean_fluorescence: npt.NDArray[np.float64],
) -> float:
    """Fit shared tau with a log-linear model.

    Args:
        frame_numbers: One-based frame number for each baseline sample.
        mean_fluorescence: Mean fluorescence across ROIs for each sample.

    Returns:
        Shared exponential ``tau`` from ``polyfit(t, log(F), 1)``.

    Raises:
        ValueError: If any fit sample is nonfinite or nonpositive.

    This intentionally fits in log space with no tau bounds, then the usual
    ROI-amplitude step uses the fitted tau. twopy fails loudly when corrected
    fluorescence cannot define a real-valued logarithm.
    """
    if np.any(~np.isfinite(mean_fluorescence)):
        msg = "log_linear dF/F fit requires finite baseline fluorescence"
        raise ValueError(msg)
    if mean_fluorescence.size == 0:
        msg = "log_linear dF/F fit requires at least one baseline sample"
        raise ValueError(msg)
    if np.any(mean_fluorescence <= 0):
        msg = "log_linear dF/F fit requires positive baseline fluorescence"
        raise ValueError(msg)
    if mean_fluorescence.size == 1:
        return 0.0

    slope, _intercept = np.polyfit(
        frame_numbers,
        np.log(mean_fluorescence),
        deg=1,
    )
    return float(slope)


def _fit_shared_tau_with_bounded_log_amplitude(
    *,
    frame_numbers: npt.NDArray[np.float64],
    mean_fluorescence: npt.NDArray[np.float64],
) -> float:
    """Fit shared tau using a bounded log-amplitude contract.

    Args:
        frame_numbers: One-based frame number for each baseline sample.
        mean_fluorescence: Mean fluorescence across ROIs for each sample.

    Returns:
        Shared exponential ``tau``.

    Raises:
        ValueError: If any fit sample is nonfinite or the first sample cannot
            define a positive log-amplitude bound.

    Its log-amplitude bound is ``[0, 10 * log(first baseline)]``. That contract
    makes the first baseline sample unusually important, so this mode
    intentionally fails on nonfinite, nonpositive, or too-small first samples
    instead of silently switching to the direct bounded-tau behavior.
    """
    if np.any(~np.isfinite(mean_fluorescence)):
        msg = (
            "direct_bounded_tau_and_log_amplitude dF/F fit requires finite "
            "baseline fluorescence"
        )
        raise ValueError(msg)
    if mean_fluorescence.size == 0:
        msg = (
            "direct_bounded_tau_and_log_amplitude dF/F fit requires at least "
            "one baseline sample"
        )
        raise ValueError(msg)
    first_value = float(mean_fluorescence[0])
    if first_value <= 1.0:
        msg = (
            "direct_bounded_tau_and_log_amplitude dF/F fit requires first "
            "baseline fluorescence greater than 1.0 to define an ordered "
            "log-amplitude bound; "
            f"got {first_value}"
        )
        raise ValueError(msg)
    if np.any(mean_fluorescence <= 0):
        msg = (
            "direct_bounded_tau_and_log_amplitude dF/F fit requires positive "
            "baseline fluorescence"
        )
        raise ValueError(msg)
    if mean_fluorescence.size == 1:
        return 0.0

    initial_log_amplitude = float(np.log(first_value))
    upper_log_amplitude = 10.0 * initial_log_amplitude

    def residual(parameters: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return direct exponential residuals for bounded least squares."""
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
    baseline_values: npt.NDArray[np.float64],
    baseline_frame_numbers: npt.NDArray[np.float64],
    tau: float,
) -> npt.NDArray[np.float64]:
    """Estimate one exponential amplitude per ROI.

    Args:
        baseline_values: Mean baseline fluorescence shaped
            ``(baseline_windows, rois)``.
        baseline_frame_numbers: One-based fit frame number per baseline sample.
        tau: Shared exponential decay constant.

    Returns:
        One amplitude per ROI.
    """
    decay = np.exp(tau * baseline_frame_numbers)[:, None]
    if tau == 0:
        return baseline_values.mean(axis=0)
    return (baseline_values / decay).mean(axis=0)


def _evaluate_baseline(
    *,
    frame_count: int,
    start_frame: int,
    amplitudes: npt.NDArray[np.float64],
    tau: float,
) -> npt.NDArray[np.float64]:
    """Evaluate the fitted baseline for every frame and ROI.

    Args:
        frame_count: Number of fluorescence frames.
        start_frame: Absolute frame index for the first fluorescence frame.
        amplitudes: Per-ROI amplitudes.
        tau: Shared exponential decay constant.

    Returns:
        Baseline shaped ``(frames, rois)``.
    """
    frame_numbers = np.arange(
        start_frame + 1,
        start_frame + frame_count + 1,
        dtype=np.float64,
    )
    return amplitudes[None, :] * np.exp(tau * frame_numbers[:, None])
