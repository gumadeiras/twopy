"""Compare twopy ROI dF/F outputs against prior saved-analysis traces.

Inputs: a converted ``RecordingData`` object and a read-only saved-analysis
object.
Outputs: computed twopy dF/F, saved traces, and simple error metrics.

This module is an audit layer. It does not make saved-analysis files part of
normal analysis; it only checks whether twopy analysis over converted objects
matches prior saved outputs for the same ROI masks and frame windows.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twopy.analysis.background_subtraction import (
    BackgroundCorrectionMethod,
    extract_background_corrected_roi_traces,
)
from twopy.analysis.dff import (
    DeltaFOverFFitMode,
    RoiDeltaFOverF,
    compute_roi_delta_f_over_f,
)
from twopy.analysis.epoch_mapping import interpolate_stimulus_epochs_to_frame_windows
from twopy.analysis.trials import FrameWindow
from twopy.converted import RecordingData, recording_frame_rate_hz
from twopy.parity.roi import SavedAnalysisRoiSet, roi_set_from_saved_analysis
from twopy.parity.saved_analysis import SavedAnalysisLastRoi

__all__ = [
    "SavedAnalysisDeltaFOverFComparison",
    "compare_saved_analysis_delta_f_over_f",
    "saved_analysis_epoch_windows",
]


@dataclass(frozen=True)
class SavedAnalysisDeltaFOverFComparison:
    """Computed dF/F beside saved reference traces and error metrics.

    Inputs: twopy-computed dF/F and saved-analysis ROI traces with the same
    frame and ROI shape.
    Outputs: absolute errors, per-ROI correlations, and summary metrics.

    Keeping the full arrays beside the summaries makes a parity failure
    inspectable: callers can plot one ROI, inspect one frame window, or report
    aggregate errors without recomputing extraction.
    """

    computed: RoiDeltaFOverF
    saved_values: npt.NDArray[np.float64]
    absolute_error: npt.NDArray[np.float64]
    correlation_by_roi: npt.NDArray[np.float64]
    roi_set: SavedAnalysisRoiSet
    interleave_windows: tuple[FrameWindow, ...]
    mean_absolute_error: float
    median_absolute_error: float
    max_absolute_error: float


def compare_saved_analysis_delta_f_over_f(
    recording: RecordingData,
    saved_analysis: SavedAnalysisLastRoi,
    *,
    saved_trace_index: int = 0,
    baseline_epoch_number: int = 1,
    background_method: BackgroundCorrectionMethod = "movie_global_percentile",
    baseline_sample_seconds: float | None = 1.0,
    fit_mode: DeltaFOverFFitMode = "direct_bounded_tau_and_log_amplitude",
    trace_start_frame: int | None = None,
    trace_stop_frame: int | None = None,
    data_rate_hz: float | None = None,
    chunk_frames: int = 128,
) -> SavedAnalysisDeltaFOverFComparison:
    """Compute twopy dF/F and compare it to one saved trace matrix.

    Args:
        recording: Loaded converted recording.
        saved_analysis: Read-only saved-analysis data.
        saved_trace_index: Which saved trace cell to compare.
        baseline_epoch_number: One-based saved epoch number used as baseline.
        background_method: twopy background method used before dF/F.
        baseline_sample_seconds: Seconds from the end of each baseline window.
            ``None`` uses each full baseline window.
        fit_mode: dF/F exponential fit mode.
        trace_start_frame: Optional absolute movie frame for saved row zero.
            When omitted, twopy infers the stimulus-bounded frame range from
            converted photodiode and stimulus data.
        trace_stop_frame: Optional exclusive absolute stop frame. When omitted
            with an explicit start, the saved trace length defines the stop.
        data_rate_hz: Optional imaging frame rate. When omitted, twopy reads
            ``acq.frameRate`` from converted acquisition metadata.
        chunk_frames: Movie frames to read per HDF5 chunk during ROI extraction.

    Returns:
        ``SavedAnalysisDeltaFOverFComparison`` with computed values and metrics.

    Raises:
        ValueError: If saved traces, ROI masks, or windows do not agree with the
            converted recording frame contract.

    The comparison uses saved ROI masks and saved baseline windows, then runs
    the normal twopy analysis path over converted HDF5 data.
    """
    saved_values = _saved_trace_values(saved_analysis, saved_trace_index)
    frame_start, frame_stop = _saved_trace_frame_range(
        recording=recording,
        saved_frame_count=saved_values.shape[0],
        trace_start_frame=trace_start_frame,
        trace_stop_frame=trace_stop_frame,
    )
    roi_set = roi_set_from_saved_analysis(saved_analysis, recording)
    _validate_saved_trace_roi_count(saved_values, roi_set)

    traces = extract_background_corrected_roi_traces(
        recording,
        roi_set.roi_set,
        method=background_method,
        start_frame=frame_start,
        stop_frame=frame_stop,
        chunk_frames=chunk_frames,
    )
    interleave_windows = saved_analysis_epoch_windows(
        saved_analysis,
        epoch_number=baseline_epoch_number,
        trace_start_frame=frame_start,
        trace_stop_frame=frame_stop,
    )
    computed = compute_roi_delta_f_over_f(
        traces,
        interleave_windows,
        data_rate_hz=(
            recording_frame_rate_hz(recording)
            if data_rate_hz is None
            else float(data_rate_hz)
        ),
        baseline_sample_seconds=baseline_sample_seconds,
        fit_mode=fit_mode,
    )
    _validate_comparison_shapes(computed.values, saved_values)

    absolute_error = np.abs(computed.values - saved_values)
    finite_errors = absolute_error[np.isfinite(absolute_error)]
    return SavedAnalysisDeltaFOverFComparison(
        computed=computed,
        saved_values=saved_values,
        absolute_error=absolute_error,
        correlation_by_roi=_correlation_by_roi(computed.values, saved_values),
        roi_set=roi_set,
        interleave_windows=interleave_windows,
        mean_absolute_error=float(np.mean(finite_errors)),
        median_absolute_error=float(np.median(finite_errors)),
        max_absolute_error=float(np.max(finite_errors)),
    )


def saved_analysis_epoch_windows(
    saved_analysis: SavedAnalysisLastRoi,
    *,
    epoch_number: int,
    trace_start_frame: int,
    trace_stop_frame: int,
) -> tuple[FrameWindow, ...]:
    """Convert saved epoch starts and durations to movie-frame windows.

    Args:
        saved_analysis: Read-only saved-analysis data.
        epoch_number: One-based saved epoch number.
        trace_start_frame: Absolute movie frame for saved trace row zero.
        trace_stop_frame: Exclusive absolute movie frame after the saved trace.

    Returns:
        Tuple of ``FrameWindow`` objects in absolute movie coordinates.

    Saved epoch starts are one-based row numbers inside the saved trace matrix.
    twopy analysis uses zero-based half-open movie-frame windows, so each saved
    start is shifted by ``trace_start_frame - 1`` and each duration defines the
    exclusive stop.
    """
    epoch_index = epoch_number - 1
    if epoch_index < 0 or epoch_index >= len(saved_analysis.epoch_start_times):
        msg = (
            f"Saved analysis has {len(saved_analysis.epoch_start_times)} epochs; "
            f"cannot read epoch {epoch_number}"
        )
        raise ValueError(msg)

    starts = saved_analysis.epoch_start_times[epoch_index]
    durations = saved_analysis.epoch_durations[epoch_index]
    if starts.shape != durations.shape:
        msg = (
            f"Saved epoch {epoch_number} starts and durations differ: "
            f"{starts.shape} versus {durations.shape}"
        )
        raise ValueError(msg)

    windows: list[FrameWindow] = []
    for index, (start, duration) in enumerate(zip(starts, durations, strict=True)):
        local_start = int(start) - 1
        duration_frames = int(duration)
        absolute_start = trace_start_frame + local_start
        absolute_stop = absolute_start + duration_frames
        if local_start < 0 or duration_frames <= 0 or absolute_stop > trace_stop_frame:
            msg = (
                f"Saved epoch {epoch_number} window {index} is outside trace "
                f"range [{trace_start_frame}, {trace_stop_frame}): "
                f"start={int(start)}, duration={duration_frames}"
            )
            raise ValueError(msg)
        windows.append(
            FrameWindow(
                index=index,
                start_frame=absolute_start,
                stop_frame=absolute_stop,
                label=f"saved_epoch_{epoch_number:04d}_{index + 1:04d}",
            ),
        )
    return tuple(windows)


def _saved_trace_values(
    saved_analysis: SavedAnalysisLastRoi,
    saved_trace_index: int,
) -> npt.NDArray[np.float64]:
    """Return one saved trace matrix as float64 values.

    Args:
        saved_analysis: Read-only saved-analysis data.
        saved_trace_index: Trace cell index to load.

    Returns:
        Saved trace matrix shaped ``(frames, rois)``.
    """
    if saved_trace_index < 0 or saved_trace_index >= len(
        saved_analysis.time_by_rois_initial,
    ):
        msg = (
            f"Saved analysis has {len(saved_analysis.time_by_rois_initial)} "
            f"trace matrices; cannot read index {saved_trace_index}"
        )
        raise ValueError(msg)
    values = np.asarray(
        saved_analysis.time_by_rois_initial[saved_trace_index],
        dtype=np.float64,
    )
    if values.ndim != 2:
        msg = f"Saved trace matrix must have shape (frames, rois); got {values.shape}"
        raise ValueError(msg)
    return values


def _saved_trace_frame_range(
    *,
    recording: RecordingData,
    saved_frame_count: int,
    trace_start_frame: int | None,
    trace_stop_frame: int | None,
) -> tuple[int, int]:
    """Resolve the absolute movie-frame range for saved trace rows.

    Args:
        recording: Converted recording.
        saved_frame_count: Number of saved trace rows.
        trace_start_frame: Optional explicit start frame.
        trace_stop_frame: Optional explicit stop frame.

    Returns:
        ``(start, stop)`` in absolute movie-frame coordinates.
    """
    if trace_start_frame is None:
        if trace_stop_frame is not None:
            msg = "trace_stop_frame requires trace_start_frame"
            raise ValueError(msg)
        mapping = interpolate_stimulus_epochs_to_frame_windows(recording)
        start = mapping.start_frame
        stop = mapping.stop_frame
        if stop - start != saved_frame_count:
            msg = (
                "Saved trace length does not match interpolated stimulus frame "
                f"range: saved={saved_frame_count}, inferred={stop - start}"
            )
            raise ValueError(msg)
        return start, stop

    start = int(trace_start_frame)
    stop = (
        start + saved_frame_count if trace_stop_frame is None else int(trace_stop_frame)
    )
    if stop - start != saved_frame_count:
        msg = (
            "Saved trace frame range length must match saved rows: "
            f"range={stop - start}, saved={saved_frame_count}"
        )
        raise ValueError(msg)
    if start < 0 or stop > recording.movie.shape[0] or start >= stop:
        msg = (
            f"Saved trace frame range [{start}, {stop}) is invalid for "
            f"{recording.movie.shape[0]} movie frames"
        )
        raise ValueError(msg)
    return start, stop


def _validate_saved_trace_roi_count(
    saved_values: npt.NDArray[np.float64],
    roi_set: SavedAnalysisRoiSet,
) -> None:
    """Confirm saved traces and saved ROI masks describe the same ROI count.

    Args:
        saved_values: Saved trace matrix.
        roi_set: ROI set derived from the saved mask.

    Returns:
        None.
    """
    roi_count = roi_set.roi_set.masks.shape[0]
    if saved_values.shape[1] != roi_count:
        msg = (
            "Saved trace ROI count does not match saved ROI mask labels: "
            f"traces={saved_values.shape[1]}, masks={roi_count}"
        )
        raise ValueError(msg)


def _validate_comparison_shapes(
    computed_values: npt.NDArray[np.float64],
    saved_values: npt.NDArray[np.float64],
) -> None:
    """Confirm computed and saved dF/F matrices can be compared.

    Args:
        computed_values: twopy-computed dF/F values.
        saved_values: Saved trace matrix.

    Returns:
        None.
    """
    if computed_values.shape != saved_values.shape:
        msg = (
            "Computed dF/F shape does not match saved traces: "
            f"computed={computed_values.shape}, saved={saved_values.shape}"
        )
        raise ValueError(msg)


def _correlation_by_roi(
    computed_values: npt.NDArray[np.float64],
    saved_values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute one Pearson correlation per ROI.

    Args:
        computed_values: twopy-computed dF/F matrix.
        saved_values: Saved dF/F matrix.

    Returns:
        One correlation per ROI, with ``NaN`` when either trace is constant or
        has fewer than two finite samples.
    """
    correlations = np.full(computed_values.shape[1], np.nan, dtype=np.float64)
    for roi_index in range(computed_values.shape[1]):
        computed = computed_values[:, roi_index]
        saved = saved_values[:, roi_index]
        finite = np.isfinite(computed) & np.isfinite(saved)
        if np.count_nonzero(finite) < 2:
            continue
        computed_finite = computed[finite]
        saved_finite = saved[finite]
        if np.std(computed_finite) == 0 or np.std(saved_finite) == 0:
            continue
        correlations[roi_index] = float(
            np.corrcoef(computed_finite, saved_finite)[0, 1],
        )
    return correlations
