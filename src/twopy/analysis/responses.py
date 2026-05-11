"""Group ROI responses by stimulus epoch and trial.

Inputs: computed ROI dF/F traces and stimulus-labeled frame windows.
Outputs: per-trial response arrays and compact per-ROI summary rows.

This module keeps response grouping independent from plotting, napari, and file
storage. It only works with converted twopy analysis objects that already have
explicit frame coordinates.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.trials import EpochFrameWindow

__all__ = [
    "GroupedRoiResponseSummary",
    "GroupedRoiResponses",
    "RoiResponseTraceSummary",
    "RoiResponseSummary",
    "RoiResponseTrial",
    "group_delta_f_over_f_by_epoch",
    "summarize_epoch_roi_responses",
    "summarize_grouped_responses",
    "summarize_roi_response_trials",
    "validate_grouped_roi_responses",
]


@dataclass(frozen=True)
class RoiResponseTrial:
    """One ROI response matrix for one stimulus trial.

    Inputs: one stimulus-labeled frame window and a slice of ROI dF/F.
    Outputs: frame-by-ROI values plus absolute frame numbers and relative time.

    ``trial_index`` is one-based within each epoch because that is the number a
    scientist naturally uses when checking repeated presentations.
    """

    epoch_number: int
    epoch_name: str
    trial_index: int
    window_index: int
    start_frame: int
    stop_frame: int
    frame_numbers: npt.NDArray[np.int64]
    time_seconds: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]


@dataclass(frozen=True)
class GroupedRoiResponses:
    """ROI responses grouped by epoch and trial.

    Inputs: one dF/F result plus epoch windows.
    Outputs: trial response objects, ROI labels, frame rate, and optional
    response-window settings used to create the trials.

    The values stay as one matrix per trial instead of being padded into a
    larger cube. This avoids fake samples when trial lengths differ.
    """

    roi_labels: tuple[str, ...]
    data_rate_hz: float
    trials: tuple[RoiResponseTrial, ...]
    pre_window_seconds: float | None = None
    post_window_seconds: float | None = None
    response_window_auto: bool | None = None


@dataclass(frozen=True)
class RoiResponseTraceSummary:
    """Mean and SEM traces for repeated ROI response trials.

    Inputs: grouped response trials from one stimulus epoch.
    Outputs: ROI-by-frame mean and SEM arrays on a shared relative time axis.

    The summary aligns trials by epoch-relative seconds before averaging so
    responses with slightly different frame coverage do not shift peaks.
    """

    time_seconds: npt.NDArray[np.float64]
    mean_values: npt.NDArray[np.float64]
    sem_values: npt.NDArray[np.float64]


@dataclass(frozen=True)
class RoiResponseSummary:
    """Compact response metrics for one ROI in one trial.

    Inputs: one trial response matrix.
    Outputs: mean, peak, minimum, and frame metadata for one ROI.

    These rows are intentionally simple so they can be written to CSV and
    inspected without custom Python code.
    """

    epoch_number: int
    epoch_name: str
    trial_index: int
    window_index: int
    roi_label: str
    start_frame: int
    stop_frame: int
    frame_count: int
    mean_response: float
    peak_response: float
    min_response: float


@dataclass(frozen=True)
class GroupedRoiResponseSummary:
    """Response metrics collapsed across trials for one epoch and ROI.

    Inputs: trial-level response summaries for one ``(epoch, ROI)`` pair.
    Outputs: mean response, SEM, and the trial count behind those values.

    A single usable trial gets SEM ``0`` because there is no across-trial spread
    to estimate. Empty or all-NaN groups are not emitted.
    """

    epoch_number: int
    epoch_name: str
    roi_label: str
    mean_response: float
    sem_response: float
    n_trials: int


def group_delta_f_over_f_by_epoch(
    dff: RoiDeltaFOverF,
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    data_rate_hz: float,
    pre_window_seconds: float = 0.0,
    post_window_seconds: float = 0.0,
) -> GroupedRoiResponses:
    """Group dF/F values by stimulus epoch windows.

    Args:
        dff: Computed ROI dF/F with absolute frame coordinates.
        epoch_windows: Stimulus-labeled windows to slice.
        data_rate_hz: Imaging frame rate in hertz for relative trial time.
        pre_window_seconds: Seconds before each epoch window to include. This
            is useful for plots that show the gray baseline before
            stimulus onset.
        post_window_seconds: Seconds after each epoch window to include. This
            is useful for plots that show the response returning into the next
            gray interleave.

    Returns:
        ``GroupedRoiResponses`` with one response object per input window.

    Raises:
        ValueError: If the dF/F shape, frame rate, or windows are invalid.

    The function only groups existing dF/F values. It does not decide which
    windows are trials and does not recompute fluorescence.
    """
    _validate_grouping_inputs(
        dff,
        epoch_windows,
        data_rate_hz,
        pre_window_seconds,
        post_window_seconds,
    )

    trials: list[RoiResponseTrial] = []
    epoch_counts: dict[tuple[int, str], int] = {}
    pre_window_frames = int(round(pre_window_seconds * data_rate_hz))
    post_window_frames = int(round(post_window_seconds * data_rate_hz))
    for epoch_window in epoch_windows:
        key = (epoch_window.epoch_number, epoch_window.epoch_name)
        epoch_counts[key] = epoch_counts.get(key, 0) + 1
        window = epoch_window.window
        response_start_frame = max(
            window.start_frame - pre_window_frames, dff.start_frame
        )
        response_stop_frame = min(
            window.stop_frame + post_window_frames,
            dff.stop_frame,
        )
        local_start = response_start_frame - dff.start_frame
        local_stop = response_stop_frame - dff.start_frame
        frame_numbers = np.arange(
            response_start_frame, response_stop_frame, dtype=np.int64
        )
        time_seconds = (frame_numbers - window.start_frame) / data_rate_hz
        trials.append(
            RoiResponseTrial(
                epoch_number=epoch_window.epoch_number,
                epoch_name=epoch_window.epoch_name,
                trial_index=epoch_counts[key],
                window_index=window.index,
                start_frame=response_start_frame,
                stop_frame=response_stop_frame,
                frame_numbers=frame_numbers,
                time_seconds=time_seconds.astype(np.float64, copy=False),
                values=dff.values[local_start:local_stop, :],
            ),
        )

    return GroupedRoiResponses(
        roi_labels=dff.labels,
        data_rate_hz=float(data_rate_hz),
        trials=tuple(trials),
        pre_window_seconds=float(pre_window_seconds),
        post_window_seconds=float(post_window_seconds),
    )


def summarize_grouped_responses(
    grouped: GroupedRoiResponses,
) -> tuple[RoiResponseSummary, ...]:
    """Summarize grouped responses with one row per trial and ROI.

    Args:
        grouped: Grouped response object.

    Returns:
        Tuple of compact response summary rows.

    The summary uses NaN-aware metrics so motion-masked dF/F frames do not make
    an entire trial unusable.
    """
    validate_grouped_roi_responses(grouped)
    summaries: list[RoiResponseSummary] = []
    for trial in grouped.trials:
        for roi_index, roi_label in enumerate(grouped.roi_labels):
            values = trial.values[:, roi_index]
            summaries.append(
                RoiResponseSummary(
                    epoch_number=trial.epoch_number,
                    epoch_name=trial.epoch_name,
                    trial_index=trial.trial_index,
                    window_index=trial.window_index,
                    roi_label=roi_label,
                    start_frame=trial.start_frame,
                    stop_frame=trial.stop_frame,
                    frame_count=trial.stop_frame - trial.start_frame,
                    mean_response=float(np.nanmean(values)),
                    peak_response=float(np.nanmax(values)),
                    min_response=float(np.nanmin(values)),
                ),
            )
    return tuple(summaries)


def summarize_epoch_roi_responses(
    grouped: GroupedRoiResponses,
) -> tuple[GroupedRoiResponseSummary, ...]:
    """Summarize responses across trials with one row per epoch and ROI.

    Args:
        grouped: Grouped response object.

    Returns:
        Tuple of grouped summary rows.

    The grouped mean and SEM are computed from trial-level mean responses, not
    from every frame pooled together. That keeps each trial at equal weight even
    if window lengths differ slightly.
    """
    trial_summaries = summarize_grouped_responses(grouped)
    grouped_values: dict[tuple[int, str, str], list[float]] = {}
    for summary in trial_summaries:
        if np.isnan(summary.mean_response):
            continue
        key = (summary.epoch_number, summary.epoch_name, summary.roi_label)
        grouped_values.setdefault(key, []).append(summary.mean_response)

    rows: list[GroupedRoiResponseSummary] = []
    for key, values in grouped_values.items():
        epoch_number, epoch_name, roi_label = key
        response_values = np.asarray(values, dtype=np.float64)
        trial_count = int(response_values.size)
        sem = (
            float(np.nanstd(response_values, ddof=1) / np.sqrt(trial_count))
            if trial_count > 1
            else 0.0
        )
        rows.append(
            GroupedRoiResponseSummary(
                epoch_number=epoch_number,
                epoch_name=epoch_name,
                roi_label=roi_label,
                mean_response=float(np.nanmean(response_values)),
                sem_response=sem,
                n_trials=trial_count,
            ),
        )
    return tuple(rows)


def summarize_roi_response_trials(
    trials: Sequence[RoiResponseTrial],
    *,
    roi_labels: tuple[str, ...],
    data_rate_hz: float,
) -> RoiResponseTraceSummary:
    """Summarize repeated response trials into ROI mean and SEM traces.

    Args:
        trials: Response trials from one stimulus epoch.
        roi_labels: ROI labels defining the expected response width.
        data_rate_hz: Imaging frame rate in hertz.

    Returns:
        Shared time axis plus ROI-by-frame mean and SEM arrays.

    Raises:
        ValueError: If no trials are provided or trial axes are malformed.
    """
    if len(trials) == 0:
        msg = "At least one trial is required for response trace summary"
        raise ValueError(msg)
    for trial in trials:
        _validate_grouped_response_trial(trial, roi_labels)
    if data_rate_hz <= 0:
        msg = f"data_rate_hz must be positive; got {data_rate_hz}"
        raise ValueError(msg)

    trial_tuple = tuple(trials)
    time_seconds = _shared_trial_time_axis(trial_tuple, data_rate_hz)
    first_time = float(time_seconds[0])
    max_frames = len(time_seconds)
    roi_count = len(roi_labels)
    sums = np.zeros((roi_count, max_frames), dtype=np.float64)
    squared_sums = np.zeros((roi_count, max_frames), dtype=np.float64)
    valid_counts = np.zeros((roi_count, max_frames), dtype=np.int64)
    for trial in trial_tuple:
        offsets = np.rint((trial.time_seconds - first_time) * data_rate_hz).astype(
            np.int64,
            copy=False,
        )
        values_by_roi = trial.values.T
        finite = np.isfinite(values_by_roi)
        finite_values = np.where(finite, values_by_roi, 0.0)
        sums[:, offsets] += finite_values
        squared_sums[:, offsets] += finite_values * finite_values
        valid_counts[:, offsets] += finite

    mean_values = np.full((roi_count, max_frames), np.nan, dtype=np.float64)
    valid_mask = valid_counts > 0
    mean_values[valid_mask] = sums[valid_mask] / valid_counts[valid_mask]
    sem_values = np.zeros((roi_count, max_frames), dtype=np.float64)
    variable_mask = valid_counts > 1
    if np.any(variable_mask):
        counts = valid_counts[variable_mask].astype(np.float64)
        variance = squared_sums[variable_mask] - (sums[variable_mask] ** 2 / counts)
        variance /= counts - 1.0
        variance = np.maximum(variance, 0.0)
        sem_values[variable_mask] = np.sqrt(variance) / np.sqrt(counts)
    return RoiResponseTraceSummary(
        time_seconds=time_seconds,
        mean_values=mean_values,
        sem_values=sem_values,
    )


def validate_grouped_roi_responses(grouped: GroupedRoiResponses) -> None:
    """Validate the shared grouped-response data contract.

    Args:
        grouped: Grouped response object to check.

    Returns:
        None.

    Raises:
        ValueError: If the frame rate, trial vectors, or response matrices are
            inconsistent.

    Grouped responses cross processing, plotting, summaries, and HDF5
    persistence. Keeping this contract in one helper makes every downstream
    path reject malformed trial data the same way before it computes with
    misaligned time, frame, or ROI axes.
    """
    if not np.isfinite(grouped.data_rate_hz) or grouped.data_rate_hz <= 0:
        msg = f"data_rate_hz must be positive; got {grouped.data_rate_hz}"
        raise ValueError(msg)
    for trial in grouped.trials:
        _validate_grouped_response_trial(trial, grouped.roi_labels)


def _validate_grouped_response_trial(
    trial: RoiResponseTrial,
    roi_labels: tuple[str, ...],
) -> None:
    """Validate one grouped response trial against shared ROI labels."""
    if not np.issubdtype(trial.values.dtype, np.floating):
        msg = f"trial values must have floating dtype; got {trial.values.dtype}"
        raise ValueError(msg)
    if not np.issubdtype(trial.frame_numbers.dtype, np.integer):
        msg = (
            "trial frame_numbers must have integer dtype; "
            f"got {trial.frame_numbers.dtype}"
        )
        raise ValueError(msg)
    if not np.issubdtype(trial.time_seconds.dtype, np.floating):
        msg = (
            "trial time_seconds must have floating dtype; "
            f"got {trial.time_seconds.dtype}"
        )
        raise ValueError(msg)
    if trial.values.ndim != 2:
        msg = f"trial values must have shape (frames, rois); got {trial.values.shape}"
        raise ValueError(msg)
    if trial.frame_numbers.ndim != 1:
        msg = (
            "trial frame_numbers must be one-dimensional; "
            f"got {trial.frame_numbers.shape}"
        )
        raise ValueError(msg)
    if trial.time_seconds.ndim != 1:
        msg = (
            "trial time_seconds must be one-dimensional; "
            f"got {trial.time_seconds.shape}"
        )
        raise ValueError(msg)
    if trial.values.shape[0] == 0:
        msg = "trial values must include at least one frame"
        raise ValueError(msg)
    if trial.values.shape[1] != len(roi_labels):
        msg = (
            "trial response width does not match ROI labels: "
            f"{trial.values.shape[1]} values, {len(roi_labels)} labels"
        )
        raise ValueError(msg)
    if trial.values.shape[0] != trial.time_seconds.size:
        msg = (
            "trial response frame count does not match time_seconds: "
            f"{trial.values.shape[0]} values, {trial.time_seconds.size} timepoints"
        )
        raise ValueError(msg)
    if trial.values.shape[0] != trial.frame_numbers.size:
        msg = (
            "trial response frame count does not match frame_numbers: "
            f"{trial.values.shape[0]} values, {trial.frame_numbers.size} frame numbers"
        )
        raise ValueError(msg)
    expected_frame_count = trial.stop_frame - trial.start_frame
    if expected_frame_count != trial.values.shape[0]:
        msg = (
            "trial response frame count does not match frame range: "
            f"{trial.values.shape[0]} values for "
            f"[{trial.start_frame}, {trial.stop_frame})"
        )
        raise ValueError(msg)
    expected_frame_numbers = np.arange(
        trial.start_frame,
        trial.stop_frame,
        dtype=np.int64,
    )
    if not np.array_equal(trial.frame_numbers, expected_frame_numbers):
        msg = "trial frame_numbers must match the half-open frame range"
        raise ValueError(msg)
    if not np.all(np.isfinite(trial.time_seconds)):
        msg = "trial time_seconds must be finite"
        raise ValueError(msg)


def _validate_grouping_inputs(
    dff: RoiDeltaFOverF,
    epoch_windows: Sequence[EpochFrameWindow],
    data_rate_hz: float,
    pre_window_seconds: float,
    post_window_seconds: float,
) -> None:
    """Validate response grouping inputs.

    Args:
        dff: Candidate dF/F result.
        epoch_windows: Candidate windows.
        data_rate_hz: Candidate frame rate.
        pre_window_seconds: Candidate prestimulus duration.
        post_window_seconds: Candidate poststimulus duration.

    Returns:
        None.
    """
    if dff.values.ndim != 2:
        msg = f"dF/F values must have shape (frames, rois); got {dff.values.shape}"
        raise ValueError(msg)
    if dff.values.shape[1] != len(dff.labels):
        msg = (
            f"dF/F labels length {len(dff.labels)} does not match "
            f"{dff.values.shape[1]} ROIs"
        )
        raise ValueError(msg)
    if data_rate_hz <= 0:
        msg = f"data_rate_hz must be positive; got {data_rate_hz}"
        raise ValueError(msg)
    if pre_window_seconds < 0:
        msg = f"pre_window_seconds must be nonnegative; got {pre_window_seconds}"
        raise ValueError(msg)
    if post_window_seconds < 0:
        msg = f"post_window_seconds must be nonnegative; got {post_window_seconds}"
        raise ValueError(msg)
    for epoch_window in epoch_windows:
        window = epoch_window.window
        if window.start_frame < dff.start_frame or window.stop_frame > dff.stop_frame:
            msg = (
                f"Window [{window.start_frame}, {window.stop_frame}) is outside "
                f"dF/F range [{dff.start_frame}, {dff.stop_frame})"
            )
            raise ValueError(msg)
        if window.start_frame >= window.stop_frame:
            msg = (
                f"Window [{window.start_frame}, {window.stop_frame}) must have "
                "positive duration"
            )
            raise ValueError(msg)


def _shared_trial_time_axis(
    trials: tuple[RoiResponseTrial, ...],
    data_rate_hz: float,
) -> npt.NDArray[np.float64]:
    """Return one relative time axis that covers all supplied trials."""
    first_time = min(float(trial.time_seconds[0]) for trial in trials)
    last_time = max(float(trial.time_seconds[-1]) for trial in trials)
    frame_count = int(round((last_time - first_time) * data_rate_hz)) + 1
    return first_time + (np.arange(frame_count, dtype=np.float64) / data_rate_hz)
