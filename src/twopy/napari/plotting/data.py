"""Build plot-ready response data for the twopy napari adapter.

Inputs: grouped response analysis outputs and saved analysis HDF5 files.
Outputs: mean/SEM traces with direct response time vectors for plotting.

This module has no Qt imports. It keeps response-data preparation testable and
separate from the napari widgets that draw and control the plots.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.persistence import load_analysis_outputs
from twopy.analysis.responses import (
    GroupedRoiResponses,
    RoiResponseTrial,
    group_delta_f_over_f_by_epoch,
    is_gray_epoch_name,
)
from twopy.analysis.trials import EpochFrameWindow
from twopy.analysis.workflow import (
    DEFAULT_RESPONSE_POST_WINDOW_SECONDS,
    DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
)
from twopy.converted import RecordingData

__all__ = [
    "EpochResponsePlotData",
    "ResponsePlotData",
    "default_analysis_output_path",
    "load_response_plot_data",
    "response_plot_data_from_grouped",
]


@dataclass(frozen=True)
class EpochResponsePlotData:
    """Mean and SEM traces for one stimulus epoch type.

    Inputs: grouped response trials for one epoch.
    Outputs: one mean and SEM trace per ROI on a shared time axis.
    """

    epoch_name: str
    epoch_number: int
    roi_labels: tuple[str, ...]
    time_seconds: npt.NDArray[np.float64]
    mean_values: npt.NDArray[np.float64]
    sem_values: npt.NDArray[np.float64]


@dataclass(frozen=True)
class ResponsePlotData:
    """Plot-ready ROI responses grouped by stimulus epoch.

    Inputs: grouped response analysis output.
    Outputs: epoch plot data plus the source analysis path.

    Each epoch carries its own direct ``time_seconds`` vector. The analysis HDF5
    stores the same vector per trial, so the plot never has to infer time from
    array position alone.
    """

    source_path: Path | None
    epochs: tuple[EpochResponsePlotData, ...]


def response_plot_data_from_grouped(
    grouped: GroupedRoiResponses,
    *,
    source_path: Path | None = None,
) -> ResponsePlotData:
    """Summarize grouped responses into mean and SEM traces for plotting.

    Args:
        grouped: Grouped response object from analysis.
        source_path: Optional analysis output path used for display.

    Returns:
        Plot-ready data with one item per stimulus epoch type.
    """
    epoch_keys: list[tuple[int, str]] = []
    trials_by_epoch: dict[tuple[int, str], list[RoiResponseTrial]] = {}
    for trial in grouped.trials:
        key = (trial.epoch_number, trial.epoch_name)
        if key not in trials_by_epoch:
            epoch_keys.append(key)
            trials_by_epoch[key] = []
        trials_by_epoch[key].append(trial)

    epochs = tuple(
        _epoch_plot_data(
            epoch_number=epoch_number,
            epoch_name=epoch_name,
            trials=tuple(trials_by_epoch[(epoch_number, epoch_name)]),
            roi_labels=grouped.roi_labels,
            data_rate_hz=grouped.data_rate_hz,
        )
        for epoch_number, epoch_name in sorted(epoch_keys)
    )
    return ResponsePlotData(source_path=source_path, epochs=epochs)


def default_analysis_output_path(recording: RecordingData) -> Path:
    """Return the default analysis output path for one recording.

    Args:
        recording: Loaded converted recording.

    Returns:
        Path to ``analysis_outputs.h5`` beside ``recording_data.h5``.
    """
    return recording.path.expanduser().parent / "analysis_outputs.h5"


def load_response_plot_data(path: Path) -> ResponsePlotData | str:
    """Load response plot data from an analysis output file.

    Args:
        path: Candidate ``analysis_outputs.h5`` path.

    Returns:
        Plot data, or a status message for the user.
    """
    if not path.is_file():
        return f"No analysis output found: {path}"
    outputs = load_analysis_outputs(path)
    if outputs.dff is not None and len(outputs.epoch_windows) > 0:
        data_rate_hz = _analysis_output_data_rate_hz(
            grouped=outputs.grouped_responses,
            dff=outputs.dff,
        )
        if data_rate_hz is not None:
            post_window_seconds = _response_post_window_seconds(outputs.epoch_windows)
            grouped = group_delta_f_over_f_by_epoch(
                outputs.dff,
                outputs.epoch_windows,
                data_rate_hz=data_rate_hz,
                pre_window_seconds=DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
                post_window_seconds=post_window_seconds,
            )
            return response_plot_data_from_grouped(grouped, source_path=outputs.path)
    if outputs.grouped_responses is None:
        return f"No grouped responses in: {path}"
    return response_plot_data_from_grouped(
        outputs.grouped_responses,
        source_path=outputs.path,
    )


def _epoch_plot_data(
    *,
    epoch_number: int,
    epoch_name: str,
    trials: tuple[RoiResponseTrial, ...],
    roi_labels: tuple[str, ...],
    data_rate_hz: float,
) -> EpochResponsePlotData:
    """Build plot data for one epoch type.

    Args:
        epoch_number: Stimulus epoch number.
        epoch_name: Stimulus epoch name.
        trials: Trial responses for this epoch type.
        roi_labels: ROI labels in column order.
        data_rate_hz: Imaging frame rate in hertz.

    Returns:
        Mean and SEM traces with shape ``(rois, frames)``.
    """
    time_seconds = _shared_time_axis(trials, data_rate_hz)
    first_time = float(time_seconds[0])
    max_frames = len(time_seconds)
    trial_count = len(trials)
    roi_count = len(roi_labels)
    values = np.full((trial_count, max_frames, roi_count), np.nan, dtype=np.float64)
    for trial_index, trial in enumerate(trials):
        offsets = np.rint((trial.time_seconds - first_time) * data_rate_hz).astype(
            np.int64,
            copy=False,
        )
        values[trial_index, offsets, :] = trial.values

    mean_values = np.nanmean(values, axis=0).T
    valid_counts = np.sum(~np.isnan(values), axis=0).T
    sem_values = np.zeros((roi_count, max_frames), dtype=np.float64)
    variable_mask = valid_counts > 1
    if np.any(variable_mask):
        std_values = np.nanstd(values, axis=0, ddof=1).T
        sem_values[variable_mask] = std_values[variable_mask] / np.sqrt(
            valid_counts[variable_mask]
        )
    return EpochResponsePlotData(
        epoch_name=epoch_name,
        epoch_number=epoch_number,
        roi_labels=roi_labels,
        time_seconds=time_seconds,
        mean_values=mean_values,
        sem_values=sem_values,
    )


def _shared_time_axis(
    trials: tuple[RoiResponseTrial, ...],
    data_rate_hz: float,
) -> npt.NDArray[np.float64]:
    """Return one relative time axis that covers all trials in an epoch.

    Args:
        trials: Trial responses for one epoch type.
        data_rate_hz: Imaging frame rate in hertz.

    Returns:
        Relative time in seconds, usually starting before stimulus onset and
        ending after stimulus offset when surrounding gray frames are available.
    """
    first_time = min(float(trial.time_seconds[0]) for trial in trials)
    last_time = max(float(trial.time_seconds[-1]) for trial in trials)
    frame_count = int(round((last_time - first_time) * data_rate_hz)) + 1
    return first_time + (np.arange(frame_count, dtype=np.float64) / data_rate_hz)


def _analysis_output_data_rate_hz(
    *,
    grouped: GroupedRoiResponses | None,
    dff: RoiDeltaFOverF,
) -> float | None:
    """Return the saved response frame rate when available.

    Args:
        grouped: Optional grouped responses loaded from analysis outputs.
        dff: dF/F result with audit metadata.

    Returns:
        Data rate in hertz, or ``None`` when unavailable.
    """
    if grouped is not None:
        return grouped.data_rate_hz
    value = dff.metadata.get("data_rate_hz")
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _response_post_window_seconds(
    epoch_windows: Sequence[EpochFrameWindow],
) -> float:
    """Return the default post-epoch plotting context from available windows.

    Args:
        epoch_windows: Iterable of epoch-window objects with ``epoch_name``.

    Returns:
        Two seconds when a gray interleave epoch exists, otherwise zero.
    """
    for epoch_window in epoch_windows:
        if is_gray_epoch_name(epoch_window.epoch_name):
            return DEFAULT_RESPONSE_POST_WINDOW_SECONDS
    return 0.0
