"""Build plot-ready response data for the twopy napari adapter.

Inputs: grouped response analysis outputs and saved analysis HDF5 files.
Outputs: mean/SEM traces with direct response time vectors for plotting.

This module has no Qt imports. It keeps response-data preparation testable and
separate from the napari widgets that draw and control the plots.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from twopy.analysis.background_subtraction import BackgroundCorrectedRoiTraces
from twopy.analysis.dff import DeltaFOverFFitMode, RoiDeltaFOverF
from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.epoch_mapping import interpolate_stimulus_epochs_to_frame_windows
from twopy.analysis.persistence import load_analysis_outputs
from twopy.analysis.response_processing import (
    ResponseProcessingOptions,
    mask_grouped_roi_responses_by_included_rois,
)
from twopy.analysis.response_window_options import (
    DEFAULT_RESPONSE_POST_WINDOW_SECONDS,
    DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
    ResponseWindowOptions,
    resolve_response_window_seconds,
)
from twopy.analysis.responses import (
    GroupedRoiResponses,
    RoiResponseTrial,
    group_delta_f_over_f_by_epoch,
)
from twopy.analysis.trials import EpochFrameWindow, is_baseline_epoch_name
from twopy.converted import RecordingData, recording_frame_rate_hz
from twopy.stimulus import stimulus_epoch_names_by_number
from twopy.typing_guards import require_string_choice

__all__ = [
    "EpochResponsePlotData",
    "ResponsePlotData",
    "default_analysis_output_path",
    "load_response_plot_data",
    "response_plot_baseline_window_limit_for_recording",
    "response_plot_post_window_seconds_for_recording",
    "response_plot_post_window_seconds",
    "response_plot_window_seconds_for_recording",
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
    delta_f_over_f_options: DeltaFOverFOptions | None = None
    response_window_options: ResponseWindowOptions | None = None
    response_processing_options: ResponseProcessingOptions | None = None


def response_plot_data_from_grouped(
    grouped: GroupedRoiResponses,
    *,
    source_path: Path | None = None,
    delta_f_over_f_options: DeltaFOverFOptions | None = None,
    response_window_options: ResponseWindowOptions | None = None,
    response_processing_options: ResponseProcessingOptions | None = None,
) -> ResponsePlotData:
    """Summarize grouped responses into mean and SEM traces for plotting.

    Args:
        grouped: Grouped response object from analysis.
        source_path: Optional analysis output path used for display.
        delta_f_over_f_options: Optional dF/F settings used to produce the
            grouped responses.
        response_window_options: Optional response-window settings used to
            produce the grouped responses.
        response_processing_options: Optional processing settings used to
            produce the grouped responses.

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
    return ResponsePlotData(
        source_path=source_path,
        epochs=epochs,
        delta_f_over_f_options=delta_f_over_f_options,
        response_window_options=(
            response_window_options
            if response_window_options is not None
            else _response_window_options_from_grouped(grouped)
        ),
        response_processing_options=response_processing_options,
    )


def default_analysis_output_path(recording: RecordingData) -> Path:
    """Return the default analysis output path for one recording.

    Args:
        recording: Loaded converted recording.

    Returns:
        Path to ``analysis_outputs.h5`` beside ``recording_data.h5``.
    """
    return recording.path.expanduser().parent / "analysis_outputs.h5"


def response_plot_post_window_seconds(epoch_names: Iterable[str]) -> float:
    """Return post-epoch context for response plots from epoch names.

    Args:
        epoch_names: Stimulus epoch names available for one recording or saved
            analysis output.

    Returns:
        Two seconds when a baseline-like gray/interleave epoch exists, otherwise zero.

    Response plots show the return into baseline context when the recording has
    such a baseline epoch. Keeping this rule shared prevents saved-output plots
    and live recompute previews from showing different x-axis ranges.
    """
    for epoch_name in epoch_names:
        if is_baseline_epoch_name(epoch_name):
            return DEFAULT_RESPONSE_POST_WINDOW_SECONDS
    return 0.0


def response_plot_post_window_seconds_for_recording(recording: RecordingData) -> float:
    """Return post-epoch plot context for one loaded recording.

    Args:
        recording: Loaded converted recording with stimulus epoch metadata.

    Returns:
        Two seconds when a baseline-like gray/interleave epoch exists, otherwise zero.
    """
    return response_plot_post_window_seconds(
        stimulus_epoch_names_by_number(recording).values(),
    )


def _response_plot_baseline_window_seconds_for_recording(
    recording: RecordingData,
) -> float | None:
    """Return the shortest named baseline epoch duration for one recording.

    Args:
        recording: Loaded converted recording with stimulus epoch metadata.

    Returns:
        Duration in seconds for the shortest gray/grey/interleave epoch window,
        or ``None`` when no such named window is available.

    Manual Plot-tab response windows are capped by this value so pre/post
    context cannot exceed the available baseline epoch duration.
    """
    mapping = interpolate_stimulus_epochs_to_frame_windows(recording)
    frame_rate_hz = recording_frame_rate_hz(recording)
    durations = tuple(
        (epoch_window.window.stop_frame - epoch_window.window.start_frame)
        / frame_rate_hz
        for epoch_window in mapping.windows
        if is_baseline_epoch_name(epoch_window.epoch_name)
    )
    if len(durations) == 0:
        return None
    return max(0.0, min(durations))


def response_plot_baseline_window_limit_for_recording(
    recording: RecordingData,
) -> float | None:
    """Return the manual response-window cap for one recording.

    Args:
        recording: Loaded converted recording with stimulus timing metadata.

    Returns:
        Shortest named gray/interleave epoch duration in seconds, or ``None``
        when no reliable cap is available.
    """
    try:
        return _response_plot_baseline_window_seconds_for_recording(recording)
    except ValueError:
        return None


def response_plot_window_seconds_for_recording(
    recording: RecordingData,
    options: ResponseWindowOptions,
) -> tuple[float, float]:
    """Resolve Plot-tab response-window options for one recording.

    Args:
        recording: Loaded converted recording.
        options: User-selected automatic/manual response-window options.

    Returns:
        ``(pre_window_seconds, post_window_seconds)`` for response grouping.
    """
    return resolve_response_window_seconds(
        options,
        automatic_pre_window_seconds=DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
        automatic_post_window_seconds=response_plot_post_window_seconds_for_recording(
            recording,
        ),
        max_window_seconds=response_plot_baseline_window_limit_for_recording(
            recording,
        ),
    )


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
    if outputs.grouped_responses is not None:
        saved_window_options = _response_window_options_from_grouped(
            outputs.grouped_responses,
        )
    else:
        saved_window_options = None
    if outputs.grouped_responses is not None and saved_window_options is not None:
        return response_plot_data_from_grouped(
            outputs.grouped_responses,
            source_path=outputs.path,
            delta_f_over_f_options=_delta_f_over_f_options_from_outputs(
                traces=outputs.traces,
                dff=outputs.dff,
            ),
            response_window_options=saved_window_options,
            response_processing_options=outputs.response_processing_options,
        )
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
            if outputs.correlation_scores is not None:
                grouped = mask_grouped_roi_responses_by_included_rois(
                    grouped,
                    included_mask=outputs.correlation_scores.included_mask,
                )
            return response_plot_data_from_grouped(
                grouped,
                source_path=outputs.path,
                delta_f_over_f_options=_delta_f_over_f_options_from_outputs(
                    traces=outputs.traces,
                    dff=outputs.dff,
                ),
                response_window_options=_response_window_options_from_grouped(grouped),
                response_processing_options=outputs.response_processing_options,
            )
    if outputs.grouped_responses is None:
        return f"No grouped responses in: {path}"
    return response_plot_data_from_grouped(
        outputs.grouped_responses,
        source_path=outputs.path,
        delta_f_over_f_options=_delta_f_over_f_options_from_outputs(
            traces=outputs.traces,
            dff=outputs.dff,
        ),
        response_window_options=_response_window_options_from_grouped(
            outputs.grouped_responses,
        ),
        response_processing_options=outputs.response_processing_options,
    )


def _response_window_options_from_grouped(
    grouped: GroupedRoiResponses,
) -> ResponseWindowOptions | None:
    """Return saved response-window options from grouped metadata."""
    pre_window_seconds = grouped.pre_window_seconds
    post_window_seconds = grouped.post_window_seconds
    if pre_window_seconds is None or post_window_seconds is None:
        return None
    return ResponseWindowOptions(
        auto=bool(grouped.response_window_auto)
        if grouped.response_window_auto is not None
        else False,
        pre_window_seconds=float(pre_window_seconds),
        post_window_seconds=float(post_window_seconds),
    )


def _delta_f_over_f_options_from_outputs(
    *,
    traces: BackgroundCorrectedRoiTraces | None,
    dff: RoiDeltaFOverF | None,
) -> DeltaFOverFOptions | None:
    """Return inspectable dF/F options from saved analysis metadata.

    Args:
        traces: Optional saved trace outputs carrying the background method.
        dff: Optional saved dF/F outputs carrying baseline and fit metadata.

    Returns:
        ``DeltaFOverFOptions`` when saved outputs expose dF/F settings,
        otherwise ``None``.
    """
    if traces is None and dff is None:
        return None

    defaults = DeltaFOverFOptions()
    background_method = (
        traces.method if traces is not None else defaults.background_method
    )
    if dff is None:
        return DeltaFOverFOptions(background_method=background_method)

    return DeltaFOverFOptions(
        baseline_epoch_number=_saved_baseline_epoch_number(dff),
        baseline_epoch_name=_saved_baseline_epoch_name(dff),
        background_method=background_method,
        baseline_sample_seconds=_saved_baseline_sample_seconds(dff),
        fit_mode=_saved_fit_mode(dff),
        apply_motion_mask="motion_artifact_masked_frame_count" in dff.metadata,
    )


def _saved_baseline_sample_seconds(dff: RoiDeltaFOverF) -> float | None:
    """Return the saved baseline sampling window from dF/F metadata.

    Args:
        dff: Saved dF/F object with audit metadata.

    Returns:
        Seconds from each baseline window end, or ``None`` for full windows.
    """
    value = dff.metadata.get("baseline_sample_seconds")
    if value == "full":
        return None
    if isinstance(value, bool):
        return DeltaFOverFOptions().baseline_sample_seconds
    if isinstance(value, int | float):
        return float(value)
    return DeltaFOverFOptions().baseline_sample_seconds


def _saved_baseline_epoch_number(dff: RoiDeltaFOverF) -> int:
    """Return the saved baseline epoch number selector from dF/F metadata."""
    value = dff.metadata.get("baseline_epoch_number")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return DeltaFOverFOptions().baseline_epoch_number


def _saved_baseline_epoch_name(dff: RoiDeltaFOverF) -> str | None:
    """Return the saved baseline epoch name selector from dF/F metadata."""
    value = dff.metadata.get("baseline_epoch_name")
    if isinstance(value, str):
        return value
    return DeltaFOverFOptions().baseline_epoch_name


def _saved_fit_mode(dff: RoiDeltaFOverF) -> DeltaFOverFFitMode:
    """Return the saved dF/F fit mode from metadata when recognized.

    Args:
        dff: Saved dF/F object with audit metadata.

    Returns:
        Recognized fit mode, otherwise the current default.
    """
    value = dff.metadata.get("fit_mode")
    if isinstance(value, str) and value in {
        "direct_bounded_tau",
        "direct_bounded_tau_and_log_amplitude",
        "log_linear",
    }:
        return require_string_choice(
            value,
            name="dF/F fit mode",
            allowed=(
                "direct_bounded_tau",
                "direct_bounded_tau_and_log_amplitude",
                "log_linear",
            ),
        )
    return DeltaFOverFOptions().fit_mode


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

    valid_counts = np.sum(~np.isnan(values), axis=0).T
    value_sums = np.nansum(values, axis=0).T
    mean_values = np.full((roi_count, max_frames), np.nan, dtype=np.float64)
    valid_mask = valid_counts > 0
    mean_values[valid_mask] = value_sums[valid_mask] / valid_counts[valid_mask]
    sem_values = np.zeros((roi_count, max_frames), dtype=np.float64)
    variable_mask = valid_counts > 1
    if np.any(variable_mask):
        values_by_roi_frame = np.moveaxis(values, 0, -1).transpose(1, 0, 2)
        std_values = np.nanstd(values_by_roi_frame[variable_mask], axis=1, ddof=1)
        sem_values[variable_mask] = std_values / np.sqrt(valid_counts[variable_mask])
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
    return response_plot_post_window_seconds(
        epoch_window.epoch_name for epoch_window in epoch_windows
    )
