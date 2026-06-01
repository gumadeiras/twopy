"""Load response plot data for the twopy napari adapter.

This module keeps saved-analysis loading and Plot-tab defaults near the napari
adapter. GUI-neutral response plot data objects live in
``twopy.analysis.response_plotting``.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from twopy.analysis import response_plotting
from twopy.analysis.background_subtraction import BackgroundCorrectedRoiTraces
from twopy.analysis.dff import DeltaFOverFFitMode, RoiDeltaFOverF
from twopy.analysis.dff_options import DeltaFOverFBaselineMode, DeltaFOverFOptions
from twopy.analysis.persistence import LoadedAnalysisOutputs, load_analysis_outputs
from twopy.analysis.response_processing import (
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
    group_delta_f_over_f_by_epoch,
)
from twopy.analysis.timing import resolve_recording_timing
from twopy.analysis.trials import EpochFrameWindow, is_baseline_epoch_name
from twopy.converted import RecordingData
from twopy.filenames import ANALYSIS_OUTPUT_FILENAME
from twopy.stimulus import stimulus_epoch_names_by_number
from twopy.typing_guards import require_string_choice

__all__ = [
    "default_analysis_output_path",
    "load_response_plot_data",
    "response_plot_baseline_window_limit_for_recording",
    "response_plot_min_epoch_duration_for_recording",
    "response_plot_min_epoch_duration_seconds",
    "response_plot_post_window_seconds_for_recording",
    "response_plot_post_window_seconds",
    "response_plot_window_seconds_for_recording",
]


def default_analysis_output_path(recording: RecordingData) -> Path:
    """Return the default analysis output path for one recording.

    Args:
        recording: Loaded converted recording.

    Returns:
        Path to ``analysis_outputs.h5`` beside ``recording_data.h5``.
    """
    return recording.path.expanduser().parent / ANALYSIS_OUTPUT_FILENAME


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
    timing = resolve_recording_timing(recording)
    durations = tuple(
        (epoch_window.window.stop_frame - epoch_window.window.start_frame)
        / timing.frame_rate_hz
        for epoch_window in timing.epoch_windows
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


def response_plot_min_epoch_duration_for_recording(
    recording: RecordingData,
) -> float | None:
    """Return the shortest stimulus epoch duration for one recording.

    Args:
        recording: Loaded converted recording with stimulus timing metadata.

    Returns:
        Shortest positive photodiode-aligned epoch duration in seconds, or
        ``None`` when timing evidence is unavailable.
    """
    try:
        timing = resolve_recording_timing(recording)
    except ValueError:
        return None
    durations = tuple(
        (epoch_window.window.stop_frame - epoch_window.window.start_frame)
        / timing.frame_rate_hz
        for epoch_window in timing.epoch_windows
        if epoch_window.window.stop_frame > epoch_window.window.start_frame
    )
    return _minimum_positive_duration(durations)


def response_plot_min_epoch_duration_seconds(
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    data_rate_hz: float,
) -> float | None:
    """Return the shortest stimulus epoch duration from exact frame windows.

    Args:
        epoch_windows: Photodiode-aligned stimulus epoch windows.
        data_rate_hz: Imaging frame rate in hertz.

    Returns:
        Shortest positive epoch duration in seconds, or ``None`` when no
        positive duration can be computed.
    """
    if not np.isfinite(data_rate_hz) or data_rate_hz <= 0.0:
        return None
    return _minimum_positive_duration(
        (epoch.window.stop_frame - epoch.window.start_frame) / data_rate_hz
        for epoch in epoch_windows
        if epoch.window.stop_frame > epoch.window.start_frame
    )


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


def load_response_plot_data(path: Path) -> response_plotting.ResponsePlotData | str:
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
        saved_window_options = (
            response_plotting.response_plot_window_options_from_grouped(
                outputs.grouped_responses,
            )
        )
    else:
        saved_window_options = None
    if outputs.grouped_responses is not None and saved_window_options is not None:
        return _response_plot_data_from_outputs(
            outputs,
            outputs.grouped_responses,
            response_window_options=saved_window_options,
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
            return _response_plot_data_from_outputs(
                outputs,
                grouped,
                response_window_options=response_plotting.response_plot_window_options_from_grouped(
                    grouped,
                ),
            )
    if outputs.grouped_responses is None:
        return f"No grouped responses in: {path}"
    return _response_plot_data_from_outputs(
        outputs,
        outputs.grouped_responses,
        response_window_options=response_plotting.response_plot_window_options_from_grouped(
            outputs.grouped_responses,
        ),
    )


def _response_plot_data_from_outputs(
    outputs: LoadedAnalysisOutputs,
    grouped: GroupedRoiResponses,
    *,
    response_window_options: ResponseWindowOptions | None,
) -> response_plotting.ResponsePlotData:
    """Return plot data with shared saved-output metadata attached."""
    return response_plotting.response_plot_data_from_grouped(
        grouped,
        source_path=outputs.path,
        epoch_windows=outputs.epoch_windows,
        delta_f_over_f_options=_delta_f_over_f_options_from_outputs(
            traces=outputs.traces,
            dff=outputs.dff,
        ),
        response_window_options=response_window_options,
        response_processing_options=outputs.response_processing_options,
        correlation_scores=outputs.correlation_scores,
        correlation_window_stop_default_seconds=_correlation_stop_default_from_outputs(
            outputs,
        ),
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
        baseline_mode=_saved_baseline_mode(dff),
        baseline_epoch_number=_saved_baseline_epoch_number(dff),
        baseline_epoch_name=_saved_baseline_epoch_name(dff),
        background_method=background_method,
        baseline_sample_seconds=_saved_baseline_sample_seconds(dff),
        fit_mode=_saved_fit_mode(dff),
        apply_motion_mask="motion_artifact_masked_frame_count" in dff.metadata,
    )


def _saved_baseline_mode(dff: RoiDeltaFOverF) -> DeltaFOverFBaselineMode:
    """Return the saved baseline-selection mode from dF/F metadata."""
    value = dff.metadata.get("baseline_mode")
    if isinstance(value, str) and value in {"epoch", "no_baseline_epoch"}:
        return require_string_choice(
            value,
            name="dF/F baseline mode",
            allowed=("epoch", "no_baseline_epoch"),
        )
    return DeltaFOverFOptions().baseline_mode


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


def _saved_baseline_epoch_number(dff: RoiDeltaFOverF) -> int | None:
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


def _correlation_stop_default_from_outputs(
    outputs: LoadedAnalysisOutputs,
) -> float | None:
    """Return the exact saved-analysis correlation stop default."""
    if outputs.grouped_responses is None or len(outputs.epoch_windows) == 0:
        return None
    return response_plot_min_epoch_duration_seconds(
        outputs.epoch_windows,
        data_rate_hz=outputs.grouped_responses.data_rate_hz,
    )


def _minimum_positive_duration(durations: Iterable[float]) -> float | None:
    """Return the smallest finite positive duration from candidates."""
    positive = tuple(
        float(duration)
        for duration in durations
        if np.isfinite(duration) and duration > 0.0
    )
    if len(positive) == 0:
        return None
    return min(positive)


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
