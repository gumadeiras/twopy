"""Resolve shared response-analysis timing context.

Inputs: converted recordings, optional caller-supplied windows, and processing
options.
Outputs: explicit frame ranges, baseline windows, frame rate, and validated
processing settings used by response analysis.

The context object lets interactive callers reuse movie-derived ROI traces
without duplicating workflow timing rules.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from twopy.analysis.dff_options import DeltaFOverFBaselineMode
from twopy.analysis.epoch_mapping import (
    InterpolatedEpochMapping,
    interpolate_stimulus_epochs_to_frame_windows,
)
from twopy.analysis.response_processing import (
    ResponseProcessingOptions,
    validate_response_processing_options,
)
from twopy.analysis.response_window_options import (
    DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
)
from twopy.analysis.trials import (
    EpochFrameWindow,
    FrameWindow,
    default_baseline_epoch_number,
    no_baseline_epoch_frame_windows,
    resolve_baseline_frame_windows,
)
from twopy.converted import RecordingData, recording_frame_rate_hz
from twopy.stimulus import stimulus_epoch_names_by_number

__all__ = [
    "ResponseAnalysisContext",
    "default_recording_baseline_epoch_number",
    "resolve_response_analysis_context",
]


@dataclass(frozen=True)
class ResponseAnalysisContext:
    """Resolved frame windows and options for one response-analysis pass.

    Inputs: one converted recording plus user-selected analysis settings.
    Outputs: explicit frame ranges, baseline windows, frame rate, and validated
    processing options shared by trace extraction and downstream response math.
    """

    frame_rate_hz: float
    processing_options: ResponseProcessingOptions
    epoch_mapping: InterpolatedEpochMapping | None
    epoch_windows: tuple[EpochFrameWindow, ...]
    baseline_windows: tuple[FrameWindow, ...]
    baseline_mode: DeltaFOverFBaselineMode
    baseline_epoch_number: int | None
    baseline_epoch_name: str | None
    trace_start: int
    trace_stop: int
    response_pre_window_seconds: float
    response_post_window_seconds: float


def resolve_response_analysis_context(
    recording: RecordingData,
    *,
    epoch_windows: Sequence[EpochFrameWindow] | None = None,
    baseline_windows: Sequence[FrameWindow] | None = None,
    baseline_mode: DeltaFOverFBaselineMode = "epoch",
    baseline_epoch_number: int | None = None,
    baseline_epoch_name: str | None = None,
    data_rate_hz: float | None = None,
    response_pre_window_seconds: float = DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
    response_post_window_seconds: float = 0.0,
    response_processing_options: ResponseProcessingOptions | None = None,
) -> ResponseAnalysisContext:
    """Resolve the shared timing context for response analysis.

    Args:
        recording: Loaded converted recording.
        epoch_windows: Optional explicit epoch windows. When omitted, twopy
            interpolates epochs from converted stimulus data.
        baseline_windows: Optional explicit baseline windows. When omitted,
            twopy selects baseline windows from epoch name or number.
        baseline_mode: Baseline selection mode. ``epoch`` uses selected epoch
            windows, while ``no_baseline_epoch`` uses one continuous span from
            the selected epoch number onward.
        baseline_epoch_number: Optional baseline epoch number selector. ``None``
            uses the recording-level default when no explicit windows or name
            selector are supplied.
        baseline_epoch_name: Optional baseline epoch name selector.
        data_rate_hz: Optional frame rate. When omitted, twopy reads converted
            acquisition metadata.
        response_pre_window_seconds: Seconds before each stimulus window to
            include in grouped responses.
        response_post_window_seconds: Seconds after each stimulus window to
            include in grouped responses.
        response_processing_options: Optional smoothing, low-pass, and
            correlation-QC settings to validate for this frame rate.

    Returns:
        Resolved analysis context shared by trace extraction and response math.
    """
    frame_rate_hz = (
        recording_frame_rate_hz(recording)
        if data_rate_hz is None
        else float(data_rate_hz)
    )
    processing_options = _resolve_response_processing_options(
        response_processing_options,
        data_rate_hz=frame_rate_hz,
    )
    mapping, resolved_epoch_windows = _resolve_epoch_windows(recording, epoch_windows)
    trace_start, trace_stop = _frame_range_for_epoch_windows(
        resolved_epoch_windows,
        frame_count=recording.movie.shape[0],
        data_rate_hz=frame_rate_hz,
        pre_window_seconds=response_pre_window_seconds,
        post_window_seconds=response_post_window_seconds,
    )
    resolved_baseline_epoch_number = _resolve_baseline_epoch_number(
        recording,
        baseline_windows=baseline_windows,
        baseline_epoch_name=baseline_epoch_name,
        baseline_epoch_number=baseline_epoch_number,
    )
    resolved_baseline_windows = _resolve_baseline_frame_windows(
        resolved_epoch_windows,
        baseline_windows=baseline_windows,
        baseline_mode=baseline_mode,
        epoch_name=baseline_epoch_name,
        epoch_number=resolved_baseline_epoch_number,
    )
    if len(resolved_baseline_windows) == 0:
        msg = "No baseline windows matched the requested selector"
        raise ValueError(msg)
    return ResponseAnalysisContext(
        frame_rate_hz=frame_rate_hz,
        processing_options=processing_options,
        epoch_mapping=mapping,
        epoch_windows=tuple(resolved_epoch_windows),
        baseline_windows=resolved_baseline_windows,
        baseline_mode=baseline_mode,
        baseline_epoch_number=resolved_baseline_epoch_number,
        baseline_epoch_name=baseline_epoch_name,
        trace_start=trace_start,
        trace_stop=trace_stop,
        response_pre_window_seconds=response_pre_window_seconds,
        response_post_window_seconds=response_post_window_seconds,
    )


def default_recording_baseline_epoch_number(recording: RecordingData) -> int:
    """Return the shared default baseline epoch for one recording.

    Args:
        recording: Loaded converted recording with stimulus parameter metadata.

    Returns:
        First epoch whose name looks like gray/grey/interleave, otherwise
        epoch one.

    GUI and script workflows both use this helper so default baseline selection
    cannot drift between entrypoints.
    """
    return default_baseline_epoch_number(stimulus_epoch_names_by_number(recording))


def _resolve_baseline_frame_windows(
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    baseline_windows: Sequence[FrameWindow] | None,
    baseline_mode: DeltaFOverFBaselineMode,
    epoch_name: str | None,
    epoch_number: int | None,
) -> tuple[FrameWindow, ...]:
    """Return baseline windows for the selected dF/F baseline mode."""
    if baseline_mode == "epoch":
        return resolve_baseline_frame_windows(
            epoch_windows,
            baseline_windows=baseline_windows,
            epoch_name=epoch_name,
            epoch_number=epoch_number,
        )
    if baseline_mode == "no_baseline_epoch":
        if baseline_windows is not None:
            return tuple(baseline_windows)
        resolved_epoch_number = (
            _first_epoch_number_named(epoch_windows, epoch_name)
            if epoch_name is not None
            else epoch_number
        )
        if resolved_epoch_number is None:
            msg = "no_baseline_epoch baseline mode requires an epoch selector"
            raise ValueError(msg)
        return no_baseline_epoch_frame_windows(
            epoch_windows,
            start_epoch_number=resolved_epoch_number,
        )

    msg = f"Unknown dF/F baseline mode: {baseline_mode!r}"
    raise ValueError(msg)


def _resolve_baseline_epoch_number(
    recording: RecordingData,
    *,
    baseline_windows: Sequence[FrameWindow] | None,
    baseline_epoch_name: str | None,
    baseline_epoch_number: int | None,
) -> int | None:
    """Return the concrete epoch-number selector for this analysis request."""
    if baseline_windows is not None:
        return baseline_epoch_number
    if baseline_epoch_number is not None:
        return baseline_epoch_number
    if baseline_epoch_name is not None:
        return None
    return default_recording_baseline_epoch_number(recording)


def _first_epoch_number_named(
    epoch_windows: Sequence[EpochFrameWindow],
    epoch_name: str,
) -> int | None:
    """Return the first epoch number whose name exactly matches."""
    for epoch_window in epoch_windows:
        if epoch_window.epoch_name == epoch_name:
            return epoch_window.epoch_number
    return None


def _resolve_response_processing_options(
    options: ResponseProcessingOptions | None,
    *,
    data_rate_hz: float,
) -> ResponseProcessingOptions:
    """Return validated response-processing options for one run."""
    resolved = ResponseProcessingOptions() if options is None else options
    validate_response_processing_options(resolved, data_rate_hz=data_rate_hz)
    return resolved


def _resolve_epoch_windows(
    recording: RecordingData,
    epoch_windows: Sequence[EpochFrameWindow] | None,
) -> tuple[InterpolatedEpochMapping | None, tuple[EpochFrameWindow, ...]]:
    """Resolve epoch windows from caller input or converted stimulus data."""
    if epoch_windows is not None:
        return None, tuple(epoch_windows)
    mapping = interpolate_stimulus_epochs_to_frame_windows(recording)
    return mapping, mapping.windows


def _frame_range_for_epoch_windows(
    epoch_windows: Sequence[EpochFrameWindow],
    *,
    frame_count: int,
    data_rate_hz: float,
    pre_window_seconds: float,
    post_window_seconds: float,
) -> tuple[int, int]:
    """Return the smallest frame range covering all epoch windows."""
    if len(epoch_windows) == 0:
        msg = "At least one epoch window is required for response analysis"
        raise ValueError(msg)
    pre_frames = int(round(pre_window_seconds * data_rate_hz))
    post_frames = int(round(post_window_seconds * data_rate_hz))
    starts = [
        max(epoch_window.window.start_frame - pre_frames, 0)
        for epoch_window in epoch_windows
    ]
    stops = [
        min(epoch_window.window.stop_frame + post_frames, frame_count)
        for epoch_window in epoch_windows
    ]
    return min(starts), max(stops)
