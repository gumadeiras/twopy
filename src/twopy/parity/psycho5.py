"""Parity helpers for reproducing psycho5-default response-analysis choices.

Inputs: converted twopy recordings, ROI masks, and optional precomputed epoch
windows.
Outputs: twopy response-analysis objects computed with psycho5's default
baseline-selection and dF/F assumptions.

This module is deliberately inside ``twopy.parity`` rather than normal twopy
analysis. Normal analysis should use ``twopy.analysis`` APIs and native
fit-mode names.
"""

from collections.abc import Sequence
from pathlib import Path

from twopy.analysis.background_subtraction import BackgroundCorrectionMethod
from twopy.analysis.epoch_mapping import interpolate_stimulus_epochs_to_frame_windows
from twopy.analysis.response_processing import ResponseProcessingOptions
from twopy.analysis.trials import (
    EpochFrameWindow,
    default_baseline_epoch_number,
    select_baseline_frame_windows,
)
from twopy.analysis.workflow import (
    AnalysisResponseComputation,
    AnalysisResponseRun,
    analyze_recording_responses,
    compute_recording_responses,
)
from twopy.converted import RecordingData, load_converted_recording
from twopy.roi import RoiSet
from twopy.spatial import SpatialDomain
from twopy.stimulus import stimulus_epoch_names_by_number

__all__ = [
    "analyze_psycho5_default_recording_responses",
    "compute_psycho5_default_recording_responses",
    "psycho5_default_interleave_epoch_numbers",
]


def compute_psycho5_default_recording_responses(
    recording: RecordingData,
    roi_set: RoiSet,
    *,
    epoch_windows: Sequence[EpochFrameWindow] | None = None,
    background_method: BackgroundCorrectionMethod = "movie_global_percentile",
    apply_motion_mask: bool = True,
    data_rate_hz: float | None = None,
    chunk_frames: int = 128,
    spatial_domain: SpatialDomain = "alignment_valid_crop",
    response_pre_window_seconds: float = 0.0,
    response_post_window_seconds: float = 0.0,
    response_processing_options: ResponseProcessingOptions | None = None,
) -> AnalysisResponseComputation:
    """Compute twopy responses with psycho5-default analysis assumptions.

    Args:
        recording: Loaded converted recording.
        roi_set: ROI masks to extract from the recording.
        epoch_windows: Optional explicit epoch windows.
        background_method: Background correction method before dF/F.
        apply_motion_mask: Whether to mark converted high-motion frames as NaN.
        data_rate_hz: Optional frame rate.
        chunk_frames: Number of movie frames to read per HDF5 chunk.
        spatial_domain: Spatial domain used for trace extraction.
        response_pre_window_seconds: Seconds before each stimulus window to
            include in grouped responses.
        response_post_window_seconds: Seconds after each stimulus window to
            include in grouped responses.
        response_processing_options: Optional post-dF/F processing settings.

    Returns:
        In-memory response-analysis objects.

    The parity path uses psycho5's default interleave epoch selection, the full
    baseline window, and log-linear tau fitting. It is intentionally separated
    from native twopy defaults.
    """
    resolved_epoch_windows = _resolve_epoch_windows(recording, epoch_windows)
    baseline_windows = select_baseline_frame_windows(
        resolved_epoch_windows,
        epoch_number=None,
        epoch_numbers=psycho5_default_interleave_epoch_numbers(recording),
    )
    return compute_recording_responses(
        recording,
        roi_set,
        epoch_windows=resolved_epoch_windows,
        baseline_windows=baseline_windows,
        background_method=background_method,
        baseline_sample_seconds=None,
        fit_mode="log_linear",
        apply_motion_mask=apply_motion_mask,
        data_rate_hz=data_rate_hz,
        chunk_frames=chunk_frames,
        spatial_domain=spatial_domain,
        response_pre_window_seconds=response_pre_window_seconds,
        response_post_window_seconds=response_post_window_seconds,
        response_processing_options=response_processing_options,
    )


def analyze_psycho5_default_recording_responses(
    recording_data_path: Path,
    roi_set: RoiSet | Path,
    *,
    output_path: Path | None = None,
    response_summary_trials_csv_path: Path | None = None,
    response_summary_grouped_csv_path: Path | None = None,
    write_summary_csv: bool = True,
    epoch_windows: Sequence[EpochFrameWindow] | None = None,
    background_method: BackgroundCorrectionMethod = "movie_global_percentile",
    apply_motion_mask: bool = True,
    data_rate_hz: float | None = None,
    chunk_frames: int = 128,
    spatial_domain: SpatialDomain = "alignment_valid_crop",
    response_pre_window_seconds: float = 0.0,
    response_post_window_seconds: float = 0.0,
    response_processing_options: ResponseProcessingOptions | None = None,
) -> AnalysisResponseRun:
    """Run and persist twopy responses with psycho5-default assumptions.

    Args:
        recording_data_path: Path to converted ``recording_data.h5``.
        roi_set: ``RoiSet`` object or path to a saved ROI HDF5 file.
        output_path: Optional destination HDF5 path.
        response_summary_trials_csv_path: Optional trial-level CSV path.
        response_summary_grouped_csv_path: Optional grouped CSV path.
        write_summary_csv: Whether to write response time-series CSV files.
        epoch_windows: Optional explicit epoch windows.
        background_method: Background correction method before dF/F.
        apply_motion_mask: Whether to mark converted high-motion frames as NaN.
        data_rate_hz: Optional frame rate.
        chunk_frames: Number of movie frames to read per HDF5 chunk.
        spatial_domain: Spatial domain used for trace extraction.
        response_pre_window_seconds: Seconds before each stimulus window.
        response_post_window_seconds: Seconds after each stimulus window.
        response_processing_options: Optional post-dF/F processing settings.

    Returns:
        ``AnalysisResponseRun`` with computed objects and output paths.
    """
    recording = load_converted_recording(recording_data_path)
    resolved_epoch_windows = _resolve_epoch_windows(recording, epoch_windows)
    baseline_windows = select_baseline_frame_windows(
        resolved_epoch_windows,
        epoch_number=None,
        epoch_numbers=psycho5_default_interleave_epoch_numbers(recording),
    )
    return analyze_recording_responses(
        recording_data_path,
        roi_set,
        output_path=output_path,
        response_summary_trials_csv_path=response_summary_trials_csv_path,
        response_summary_grouped_csv_path=response_summary_grouped_csv_path,
        write_summary_csv=write_summary_csv,
        epoch_windows=resolved_epoch_windows,
        baseline_windows=baseline_windows,
        background_method=background_method,
        baseline_sample_seconds=None,
        fit_mode="log_linear",
        apply_motion_mask=apply_motion_mask,
        data_rate_hz=data_rate_hz,
        chunk_frames=chunk_frames,
        spatial_domain=spatial_domain,
        response_pre_window_seconds=response_pre_window_seconds,
        response_post_window_seconds=response_post_window_seconds,
        response_processing_options=response_processing_options,
    )


def psycho5_default_interleave_epoch_numbers(
    recording: RecordingData,
) -> tuple[int, ...]:
    """Return psycho5's default interleave epoch selection for one recording.

    Args:
        recording: Loaded converted recording with decoded stimulus parameters.

    Returns:
        One or more one-based epoch numbers to use as gray-interleave baselines.

    psycho5 first uses ``paramsHere(end).nextEpoch`` when present. Otherwise it
    selects epochs named exactly ``Gray Interleave`` and reverses the list when
    epoch one is the first of multiple gray interleaves.
    """
    if recording.stimulus_parameters:
        next_epoch = recording.stimulus_parameters[-1].get("nextEpoch")
        if next_epoch is not None:
            return _epoch_numbers_from_source_value(next_epoch, name="nextEpoch")

    exact_gray_epochs = tuple(
        index
        for index, parameters in enumerate(recording.stimulus_parameters, start=1)
        if parameters.get("epochName") == "Gray Interleave"
    )
    if len(exact_gray_epochs) > 1 and exact_gray_epochs[0] == 1:
        return exact_gray_epochs[::-1]
    if exact_gray_epochs:
        return exact_gray_epochs

    return (default_baseline_epoch_number(stimulus_epoch_names_by_number(recording)),)


def _resolve_epoch_windows(
    recording: RecordingData,
    epoch_windows: Sequence[EpochFrameWindow] | None,
) -> tuple[EpochFrameWindow, ...]:
    """Resolve parity epoch windows from caller input or converted stimulus data."""
    if epoch_windows is not None:
        return tuple(epoch_windows)
    return interpolate_stimulus_epochs_to_frame_windows(recording).windows


def _epoch_numbers_from_source_value(value: object, *, name: str) -> tuple[int, ...]:
    """Decode one MATLAB-like epoch-number value from converted metadata."""
    if isinstance(value, bool):
        msg = f"{name} must be an epoch number, not a boolean"
        raise ValueError(msg)
    if isinstance(value, int | float):
        return (_source_epoch_number(value, name=name),)
    if isinstance(value, list | tuple):
        epoch_numbers = tuple(_source_epoch_number(item, name=name) for item in value)
        if not epoch_numbers:
            msg = f"{name} must contain at least one epoch number"
            raise ValueError(msg)
        return epoch_numbers

    msg = f"{name} must be an epoch number or list of epoch numbers; got {value!r}"
    raise ValueError(msg)


def _source_epoch_number(value: object, *, name: str) -> int:
    """Validate one one-based epoch number from source metadata."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{name} entries must be numeric epoch numbers; got {value!r}"
        raise ValueError(msg)
    epoch_number = int(value)
    if epoch_number < 1 or float(value) != float(epoch_number):
        msg = f"{name} entries must be positive integer epoch numbers; got {value!r}"
        raise ValueError(msg)
    return epoch_number
