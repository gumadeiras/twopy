"""Script-facing workflow for ROI response analysis.

Inputs: a converted recording, ROI masks, and analysis parameters.
Outputs: computed traces, dF/F, grouped responses, and saved analysis files.

This module wires together existing twopy analysis helpers. It does not read
source microscope files; callers must convert recordings before using it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twopy.analysis.background_subtraction import (
    BackgroundCorrectedRoiTraces,
    BackgroundCorrectionMethod,
    extract_background_corrected_roi_traces,
)
from twopy.analysis.dff import (
    DeltaFOverFFitMode,
    RoiDeltaFOverF,
    compute_roi_delta_f_over_f,
)
from twopy.analysis.epoch_mapping import (
    InterpolatedEpochMapping,
    interpolate_stimulus_epochs_to_frame_windows,
)
from twopy.analysis.motion import apply_motion_artifact_mask_to_delta_f_over_f
from twopy.analysis.persistence import save_analysis_outputs
from twopy.analysis.responses import GroupedRoiResponses, group_delta_f_over_f_by_epoch
from twopy.analysis.trials import (
    EpochFrameWindow,
    FrameWindow,
    select_epoch_frame_windows,
)
from twopy.converted import RecordingData, load_converted_recording
from twopy.roi import RoiSet, load_roi_set
from twopy.spatial import SpatialDomain

__all__ = [
    "AnalysisResponseRun",
    "analyze_recording_responses",
]


@dataclass(frozen=True)
class AnalysisResponseRun:
    """Outputs from one script-facing response-analysis run.

    Inputs: one converted recording and one ROI set.
    Outputs: all computed analysis objects plus file paths written by the run.

    Returning the objects lets scripts inspect results immediately while the
    HDF5/CSV files keep the same run auditable and reloadable later.
    """

    recording: RecordingData
    roi_set: RoiSet
    traces: BackgroundCorrectedRoiTraces
    dff: RoiDeltaFOverF
    epoch_mapping: InterpolatedEpochMapping | None
    epoch_windows: tuple[EpochFrameWindow, ...]
    interleave_windows: tuple[FrameWindow, ...]
    grouped_responses: GroupedRoiResponses
    output_path: Path
    response_summary_csv_path: Path | None


def analyze_recording_responses(
    recording_data_path: Path,
    roi_set: RoiSet | Path,
    *,
    output_path: Path | None = None,
    response_summary_csv_path: Path | None = None,
    write_summary_csv: bool = True,
    epoch_windows: Sequence[EpochFrameWindow] | None = None,
    interleave_epoch_number: int | None = 1,
    interleave_epoch_name: str | None = None,
    background_method: BackgroundCorrectionMethod = "movie_global_percentile",
    seconds_interleave_use: float | None = 1.0,
    fit_mode: DeltaFOverFFitMode = "robust",
    apply_motion_mask: bool = True,
    data_rate_hz: float | None = None,
    chunk_frames: int = 128,
    spatial_domain: SpatialDomain = "alignment_valid_crop",
) -> AnalysisResponseRun:
    """Run the standard ROI response analysis workflow.

    Args:
        recording_data_path: Path to converted ``recording_data.h5``.
        roi_set: ``RoiSet`` object or path to a saved ROI HDF5 file.
        output_path: Optional destination HDF5 path. When omitted, output is
            written beside ``recording_data.h5`` as ``analysis_outputs.h5``.
        response_summary_csv_path: Optional CSV path. When omitted and
            ``write_summary_csv`` is true, output is written beside the HDF5 file
            as ``response_summary.csv``.
        write_summary_csv: Whether to write the compact CSV summary.
        epoch_windows: Optional explicit epoch windows. When omitted, twopy
            interpolates epochs from converted stimulus data.
        interleave_epoch_number: Optional baseline epoch number selector.
        interleave_epoch_name: Optional baseline epoch name selector.
        background_method: Background correction method before dF/F.
        seconds_interleave_use: Seconds from the end of each interleave window.
        fit_mode: dF/F exponential fit mode.
        apply_motion_mask: Whether to mark converted high-motion frames as NaN
            after dF/F.
        data_rate_hz: Optional frame rate. When omitted, twopy reads
            ``acq.frameRate`` from converted metadata.
        chunk_frames: Number of movie frames to read per HDF5 chunk.
        spatial_domain: Spatial domain used for trace extraction.

    Returns:
        ``AnalysisResponseRun`` with computed objects and output paths.

    Raises:
        ValueError: If no interleave windows match the selector.

    This is the main script API for a converted recording: load, extract traces,
    compute dF/F, group responses, and persist the results.
    """
    recording = load_converted_recording(recording_data_path)
    loaded_roi_set = _resolve_roi_set(roi_set)
    frame_rate_hz = (
        _recording_frame_rate_hz(recording)
        if data_rate_hz is None
        else float(data_rate_hz)
    )
    mapping, resolved_epoch_windows = _resolve_epoch_windows(recording, epoch_windows)
    trace_start, trace_stop = _frame_range_for_epoch_windows(resolved_epoch_windows)
    interleave_windows = select_epoch_frame_windows(
        resolved_epoch_windows,
        epoch_name=interleave_epoch_name,
        epoch_number=interleave_epoch_number,
    )
    if len(interleave_windows) == 0:
        msg = "No interleave windows matched the requested selector"
        raise ValueError(msg)

    traces = extract_background_corrected_roi_traces(
        recording,
        loaded_roi_set,
        method=background_method,
        start_frame=trace_start,
        stop_frame=trace_stop,
        chunk_frames=chunk_frames,
        spatial_domain=spatial_domain,
    )
    dff = compute_roi_delta_f_over_f(
        traces,
        interleave_windows,
        data_rate_hz=frame_rate_hz,
        seconds_interleave_use=seconds_interleave_use,
        fit_mode=fit_mode,
    )
    if apply_motion_mask:
        dff = apply_motion_artifact_mask_to_delta_f_over_f(dff, recording)

    grouped_responses = group_delta_f_over_f_by_epoch(
        dff,
        resolved_epoch_windows,
        data_rate_hz=frame_rate_hz,
    )
    resolved_output_path = _resolve_output_path(recording.path, output_path)
    resolved_summary_path = _resolve_summary_csv_path(
        output_path=resolved_output_path,
        response_summary_csv_path=response_summary_csv_path,
        write_summary_csv=write_summary_csv,
    )
    save_analysis_outputs(
        resolved_output_path,
        roi_set=loaded_roi_set,
        traces=traces,
        dff=dff,
        epoch_windows=resolved_epoch_windows,
        grouped_responses=grouped_responses,
        response_summary_csv=resolved_summary_path,
    )

    return AnalysisResponseRun(
        recording=recording,
        roi_set=loaded_roi_set,
        traces=traces,
        dff=dff,
        epoch_mapping=mapping,
        epoch_windows=resolved_epoch_windows,
        interleave_windows=interleave_windows,
        grouped_responses=grouped_responses,
        output_path=resolved_output_path,
        response_summary_csv_path=resolved_summary_path,
    )


def _resolve_roi_set(roi_set: RoiSet | Path) -> RoiSet:
    """Return a ROI set from an object or HDF5 path.

    Args:
        roi_set: Existing ``RoiSet`` or path to one saved by ``save_roi_set``.

    Returns:
        ``RoiSet`` ready for trace extraction.
    """
    if isinstance(roi_set, Path):
        return load_roi_set(roi_set)
    return roi_set


def _resolve_epoch_windows(
    recording: RecordingData,
    epoch_windows: Sequence[EpochFrameWindow] | None,
) -> tuple[InterpolatedEpochMapping | None, tuple[EpochFrameWindow, ...]]:
    """Resolve epoch windows from caller input or converted stimulus data.

    Args:
        recording: Loaded converted recording.
        epoch_windows: Optional explicit epoch windows.

    Returns:
        ``(mapping, windows)``. ``mapping`` is ``None`` when windows were passed
        explicitly.
    """
    if epoch_windows is not None:
        return None, tuple(epoch_windows)
    mapping = interpolate_stimulus_epochs_to_frame_windows(recording)
    return mapping, mapping.windows


def _frame_range_for_epoch_windows(
    epoch_windows: Sequence[EpochFrameWindow],
) -> tuple[int, int]:
    """Return the smallest frame range covering all epoch windows.

    Args:
        epoch_windows: Windows used for response analysis.

    Returns:
        ``(start, stop)`` absolute movie-frame range.
    """
    if len(epoch_windows) == 0:
        msg = "At least one epoch window is required for response analysis"
        raise ValueError(msg)
    starts = [epoch_window.window.start_frame for epoch_window in epoch_windows]
    stops = [epoch_window.window.stop_frame for epoch_window in epoch_windows]
    return min(starts), max(stops)


def _recording_frame_rate_hz(recording: RecordingData) -> float:
    """Read the imaging frame rate from converted metadata.

    Args:
        recording: Loaded converted recording.

    Returns:
        Frame rate in hertz.
    """
    try:
        value = recording.acquisition_metadata["acq.frameRate"]
    except KeyError as error:
        msg = "Recording metadata is missing required field 'acq.frameRate'"
        raise ValueError(msg) from error
    if isinstance(value, str | bytes | int | float | np.generic):
        return float(value)

    value_type = type(value).__name__
    msg = f"Recording metadata field 'acq.frameRate' must be numeric, got {value_type}"
    raise ValueError(msg)


def _resolve_output_path(recording_data_path: Path, output_path: Path | None) -> Path:
    """Resolve the analysis HDF5 output path.

    Args:
        recording_data_path: Converted recording manifest path.
        output_path: Optional caller path.

    Returns:
        Expanded output path.
    """
    if output_path is not None:
        return output_path.expanduser()
    return recording_data_path.expanduser().parent / "analysis_outputs.h5"


def _resolve_summary_csv_path(
    *,
    output_path: Path,
    response_summary_csv_path: Path | None,
    write_summary_csv: bool,
) -> Path | None:
    """Resolve the optional response summary CSV path.

    Args:
        output_path: Analysis HDF5 output path.
        response_summary_csv_path: Optional caller path.
        write_summary_csv: Whether a CSV should be written.

    Returns:
        CSV path, or ``None`` when disabled.
    """
    if not write_summary_csv:
        return None
    if response_summary_csv_path is not None:
        return response_summary_csv_path.expanduser()
    return output_path.parent / "response_summary.csv"
