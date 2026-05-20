"""Persist twopy analysis outputs to inspectable files.

Inputs: ROI sets, extracted traces, dF/F results, epoch windows, and grouped
responses.
Outputs: one HDF5 analysis file plus optional CSV response time series.

This module only saves twopy-owned Python objects. It does not read source
MATLAB/TIFF files or decide analysis parameters.
"""

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from twopy.analysis.background_subtraction import (
    BackgroundCorrectedRoiTraces,
    BackgroundCorrectionMethod,
)
from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.response_processing import (
    ResponseProcessingOptions,
    RoiCorrelationScores,
    RoiNormalizationFactors,
)
from twopy.analysis.response_processing.persistence import (
    read_response_processing_group,
    write_response_processing_group,
)
from twopy.analysis.responses import (
    GroupedRoiResponses,
    RoiResponseTrial,
    finite_mean_and_sem,
    validate_grouped_roi_responses,
)
from twopy.analysis.trials import EpochFrameWindow, FrameWindow
from twopy.hdf5_utils import (
    plain_hdf5_attr_value,
)
from twopy.hdf5_utils import (
    read_string_dataset as _read_string_dataset,
)
from twopy.hdf5_utils import (
    write_string_dataset as _write_string_dataset,
)
from twopy.roi import RoiSet, TraceStatistic, make_roi_set
from twopy.typing_guards import (
    require_bool_array,
    require_float64_array,
    require_int64_array,
    require_string_choice,
)

__all__ = [
    "ANALYSIS_OUTPUT_FILE_FORMAT",
    "LoadedAnalysisOutputs",
    "load_analysis_outputs",
    "save_analysis_outputs",
    "write_response_summary_grouped_csv",
    "write_response_summary_trials_csv",
]

ANALYSIS_OUTPUT_FILE_FORMAT = "twopy-analysis-outputs"
_TRIAL_RESPONSE_CSV_BASE_COLUMNS = (
    "epoch_number",
    "epoch_name",
    "trial_index",
    "window_index",
    "roi_label",
    "start_frame",
    "stop_frame",
    "frame_count",
)
_GROUPED_RESPONSE_CSV_BASE_COLUMNS = (
    "epoch_number",
    "epoch_name",
    "roi_label",
    "statistic",
)
_TRACE_STATISTICS: tuple[TraceStatistic, ...] = ("mean",)
_BACKGROUND_CORRECTION_METHODS: tuple[BackgroundCorrectionMethod, ...] = (
    "none",
    "movie_global_percentile",
    "movie_y_stripe_percentile",
    "roi_y_stripe_percentile",
)


@dataclass(frozen=True)
class LoadedAnalysisOutputs:
    """Analysis objects loaded from one twopy analysis HDF5 file.

    Inputs: one file written by ``save_analysis_outputs``.
    Outputs: optional ROI, trace, dF/F, window, and response objects.

    Groups are optional because scripts can save only the products they have
    computed. Missing groups load as ``None`` or an empty window tuple.
    """

    path: Path
    roi_set: RoiSet | None
    traces: BackgroundCorrectedRoiTraces | None
    dff: RoiDeltaFOverF | None
    epoch_windows: tuple[EpochFrameWindow, ...]
    baseline_windows: tuple[FrameWindow, ...]
    grouped_responses: GroupedRoiResponses | None
    response_processing_options: ResponseProcessingOptions | None
    normalization_factors: RoiNormalizationFactors | None
    correlation_scores: RoiCorrelationScores | None


def save_analysis_outputs(
    path: Path,
    *,
    roi_set: RoiSet | None = None,
    traces: BackgroundCorrectedRoiTraces | None = None,
    dff: RoiDeltaFOverF | None = None,
    epoch_windows: Sequence[EpochFrameWindow] = (),
    baseline_windows: Sequence[FrameWindow] = (),
    grouped_responses: GroupedRoiResponses | None = None,
    response_processing_options: ResponseProcessingOptions | None = None,
    normalization_factors: RoiNormalizationFactors | None = None,
    correlation_scores: RoiCorrelationScores | None = None,
    response_summary_trials_csv: Path | None = None,
    response_summary_grouped_csv: Path | None = None,
) -> None:
    """Save analysis outputs to one HDF5 file and optional CSV time series.

    Args:
        path: Destination HDF5 path.
        roi_set: Optional ROI masks and labels.
        traces: Optional raw/background/corrected fluorescence traces.
        dff: Optional ROI dF/F result.
        epoch_windows: Optional stimulus windows used for response grouping.
        baseline_windows: Optional baseline windows used for dF/F fitting.
        grouped_responses: Optional grouped trial responses.
        response_processing_options: Optional smoothing, low-pass, and
            correlation-QC settings used for these outputs.
        normalization_factors: Optional per-ROI factors used for epoch-peak
            normalization.
        correlation_scores: Optional ROI-level correlation QC scores.
        response_summary_trials_csv: Optional CSV path for one row per trial
            and ROI, with one column per relative response timepoint.
        response_summary_grouped_csv: Optional CSV path for one row per epoch,
            ROI, and statistic, with one column per relative response
            timepoint.

    Returns:
        None.

    The HDF5 file stores arrays with gzip compression where arrays can be large.
    The HDF5 file is the complete audit trail. CSV files duplicate response
    time series in a spreadsheet-friendly layout for quick plotting and checks.
    """
    summary_outputs = (
        response_summary_trials_csv,
        response_summary_grouped_csv,
    )
    if any(path is not None for path in summary_outputs) and grouped_responses is None:
        msg = "response summary CSV outputs require grouped_responses"
        raise ValueError(msg)
    if (
        correlation_scores is not None or normalization_factors is not None
    ) and response_processing_options is None:
        msg = "response-processing audit outputs require response_processing_options"
        raise ValueError(msg)
    if grouped_responses is not None:
        validate_grouped_roi_responses(grouped_responses)

    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = ANALYSIS_OUTPUT_FILE_FORMAT
        if roi_set is not None:
            _write_roi_set(h5_file.create_group("roi_set"), roi_set)
        if traces is not None:
            _write_background_traces(h5_file.create_group("traces"), traces)
        if dff is not None:
            _write_dff(h5_file.create_group("dff"), dff)
        if len(epoch_windows) > 0:
            _write_epoch_windows(h5_file.create_group("epoch_windows"), epoch_windows)
        if len(baseline_windows) > 0:
            _write_frame_windows(
                h5_file.create_group("baseline_windows"),
                baseline_windows,
            )
        if grouped_responses is not None:
            _write_grouped_responses(
                h5_file.create_group("responses"),
                grouped_responses,
            )
        if response_processing_options is not None:
            write_response_processing_group(
                h5_file.create_group("response_processing"),
                options=response_processing_options,
                normalization_factors=normalization_factors,
                correlation_scores=correlation_scores,
            )

    if grouped_responses is not None:
        if response_summary_trials_csv is not None:
            write_response_summary_trials_csv(
                grouped_responses,
                response_summary_trials_csv,
            )
        if response_summary_grouped_csv is not None:
            write_response_summary_grouped_csv(
                grouped_responses,
                response_summary_grouped_csv,
            )


def load_analysis_outputs(path: Path) -> LoadedAnalysisOutputs:
    """Load twopy analysis outputs from HDF5.

    Args:
        path: Analysis HDF5 file written by ``save_analysis_outputs``.

    Returns:
        ``LoadedAnalysisOutputs`` with present groups reconstructed as typed
        Python objects.

    Raises:
        ValueError: If the file is not a twopy analysis output file.
    """
    input_path = path.expanduser()
    with h5py.File(input_path, "r") as h5_file:
        _require_analysis_format(h5_file, input_path)
        response_processing = (
            read_response_processing_group(h5_file["response_processing"])
            if "response_processing" in h5_file
            else None
        )
        return LoadedAnalysisOutputs(
            path=input_path,
            roi_set=(
                _read_roi_set(h5_file["roi_set"]) if "roi_set" in h5_file else None
            ),
            traces=(
                _read_background_traces(h5_file["traces"])
                if "traces" in h5_file
                else None
            ),
            dff=_read_dff(h5_file["dff"]) if "dff" in h5_file else None,
            epoch_windows=(
                _read_epoch_windows(h5_file["epoch_windows"])
                if "epoch_windows" in h5_file
                else ()
            ),
            baseline_windows=(
                _read_frame_windows(
                    h5_file["baseline_windows"], name="baseline_windows"
                )
                if "baseline_windows" in h5_file
                else ()
            ),
            grouped_responses=(
                _read_grouped_responses(h5_file["responses"])
                if "responses" in h5_file
                else None
            ),
            response_processing_options=(
                response_processing.options if response_processing is not None else None
            ),
            normalization_factors=(
                response_processing.normalization_factors
                if response_processing is not None
                else None
            ),
            correlation_scores=(
                response_processing.correlation_scores
                if response_processing is not None
                else None
            ),
        )


def write_response_summary_trials_csv(
    grouped_responses: GroupedRoiResponses,
    path: Path,
) -> None:
    """Write trial-level response time series to CSV.

    Args:
        grouped_responses: Grouped response object to write.
        path: Destination CSV path.

    Returns:
        None.
    """
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_grouped_roi_responses(grouped_responses)
    time_columns = _time_columns(grouped_responses.trials)
    fieldnames = (*_TRIAL_RESPONSE_CSV_BASE_COLUMNS, *time_columns)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for trial in grouped_responses.trials:
            for roi_index, roi_label in enumerate(grouped_responses.roi_labels):
                writer.writerow(
                    _trial_response_row(
                        trial,
                        roi_index=roi_index,
                        roi_label=roi_label,
                        time_columns=time_columns,
                    ),
                )


def write_response_summary_grouped_csv(
    grouped_responses: GroupedRoiResponses,
    path: Path,
) -> None:
    """Write epoch-level response time series to CSV.

    Args:
        grouped_responses: Grouped response object to write.
        path: Destination CSV path.

    Returns:
        None.
    """
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_grouped_roi_responses(grouped_responses)
    time_columns = _time_columns(grouped_responses.trials)
    fieldnames = (*_GROUPED_RESPONSE_CSV_BASE_COLUMNS, *time_columns)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in _grouped_response_rows(grouped_responses, time_columns):
            writer.writerow(row)


def _write_roi_set(group: h5py.Group, roi_set: RoiSet) -> None:
    """Write ROI masks and labels to an HDF5 group.

    Args:
        group: Destination HDF5 group.
        roi_set: ROI masks and labels.

    Returns:
        None.
    """
    group.create_dataset("masks", data=roi_set.masks, compression="gzip")
    _write_string_dataset(group, "labels", roi_set.labels)


def _read_roi_set(group: h5py.Group) -> RoiSet:
    """Read a persisted ROI set.

    Args:
        group: HDF5 ``roi_set`` group.

    Returns:
        ``RoiSet`` with validated masks and labels.
    """
    return make_roi_set(
        require_bool_array(group["masks"][()], name="roi_set/masks", ndim=3),
        labels=_read_string_dataset(group, "labels"),
    )


def _write_background_traces(
    group: h5py.Group,
    traces: BackgroundCorrectedRoiTraces,
) -> None:
    """Write background-corrected trace arrays.

    Args:
        group: Destination HDF5 group.
        traces: Trace object to persist.

    Returns:
        None.
    """
    group.attrs["start_frame"] = traces.start_frame
    group.attrs["stop_frame"] = traces.stop_frame
    group.attrs["statistic"] = traces.statistic
    group.attrs["method"] = traces.method
    _write_metadata(group.create_group("metadata"), traces.metadata)
    _write_string_dataset(group, "labels", traces.labels)
    group.create_dataset("raw_values", data=traces.raw_values, compression="gzip")
    group.create_dataset(
        "background_values",
        data=traces.background_values,
        compression="gzip",
    )
    group.create_dataset(
        "corrected_values",
        data=traces.corrected_values,
        compression="gzip",
    )


def _read_background_traces(group: h5py.Group) -> BackgroundCorrectedRoiTraces:
    """Read persisted background-corrected traces.

    Args:
        group: HDF5 ``traces`` group.

    Returns:
        ``BackgroundCorrectedRoiTraces`` reconstructed from arrays and attrs.
    """
    return BackgroundCorrectedRoiTraces(
        raw_values=require_float64_array(
            group["raw_values"][()],
            name="traces/raw_values",
            ndim=2,
        ),
        background_values=require_float64_array(
            group["background_values"][()],
            name="traces/background_values",
            ndim=2,
        ),
        corrected_values=require_float64_array(
            group["corrected_values"][()],
            name="traces/corrected_values",
            ndim=2,
        ),
        labels=_read_string_dataset(group, "labels"),
        start_frame=int(group.attrs["start_frame"]),
        stop_frame=int(group.attrs["stop_frame"]),
        statistic=require_string_choice(
            plain_hdf5_attr_value(group.attrs["statistic"], scalar_only=True),
            name="traces statistic",
            allowed=_TRACE_STATISTICS,
        ),
        method=require_string_choice(
            plain_hdf5_attr_value(group.attrs["method"], scalar_only=True),
            name="traces background correction method",
            allowed=_BACKGROUND_CORRECTION_METHODS,
        ),
        metadata=_read_metadata(group["metadata"]),
    )


def _write_dff(group: h5py.Group, dff: RoiDeltaFOverF) -> None:
    """Write ROI dF/F arrays and fit metadata.

    Args:
        group: Destination HDF5 group.
        dff: dF/F object to persist.

    Returns:
        None.
    """
    group.attrs["start_frame"] = dff.start_frame
    group.attrs["stop_frame"] = dff.stop_frame
    group.attrs["tau"] = dff.tau
    _write_metadata(group.create_group("metadata"), dff.metadata)
    _write_string_dataset(group, "labels", dff.labels)
    group.create_dataset("fluorescence", data=dff.fluorescence, compression="gzip")
    group.create_dataset("baseline", data=dff.baseline, compression="gzip")
    group.create_dataset("values", data=dff.values, compression="gzip")
    group.create_dataset("amplitudes", data=dff.amplitudes)
    group.create_dataset("baseline_frame_numbers", data=dff.baseline_frame_numbers)
    group.create_dataset(
        "baseline_fluorescence",
        data=dff.baseline_fluorescence,
        compression="gzip",
    )


def _read_dff(group: h5py.Group) -> RoiDeltaFOverF:
    """Read persisted ROI dF/F arrays and metadata.

    Args:
        group: HDF5 ``dff`` group.

    Returns:
        ``RoiDeltaFOverF`` reconstructed from saved datasets.
    """
    return RoiDeltaFOverF(
        fluorescence=require_float64_array(
            group["fluorescence"][()],
            name="dff/fluorescence",
            ndim=2,
        ),
        baseline=require_float64_array(
            group["baseline"][()],
            name="dff/baseline",
            ndim=2,
        ),
        values=require_float64_array(group["values"][()], name="dff/values", ndim=2),
        labels=_read_string_dataset(group, "labels"),
        start_frame=int(group.attrs["start_frame"]),
        stop_frame=int(group.attrs["stop_frame"]),
        tau=float(group.attrs["tau"]),
        amplitudes=require_float64_array(
            group["amplitudes"][()],
            name="dff/amplitudes",
            ndim=1,
        ),
        baseline_frame_numbers=require_float64_array(
            group["baseline_frame_numbers"][()],
            name="dff/baseline_frame_numbers",
            ndim=1,
        ),
        baseline_fluorescence=require_float64_array(
            group["baseline_fluorescence"][()],
            name="dff/baseline_fluorescence",
            ndim=2,
        ),
        metadata=_read_metadata(group["metadata"]),
    )


def _write_epoch_windows(
    group: h5py.Group,
    epoch_windows: Sequence[EpochFrameWindow],
) -> None:
    """Write stimulus epoch frame windows as column datasets.

    Args:
        group: Destination HDF5 group.
        epoch_windows: Epoch windows to persist.

    Returns:
        None.
    """
    _write_frame_windows(group, tuple(item.window for item in epoch_windows))
    group.create_dataset(
        "epoch_number",
        data=np.asarray([item.epoch_number for item in epoch_windows], dtype=np.int64),
    )
    _write_string_dataset(
        group,
        "epoch_name",
        tuple(item.epoch_name for item in epoch_windows),
    )


def _read_epoch_windows(group: h5py.Group) -> tuple[EpochFrameWindow, ...]:
    """Read persisted epoch windows.

    Args:
        group: HDF5 ``epoch_windows`` group.

    Returns:
        Tuple of stimulus-labeled frame windows.
    """
    windows = _read_frame_windows(group, name="epoch_windows")
    epoch_numbers = require_int64_array(
        group["epoch_number"][()],
        name="epoch_windows/epoch_number",
        ndim=1,
    )
    epoch_names = _read_string_dataset(group, "epoch_name")
    return tuple(
        EpochFrameWindow(
            window=window,
            epoch_number=int(epoch_number),
            epoch_name=epoch_name,
        )
        for (
            window,
            epoch_number,
            epoch_name,
        ) in zip(
            windows,
            epoch_numbers,
            epoch_names,
            strict=True,
        )
    )


def _write_frame_windows(group: h5py.Group, windows: Sequence[FrameWindow]) -> None:
    """Write plain frame windows as column datasets.

    Args:
        group: Destination HDF5 group.
        windows: Frame windows to persist.

    Returns:
        None.
    """
    group.create_dataset(
        "window_index",
        data=np.asarray([window.index for window in windows], dtype=np.int64),
    )
    group.create_dataset(
        "start_frame",
        data=np.asarray([window.start_frame for window in windows], dtype=np.int64),
    )
    group.create_dataset(
        "stop_frame",
        data=np.asarray([window.stop_frame for window in windows], dtype=np.int64),
    )
    _write_string_dataset(group, "label", tuple(window.label for window in windows))


def _read_frame_windows(group: h5py.Group, *, name: str) -> tuple[FrameWindow, ...]:
    """Read persisted plain frame windows.

    Args:
        group: HDF5 group carrying frame-window datasets.
        name: Group name used in validation messages.

    Returns:
        Tuple of frame windows.
    """
    window_indices = require_int64_array(
        group["window_index"][()],
        name=f"{name}/window_index",
        ndim=1,
    )
    start_frames = require_int64_array(
        group["start_frame"][()],
        name=f"{name}/start_frame",
        ndim=1,
    )
    stop_frames = require_int64_array(
        group["stop_frame"][()],
        name=f"{name}/stop_frame",
        ndim=1,
    )
    labels = _read_string_dataset(group, "label")
    return tuple(
        FrameWindow(
            index=int(window_index),
            start_frame=int(start_frame),
            stop_frame=int(stop_frame),
            label=label,
        )
        for window_index, start_frame, stop_frame, label in zip(
            window_indices,
            start_frames,
            stop_frames,
            labels,
            strict=True,
        )
    )


def _write_grouped_responses(
    group: h5py.Group,
    grouped: GroupedRoiResponses,
) -> None:
    """Write grouped response arrays.

    Args:
        group: Destination HDF5 group.
        grouped: Grouped responses to persist.

    Returns:
        None.

    Trial lengths can differ, so each trial gets its own subgroup instead of a
    padded cube with artificial fill values.
    """
    group.attrs["data_rate_hz"] = grouped.data_rate_hz
    if grouped.pre_window_seconds is not None:
        group.attrs["pre_window_seconds"] = grouped.pre_window_seconds
    if grouped.post_window_seconds is not None:
        group.attrs["post_window_seconds"] = grouped.post_window_seconds
    if grouped.response_window_auto is not None:
        group.attrs["response_window_auto"] = grouped.response_window_auto
    _write_string_dataset(group, "roi_labels", grouped.roi_labels)
    trials_group = group.create_group("trials")
    for trial_position, trial in enumerate(grouped.trials):
        trial_group = trials_group.create_group(f"trial_{trial_position + 1:04d}")
        trial_group.attrs["epoch_number"] = trial.epoch_number
        trial_group.attrs["epoch_name"] = trial.epoch_name
        trial_group.attrs["trial_index"] = trial.trial_index
        trial_group.attrs["window_index"] = trial.window_index
        trial_group.attrs["start_frame"] = trial.start_frame
        trial_group.attrs["stop_frame"] = trial.stop_frame
        trial_group.create_dataset("frame_numbers", data=trial.frame_numbers)
        trial_group.create_dataset("time_seconds", data=trial.time_seconds)
        trial_group.create_dataset("values", data=trial.values, compression="gzip")


def _read_grouped_responses(group: h5py.Group) -> GroupedRoiResponses:
    """Read persisted grouped response arrays.

    Args:
        group: HDF5 ``responses`` group.

    Returns:
        ``GroupedRoiResponses`` with one object per trial subgroup.
    """
    trials_group = group["trials"]
    grouped = GroupedRoiResponses(
        roi_labels=_read_string_dataset(group, "roi_labels"),
        data_rate_hz=float(group.attrs["data_rate_hz"]),
        trials=tuple(
            _read_response_trial(trials_group[name])
            for name in sorted(trials_group.keys())
        ),
        pre_window_seconds=_optional_float_attr(group, "pre_window_seconds"),
        post_window_seconds=_optional_float_attr(group, "post_window_seconds"),
        response_window_auto=_optional_bool_attr(group, "response_window_auto"),
    )
    validate_grouped_roi_responses(grouped)
    return grouped


def _read_response_trial(group: h5py.Group) -> RoiResponseTrial:
    """Read one persisted response trial group.

    Args:
        group: HDF5 trial subgroup.

    Returns:
        ``RoiResponseTrial`` with frame/time vectors and values.
    """
    return RoiResponseTrial(
        epoch_number=int(group.attrs["epoch_number"]),
        epoch_name=str(
            plain_hdf5_attr_value(group.attrs["epoch_name"], scalar_only=True),
        ),
        trial_index=int(group.attrs["trial_index"]),
        window_index=int(group.attrs["window_index"]),
        start_frame=int(group.attrs["start_frame"]),
        stop_frame=int(group.attrs["stop_frame"]),
        frame_numbers=require_int64_array(
            group["frame_numbers"][()],
            name="responses/trial/frame_numbers",
            ndim=1,
        ),
        time_seconds=require_float64_array(
            group["time_seconds"][()],
            name="responses/trial/time_seconds",
            ndim=1,
        ),
        values=require_float64_array(
            group["values"][()],
            name="responses/trial/values",
            ndim=2,
        ),
    )


def _optional_float_attr(group: h5py.Group, name: str) -> float | None:
    """Read one optional float HDF5 attribute.

    Args:
        group: HDF5 group carrying attrs.
        name: Attribute name to read.

    Returns:
        Float value, or ``None`` when the attribute is absent.
    """
    if name not in group.attrs:
        return None
    return float(group.attrs[name])


def _optional_bool_attr(group: h5py.Group, name: str) -> bool | None:
    """Read one optional boolean HDF5 attribute.

    Args:
        group: HDF5 group carrying attrs.
        name: Attribute name to read.

    Returns:
        Boolean value, or ``None`` when the attribute is absent.
    """
    if name not in group.attrs:
        return None
    value = group.attrs[name]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _write_metadata(
    group: h5py.Group,
    metadata: Mapping[str, str | int | float | bool],
) -> None:
    """Write simple metadata values as HDF5 attributes.

    Args:
        group: Destination HDF5 group.
        metadata: Metadata values from analysis objects.

    Returns:
        None.
    """
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool | np.generic):
            group.attrs[key] = value
        else:
            group.attrs[key] = json.dumps(value)


def _read_metadata(group: h5py.Group) -> dict[str, str | int | float | bool]:
    """Read simple metadata attributes.

    Args:
        group: HDF5 metadata group.

    Returns:
        Plain Python metadata dictionary.
    """
    return {
        key: plain_hdf5_attr_value(value, scalar_only=True)
        for key, value in group.attrs.items()
    }


def _time_columns(trials: Sequence[RoiResponseTrial]) -> tuple[str, ...]:
    """Return stable CSV column names for every relative response timepoint.

    Args:
        trials: Response trials whose time axes should become CSV columns.

    Returns:
        Sorted column names, one per unique relative time in seconds.

    Trials can include pre-window or post-window context and different epoch
    types can have different lengths. A shared union of timepoints lets one CSV
    hold all rows without padding arrays in memory.
    """
    time_values = sorted(
        {float(time) for trial in trials for time in trial.time_seconds},
    )
    columns = tuple(_time_column_name(time) for time in time_values)
    if len(columns) != len(set(columns)):
        msg = "Response time columns are not unique after formatting"
        raise ValueError(msg)
    return columns


def _time_column_name(time_seconds: float) -> str:
    """Format one relative response time as a CSV column name.

    Args:
        time_seconds: Relative time in seconds.

    Returns:
        Human-readable CSV column name.
    """
    return f"time_s_{time_seconds:.6f}"


def _trial_response_row(
    trial: RoiResponseTrial,
    *,
    roi_index: int,
    roi_label: str,
    time_columns: tuple[str, ...],
) -> dict[str, str | int | float]:
    """Convert one trial and ROI response trace to a CSV row.

    Args:
        trial: Trial response object.
        roi_index: Zero-based ROI column index in ``trial.values``.
        roi_label: Display label for that ROI.
        time_columns: Complete CSV time-column union.

    Returns:
        Dictionary keyed by trial-response CSV column name.
    """
    row: dict[str, str | int | float] = {
        "epoch_number": trial.epoch_number,
        "epoch_name": trial.epoch_name,
        "trial_index": trial.trial_index,
        "window_index": trial.window_index,
        "roi_label": roi_label,
        "start_frame": trial.start_frame,
        "stop_frame": trial.stop_frame,
        "frame_count": trial.stop_frame - trial.start_frame,
    }
    row.update(dict.fromkeys(time_columns, ""))
    for time_seconds, value in zip(
        trial.time_seconds,
        trial.values[:, roi_index],
        strict=True,
    ):
        row[_time_column_name(float(time_seconds))] = float(value)
    return row


def _grouped_response_rows(
    grouped: GroupedRoiResponses,
    time_columns: tuple[str, ...],
) -> tuple[dict[str, str | int | float], ...]:
    """Convert grouped trial responses to mean, SEM, and count CSV rows.

    Args:
        grouped: Grouped response object.
        time_columns: Complete CSV time-column union.

    Returns:
        Rows keyed by epoch, ROI, statistic, and relative timepoint.

    Counts are per timepoint rather than one scalar because pre/post context or
    clipped recording edges can make some trials shorter than others.
    """
    rows: list[dict[str, str | int | float]] = []
    for epoch_number, epoch_name, trials in _trials_by_epoch(grouped.trials):
        for roi_index, roi_label in enumerate(grouped.roi_labels):
            values_by_time = _values_by_time_column(
                trials,
                roi_index=roi_index,
                time_columns=time_columns,
            )
            rows.extend(
                _epoch_roi_statistic_rows(
                    epoch_number=epoch_number,
                    epoch_name=epoch_name,
                    roi_label=roi_label,
                    values_by_time=values_by_time,
                    time_columns=time_columns,
                ),
            )
    return tuple(rows)


def _trials_by_epoch(
    trials: Sequence[RoiResponseTrial],
) -> tuple[tuple[int, str, tuple[RoiResponseTrial, ...]], ...]:
    """Group trials by epoch while preserving first-seen epoch order.

    Args:
        trials: Trial responses to group.

    Returns:
        Tuple of ``(epoch_number, epoch_name, trials)`` groups.
    """
    grouped: dict[tuple[int, str], list[RoiResponseTrial]] = {}
    for trial in trials:
        grouped.setdefault((trial.epoch_number, trial.epoch_name), []).append(trial)
    return tuple(
        (epoch_number, epoch_name, tuple(epoch_trials))
        for (epoch_number, epoch_name), epoch_trials in grouped.items()
    )


def _values_by_time_column(
    trials: Sequence[RoiResponseTrial],
    *,
    roi_index: int,
    time_columns: tuple[str, ...],
) -> dict[str, list[float]]:
    """Collect repeated-trial values for one ROI by relative time column.

    Args:
        trials: Trials from one epoch type.
        roi_index: Zero-based ROI column index.
        time_columns: Complete CSV time-column union.

    Returns:
        Mapping from time column to observed response values.
    """
    values_by_time = {column: [] for column in time_columns}
    for trial in trials:
        for time_seconds, value in zip(
            trial.time_seconds,
            trial.values[:, roi_index],
            strict=True,
        ):
            if np.isfinite(value):
                values_by_time[_time_column_name(float(time_seconds))].append(
                    float(value),
                )
    return values_by_time


def _epoch_roi_statistic_rows(
    *,
    epoch_number: int,
    epoch_name: str,
    roi_label: str,
    values_by_time: Mapping[str, Sequence[float]],
    time_columns: tuple[str, ...],
) -> tuple[dict[str, str | int | float], ...]:
    """Build grouped mean, SEM, and count rows for one epoch and ROI.

    Args:
        epoch_number: Stimulus epoch number.
        epoch_name: Stimulus epoch name.
        roi_label: ROI label.
        values_by_time: Response values keyed by relative time column.
        time_columns: Complete CSV time-column union.

    Returns:
        Three CSV rows: ``mean``, ``sem``, and ``n_trials``.
    """
    base_row: dict[str, str | int | float] = {
        "epoch_number": epoch_number,
        "epoch_name": epoch_name,
        "roi_label": roi_label,
    }
    mean_row = {**base_row, "statistic": "mean"}
    sem_row = {**base_row, "statistic": "sem"}
    count_row = {**base_row, "statistic": "n_trials"}
    for column in time_columns:
        values = np.asarray(values_by_time[column], dtype=np.float64)
        if values.size > 0:
            mean, sem = finite_mean_and_sem(values, axis=0)
            mean_row[column] = float(mean)
            sem_row[column] = float(sem)
        else:
            mean_row[column] = ""
            sem_row[column] = ""
        count_row[column] = int(values.size)
    return mean_row, sem_row, count_row


def _require_analysis_format(h5_file: h5py.File, path: Path) -> None:
    """Validate the twopy analysis HDF5 format marker.

    Args:
        h5_file: Open analysis HDF5 file.
        path: Path used in error messages.

    Returns:
        None.
    """
    actual = str(plain_hdf5_attr_value(h5_file.attrs.get("twopy_format", "")))
    if actual != ANALYSIS_OUTPUT_FILE_FORMAT:
        msg = f"Expected {ANALYSIS_OUTPUT_FILE_FORMAT!r} file at {path}"
        raise ValueError(msg)
