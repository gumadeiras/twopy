"""Persist twopy analysis outputs to inspectable files.

Inputs: ROI sets, extracted traces, dF/F results, epoch windows, and grouped
responses.
Outputs: one HDF5 analysis file plus optional CSV response summaries.

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
from twopy.analysis.responses import (
    GroupedRoiResponses,
    GroupedRoiResponseSummary,
    RoiResponseSummary,
    RoiResponseTrial,
    summarize_epoch_roi_responses,
    summarize_grouped_responses,
)
from twopy.analysis.trials import EpochFrameWindow, FrameWindow
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
_SUMMARY_CSV_COLUMNS = (
    "epoch_number",
    "epoch_name",
    "trial_index",
    "window_index",
    "roi_label",
    "start_frame",
    "stop_frame",
    "frame_count",
    "mean_response",
    "peak_response",
    "min_response",
)
_GROUPED_SUMMARY_CSV_COLUMNS = (
    "epoch_number",
    "epoch_name",
    "roi_label",
    "mean_response",
    "sem_response",
    "n_trials",
)
_TRACE_STATISTICS: tuple[TraceStatistic, ...] = ("mean",)
_BACKGROUND_CORRECTION_METHODS: tuple[BackgroundCorrectionMethod, ...] = (
    "none",
    "movie_global_percentile",
    "roi_local_percentile_y",
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
    grouped_responses: GroupedRoiResponses | None


def save_analysis_outputs(
    path: Path,
    *,
    roi_set: RoiSet | None = None,
    traces: BackgroundCorrectedRoiTraces | None = None,
    dff: RoiDeltaFOverF | None = None,
    epoch_windows: Sequence[EpochFrameWindow] = (),
    grouped_responses: GroupedRoiResponses | None = None,
    response_summary_trials_csv: Path | None = None,
    response_summary_grouped_csv: Path | None = None,
) -> None:
    """Save analysis outputs to one HDF5 file and optional CSV summaries.

    Args:
        path: Destination HDF5 path.
        roi_set: Optional ROI masks and labels.
        traces: Optional raw/background/corrected fluorescence traces.
        dff: Optional ROI dF/F result.
        epoch_windows: Optional stimulus windows used for response grouping.
        grouped_responses: Optional grouped trial responses.
        response_summary_trials_csv: Optional CSV path for one row per trial
            and ROI.
        response_summary_grouped_csv: Optional CSV path for one row per epoch
            and ROI, collapsed across trials.

    Returns:
        None.

    The HDF5 file stores arrays with gzip compression where arrays can be large.
    CSV files are intentionally small summaries; frame-by-frame values stay in
    HDF5.
    """
    summary_outputs = (
        response_summary_trials_csv,
        response_summary_grouped_csv,
    )
    if any(path is not None for path in summary_outputs) and grouped_responses is None:
        msg = "response summary CSV outputs require grouped_responses"
        raise ValueError(msg)

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
        if grouped_responses is not None:
            _write_grouped_responses(
                h5_file.create_group("responses"),
                grouped_responses,
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
            grouped_responses=(
                _read_grouped_responses(h5_file["responses"])
                if "responses" in h5_file
                else None
            ),
        )


def write_response_summary_trials_csv(
    grouped_responses: GroupedRoiResponses,
    path: Path,
) -> None:
    """Write trial-level response summaries to CSV.

    Args:
        grouped_responses: Grouped response object to summarize.
        path: Destination CSV path.

    Returns:
        None.
    """
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = summarize_grouped_responses(grouped_responses)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_SUMMARY_CSV_COLUMNS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(_summary_row(summary))


def write_response_summary_grouped_csv(
    grouped_responses: GroupedRoiResponses,
    path: Path,
) -> None:
    """Write epoch-level response summaries to CSV.

    Args:
        grouped_responses: Grouped response object to summarize.
        path: Destination CSV path.

    Returns:
        None.
    """
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = summarize_epoch_roi_responses(grouped_responses)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_GROUPED_SUMMARY_CSV_COLUMNS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(_grouped_summary_row(summary))


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
            _plain_attr_value(group.attrs["statistic"]),
            name="traces statistic",
            allowed=_TRACE_STATISTICS,
        ),
        method=require_string_choice(
            _plain_attr_value(group.attrs["method"]),
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
    group.create_dataset("interleave_frame_numbers", data=dff.interleave_frame_numbers)
    group.create_dataset(
        "interleave_fluorescence",
        data=dff.interleave_fluorescence,
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
        interleave_frame_numbers=require_float64_array(
            group["interleave_frame_numbers"][()],
            name="dff/interleave_frame_numbers",
            ndim=1,
        ),
        interleave_fluorescence=require_float64_array(
            group["interleave_fluorescence"][()],
            name="dff/interleave_fluorescence",
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
    group.create_dataset(
        "window_index",
        data=np.asarray([item.window.index for item in epoch_windows], dtype=np.int64),
    )
    group.create_dataset(
        "start_frame",
        data=np.asarray(
            [item.window.start_frame for item in epoch_windows],
            dtype=np.int64,
        ),
    )
    group.create_dataset(
        "stop_frame",
        data=np.asarray(
            [item.window.stop_frame for item in epoch_windows],
            dtype=np.int64,
        ),
    )
    group.create_dataset(
        "epoch_number",
        data=np.asarray([item.epoch_number for item in epoch_windows], dtype=np.int64),
    )
    _write_string_dataset(
        group,
        "epoch_name",
        tuple(item.epoch_name for item in epoch_windows),
    )
    _write_string_dataset(
        group,
        "label",
        tuple(item.window.label for item in epoch_windows),
    )


def _read_epoch_windows(group: h5py.Group) -> tuple[EpochFrameWindow, ...]:
    """Read persisted epoch windows.

    Args:
        group: HDF5 ``epoch_windows`` group.

    Returns:
        Tuple of stimulus-labeled frame windows.
    """
    window_indices = require_int64_array(
        group["window_index"][()],
        name="epoch_windows/window_index",
        ndim=1,
    )
    start_frames = require_int64_array(
        group["start_frame"][()],
        name="epoch_windows/start_frame",
        ndim=1,
    )
    stop_frames = require_int64_array(
        group["stop_frame"][()],
        name="epoch_windows/stop_frame",
        ndim=1,
    )
    epoch_numbers = require_int64_array(
        group["epoch_number"][()],
        name="epoch_windows/epoch_number",
        ndim=1,
    )
    epoch_names = _read_string_dataset(group, "epoch_name")
    labels = _read_string_dataset(group, "label")
    return tuple(
        EpochFrameWindow(
            window=FrameWindow(
                index=int(window_index),
                start_frame=int(start_frame),
                stop_frame=int(stop_frame),
                label=label,
            ),
            epoch_number=int(epoch_number),
            epoch_name=epoch_name,
        )
        for (
            window_index,
            start_frame,
            stop_frame,
            epoch_number,
            epoch_name,
            label,
        ) in zip(
            window_indices,
            start_frames,
            stop_frames,
            epoch_numbers,
            epoch_names,
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
    return GroupedRoiResponses(
        roi_labels=_read_string_dataset(group, "roi_labels"),
        data_rate_hz=float(group.attrs["data_rate_hz"]),
        trials=tuple(
            _read_response_trial(trials_group[name])
            for name in sorted(trials_group.keys())
        ),
    )


def _read_response_trial(group: h5py.Group) -> RoiResponseTrial:
    """Read one persisted response trial group.

    Args:
        group: HDF5 trial subgroup.

    Returns:
        ``RoiResponseTrial`` with frame/time vectors and values.
    """
    return RoiResponseTrial(
        epoch_number=int(group.attrs["epoch_number"]),
        epoch_name=str(_plain_attr_value(group.attrs["epoch_name"])),
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
    return {key: _plain_attr_value(value) for key, value in group.attrs.items()}


def _write_string_dataset(
    group: h5py.Group,
    name: str,
    values: tuple[str, ...],
) -> None:
    """Write a tuple of strings as a UTF-8 HDF5 dataset.

    Args:
        group: Destination HDF5 group.
        name: Dataset name.
        values: String values to write.

    Returns:
        None.
    """
    group.create_dataset(
        name,
        data=np.asarray(values, dtype=h5py.string_dtype("utf-8")),
    )


def _read_string_dataset(group: h5py.Group, name: str) -> tuple[str, ...]:
    """Read a UTF-8 string dataset.

    Args:
        group: HDF5 group containing the dataset.
        name: Dataset name.

    Returns:
        Tuple of decoded strings.
    """
    return tuple(_decode_string(value) for value in group[name][()])


def _summary_row(summary: RoiResponseSummary) -> dict[str, str | int | float]:
    """Convert one summary object to a CSV row.

    Args:
        summary: Response summary object.

    Returns:
        Dictionary keyed by CSV column name.
    """
    return {
        "epoch_number": summary.epoch_number,
        "epoch_name": summary.epoch_name,
        "trial_index": summary.trial_index,
        "window_index": summary.window_index,
        "roi_label": summary.roi_label,
        "start_frame": summary.start_frame,
        "stop_frame": summary.stop_frame,
        "frame_count": summary.frame_count,
        "mean_response": summary.mean_response,
        "peak_response": summary.peak_response,
        "min_response": summary.min_response,
    }


def _grouped_summary_row(
    summary: GroupedRoiResponseSummary,
) -> dict[str, str | int | float]:
    """Convert one grouped summary object to a CSV row.

    Args:
        summary: Grouped response summary object.

    Returns:
        Dictionary keyed by grouped CSV column name.
    """
    return {
        "epoch_number": summary.epoch_number,
        "epoch_name": summary.epoch_name,
        "roi_label": summary.roi_label,
        "mean_response": summary.mean_response,
        "sem_response": summary.sem_response,
        "n_trials": summary.n_trials,
    }


def _require_analysis_format(h5_file: h5py.File, path: Path) -> None:
    """Validate the twopy analysis HDF5 format marker.

    Args:
        h5_file: Open analysis HDF5 file.
        path: Path used in error messages.

    Returns:
        None.
    """
    actual = str(_plain_attr_value(h5_file.attrs.get("twopy_format", "")))
    if actual != ANALYSIS_OUTPUT_FILE_FORMAT:
        msg = f"Expected {ANALYSIS_OUTPUT_FILE_FORMAT!r} file at {path}"
        raise ValueError(msg)


def _plain_attr_value(value: object) -> str | int | float | bool:
    """Convert one HDF5 attribute value to a plain Python scalar.

    Args:
        value: Raw HDF5 attribute value.

    Returns:
        Plain string, integer, float, or boolean.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer | int):
        return int(value)
    if isinstance(value, np.floating | float):
        return float(value)
    return str(value)


def _decode_string(value: object) -> str:
    """Decode one HDF5 string scalar.

    Args:
        value: Raw scalar read from a string dataset.

    Returns:
        Decoded text.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
