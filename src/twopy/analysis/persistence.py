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
from pathlib import Path

import h5py
import numpy as np

from twopy.analysis.background_subtraction import BackgroundCorrectedRoiTraces
from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.responses import (
    GroupedRoiResponses,
    RoiResponseSummary,
    summarize_grouped_responses,
)
from twopy.analysis.trials import EpochFrameWindow
from twopy.roi import RoiSet

__all__ = [
    "ANALYSIS_OUTPUT_FILE_FORMAT",
    "save_analysis_outputs",
    "write_response_summary_csv",
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


def save_analysis_outputs(
    path: Path,
    *,
    roi_set: RoiSet | None = None,
    traces: BackgroundCorrectedRoiTraces | None = None,
    dff: RoiDeltaFOverF | None = None,
    epoch_windows: Sequence[EpochFrameWindow] = (),
    grouped_responses: GroupedRoiResponses | None = None,
    response_summary_csv: Path | None = None,
) -> None:
    """Save analysis outputs to one HDF5 file and optional CSV summary.

    Args:
        path: Destination HDF5 path.
        roi_set: Optional ROI masks and labels.
        traces: Optional raw/background/corrected fluorescence traces.
        dff: Optional ROI dF/F result.
        epoch_windows: Optional stimulus windows used for response grouping.
        grouped_responses: Optional grouped trial responses.
        response_summary_csv: Optional CSV path for one-row-per-trial-ROI
            response summaries.

    Returns:
        None.

    The HDF5 file stores arrays with gzip compression where arrays can be large.
    The CSV is intentionally small and summary-only; frame-by-frame values stay
    in HDF5.
    """
    summary_output: tuple[GroupedRoiResponses, Path] | None = None
    if response_summary_csv is not None:
        if grouped_responses is None:
            msg = "response_summary_csv requires grouped_responses"
            raise ValueError(msg)
        summary_output = (grouped_responses, response_summary_csv)

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

    if summary_output is not None:
        summary_grouped_responses, summary_csv_path = summary_output
        write_response_summary_csv(summary_grouped_responses, summary_csv_path)


def write_response_summary_csv(
    grouped_responses: GroupedRoiResponses,
    path: Path,
) -> None:
    """Write compact response summaries to CSV.

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
