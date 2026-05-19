"""Stimulus-table assembly for temporal kernel fitting.

Inputs: converted stimulus tables, epoch windows, and ROI dF/F traces.
Outputs: regular stimulus vectors plus sparse per-ROI response samples.

This module owns recording-specific data assembly. Numeric fitting lives in
``twopy.analysis.kernels._fitting``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from twopy.analysis.kernels._fitting import (
    stimulus_dt,
    validate_regular_stimulus_times,
)
from twopy.analysis.trials import EpochFrameWindow
from twopy.analysis.workflow import AnalysisResponseComputation
from twopy.converted import Hemisphere, RecordingData, recording_hemisphere
from twopy.stimulus import stimulus_epoch_names_by_number

KernelStimulusModality = Literal["olfaction", "vision"]

_DEFAULT_STIMULUS_COLUMNS: dict[KernelStimulusModality, str] = {
    "olfaction": "stimulus_specific_05",
    "vision": "stimulus_specific_05",
}


class _KernelOptions(Protocol):
    """Small option surface needed for hemisphere resolution."""

    stimulus_modality: KernelStimulusModality
    hemisphere: Hemisphere | None


@dataclass(frozen=True)
class StimulusSegment:
    """One contiguous stimulus-table block with a stable epoch number."""

    epoch_number: int
    start: int
    stop: int


@dataclass(frozen=True)
class StimulusEpochGroup:
    """Selected epoch numbers that share one readable epoch name."""

    name: str
    epoch_numbers: tuple[int, ...]


@dataclass(frozen=True)
class AssembledKernelInput:
    """Regular stimulus vectors plus per-ROI response samples."""

    stimulus_times: npt.NDArray[np.float64]
    stimulus: npt.NDArray[np.float64]
    response_times_by_roi: tuple[npt.NDArray[np.float64], ...]
    responses_by_roi: tuple[npt.NDArray[np.float64], ...]


def default_kernel_stimulus_column(
    stimulus_modality: KernelStimulusModality,
) -> str:
    """Return the converted stimulus column used by default for one modality.

    Args:
        stimulus_modality: ``"olfaction"`` for LED antenna random noise or
            ``"vision"`` for full-field contrast flicker.

    Returns:
        Converted twopy column name.

    The current lab workflows both store their primary scalar stimulus in
    ``stimulus_specific_05`` after conversion, but the interpretation differs:
    antenna activation codes for olfaction, signed contrast for vision.
    """
    if stimulus_modality not in _DEFAULT_STIMULUS_COLUMNS:
        msg = f"Unknown kernel stimulus modality {stimulus_modality!r}."
        raise ValueError(msg)
    return _DEFAULT_STIMULUS_COLUMNS[stimulus_modality]


def resolve_kernel_hemisphere(
    computation: AnalysisResponseComputation,
    options: _KernelOptions,
) -> Hemisphere | None:
    """Return the hemisphere used for ipsi/contra remapping."""
    if options.stimulus_modality == "vision":
        return None
    if options.hemisphere is not None:
        return options.hemisphere
    return recording_hemisphere(computation.recording)


def frame_rate_from_computation(computation: AnalysisResponseComputation) -> float:
    """Return the timing frame rate used for the response computation."""
    if computation.timing is not None:
        return float(computation.timing.frame_rate_hz)
    return float(computation.grouped_responses.data_rate_hz)


def stimulus_epoch_segments(
    epoch_values: npt.NDArray[np.float64],
    *,
    baseline_epoch_number: int,
) -> tuple[StimulusSegment, ...]:
    """Return positive non-baseline contiguous epoch segments."""
    if epoch_values.ndim != 1:
        msg = "stimulus epoch values must be one-dimensional."
        raise ValueError(msg)
    segments: list[StimulusSegment] = []
    start = 0
    while start < epoch_values.size:
        epoch_number = int(round(float(epoch_values[start])))
        stop = start + 1
        while (
            stop < epoch_values.size
            and int(round(float(epoch_values[stop]))) == epoch_number
        ):
            stop += 1
        if epoch_number > 0 and epoch_number != baseline_epoch_number:
            segments.append(StimulusSegment(epoch_number, start, stop))
        start = stop
    if len(segments) == 0:
        msg = "No non-baseline stimulus epochs are available for kernel fitting."
        raise ValueError(msg)
    return tuple(segments)


def selected_epoch_numbers(
    segments: Sequence[StimulusSegment],
    *,
    discard_first: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return selected and discarded non-baseline epoch numbers."""
    ordered_numbers: list[int] = []
    for segment in segments:
        if segment.epoch_number not in ordered_numbers:
            ordered_numbers.append(segment.epoch_number)
    if discard_first:
        discarded = tuple(ordered_numbers[:1])
        selected = tuple(ordered_numbers[1:])
    else:
        discarded = ()
        selected = tuple(ordered_numbers)
    if len(selected) == 0:
        msg = "No stimulus epochs remain after the requested first-epoch discard."
        raise ValueError(msg)
    return selected, discarded


def selected_epoch_groups(
    recording: RecordingData,
    *,
    selected_epoch_numbers: tuple[int, ...],
) -> tuple[StimulusEpochGroup, ...]:
    """Group selected epoch numbers by readable epoch name."""
    names_by_number = stimulus_epoch_names_by_number(recording)
    grouped: dict[str, list[int]] = {}
    ordered_names: list[str] = []
    for epoch_number in selected_epoch_numbers:
        epoch_name = names_by_number.get(epoch_number, f"epoch_{epoch_number}")
        if epoch_name not in grouped:
            grouped[epoch_name] = []
            ordered_names.append(epoch_name)
        grouped[epoch_name].append(epoch_number)
    return tuple(
        StimulusEpochGroup(name=name, epoch_numbers=tuple(grouped[name]))
        for name in ordered_names
    )


def assemble_kernel_input(
    computation: AnalysisResponseComputation,
    *,
    segments: Sequence[StimulusSegment],
    selected_epoch_numbers: tuple[int, ...],
    time_column: int,
    stimulus_column: int,
    stimulus_modality: KernelStimulusModality,
    frame_rate_hz: float,
) -> AssembledKernelInput:
    """Build complete regular stimulus vectors and sparse ROI responses."""
    stimulus_times: list[npt.NDArray[np.float64]] = []
    stimulus_values: list[npt.NDArray[np.float64]] = []
    response_times_by_roi: list[list[npt.NDArray[np.float64]]] = [
        [] for _ in computation.roi_set.labels
    ]
    responses_by_roi: list[list[npt.NDArray[np.float64]]] = [
        [] for _ in computation.roi_set.labels
    ]
    epoch_windows_by_number = _epoch_windows_by_number(computation.epoch_windows)
    offset = 0.0
    for segment in segments:
        if segment.epoch_number not in selected_epoch_numbers:
            continue
        matching_windows = epoch_windows_by_number.get(segment.epoch_number)
        if matching_windows is None or len(matching_windows) == 0:
            msg = (
                f"No aligned response window for stimulus epoch {segment.epoch_number}."
            )
            raise ValueError(msg)
        window = matching_windows.pop(0).window
        rows = slice(segment.start, segment.stop)
        epoch_times = computation.recording.stimulus_data[rows, time_column].astype(
            np.float64,
            copy=False,
        )
        if epoch_times.size < 2:
            msg = f"Stimulus epoch {segment.epoch_number} has fewer than two samples."
            raise ValueError(msg)
        relative_stimulus_times = epoch_times - float(epoch_times[0])
        validate_regular_stimulus_times(
            relative_stimulus_times,
            context=f"stimulus epoch {segment.epoch_number}",
        )
        stimulus_times.append(relative_stimulus_times + offset)
        stimulus_values.append(
            kernel_stimulus_values(
                computation.recording.stimulus_data[rows, stimulus_column],
                stimulus_modality=stimulus_modality,
                context=f"stimulus epoch {segment.epoch_number}",
            )
        )
        frame_numbers = np.arange(
            window.start_frame,
            window.stop_frame,
            dtype=np.int64,
        )
        local_start = window.start_frame - computation.dff.start_frame
        local_stop = window.stop_frame - computation.dff.start_frame
        if local_start < 0 or local_stop > computation.dff.values.shape[0]:
            msg = (
                f"Response window for epoch {segment.epoch_number} is outside "
                "dF/F traces."
            )
            raise ValueError(msg)
        response_times = (frame_numbers - window.start_frame) / frame_rate_hz + offset
        response_values = computation.dff.values[local_start:local_stop, :]
        for roi_index in range(response_values.shape[1]):
            finite = np.isfinite(response_values[:, roi_index])
            response_times_by_roi[roi_index].append(response_times[finite])
            responses_by_roi[roi_index].append(response_values[finite, roi_index])
        offset = float(
            relative_stimulus_times[-1] + stimulus_dt(relative_stimulus_times) + offset,
        )

    if len(stimulus_times) == 0:
        msg = "No selected stimulus epochs were assembled for kernel fitting."
        raise ValueError(msg)
    return AssembledKernelInput(
        stimulus_times=np.concatenate(stimulus_times).astype(np.float64, copy=False),
        stimulus=np.concatenate(stimulus_values).astype(np.float64, copy=False),
        response_times_by_roi=tuple(
            np.concatenate(values).astype(np.float64, copy=False)
            if len(values) > 0
            else np.array([], dtype=np.float64)
            for values in response_times_by_roi
        ),
        responses_by_roi=tuple(
            np.concatenate(values).astype(np.float64, copy=False)
            if len(values) > 0
            else np.array([], dtype=np.float64)
            for values in responses_by_roi
        ),
    )


def kernel_stimulus_values(
    values: npt.NDArray[np.float64],
    *,
    stimulus_modality: KernelStimulusModality,
    context: str,
) -> npt.NDArray[np.float64]:
    """Convert one stimulus column into the stream fit by default."""
    if stimulus_modality == "olfaction":
        left_stimulus, _ = antenna_activation_to_left_right(
            values,
            context=f"{context} antenna activation",
        )
        return left_stimulus
    return visual_contrast_stimulus(values, context=f"{context} visual contrast")


def antenna_activation_to_right_stream(
    values: npt.NDArray[np.float64],
    *,
    segments: Sequence[StimulusSegment],
    selected_epoch_numbers: tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Return concatenated right-antenna streams for selected segments."""
    right_values: list[npt.NDArray[np.float64]] = []
    for segment in segments:
        if segment.epoch_number not in selected_epoch_numbers:
            continue
        _, right_stimulus = antenna_activation_to_left_right(
            values[segment.start : segment.stop],
            context=f"antenna activation epoch {segment.epoch_number}",
        )
        right_values.append(right_stimulus)
    if len(right_values) == 0:
        msg = "No right antenna stimulus epochs were assembled for kernel fitting."
        raise ValueError(msg)
    return np.concatenate(right_values).astype(np.float64, copy=False)


def map_raw_to_ipsi_contra(
    raw_left: npt.NDArray[np.float64],
    raw_right: npt.NDArray[np.float64],
    *,
    hemisphere: Hemisphere,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Map raw left/right stimulus kernels into ipsi/contra convention."""
    if hemisphere == "right":
        return raw_right, raw_left
    return raw_left, raw_right


def antenna_activation_to_left_right(
    values: npt.NDArray[np.float64],
    *,
    context: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Derive signed left/right stimulus streams from LED antenna codes."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size != values.size:
        msg = f"{context} contains non-finite stimulus values."
        raise ValueError(msg)
    rounded = np.rint(values).astype(np.int64)
    if not np.array_equal(values, rounded.astype(np.float64)):
        msg = f"{context} must contain integer LED antenna activation codes."
        raise ValueError(msg)
    unique = set(rounded.tolist())
    if not unique.issubset({0, 1, 2, 3}):
        msg = f"{context} must use LED antenna codes 0=left, 1=both, 2=right, 3=blank."
        raise ValueError(msg)
    left_binary = np.isin(rounded, (0, 1)).astype(np.float64)
    right_binary = np.isin(rounded, (1, 2)).astype(np.float64)
    return (
        binary_stimulus_to_signed(left_binary, context=f"{context} left stream"),
        binary_stimulus_to_signed(right_binary, context=f"{context} right stream"),
    )


def binary_stimulus_to_signed(
    values: npt.NDArray[np.float64],
    *,
    context: str,
) -> npt.NDArray[np.float64]:
    """Convert ``0/1`` stimulus values to ``-1/+1`` while accepting signed data."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size != values.size:
        msg = f"{context} contains non-finite stimulus values."
        raise ValueError(msg)
    unique = set(np.unique(values).tolist())
    if unique.issubset({0.0, 1.0}):
        return (2.0 * values - 1.0).astype(np.float64, copy=False)
    if unique.issubset({-1.0, 1.0}):
        return values.astype(np.float64, copy=False)
    msg = f"{context} must contain binary 0/1 or signed -1/+1 values."
    raise ValueError(msg)


def visual_contrast_stimulus(
    values: npt.NDArray[np.float64],
    *,
    context: str,
) -> npt.NDArray[np.float64]:
    """Validate one visual contrast stream from converted stimulus data."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size != values.size:
        msg = f"{context} contains non-finite stimulus values."
        raise ValueError(msg)
    if values.ndim != 1:
        msg = f"{context} must be one-dimensional."
        raise ValueError(msg)
    if np.nanmax(np.abs(values)) <= 0:
        msg = f"{context} is all zero and cannot fit a kernel."
        raise ValueError(msg)
    return values.astype(np.float64, copy=False)


def _epoch_windows_by_number(
    epoch_windows: Sequence[EpochFrameWindow],
) -> dict[int, list[EpochFrameWindow]]:
    """Group epoch windows by epoch number while preserving trial order."""
    grouped: dict[int, list[EpochFrameWindow]] = {}
    for window in epoch_windows:
        grouped.setdefault(window.epoch_number, []).append(window)
    return grouped
