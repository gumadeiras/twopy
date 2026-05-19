"""Fit stimulus-history kernels from converted recordings.

Inputs: converted stimulus tables plus ROI response traces computed by twopy.
Outputs: per-ROI temporal kernels on the stimulus sample clock. Olfactory
recordings expose left/right and ipsi/contra kernels; visual recordings expose
one contrast kernel.

This module keeps reverse-correlation analysis native to twopy data. It does
not read source MATLAB files and does not depend on napari.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from twopy.analysis.trials import EpochFrameWindow
from twopy.analysis.workflow import AnalysisResponseComputation
from twopy.converted import Hemisphere, RecordingData, recording_hemisphere
from twopy.stimulus import stimulus_column_index, stimulus_epoch_names_by_number

__all__ = [
    "Hemisphere",
    "KernelFitMethod",
    "KernelStimulusModality",
    "RecordingKernelFit",
    "StimulusKernelOptions",
    "default_kernel_stimulus_column",
    "fit_recording_stimulus_kernels",
    "fit_stimulus_kernel",
]

KernelFitMethod = Literal["ols", "xcorr"]
KernelStimulusModality = Literal["olfaction", "vision"]

_DEFAULT_STIMULUS_COLUMNS: dict[KernelStimulusModality, str] = {
    "olfaction": "stimulus_specific_05",
    "vision": "stimulus_specific_05",
}


@dataclass(frozen=True)
class StimulusKernelOptions:
    """Controls for fitting random-noise stimulus kernels.

    Args:
        stimulus_modality: Sensory modality that defines how the selected
            stimulus column should be interpreted.
        stimulus_column: Converted stimulus column to fit. ``None`` uses the
            twopy default for ``stimulus_modality``.
        baseline_epoch_number: Epoch number treated as gray or baseline and
            excluded from fitting.
        discard_first_stimulus_epoch: Whether to discard the first remaining
            non-baseline stimulus epoch.
        num_stim_past: Number of stimulus samples before a response sample.
        num_stim_future: Number of stimulus samples after a response sample.
        method: Fitting method. ``"ols"`` solves a least-squares design matrix;
            ``"xcorr"`` computes reverse correlation.
        hemisphere: Recording hemisphere used to map raw left/right streams to
            ipsi/contra. ``None`` reads the value from converted recording
            metadata.

    Returns:
        Immutable fitting options.
    """

    stimulus_modality: KernelStimulusModality = "olfaction"
    stimulus_column: str | None = None
    baseline_epoch_number: int = 1
    discard_first_stimulus_epoch: bool = True
    num_stim_past: int = 150
    num_stim_future: int = 25
    method: KernelFitMethod = "ols"
    hemisphere: Hemisphere | None = None


@dataclass(frozen=True)
class RecordingKernelFit:
    """Per-ROI temporal kernels from one recording.

    Args:
        roi_labels: Labels for the fitted ROI rows.
        time_seconds: Kernel sample times. Negative times are future stimulus
            samples; positive times are stimulus samples before the response.
        stimulus_modality: Sensory modality used to interpret the stimulus.
        stimulus_column: Converted stimulus column used for fitting.
        epoch_names: Unique readable epoch names fitted independently.
        ipsilateral: Ipsi kernels shaped ``(epoch_names, rois, samples)`` for
            olfaction, or ``None`` for vision.
        contralateral: Contra kernels shaped ``(epoch_names, rois, samples)``
            for olfaction, or ``None`` for vision.
        raw_left: Kernels fit to the raw left stimulus stream for olfaction,
            shaped ``(epoch_names, rois, samples)``, or ``None`` for vision.
        raw_right: Kernels fit to the raw right stimulus stream for olfaction,
            shaped ``(epoch_names, rois, samples)``, or ``None`` for vision.
        contrast: Kernels fit to visual signed contrast for vision, shaped
            ``(epoch_names, rois, samples)``, or ``None`` for olfaction.
        selected_epoch_numbers: Non-baseline epochs used for fitting.
        selected_epoch_numbers_by_name: Selected epoch numbers grouped by
            ``epoch_names``.
        discarded_epoch_numbers: Epochs intentionally excluded after baseline
            removal.
        response_sample_counts: Number of finite response samples used per
            epoch name and ROI.
        stimulus_sample_counts: Number of complete stimulus samples used per
            epoch name.
        method: Fitting method used for every kernel.
        hemisphere: Recording hemisphere used for ipsi/contra mapping, or
            ``None`` for vision.

    Returns:
        Immutable result object ready for plotting or CSV export.
    """

    roi_labels: tuple[str, ...]
    time_seconds: npt.NDArray[np.float64]
    stimulus_modality: KernelStimulusModality
    stimulus_column: str
    epoch_names: tuple[str, ...]
    ipsilateral: npt.NDArray[np.float64] | None
    contralateral: npt.NDArray[np.float64] | None
    raw_left: npt.NDArray[np.float64] | None
    raw_right: npt.NDArray[np.float64] | None
    contrast: npt.NDArray[np.float64] | None
    selected_epoch_numbers: tuple[int, ...]
    selected_epoch_numbers_by_name: tuple[tuple[int, ...], ...]
    discarded_epoch_numbers: tuple[int, ...]
    response_sample_counts: npt.NDArray[np.int64]
    stimulus_sample_counts: tuple[int, ...]
    method: KernelFitMethod
    hemisphere: Hemisphere | None


@dataclass(frozen=True)
class _StimulusSegment:
    """One contiguous stimulus-table block with a stable epoch number."""

    epoch_number: int
    start: int
    stop: int


@dataclass(frozen=True)
class _StimulusEpochGroup:
    """Selected epoch numbers that share one readable epoch name."""

    name: str
    epoch_numbers: tuple[int, ...]


@dataclass(frozen=True)
class _AssembledKernelInput:
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


def fit_recording_stimulus_kernels(
    computation: AnalysisResponseComputation,
    options: StimulusKernelOptions,
) -> RecordingKernelFit:
    """Fit left/right and ipsi/contra kernels for computed ROI responses.

    Args:
        computation: Standard twopy response computation containing converted
            stimulus data, ROI labels, dF/F traces, and frame-aligned epochs.
        options: Kernel-fitting controls.

    Returns:
        Per-ROI kernels and audit counts.

    Raises:
        ValueError: If stimulus columns, epochs, timing, or responses are not
            usable for kernel fitting.

    The native implementation keeps the complete regular stimulus stream for
    each selected epoch and samples responses from photodiode-aligned frame
    windows. This gives the fitter sparse response times without dropping
    stimulus samples.
    """
    _validate_options(options)
    recording = computation.recording
    stimulus_column_name = (
        default_kernel_stimulus_column(options.stimulus_modality)
        if options.stimulus_column is None
        else options.stimulus_column
    )
    hemisphere = _resolve_kernel_hemisphere(computation, options)
    time_column = stimulus_column_index(recording, "time_seconds")
    epoch_column = stimulus_column_index(recording, "epoch_number")
    stimulus_column = stimulus_column_index(recording, stimulus_column_name)
    frame_rate_hz = _frame_rate_from_computation(computation)

    segments = _stimulus_epoch_segments(
        recording.stimulus_data[:, epoch_column],
        baseline_epoch_number=options.baseline_epoch_number,
    )
    selected_epoch_numbers, discarded_epoch_numbers = _selected_epoch_numbers(
        segments,
        discard_first=options.discard_first_stimulus_epoch,
    )
    epoch_groups = _selected_epoch_groups(
        recording,
        selected_epoch_numbers=selected_epoch_numbers,
    )

    raw_left: npt.NDArray[np.float64] | None
    raw_right: npt.NDArray[np.float64] | None
    ipsilateral: npt.NDArray[np.float64] | None
    contralateral: npt.NDArray[np.float64] | None
    contrast: npt.NDArray[np.float64] | None
    raw_left_by_name: list[npt.NDArray[np.float64]] = []
    raw_right_by_name: list[npt.NDArray[np.float64]] = []
    ipsilateral_by_name: list[npt.NDArray[np.float64]] = []
    contralateral_by_name: list[npt.NDArray[np.float64]] = []
    contrast_by_name: list[npt.NDArray[np.float64]] = []
    response_sample_counts: list[npt.NDArray[np.int64]] = []
    stimulus_sample_counts: list[int] = []
    time_seconds: npt.NDArray[np.float64] | None = None
    for epoch_group in epoch_groups:
        assembled = _assemble_kernel_input(
            computation,
            segments=segments,
            selected_epoch_numbers=epoch_group.epoch_numbers,
            time_column=time_column,
            stimulus_column=stimulus_column,
            stimulus_modality=options.stimulus_modality,
            frame_rate_hz=frame_rate_hz,
        )
        primary_kernels, group_time_seconds = _fit_all_rois(
            assembled.stimulus_times,
            assembled.stimulus,
            assembled.response_times_by_roi,
            assembled.responses_by_roi,
            options=options,
        )
        if time_seconds is None:
            time_seconds = group_time_seconds
        elif not np.allclose(time_seconds, group_time_seconds):
            msg = "Kernel time axes differ across epoch names."
            raise ValueError(msg)
        response_sample_counts.append(
            np.asarray(
                [values.size for values in assembled.responses_by_roi],
                dtype=np.int64,
            )
        )
        stimulus_sample_counts.append(int(assembled.stimulus_times.size))
        if options.stimulus_modality == "olfaction":
            right_kernels, right_time_seconds = _fit_all_rois(
                assembled.stimulus_times,
                _antenna_activation_to_right_stream(
                    computation.recording.stimulus_data[:, stimulus_column],
                    segments=segments,
                    selected_epoch_numbers=epoch_group.epoch_numbers,
                ),
                assembled.response_times_by_roi,
                assembled.responses_by_roi,
                options=options,
            )
            if not np.allclose(group_time_seconds, right_time_seconds):
                msg = "Left and right kernel time axes differ."
                raise ValueError(msg)
            if hemisphere is None:
                msg = "Olfactory kernels require recording hemisphere metadata."
                raise ValueError(msg)
            group_ipsi, group_contra = _map_raw_to_ipsi_contra(
                primary_kernels,
                right_kernels,
                hemisphere=hemisphere,
            )
            raw_left_by_name.append(primary_kernels)
            raw_right_by_name.append(right_kernels)
            ipsilateral_by_name.append(group_ipsi)
            contralateral_by_name.append(group_contra)
        else:
            contrast_by_name.append(primary_kernels)

    if time_seconds is None:
        msg = "No selected epoch names were assembled for kernel fitting."
        raise ValueError(msg)
    if options.stimulus_modality == "olfaction":
        raw_left = np.stack(raw_left_by_name, axis=0)
        raw_right = np.stack(raw_right_by_name, axis=0)
        ipsilateral = np.stack(ipsilateral_by_name, axis=0)
        contralateral = np.stack(contralateral_by_name, axis=0)
        contrast = None
    else:
        raw_left = None
        raw_right = None
        ipsilateral = None
        contralateral = None
        contrast = np.stack(contrast_by_name, axis=0)
    return RecordingKernelFit(
        roi_labels=computation.roi_set.labels,
        time_seconds=time_seconds,
        stimulus_modality=options.stimulus_modality,
        stimulus_column=stimulus_column_name,
        epoch_names=tuple(group.name for group in epoch_groups),
        ipsilateral=ipsilateral,
        contralateral=contralateral,
        raw_left=raw_left,
        raw_right=raw_right,
        contrast=contrast,
        selected_epoch_numbers=selected_epoch_numbers,
        selected_epoch_numbers_by_name=tuple(
            group.epoch_numbers for group in epoch_groups
        ),
        discarded_epoch_numbers=discarded_epoch_numbers,
        response_sample_counts=np.stack(response_sample_counts, axis=0),
        stimulus_sample_counts=tuple(stimulus_sample_counts),
        method=options.method,
        hemisphere=hemisphere,
    )


def fit_stimulus_kernel(
    stimulus_times: npt.NDArray[np.float64],
    stimulus: npt.NDArray[np.float64],
    response_times: npt.NDArray[np.float64],
    response: npt.NDArray[np.float64],
    *,
    num_stim_past: int,
    num_stim_future: int,
    method: KernelFitMethod = "ols",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit one temporal kernel from a regular stimulus and sparse responses.

    Args:
        stimulus_times: Strictly increasing regular stimulus sample times.
        stimulus: Stimulus values at ``stimulus_times``.
        response_times: Response sample times in the same time base.
        response: Response values at ``response_times``.
        num_stim_past: Number of samples before each response included.
        num_stim_future: Number of samples after each response included.
        method: ``"ols"`` or ``"xcorr"``.

    Returns:
        ``(kernel, time_seconds)``. The kernel time axis runs from
        ``-num_stim_future * dt`` through ``num_stim_past * dt``.

    This mirrors the MATLAB filter convention: the left side of the kernel is
    future stimulus samples and should usually stay near zero.
    """
    _validate_fit_inputs(
        stimulus_times=stimulus_times,
        stimulus=stimulus,
        response_times=response_times,
        response=response,
        num_stim_past=num_stim_past,
        num_stim_future=num_stim_future,
        method=method,
    )
    stimulus_clock = np.polyfit(
        np.arange(1, stimulus_times.size + 1, dtype=np.float64),
        stimulus_times,
        deg=1,
    )
    stim_dt = float(stimulus_clock[0])
    time_offset = float(stimulus_clock[1])
    stimulus_indices_one_based = np.rint(
        (response_times - time_offset) / stim_dt,
    ).astype(np.int64)
    valid = (
        (stimulus_indices_one_based - num_stim_past > 0)
        & (stimulus_indices_one_based + num_stim_future <= stimulus.size)
        & np.isfinite(response)
    )
    if not np.any(valid):
        msg = "No response samples have a complete stimulus history/future window."
        raise ValueError(msg)
    stimulus_indices = stimulus_indices_one_based[valid] - 1
    centered_response = response[valid] - float(np.nanmean(response[valid]))
    design = _stimulus_design_matrix(
        stimulus,
        stimulus_indices,
        num_stim_past=num_stim_past,
        num_stim_future=num_stim_future,
    )
    if method == "ols":
        kernel = np.linalg.lstsq(design, centered_response, rcond=None)[0]
    elif method == "xcorr":
        kernel = centered_response @ design / centered_response.size
    else:
        msg = f"Unknown kernel fit method {method!r}."
        raise ValueError(msg)
    time_seconds = stim_dt * np.arange(
        -num_stim_future,
        num_stim_past + 1,
        dtype=np.float64,
    )
    return kernel.astype(np.float64, copy=False), time_seconds.astype(
        np.float64,
        copy=False,
    )


def _validate_options(options: StimulusKernelOptions) -> None:
    """Validate high-level kernel options before reading data."""
    if options.stimulus_modality not in {"olfaction", "vision"}:
        msg = f"Unknown kernel stimulus modality {options.stimulus_modality!r}."
        raise ValueError(msg)
    if options.baseline_epoch_number < 0:
        msg = "baseline_epoch_number must be nonnegative."
        raise ValueError(msg)
    if options.num_stim_past < 0 or options.num_stim_future < 0:
        msg = "num_stim_past and num_stim_future must be nonnegative."
        raise ValueError(msg)
    if options.method not in {"ols", "xcorr"}:
        msg = f"Unknown kernel fit method {options.method!r}."
        raise ValueError(msg)
    if options.hemisphere is not None and options.hemisphere not in {"left", "right"}:
        msg = (
            f"hemisphere must be 'left', 'right', or None; got {options.hemisphere!r}."
        )
        raise ValueError(msg)


def _resolve_kernel_hemisphere(
    computation: AnalysisResponseComputation,
    options: StimulusKernelOptions,
) -> Hemisphere | None:
    """Return the hemisphere used for ipsi/contra remapping."""
    if options.stimulus_modality == "vision":
        return None
    if options.hemisphere is not None:
        return options.hemisphere
    return recording_hemisphere(computation.recording)


def _frame_rate_from_computation(computation: AnalysisResponseComputation) -> float:
    """Return the timing frame rate used for the response computation."""
    if computation.timing is not None:
        return float(computation.timing.frame_rate_hz)
    return float(computation.grouped_responses.data_rate_hz)


def _stimulus_epoch_segments(
    epoch_values: npt.NDArray[np.float64],
    *,
    baseline_epoch_number: int,
) -> tuple[_StimulusSegment, ...]:
    """Return positive non-baseline contiguous epoch segments."""
    if epoch_values.ndim != 1:
        msg = "stimulus epoch values must be one-dimensional."
        raise ValueError(msg)
    segments: list[_StimulusSegment] = []
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
            segments.append(_StimulusSegment(epoch_number, start, stop))
        start = stop
    if len(segments) == 0:
        msg = "No non-baseline stimulus epochs are available for kernel fitting."
        raise ValueError(msg)
    return tuple(segments)


def _selected_epoch_numbers(
    segments: Sequence[_StimulusSegment],
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


def _selected_epoch_groups(
    recording: RecordingData,
    *,
    selected_epoch_numbers: tuple[int, ...],
) -> tuple[_StimulusEpochGroup, ...]:
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
        _StimulusEpochGroup(name=name, epoch_numbers=tuple(grouped[name]))
        for name in ordered_names
    )


def _assemble_kernel_input(
    computation: AnalysisResponseComputation,
    *,
    segments: Sequence[_StimulusSegment],
    selected_epoch_numbers: tuple[int, ...],
    time_column: int,
    stimulus_column: int,
    stimulus_modality: KernelStimulusModality,
    frame_rate_hz: float,
) -> _AssembledKernelInput:
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
        _validate_regular_stimulus_times(
            relative_stimulus_times,
            context=f"stimulus epoch {segment.epoch_number}",
        )
        stimulus_times.append(relative_stimulus_times + offset)
        stimulus_values.append(
            _kernel_stimulus_values(
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
            relative_stimulus_times[-1]
            + _stimulus_dt(relative_stimulus_times)
            + offset,
        )

    if len(stimulus_times) == 0:
        msg = "No selected stimulus epochs were assembled for kernel fitting."
        raise ValueError(msg)
    return _AssembledKernelInput(
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


def _kernel_stimulus_values(
    values: npt.NDArray[np.float64],
    *,
    stimulus_modality: KernelStimulusModality,
    context: str,
) -> npt.NDArray[np.float64]:
    """Convert one stimulus column into the stream fit by default."""
    if stimulus_modality == "olfaction":
        left_stimulus, _ = _antenna_activation_to_left_right(
            values,
            context=f"{context} antenna activation",
        )
        return left_stimulus
    return _visual_contrast_stimulus(values, context=f"{context} visual contrast")


def _antenna_activation_to_right_stream(
    values: npt.NDArray[np.float64],
    *,
    segments: Sequence[_StimulusSegment],
    selected_epoch_numbers: tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Return concatenated right-antenna streams for selected segments."""
    right_values: list[npt.NDArray[np.float64]] = []
    for segment in segments:
        if segment.epoch_number not in selected_epoch_numbers:
            continue
        _, right_stimulus = _antenna_activation_to_left_right(
            values[segment.start : segment.stop],
            context=f"antenna activation epoch {segment.epoch_number}",
        )
        right_values.append(right_stimulus)
    if len(right_values) == 0:
        msg = "No right antenna stimulus epochs were assembled for kernel fitting."
        raise ValueError(msg)
    return np.concatenate(right_values).astype(np.float64, copy=False)


def _fit_all_rois(
    stimulus_times: npt.NDArray[np.float64],
    stimulus: npt.NDArray[np.float64],
    response_times_by_roi: Sequence[npt.NDArray[np.float64]],
    responses_by_roi: Sequence[npt.NDArray[np.float64]],
    *,
    options: StimulusKernelOptions,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit the same stimulus stream against every ROI response."""
    kernels: list[npt.NDArray[np.float64]] = []
    time_seconds: npt.NDArray[np.float64] | None = None
    for response_times, responses in zip(
        response_times_by_roi,
        responses_by_roi,
        strict=True,
    ):
        kernel, roi_time_seconds = fit_stimulus_kernel(
            stimulus_times,
            stimulus,
            response_times,
            responses,
            num_stim_past=options.num_stim_past,
            num_stim_future=options.num_stim_future,
            method=options.method,
        )
        if time_seconds is None:
            time_seconds = roi_time_seconds
        elif not np.array_equal(time_seconds, roi_time_seconds):
            msg = "ROI kernel time axes differ."
            raise ValueError(msg)
        kernels.append(kernel)
    if time_seconds is None:
        msg = "At least one ROI is required for kernel fitting."
        raise ValueError(msg)
    return np.stack(kernels, axis=0), time_seconds


def _map_raw_to_ipsi_contra(
    raw_left: npt.NDArray[np.float64],
    raw_right: npt.NDArray[np.float64],
    *,
    hemisphere: Hemisphere,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Map raw left/right stimulus kernels into ipsi/contra convention."""
    if hemisphere == "right":
        return raw_right, raw_left
    return raw_left, raw_right


def _epoch_windows_by_number(
    epoch_windows: Sequence[EpochFrameWindow],
) -> dict[int, list[EpochFrameWindow]]:
    """Group epoch windows by epoch number while preserving trial order."""
    grouped: dict[int, list[EpochFrameWindow]] = {}
    for window in epoch_windows:
        grouped.setdefault(window.epoch_number, []).append(window)
    return grouped


def _validate_fit_inputs(
    *,
    stimulus_times: npt.NDArray[np.float64],
    stimulus: npt.NDArray[np.float64],
    response_times: npt.NDArray[np.float64],
    response: npt.NDArray[np.float64],
    num_stim_past: int,
    num_stim_future: int,
    method: KernelFitMethod,
) -> None:
    """Validate the primitive filter inputs."""
    if stimulus_times.ndim != 1 or stimulus.ndim != 1:
        msg = "stimulus_times and stimulus must be one-dimensional."
        raise ValueError(msg)
    if response_times.ndim != 1 or response.ndim != 1:
        msg = "response_times and response must be one-dimensional."
        raise ValueError(msg)
    if stimulus_times.shape != stimulus.shape:
        msg = "stimulus_times and stimulus must have the same length."
        raise ValueError(msg)
    if response_times.shape != response.shape:
        msg = "response_times and response must have the same length."
        raise ValueError(msg)
    if stimulus_times.size < 2:
        msg = "At least two stimulus samples are required."
        raise ValueError(msg)
    if response_times.size == 0:
        msg = "At least one response sample is required."
        raise ValueError(msg)
    if num_stim_past < 0 or num_stim_future < 0:
        msg = "num_stim_past and num_stim_future must be nonnegative."
        raise ValueError(msg)
    if method not in {"ols", "xcorr"}:
        msg = f"Unknown kernel fit method {method!r}."
        raise ValueError(msg)
    _validate_regular_stimulus_times(stimulus_times, context="kernel stimulus")
    if not np.all(np.diff(response_times) >= 0):
        msg = "response_times must be sorted."
        raise ValueError(msg)


def _stimulus_design_matrix(
    stimulus: npt.NDArray[np.float64],
    stimulus_indices: npt.NDArray[np.int64],
    *,
    num_stim_past: int,
    num_stim_future: int,
) -> npt.NDArray[np.float64]:
    """Return a design matrix ordered from future samples to past samples."""
    offsets = np.arange(
        num_stim_future,
        -num_stim_past - 1,
        -1,
        dtype=np.int64,
    )
    return stimulus[stimulus_indices[:, None] + offsets[None, :]]


def _binary_stimulus_to_signed(
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


def _antenna_activation_to_left_right(
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
        _binary_stimulus_to_signed(left_binary, context=f"{context} left stream"),
        _binary_stimulus_to_signed(right_binary, context=f"{context} right stream"),
    )


def _visual_contrast_stimulus(
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


def _validate_regular_stimulus_times(
    times: npt.NDArray[np.float64],
    *,
    context: str,
) -> None:
    """Check that stimulus sample times are strictly increasing and regular."""
    if times.ndim != 1:
        msg = f"{context} times must be one-dimensional."
        raise ValueError(msg)
    diffs = np.diff(times)
    if not np.all(diffs > 0):
        msg = f"{context} times must be strictly increasing."
        raise ValueError(msg)
    mean_dt = float(np.mean(diffs))
    if mean_dt <= 0:
        msg = f"{context} sample interval must be positive."
        raise ValueError(msg)
    if float(np.std(diffs)) / mean_dt >= 0.1:
        msg = f"{context} times are not regular enough for kernel fitting."
        raise ValueError(msg)


def _stimulus_dt(times: npt.NDArray[np.float64]) -> float:
    """Return the mean stimulus sample interval."""
    return float(np.mean(np.diff(times)))
