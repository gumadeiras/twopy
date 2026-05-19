"""Fit stimulus-history kernels from converted recordings.

Inputs: converted stimulus tables plus ROI response traces computed by twopy.
Outputs: per-ROI temporal kernels on the stimulus sample clock. Olfactory
recordings expose left/right and ipsi/contra kernels; visual recordings expose
one contrast kernel.

This package keeps reverse-correlation analysis native to twopy data. It does
not read source MATLAB files and does not depend on napari.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twopy.analysis.kernels._fitting import (
    KernelFitMethod,
    fit_stimulus_kernel,
    fit_stimulus_kernel_by_roi,
)
from twopy.analysis.kernels._stimulus import (
    KernelStimulusModality,
    antenna_activation_to_right_stream,
    assemble_kernel_input,
    default_kernel_stimulus_column,
    frame_rate_from_computation,
    map_raw_to_ipsi_contra,
    resolve_kernel_hemisphere,
    selected_epoch_groups,
    selected_epoch_numbers,
    stimulus_epoch_segments,
)
from twopy.analysis.workflow import AnalysisResponseComputation
from twopy.converted import Hemisphere
from twopy.stimulus import stimulus_column_index

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
        selected_epoch_numbers: Optional one-based epoch numbers to fit
            directly after baseline removal. ``None`` keeps all non-baseline
            epochs and applies ``discard_first_stimulus_epoch``.
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
    selected_epoch_numbers: tuple[int, ...] | None = None
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
        response_sample_counts: Number of response samples actually used by the
            fitter per epoch name and ROI after history/future-window filtering.
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


def fit_recording_stimulus_kernels(
    computation: AnalysisResponseComputation,
    options: StimulusKernelOptions,
) -> RecordingKernelFit:
    """Fit temporal kernels for computed ROI responses.

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
    hemisphere = resolve_kernel_hemisphere(computation, options)
    time_column = stimulus_column_index(recording, "time_seconds")
    epoch_column = stimulus_column_index(recording, "epoch_number")
    stimulus_column = stimulus_column_index(recording, stimulus_column_name)
    frame_rate_hz = frame_rate_from_computation(computation)

    segments = stimulus_epoch_segments(
        recording.stimulus_data[:, epoch_column],
        baseline_epoch_number=options.baseline_epoch_number,
    )
    selected_numbers, discarded_numbers = selected_epoch_numbers(
        segments,
        discard_first=options.discard_first_stimulus_epoch,
        included_epoch_numbers=options.selected_epoch_numbers,
    )
    epoch_groups = selected_epoch_groups(
        recording,
        selected_epoch_numbers=selected_numbers,
    )

    raw_left_by_name: list[npt.NDArray[np.float64]] = []
    raw_right_by_name: list[npt.NDArray[np.float64]] = []
    ipsilateral_by_name: list[npt.NDArray[np.float64]] = []
    contralateral_by_name: list[npt.NDArray[np.float64]] = []
    contrast_by_name: list[npt.NDArray[np.float64]] = []
    response_sample_counts: list[npt.NDArray[np.int64]] = []
    stimulus_sample_counts: list[int] = []
    time_seconds: npt.NDArray[np.float64] | None = None

    for epoch_group in epoch_groups:
        assembled = assemble_kernel_input(
            computation,
            segments=segments,
            selected_epoch_numbers=epoch_group.epoch_numbers,
            time_column=time_column,
            stimulus_column=stimulus_column,
            stimulus_modality=options.stimulus_modality,
            frame_rate_hz=frame_rate_hz,
        )
        primary_fit = fit_stimulus_kernel_by_roi(
            assembled.stimulus_times,
            assembled.stimulus,
            assembled.response_times_by_roi,
            assembled.responses_by_roi,
            num_stim_past=options.num_stim_past,
            num_stim_future=options.num_stim_future,
            method=options.method,
        )
        if time_seconds is None:
            time_seconds = primary_fit.time_seconds
        elif not np.allclose(time_seconds, primary_fit.time_seconds):
            msg = "Kernel time axes differ across epoch names."
            raise ValueError(msg)
        response_sample_counts.append(primary_fit.response_sample_counts)
        stimulus_sample_counts.append(int(assembled.stimulus_times.size))
        if options.stimulus_modality == "olfaction":
            right_fit = fit_stimulus_kernel_by_roi(
                assembled.stimulus_times,
                antenna_activation_to_right_stream(
                    computation.recording.stimulus_data[:, stimulus_column],
                    segments=segments,
                    selected_epoch_numbers=epoch_group.epoch_numbers,
                ),
                assembled.response_times_by_roi,
                assembled.responses_by_roi,
                num_stim_past=options.num_stim_past,
                num_stim_future=options.num_stim_future,
                method=options.method,
            )
            if not np.allclose(primary_fit.time_seconds, right_fit.time_seconds):
                msg = "Left and right kernel time axes differ."
                raise ValueError(msg)
            if not np.array_equal(
                primary_fit.response_sample_counts,
                right_fit.response_sample_counts,
            ):
                msg = "Left and right kernel response counts differ."
                raise ValueError(msg)
            if hemisphere is None:
                msg = "Olfactory kernels require recording hemisphere metadata."
                raise ValueError(msg)
            group_ipsi, group_contra = map_raw_to_ipsi_contra(
                primary_fit.kernels,
                right_fit.kernels,
                hemisphere=hemisphere,
            )
            raw_left_by_name.append(primary_fit.kernels)
            raw_right_by_name.append(right_fit.kernels)
            ipsilateral_by_name.append(group_ipsi)
            contralateral_by_name.append(group_contra)
        else:
            contrast_by_name.append(primary_fit.kernels)

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
        selected_epoch_numbers=selected_numbers,
        selected_epoch_numbers_by_name=tuple(
            group.epoch_numbers for group in epoch_groups
        ),
        discarded_epoch_numbers=discarded_numbers,
        response_sample_counts=np.stack(response_sample_counts, axis=0),
        stimulus_sample_counts=tuple(stimulus_sample_counts),
        method=options.method,
        hemisphere=hemisphere,
    )


def _validate_options(options: StimulusKernelOptions) -> None:
    """Validate high-level kernel options before reading data."""
    if options.stimulus_modality not in {"olfaction", "vision"}:
        msg = f"Unknown kernel stimulus modality {options.stimulus_modality!r}."
        raise ValueError(msg)
    if options.baseline_epoch_number < 0:
        msg = "baseline_epoch_number must be nonnegative."
        raise ValueError(msg)
    if options.selected_epoch_numbers is not None:
        if len(options.selected_epoch_numbers) == 0:
            msg = "selected_epoch_numbers must not be empty."
            raise ValueError(msg)
        for epoch_number in options.selected_epoch_numbers:
            if epoch_number < 1:
                msg = "selected_epoch_numbers must contain positive epoch numbers."
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
