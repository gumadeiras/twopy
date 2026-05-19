"""Fit random-noise response kernels from the Custom tab.

This workflow ships with twopy because temporal stimulus kernels are a common
inspection step for fixed/random-noise recordings. It uses converted twopy data
and current ROIs, then returns plots and CSV files with workflow provenance.
"""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from twopy.analysis import (
    Hemisphere,
    RecordingKernelFit,
    StimulusKernelOptions,
    default_kernel_stimulus_column,
    fit_recording_stimulus_kernels,
)
from twopy.custom import CustomLinePlot, CustomResult, CustomRunContext, workflow

_SAFE_FILENAME_PART = re.compile(r"[^A-Za-z0-9_.-]+")
_KERNEL_Y_LABEL = "Weight"


@dataclass(frozen=True)
class ResponseKernelParams:
    """Controls shown for Response kernels in the Custom tab.

    Args:
        roi_selector: Which ROI subset to fit.
        stimulus_modality: Sensory modality that defines how the default
            stimulus column is interpreted.
        hemisphere: Recording hemisphere used to map raw left/right streams to
            ipsi/contra for olfactory kernels.
        fit_method: Linear fitting method for the design matrix.
        num_stim_past: Number of stimulus samples before each response.
        num_stim_future: Number of future stimulus samples used as a timing QC.
        baseline_epoch_number: Gray or baseline epoch number excluded from fit.
        discard_first_stimulus_epoch: Whether to discard the first non-baseline
            stimulus epoch after baseline removal.
        output_prefix: Relative output filename prefix.

    Returns:
        Immutable GUI parameter object.
    """

    roi_selector: str = field(
        default="all_rois",
        metadata={
            "label": "ROIs",
            "description": "ROI subset used for kernel fitting.",
            "twopy_role": "roi_selector",
        },
    )
    stimulus_modality: Literal["olfaction", "vision"] = field(
        default="olfaction",
        metadata={
            "label": "Stimulus",
            "description": "Use antenna activation codes or visual signed contrast.",
        },
    )
    hemisphere: Literal["recording_metadata", "right", "left"] = field(
        default="recording_metadata",
        metadata={
            "label": "Hemisphere",
            "description": "Used only for olfactory ipsi/contra mapping.",
        },
    )
    fit_method: Literal["ols", "xcorr"] = field(
        default="ols",
        metadata={
            "label": "Fit method",
            "description": "Design-matrix least squares or reverse correlation.",
        },
    )
    num_stim_past: int = field(
        default=150,
        metadata={
            "label": "Past samples",
            "description": "Stimulus samples before each response sample.",
            "min": 0,
        },
    )
    num_stim_future: int = field(
        default=25,
        metadata={
            "label": "Future samples",
            "description": "Future stimulus samples used as negative-lag QC.",
            "min": 0,
        },
    )
    baseline_epoch_number: int = field(
        default=1,
        metadata={
            "label": "Baseline epoch",
            "description": "Epoch number excluded as gray or baseline.",
            "min": 0,
        },
    )
    discard_first_stimulus_epoch: bool = field(
        default=True,
        metadata={
            "label": "Discard first stimulus epoch",
            "description": "Drop the first non-baseline epoch before fitting.",
        },
    )
    output_prefix: str = field(
        default="response_kernels",
        metadata={
            "label": "Output prefix",
            "description": "Relative prefix for CSV outputs.",
            "twopy_role": "output_name",
        },
    )


@workflow(
    id="response-kernels",
    name="Response kernels",
    version="1.0",
    description="Fits olfactory antenna or visual contrast temporal kernels per ROI.",
    params=ResponseKernelParams,
    author="twopy",
    output_prefix="response_kernels",
)
def run(ctx: CustomRunContext, params: ResponseKernelParams) -> CustomResult:
    """Fit and plot response kernels for the active recording and ROIs.

    Args:
        ctx: Active recording, ROIs, and output helpers.
        params: Custom tab values.

    Returns:
        CSV files and line plots for the fitted kernels.
    """
    rois = ctx.rois_for_selector(params.roi_selector)
    computation = ctx.compute_standard_responses(rois)
    kernels = fit_recording_stimulus_kernels(
        computation,
        StimulusKernelOptions(
            stimulus_modality=params.stimulus_modality,
            stimulus_column=default_kernel_stimulus_column(params.stimulus_modality),
            baseline_epoch_number=params.baseline_epoch_number,
            discard_first_stimulus_epoch=params.discard_first_stimulus_epoch,
            num_stim_past=params.num_stim_past,
            num_stim_future=params.num_stim_future,
            method=params.fit_method,
            hemisphere=_selected_hemisphere(params),
        ),
    )

    if params.stimulus_modality == "vision":
        return _visual_kernel_result(ctx, params, kernels)
    return _olfactory_kernel_result(ctx, params, kernels)


def _olfactory_kernel_result(
    ctx: CustomRunContext,
    params: ResponseKernelParams,
    kernels: RecordingKernelFit,
) -> CustomResult:
    """Write olfactory ipsi/contra kernel outputs."""
    if kernels.ipsilateral is None or kernels.contralateral is None:
        msg = "Olfactory kernel result is missing ipsi/contra kernels."
        raise ValueError(msg)
    summary_path = ctx.output_path(f"{params.output_prefix}_summary.csv")
    files: list[Path] = []
    plots: list[CustomLinePlot] = []
    lag_columns = _lag_column_labels(kernels.time_seconds)
    for epoch_index, epoch_name in enumerate(kernels.epoch_names):
        stem = _kernel_output_stem(
            params.output_prefix,
            epoch_index=epoch_index,
            epoch_name=epoch_name,
            epoch_numbers=kernels.selected_epoch_numbers_by_name[epoch_index],
        )
        ipsi_path = ctx.output_path(f"{stem}_ipsi.csv")
        contra_path = ctx.output_path(f"{stem}_contra.csv")
        ipsi = kernels.ipsilateral[epoch_index]
        contra = kernels.contralateral[epoch_index]
        ctx.write_matrix_csv(
            ipsi_path,
            ipsi,
            row_labels=kernels.roi_labels,
            column_labels=lag_columns,
        )
        ctx.write_matrix_csv(
            contra_path,
            contra,
            row_labels=kernels.roi_labels,
            column_labels=lag_columns,
        )
        files.extend((ipsi_path, contra_path))
        plots.extend(
            (
                CustomLinePlot(
                    f"Ipsi kernels - {epoch_name}",
                    kernels.time_seconds,
                    ipsi,
                    kernels.roi_labels,
                    y_label=_KERNEL_Y_LABEL,
                ),
                CustomLinePlot(
                    f"Contra kernels - {epoch_name}",
                    kernels.time_seconds,
                    contra,
                    kernels.roi_labels,
                    y_label=_KERNEL_Y_LABEL,
                ),
            )
        )
    _write_kernel_summary(summary_path, kernels)
    mean_rows: list[np.ndarray] = []
    mean_labels: list[str] = []
    for epoch_index, epoch_name in enumerate(kernels.epoch_names):
        mean_rows.append(np.nanmean(kernels.ipsilateral[epoch_index], axis=0))
        mean_labels.append(f"{epoch_name} ipsi")
        mean_rows.append(np.nanmean(kernels.contralateral[epoch_index], axis=0))
        mean_labels.append(f"{epoch_name} contra")
    plots.append(
        CustomLinePlot(
            "Mean ipsi/contra kernels",
            kernels.time_seconds,
            np.stack(mean_rows, axis=0),
            tuple(mean_labels),
            y_label=_KERNEL_Y_LABEL,
        )
    )
    return CustomResult(
        message=(
            f"Fit olfactory kernels for {len(kernels.roi_labels)} ROIs across "
            f"{len(kernels.epoch_names)} epoch names."
        ),
        files=(*files, summary_path),
        plots=tuple(plots),
    )


def _visual_kernel_result(
    ctx: CustomRunContext,
    params: ResponseKernelParams,
    kernels: RecordingKernelFit,
) -> CustomResult:
    """Write visual contrast kernel outputs."""
    if kernels.contrast is None:
        msg = "Visual kernel result is missing contrast kernels."
        raise ValueError(msg)
    summary_path = ctx.output_path(f"{params.output_prefix}_summary.csv")
    files: list[Path] = []
    plots: list[CustomLinePlot] = []
    mean_rows: list[np.ndarray] = []
    lag_columns = _lag_column_labels(kernels.time_seconds)
    for epoch_index, epoch_name in enumerate(kernels.epoch_names):
        stem = _kernel_output_stem(
            params.output_prefix,
            epoch_index=epoch_index,
            epoch_name=epoch_name,
            epoch_numbers=kernels.selected_epoch_numbers_by_name[epoch_index],
        )
        contrast_path = ctx.output_path(f"{stem}_contrast.csv")
        contrast = kernels.contrast[epoch_index]
        ctx.write_matrix_csv(
            contrast_path,
            contrast,
            row_labels=kernels.roi_labels,
            column_labels=lag_columns,
        )
        files.append(contrast_path)
        plots.append(
            CustomLinePlot(
                f"Contrast kernels - {epoch_name}",
                kernels.time_seconds,
                contrast,
                kernels.roi_labels,
                y_label=_KERNEL_Y_LABEL,
            )
        )
        mean_rows.append(np.nanmean(contrast, axis=0))
    _write_kernel_summary(summary_path, kernels)
    plots.append(
        CustomLinePlot(
            "Mean contrast kernels",
            kernels.time_seconds,
            np.stack(mean_rows, axis=0),
            kernels.epoch_names,
            y_label=_KERNEL_Y_LABEL,
        )
    )
    return CustomResult(
        message=(
            f"Fit visual contrast kernels for {len(kernels.roi_labels)} ROIs "
            f"across {len(kernels.epoch_names)} epoch names from "
            f"{kernels.stimulus_column}."
        ),
        files=(*files, summary_path),
        plots=tuple(plots),
    )


def _write_kernel_summary(path: Path, kernels: RecordingKernelFit) -> None:
    """Write per-ROI response counts and fit metadata.

    Args:
        path: CSV path returned by ``ctx.output_path``.
        kernels: ``RecordingKernelFit`` result from the core fitter.

    Returns:
        None.
    """
    discarded = " ".join(str(value) for value in kernels.discarded_epoch_numbers)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "epoch_name",
                "roi_label",
                "response_samples",
                "stimulus_samples",
                "method",
                "stimulus_modality",
                "stimulus_column",
                "hemisphere",
                "selected_epochs",
                "discarded_epochs",
            )
        )
        for epoch_index, epoch_name in enumerate(kernels.epoch_names):
            selected = " ".join(
                str(value)
                for value in kernels.selected_epoch_numbers_by_name[epoch_index]
            )
            stimulus_samples = str(kernels.stimulus_sample_counts[epoch_index])
            for label, count in zip(
                kernels.roi_labels,
                kernels.response_sample_counts[epoch_index],
                strict=True,
            ):
                writer.writerow(
                    (
                        epoch_name,
                        label,
                        int(count),
                        stimulus_samples,
                        kernels.method,
                        kernels.stimulus_modality,
                        kernels.stimulus_column,
                        "" if kernels.hemisphere is None else kernels.hemisphere,
                        selected,
                        discarded,
                    )
                )


def _safe_epoch_suffix(epoch_name: str) -> str:
    """Return a readable filename suffix for one epoch name."""
    cleaned = _SAFE_FILENAME_PART.sub("_", epoch_name.strip()).strip("_")
    return cleaned or "epoch"


def _kernel_output_stem(
    output_prefix: str,
    *,
    epoch_index: int,
    epoch_name: str,
    epoch_numbers: tuple[int, ...],
) -> str:
    """Return a collision-proof output stem for one kernel epoch group."""
    epoch_number_text = "_".join(str(value) for value in epoch_numbers)
    suffix = _safe_epoch_suffix(epoch_name)
    return (
        f"{output_prefix}_group_{epoch_index + 1:02d}_"
        f"epochs_{epoch_number_text}_{suffix}"
    )


def _lag_column_labels(time_seconds: np.ndarray) -> tuple[str, ...]:
    """Return matrix CSV labels that encode actual lag seconds."""
    return tuple(f"lag_s_{_clean_lag_value(value):.6f}" for value in time_seconds)


def _clean_lag_value(value: float) -> float:
    """Avoid negative-zero labels from floating-point lag axes."""
    as_float = float(value)
    if abs(as_float) < 5e-13:
        return 0.0
    return as_float


def _selected_hemisphere(params: ResponseKernelParams) -> Hemisphere | None:
    """Return an explicit hemisphere or request recording metadata."""
    if params.hemisphere == "left":
        return "left"
    if params.hemisphere == "right":
        return "right"
    return None
