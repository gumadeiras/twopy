"""Run custom workflows without depending on napari.

The Custom tab and Python scripts both use this module to build the same run
context, call the workflow, validate outputs, and write provenance. GUI code
only adapts viewer state before the call and applies UI updates after it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.response_processing import ResponseProcessingOptions
from twopy.analysis.response_window_options import (
    DEFAULT_RESPONSE_POST_WINDOW_SECONDS,
    DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
)
from twopy.converted import RecordingData
from twopy.custom.parameters import build_parameter_object, parameter_values
from twopy.custom.provenance import validate_custom_result, write_result_provenance
from twopy.custom.types import (
    CustomLinePlot,
    CustomResult,
    CustomRunContext,
    CustomWorkflow,
    CustomWorkflowProvenance,
    workflow_provenance,
)
from twopy.roi import RoiSet

__all__ = [
    "CustomWorkflowRun",
    "run_custom_workflow",
]


@dataclass(frozen=True)
class CustomWorkflowRun:
    """Result and metadata from one custom workflow run.

    Args:
        result: Validated workflow result.
        provenance: Metadata describing the workflow, parameters, recording, and
            twopy version used for the run.
        provenance_paths: Sidecar or HDF5 metadata paths written for returned
            output files.
        output_dir: Folder the workflow was allowed to write under.

    Returns:
        Immutable run record for scripts and GUI adapters.
    """

    result: CustomResult
    provenance: CustomWorkflowProvenance
    provenance_paths: tuple[Path, ...]
    output_dir: Path


def _custom_workflow_output_dir(
    recording: RecordingData,
    workflow: CustomWorkflow,
) -> Path:
    """Return the default output folder for one workflow run.

    Args:
        recording: Converted recording analyzed by the workflow.
        workflow: Workflow being run.

    Returns:
        Versioned ``custom_outputs`` folder beside the converted recording.
    """
    return (
        recording.path.parent
        / "custom_outputs"
        / workflow.output_prefix
        / workflow.version
    )


def run_custom_workflow(
    workflow: CustomWorkflow,
    recording: RecordingData,
    *,
    roi_set: RoiSet | None = None,
    params: object | Mapping[str, object] | None = None,
    output_dir: Path | None = None,
    delta_f_over_f_options: DeltaFOverFOptions | None = None,
    response_processing_options: ResponseProcessingOptions | None = None,
    response_pre_window_seconds: float = DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
    response_post_window_seconds: float = DEFAULT_RESPONSE_POST_WINDOW_SECONDS,
    visible_roi_indices: tuple[int, ...] = (),
    visible_epoch_numbers: tuple[int, ...] = (),
    roi_colors: tuple[str, ...] = (),
) -> CustomWorkflowRun:
    """Run one discovered custom workflow on converted recording data.

    Args:
        workflow: Workflow returned by ``discover_custom_workflows``.
        recording: Converted recording to analyze.
        roi_set: Current ROI masks, when the workflow uses ROIs.
        params: Parameter dataclass instance or a mapping from parameter names
            to values. Missing mapping values use dataclass defaults. When
            omitted, default parameters are built for workflows that declare a
            params type.
        output_dir: Exact folder the workflow may write under. When omitted,
            twopy uses the standard versioned ``custom_outputs`` folder.
        delta_f_over_f_options: dF/F settings used by
            ``ctx.compute_standard_responses``.
        response_processing_options: Response post-processing settings used by
            ``ctx.compute_standard_responses``.
        response_pre_window_seconds: Seconds before each epoch included in
            response plots and grouped responses.
        response_post_window_seconds: Seconds after each epoch included in
            response plots and grouped responses.
        visible_roi_indices: Zero-based ROI rows currently visible to the user.
        visible_epoch_numbers: One-based epoch numbers currently visible.
        roi_colors: Plot colors aligned to ``roi_set.labels``.

    Returns:
        Validated result, provenance metadata, provenance paths, and output
        folder for the run.
    """
    if workflow.requires_rois and roi_set is None:
        msg = "This custom workflow needs ROIs, but no ROI Labels are available."
        raise ValueError(msg)
    params_object = _workflow_params(workflow, params)
    run_output_dir = (
        output_dir
        if output_dir is not None
        else _custom_workflow_output_dir(recording, workflow)
    )
    parameters = parameter_values(params_object)
    provenance = workflow_provenance(
        workflow,
        parameters=parameters,
        recording=recording,
    )
    context = CustomRunContext(
        recording=recording,
        roi_set=roi_set,
        output_dir=run_output_dir,
        delta_f_over_f_options=(
            delta_f_over_f_options
            if delta_f_over_f_options is not None
            else DeltaFOverFOptions()
        ),
        response_processing_options=(
            response_processing_options
            if response_processing_options is not None
            else ResponseProcessingOptions()
        ),
        response_pre_window_seconds=response_pre_window_seconds,
        response_post_window_seconds=response_post_window_seconds,
        provenance=provenance,
        visible_roi_indices=visible_roi_indices,
        visible_epoch_numbers=visible_epoch_numbers,
        roi_colors=roi_colors,
    )
    if params_object is None:
        raw_result = workflow.function(context)
    else:
        raw_result = workflow.function(context, params_object)
    result = validate_custom_result(
        raw_result,
        output_dir=run_output_dir,
        expected_roi_shape=recording.movie.shape[1:],
    )
    result = _custom_workflow_result_with_roi_colors(result, context)
    provenance_paths = write_result_provenance(result, provenance)
    return CustomWorkflowRun(
        result=result,
        provenance=provenance,
        provenance_paths=provenance_paths,
        output_dir=run_output_dir,
    )


def _workflow_params(
    workflow: CustomWorkflow,
    params: object | Mapping[str, object] | None,
) -> object | None:
    """Return the parameter object passed to the workflow function."""
    if workflow.params_type is None:
        if params is not None:
            msg = f"Workflow {workflow.id!r} does not accept parameters."
            raise ValueError(msg)
        return None
    if params is None:
        return workflow.params_type()
    if isinstance(params, Mapping):
        return build_parameter_object(workflow.params_type, _parameter_mapping(params))
    if not isinstance(params, workflow.params_type):
        msg = (
            f"Workflow {workflow.id!r} expects parameters of type "
            f"{workflow.params_type.__name__} or a parameter mapping."
        )
        raise ValueError(msg)
    return params


def _parameter_mapping(values: object) -> dict[str, object]:
    """Return parameter values with validated string keys."""
    if not isinstance(values, Mapping):
        msg = "Workflow parameters must be a mapping."
        raise ValueError(msg)
    checked: dict[str, object] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            msg = f"Workflow parameter names must be strings; got {key!r}."
            raise ValueError(msg)
        checked[key] = value
    return checked


def _custom_workflow_result_with_roi_colors(
    result: CustomResult,
    context: CustomRunContext,
) -> CustomResult:
    """Apply current ROI colors to ROI-labeled custom line plots."""
    updated_plots: list[CustomLinePlot] = []
    changed = False
    for plot in result.plots:
        updated = _custom_line_plot_with_roi_colors(plot, context)
        updated_plots.append(updated)
        if updated is not plot:
            changed = True
    if not changed:
        return result
    return replace(result, plots=tuple(updated_plots))


def _custom_line_plot_with_roi_colors(
    plot: CustomLinePlot,
    context: CustomRunContext,
) -> CustomLinePlot:
    """Return one plot with ROI colors when labels identify current ROIs."""
    if len(plot.colors) > 0 or len(plot.labels) == 0:
        return plot
    colors = context.roi_colors_for_labels(plot.labels)
    if len(colors) != len(plot.labels):
        return plot
    return replace(plot, colors=colors)
