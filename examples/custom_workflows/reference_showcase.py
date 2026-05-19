"""Reference workflow showing the full public custom workflow API.

Copy this file outside the repo, add that folder to ``custom_workflow_paths`` in
``config.yml``, and edit it into a real lab workflow. The math is intentionally
simple so the API is easy to inspect.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from twopy.custom import (
    CustomLinePlot,
    CustomResult,
    CustomRunContext,
    CustomTable,
    workflow,
)


class SummaryMode(Enum):
    """Example enum that becomes a dropdown."""

    COMPACT = "compact"
    VERBOSE = "verbose"


@dataclass(frozen=True)
class ReferenceParams:
    """Example parameters for every Custom tab control type.

    Args:
        show_standard_response_plot: Whether to update the response dock.
        roi_selector: ROI subset to analyze.
        max_rois_in_response_plot: Maximum ROI count shown in the response dock.
        amplitude_scale: Scale applied to the summary metric.
        note: Free text saved into a note file.
        table_name: CSV path below ``ctx.output_dir``.
        selected_epoch: Main epoch to summarize.
        baseline_epoch: Example baseline epoch selector.
        comparison_epoch: Example comparison epoch selector.
        stimulus_column: Example converted stimulus column selector.
        metric: Metric used by ``ctx.epoch_metric``.
        window_start_seconds: Epoch-relative metric start.
        window_stop_seconds: Epoch-relative metric stop.
        response_window_start_seconds: Response-plot window start.
        response_window_stop_seconds: Response-plot window stop.
        highlight_threshold: Absolute metric threshold for highlighted rows.
        mode: Enum dropdown that changes table columns.

    Returns:
        GUI parameter object.
    """

    show_standard_response_plot: bool = field(
        default=True,
        metadata={
            "label": "Show response plot",
            "description": "Return standard ResponsePlotData to the main plot dock.",
        },
    )
    roi_selector: str = field(
        default="all_rois",
        metadata={
            "label": "ROIs",
            "description": "Example standard ROI selector.",
            "twopy_role": "roi_selector",
        },
    )
    max_rois_in_response_plot: int = field(
        default=6,
        metadata={
            "label": "Max plotted ROIs",
            "description": "Caps standard response plots for quick inspection.",
            "max": 50,
            "twopy_role": "roi_limit",
        },
    )
    amplitude_scale: float = field(
        default=1.0,
        metadata={
            "label": "Amplitude scale",
            "description": "Example numeric transform applied to the metric.",
            "min": 0.0,
            "max": 10.0,
            "step": 0.1,
        },
    )
    note: str = field(
        default="reference run",
        metadata={"label": "Note"},
    )
    table_name: Path = field(
        default=Path("reference_summary.csv"),
        metadata={
            "label": "Table file",
            "description": "Relative CSV path below the workflow output folder.",
            "twopy_role": "output_name",
        },
    )
    selected_epoch: str = field(
        default="",
        metadata={
            "label": "Epoch",
            "description": "Recording epoch summarized by this workflow.",
            "twopy_role": "epoch",
        },
    )
    baseline_epoch: str = field(
        default="",
        metadata={
            "label": "Baseline epoch",
            "description": "Example baseline-like recording epoch selector.",
            "twopy_role": "baseline_epoch",
        },
    )
    comparison_epoch: str = field(
        default="",
        metadata={
            "label": "Comparison epoch",
            "description": "Example comparison recording epoch selector.",
            "twopy_role": "comparison_epoch",
        },
    )
    stimulus_column: str = field(
        default="stimulus_specific_01",
        metadata={
            "label": "Stimulus column",
            "description": "Example converted stimulus/data column selector.",
            "twopy_role": "stimulus_column",
        },
    )
    metric: str = field(
        default="mean",
        metadata={"label": "Metric", "twopy_role": "response_metric"},
    )
    window_start_seconds: float = field(
        default=0.0,
        metadata={
            "label": "Window start (s)",
            "description": "Epoch-relative start time used for the metric.",
            "twopy_role": "epoch_window_start",
        },
    )
    window_stop_seconds: float = field(
        default=3.0,
        metadata={
            "label": "Window end (s)",
            "description": "Epoch-relative end time used for the metric.",
            "twopy_role": "epoch_window_stop",
        },
    )
    response_window_start_seconds: float = field(
        default=-1.0,
        metadata={
            "label": "Response window start (s)",
            "description": "Response plot-relative window start.",
            "twopy_role": "response_window_start",
        },
    )
    response_window_stop_seconds: float = field(
        default=3.0,
        metadata={
            "label": "Response window end (s)",
            "description": "Response plot-relative window end.",
            "twopy_role": "response_window_stop",
        },
    )
    highlight_threshold: float = field(
        default=0.5,
        metadata={
            "label": "Highlight threshold",
            "description": (
                "Rows with absolute metric at or above this value are highlighted."
            ),
            "twopy_role": "table_highlight_threshold",
        },
    )
    mode: SummaryMode = field(
        default=SummaryMode.COMPACT,
        metadata={"label": "Mode"},
    )
    return_current_rois: bool = field(
        default=False,
        metadata={
            "label": "Return current ROIs",
            "description": "Shows how a workflow can send ROIs back to napari.",
        },
    )


@workflow(
    id="reference-showcase",
    name="Reference showcase",
    version="1.0",
    description=(
        "Demonstrates parameters, files, tables, plots, ROIs, and response plot output."
    ),
    params=ReferenceParams,
)
def run(ctx: CustomRunContext, params: ReferenceParams) -> CustomResult:
    """Run a small example analysis over the selected ROIs.

    Args:
        ctx: Active recording, ROIs, and output helpers.
        params: GUI parameter values.

    Returns:
        Result using every supported output type.
    """
    rois = ctx.rois_for_selector(params.roi_selector)
    computation = ctx.compute_standard_responses(rois)
    epoch = params.selected_epoch or _first_epoch_label(ctx)
    baseline_epoch = params.baseline_epoch or _first_epoch_label(ctx)
    comparison_epoch = params.comparison_epoch or epoch
    window_seconds = ctx.epoch_window(
        params.window_start_seconds,
        params.window_stop_seconds,
    )
    response_window_seconds = ctx.response_window(
        params.response_window_start_seconds,
        params.response_window_stop_seconds,
    )
    metric_values = (
        ctx.epoch_metric(
            computation.grouped_responses,
            epoch,
            params.metric,
            window_seconds=window_seconds,
        )
        * params.amplitude_scale
    )
    highlighted_rows = tuple(
        int(index)
        for index in np.flatnonzero(np.abs(metric_values) >= params.highlight_threshold)
    )

    roi_count = len(rois.labels)
    table_path = ctx.output_path(params.table_name)
    columns: dict[str, tuple[float | str, ...]] = {
        "epoch": (str(epoch),) * roi_count,
        "baseline_epoch": (str(baseline_epoch),) * roi_count,
        "comparison_epoch": (str(comparison_epoch),) * roi_count,
        "scaled_metric": tuple(float(value) for value in metric_values),
    }
    if params.mode is SummaryMode.VERBOSE:
        columns["roi_index"] = tuple(float(index) for index in range(roi_count))
    ctx.write_roi_table(table_path, columns, roi_labels=rois.labels)

    note_path = ctx.output_path("reference_note.txt")
    note_path.write_text(
        (
            f"{params.note}\n"
            f"metric={params.metric}\n"
            f"epoch={epoch}\n"
            f"response_window={response_window_seconds}\n"
        ),
        encoding="utf-8",
    )

    metric_plot = CustomLinePlot(
        "Scaled metric by ROI",
        np.arange(roi_count, dtype=np.float64),
        metric_values.astype(np.float64, copy=False),
        ("scaled metric",),
    )
    response_plot_data = None
    if params.show_standard_response_plot:
        response_plot_data = ctx.response_plot_data(
            computation.grouped_responses,
            source_path=table_path,
            max_rois=params.max_rois_in_response_plot,
        )

    return CustomResult(
        message=f"Reference workflow summarized {roi_count} ROIs.",
        files=(note_path,),
        tables=(
            CustomTable(
                "Reference summary",
                table_path,
                highlighted_rows=highlighted_rows,
            ),
        ),
        plots=(metric_plot,),
        roi_set=rois if params.return_current_rois else None,
        response_plot_data=response_plot_data,
    )


def _first_epoch_label(ctx: CustomRunContext) -> str:
    """Return the first epoch label available to workflows."""
    choices = ctx.epoch_choices()
    if len(choices) == 0:
        msg = "No stimulus epochs are available for the reference workflow."
        raise ValueError(msg)
    return choices[0].label
