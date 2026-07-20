"""Compute direction selectivity from two selected stimulus epochs.

This workflow ships with twopy because DSI is a common response summary. It
still runs through the Custom tab, so its outputs record the same workflow
version and source hash as lab-local workflows.
"""

from dataclasses import dataclass, field

import numpy as np

from twopy.custom import CustomResult, CustomRunContext, CustomTable, workflow


@dataclass(frozen=True)
class DirectionSelectivityParams:
    """Controls shown for Direction selectivity in the Custom tab.

    Args:
        preferred_epoch: Epoch treated as the preferred direction.
        null_epoch: Epoch treated as the opposite direction.
        metric: Response metric used before computing DSI.
        roi_selector: Which ROI subset to analyze.
        window_start_seconds: Epoch-relative metric start time.
        window_stop_seconds: Epoch-relative metric stop time.
        rectify_responses: Whether negative preferred/null responses are set
            to zero before computing DSI.
        dsi_threshold: Minimum absolute DSI to show and highlight.
        output_name: CSV filename under the workflow output folder.

    Returns:
        Immutable GUI parameter object.
    """

    preferred_epoch: str = field(
        default="LR",
        metadata={
            "label": "Preferred epoch",
            "description": "Epoch name or number for the preferred direction.",
            "twopy_role": "epoch",
        },
    )
    null_epoch: str = field(
        default="RL",
        metadata={
            "label": "Null epoch",
            "description": "Epoch name or number for the opposite direction.",
            "twopy_role": "epoch",
        },
    )
    metric: str = field(
        default="mean",
        metadata={"label": "Metric", "twopy_role": "response_metric"},
    )
    roi_selector: str = field(
        default="visible_rois",
        metadata={
            "label": "ROIs",
            "description": "ROI subset used for DSI computation.",
            "twopy_role": "roi_selector",
        },
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
        default=3.3,
        metadata={
            "label": "Window end (s)",
            "description": "Epoch-relative end time used for the metric.",
            "twopy_role": "epoch_window_stop",
        },
    )
    rectify_responses: bool = field(
        default=True,
        metadata={
            "label": "Rectify responses",
            "description": (
                "Set negative preferred and null responses to zero before DSI."
            ),
        },
    )
    dsi_threshold: float = field(
        default=0.1,
        metadata={
            "label": "DSI show threshold",
            "description": (
                "Only ROIs with absolute DSI at or above this value are shown."
            ),
            "max": 1.0,
            "twopy_role": "table_highlight_threshold",
        },
    )
    output_name: str = field(
        default="direction_selectivity.csv",
        metadata={
            "label": "Output file",
            "description": "Relative CSV path below the workflow output folder.",
            "twopy_role": "output_name",
        },
    )


@workflow(
    id="direction-selectivity",
    name="Direction selectivity",
    version="1.0",
    description="Computes a direction-selectivity index for each current ROI.",
    params=DirectionSelectivityParams,
    author="twopy",
)
def run(
    ctx: CustomRunContext,
    params: DirectionSelectivityParams,
) -> CustomResult:
    """Compute DSI for the selected ROI set.

    Args:
        ctx: Active recording, ROIs, and output helpers.
        params: Custom tab values.

    Returns:
        Table output plus ROI visibility rows for the current response plot.
    """
    selected_roi_indices = ctx.roi_indices_for_selector(params.roi_selector)
    rois = ctx.rois_for_selector(params.roi_selector)
    computation = ctx.compute_standard_responses(rois)
    window_seconds = ctx.epoch_window(
        params.window_start_seconds,
        params.window_stop_seconds,
    )
    preferred = ctx.epoch_metric(
        computation.grouped_responses,
        params.preferred_epoch,
        params.metric,
        window_seconds=window_seconds,
    )
    null = ctx.epoch_metric(
        computation.grouped_responses,
        params.null_epoch,
        params.metric,
        window_seconds=window_seconds,
    )
    dsi = _direction_selectivity_index(
        preferred,
        null,
        rectify=params.rectify_responses,
    )
    csv_path = ctx.output_path(params.output_name)
    ctx.write_roi_table(
        csv_path,
        {
            "direction_selectivity_index": tuple(f"{value:.3f}" for value in dsi),
        },
        roi_labels=rois.labels,
    )
    passing_table_rows = tuple(
        int(index) for index in np.flatnonzero(np.abs(dsi) >= params.dsi_threshold)
    )
    visible_roi_indices = tuple(
        selected_roi_indices[index]
        for index in passing_table_rows
        if index < len(selected_roi_indices)
    )
    return CustomResult(
        message=(
            f"Computed DSI for {len(rois.labels)} ROIs. "
            f"Showing {len(visible_roi_indices)} above absolute threshold."
        ),
        tables=(
            CustomTable(
                "Direction selectivity",
                csv_path,
                highlighted_rows=passing_table_rows,
            ),
        ),
        visible_roi_indices=visible_roi_indices,
    )


def _direction_selectivity_index(
    preferred: np.ndarray,
    null: np.ndarray,
    *,
    rectify: bool,
) -> np.ndarray:
    """Return ``(preferred - null) / (preferred + null)`` per ROI."""
    preferred_values = np.maximum(preferred, 0.0) if rectify else preferred
    null_values = np.maximum(null, 0.0) if rectify else null
    numerator = preferred_values - null_values
    denominator = preferred_values + null_values
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=np.abs(denominator) > 1e-12,
    )
