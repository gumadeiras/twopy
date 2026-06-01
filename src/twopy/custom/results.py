"""Output objects returned by twopy custom workflows.

Workflow files use these plain dataclasses to describe files, tables, plots,
ROI updates, and response plots that the Custom tab should display after a run.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from twopy.analysis.response_plotting import ResponsePlotData
from twopy.roi import RoiSet

__all__ = [
    "CustomLineBand",
    "CustomLinePlot",
    "CustomResult",
    "CustomTable",
]


@dataclass(frozen=True)
class CustomTable:
    """CSV or TSV table shown in the Custom tab.

    Args:
        title: Short title shown above the table.
        path: CSV or TSV file written by the workflow.
        highlighted_rows: Zero-based data rows to highlight.
        display_path: Optional path shown to the user when preview reads from a
            local cache path.

    Returns:
        Immutable table descriptor.
    """

    title: str
    path: Path
    highlighted_rows: tuple[int, ...] = ()
    display_path: Path | None = None


@dataclass(frozen=True)
class CustomLineBand:
    """Filled uncertainty band shown behind one custom line-plot series.

    Args:
        series_index: Zero-based line series that owns the band color.
        lower: One-dimensional lower bound values.
        upper: One-dimensional upper bound values.
        label: Optional legend label for the band.

    Returns:
        Immutable band descriptor.
    """

    series_index: int
    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]
    label: str = ""


@dataclass(frozen=True)
class CustomLinePlot:
    """Line plot shown in the Custom tab.

    Args:
        title: Short display title for the GUI.
        x: One-dimensional x-axis values.
        y: One-dimensional values or two-dimensional ``(series, samples)``
            values.
        labels: Optional series labels.
        y_label: Y-axis label shown beside the plot.
        bands: Optional filled uncertainty bands that share the x-axis.
        colors: Optional ``#RRGGBB`` line colors matching the plotted series.

    Returns:
        Immutable plot descriptor.
    """

    title: str
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    labels: tuple[str, ...] = ()
    y_label: str = "Value"
    bands: tuple[CustomLineBand, ...] = ()
    colors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CustomResult:
    """Outputs returned by one custom workflow run.

    Args:
        message: Status text shown after the workflow finishes.
        files: Files written by the workflow.
        tables: Tables to preview in the Custom tab.
        plots: Line plots to show in the Custom tab.
        roi_set: Optional ROIs to replace the active Labels layer.
        response_plot_data: Optional responses to show in the response dock.
        visible_roi_indices: Optional ROI rows to select in the existing
            response dock without replacing its plot data.

    Returns:
        Immutable workflow result.
    """

    message: str
    files: tuple[Path, ...] = ()
    tables: tuple[CustomTable, ...] = ()
    plots: tuple[CustomLinePlot, ...] = ()
    roi_set: RoiSet | None = None
    response_plot_data: ResponsePlotData | None = None
    visible_roi_indices: tuple[int, ...] | None = None
