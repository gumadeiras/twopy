"""Custom workflow tab for the twopy napari sidebar.

The tab lets users choose a workflow, edit its parameters, run it, and inspect
returned files, tables, and plots. Validation and analysis helpers live in
``twopy.custom`` so this module stays focused on Qt widgets.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QPalette, QWheelEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from twopy.custom import (
    CustomLineBand,
    CustomLinePlot,
    CustomParameterSpec,
    CustomResult,
    CustomTable,
    CustomWorkflow,
    WorkflowDiscoveryResult,
    build_parameter_object,
    discover_custom_workflows,
    parameter_specs,
)
from twopy.napari.text import configure_placeholder

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["CustomWorkflowPanel"]

WorkflowRunner = Callable[[CustomWorkflow, object | None], CustomResult]
ParameterSpecProvider = Callable[[CustomWorkflow], tuple[CustomParameterSpec, ...]]
_CUSTOM_CHOICE_LABELS = {
    "all_epochs": "all epochs",
    "all_rois": "all ROIs",
    "recording_metadata": "auto",
    "visible_epochs": "visible epochs",
    "visible_rois": "visible ROIs",
}
_PLOT_BACKGROUND = "#20252d"
_PLOT_AXIS_COLOR = "#cfd6df"
_PLOT_GRID_COLOR = "#6d7683"
_PLOT_TRACE_COLORS = (
    "#4cc9f0",
    "#f72585",
    "#f8961e",
    "#90be6d",
    "#b5179e",
    "#43aa8b",
)
_EMPTY_RESULT_TEXT = "No custom workflow outputs."


class CustomWorkflowPanel(QScrollArea):
    """Qt panel that lists workflows and runs the selected one."""

    def __init__(
        self,
        *,
        workflow_paths: tuple[Path, ...],
        on_run: WorkflowRunner,
        parameter_specs_for_workflow: ParameterSpecProvider | None = None,
    ) -> None:
        """Create the Custom tab panel.

        Args:
            workflow_paths: Python files or folders to scan.
            on_run: Callback that executes the selected workflow.
            parameter_specs_for_workflow: Optional callback that can adjust
                controls for the loaded recording.
        """
        super().__init__()
        self._workflow_paths = workflow_paths
        self._on_run = on_run
        self._parameter_specs_for_workflow = parameter_specs_for_workflow
        self._discovery_result = WorkflowDiscoveryResult((), ())
        self._parameter_controls: dict[str, object] = {}

        self._workflow_combo = QComboBox()
        self._workflow_combo.currentIndexChanged.connect(
            self._selected_workflow_changed
        )
        self._reload_button = QPushButton("Reload workflows")
        self._reload_button.clicked.connect(self.reload_workflows)
        self._description_label = QLabel("")
        self._description_label.setWordWrap(True)
        self._version_label = QLabel("")
        self._version_label.setWordWrap(True)
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._parameter_group = QGroupBox("Parameters")
        self._parameter_layout = QFormLayout()
        self._parameter_group.setLayout(self._parameter_layout)
        self._run_button = QPushButton("Run")
        self._run_button.clicked.connect(self._run_selected_workflow)
        self._result_layout = QVBoxLayout()
        self._result_group = QGroupBox("Result")
        self._result_group.setLayout(self._result_layout)

        root_layout = QVBoxLayout()
        root_layout.setSpacing(8)
        top_row = QHBoxLayout()
        top_row.addWidget(self._workflow_combo)
        top_row.addWidget(self._reload_button)
        root_layout.addLayout(top_row)
        root_layout.addWidget(self._version_label)
        root_layout.addWidget(self._description_label)
        root_layout.addWidget(self._parameter_group)
        root_layout.addWidget(self._run_button)
        root_layout.addWidget(self._status_label)
        root_layout.addWidget(self._result_group)
        root_layout.addStretch(1)

        content = QWidget()
        content.setLayout(root_layout)
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidget(content)
        self.reload_workflows()

    def refresh_parameters(self) -> None:
        """Rebuild controls for the selected workflow.

        Args:
            None.

        Returns:
            None.
        """
        self._selected_workflow_changed()

    def reload_workflows(self) -> None:
        """Reload workflow files and refresh the dropdown."""
        self._discovery_result = discover_custom_workflows(self._workflow_paths)
        self._workflow_combo.blockSignals(True)
        self._workflow_combo.clear()
        for workflow in self._discovery_result.workflows:
            self._workflow_combo.addItem(_workflow_label(workflow), workflow)
        self._workflow_combo.blockSignals(False)
        self._selected_workflow_changed()
        self._set_discovery_status()

    def _selected_workflow_changed(self) -> None:
        """Show the selected workflow description and controls."""
        workflow = self._selected_workflow()
        _clear_layout(self._parameter_layout)
        self._parameter_controls.clear()
        if workflow is None:
            self._description_label.setText("No custom workflows available.")
            self._version_label.setText("")
            self._run_button.setEnabled(False)
            return
        self._run_button.setEnabled(True)
        self._description_label.setText(workflow.description)
        source = workflow.source_path
        self._version_label.setText(
            f"version {workflow.version}; source {source}",
        )
        if workflow.params_type is None:
            self._parameter_layout.addRow(QLabel("No parameters."))
            return
        for spec in self._parameter_specs(workflow):
            widget = _parameter_widget(spec)
            self._parameter_controls[spec.name] = widget
            self._parameter_layout.addRow(spec.label, widget)

    def _run_selected_workflow(self) -> None:
        """Run the selected workflow with current GUI values."""
        workflow = self._selected_workflow()
        if workflow is None:
            self._status_label.setText("No custom workflow selected.")
            self._show_empty_result()
            return
        self._show_empty_result()
        try:
            params = build_parameter_object(
                workflow.params_type,
                self._parameter_values(workflow),
            )
            result = self._on_run(workflow, params)
        except Exception as error:
            self._status_label.setText(f"Custom workflow failed: {error}")
            return
        self._status_label.setText(result.message)
        self._render_result(result)

    def _selected_workflow(self) -> CustomWorkflow | None:
        """Return the currently selected workflow."""
        data = self._workflow_combo.currentData()
        if isinstance(data, CustomWorkflow):
            return data
        return None

    def _parameter_values(self, workflow: CustomWorkflow) -> dict[str, object]:
        """Return current parameter values keyed by field name."""
        if workflow.params_type is None:
            return {}
        values: dict[str, object] = {}
        for spec in self._parameter_specs(workflow):
            widget = self._parameter_controls[spec.name]
            values[spec.name] = _read_parameter_widget(widget)
        return values

    def _parameter_specs(
        self, workflow: CustomWorkflow
    ) -> tuple[CustomParameterSpec, ...]:
        """Return parameter specs for one workflow."""
        if workflow.params_type is None:
            return ()
        if self._parameter_specs_for_workflow is not None:
            return self._parameter_specs_for_workflow(workflow)
        return parameter_specs(workflow.params_type)

    def _set_discovery_status(self) -> None:
        """Show how many workflows loaded or failed."""
        workflow_count = len(self._discovery_result.workflows)
        error_count = len(self._discovery_result.errors)
        if len(self._workflow_paths) == 0:
            self._status_label.setText("No custom workflow paths configured.")
            return
        if error_count == 0:
            self._status_label.setText(f"Loaded {workflow_count} custom workflows.")
            return
        error_lines = "\n".join(
            f"{error.path}: {error.message}" for error in self._discovery_result.errors
        )
        self._status_label.setText(
            f"Loaded {workflow_count} custom workflows; rejected {error_count}.\n"
            f"{error_lines}"
        )

    def _render_result(self, result: CustomResult) -> None:
        """Show files, tables, and plots returned by a workflow."""
        _clear_layout(self._result_layout)
        output_root = _custom_output_display_root(result)
        if output_root is not None:
            self._result_layout.addWidget(
                _wrapped_label(f"Output path: {output_root}/")
            )
        for table in result.tables:
            self._result_layout.addWidget(_wrapped_label(f"Table: {table.title}"))
            self._result_layout.addWidget(_table_widget(table))
        y_bounds = _line_plot_y_bounds(result.plots)
        for plot in result.plots:
            self._result_layout.addWidget(_line_plot_widget(plot, y_bounds=y_bounds))
        if self._result_layout.count() == 0:
            self._add_empty_result_label()

    def _show_empty_result(self) -> None:
        """Clear stale workflow outputs from the Result panel."""
        _clear_layout(self._result_layout)
        self._add_empty_result_label()

    def _add_empty_result_label(self) -> None:
        """Show the empty-state label in the Result panel."""
        self._result_layout.addWidget(_wrapped_label(_EMPTY_RESULT_TEXT))


def _workflow_label(workflow: CustomWorkflow) -> str:
    """Return the dropdown label for one workflow."""
    return f"{workflow.name} {workflow.version}"


def _custom_output_display_root(result: CustomResult) -> str | None:
    """Return the compact common output folder shown for workflow artifacts."""
    paths = (*(_table_display_path(table) for table in result.tables), *result.files)
    if len(paths) == 0:
        return None
    return _custom_output_root_text(paths[0])


def _custom_output_root_text(path: Path) -> str:
    """Return ``path`` through ``custom_outputs/<workflow>`` when present."""
    parts = path.parts
    try:
        custom_outputs_index = parts.index("custom_outputs")
    except ValueError:
        return str(path.parent)
    workflow_index = custom_outputs_index + 1
    if workflow_index >= len(parts):
        return str(path)
    return str(Path(*parts[: workflow_index + 1]))


def _parameter_widget(spec: CustomParameterSpec) -> QWidget:
    """Create one Qt widget for a parameter spec."""
    if spec.kind == "bool":
        widget = QCheckBox()
        widget.setChecked(bool(spec.default))
        return widget
    if spec.kind == "int":
        widget = QSpinBox()
        widget.setRange(
            int(spec.minimum if spec.minimum is not None else -1_000_000),
            int(spec.maximum if spec.maximum is not None else 1_000_000),
        )
        widget.setSingleStep(int(spec.step if spec.step is not None else 1))
        widget.setValue(_int_default(spec))
        return widget
    if spec.kind == "float":
        widget = QDoubleSpinBox()
        widget.setRange(
            float(spec.minimum if spec.minimum is not None else -1_000_000.0),
            float(spec.maximum if spec.maximum is not None else 1_000_000.0),
        )
        widget.setSingleStep(float(spec.step if spec.step is not None else 0.1))
        widget.setDecimals(_float_decimals(spec))
        widget.setValue(_float_default(spec))
        return widget
    if spec.kind in {"str", "path"}:
        widget = QLineEdit(str(spec.default))
        configure_placeholder(widget, _parameter_placeholder(spec))
        if spec.description:
            widget.setToolTip(spec.description)
        return widget
    if spec.kind == "choice":
        widget = QComboBox()
        for choice in spec.choices:
            widget.addItem(_choice_label(choice), choice)
        index = widget.findData(spec.default)
        if index < 0:
            default_label = _choice_label(spec.default)
            index = widget.findText(default_label)
        if index >= 0:
            widget.setCurrentIndex(index)
        return widget
    msg = f"Unsupported custom parameter kind {spec.kind!r}."
    raise ValueError(msg)


def _read_parameter_widget(widget: object) -> object:
    """Return the current value from one parameter widget."""
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QSpinBox | QDoubleSpinBox):
        return widget.value()
    if isinstance(widget, QLineEdit):
        return widget.text()
    if isinstance(widget, QComboBox):
        value = widget.currentData()
        if value is not None:
            return value
        return widget.currentText()
    msg = f"Unsupported parameter widget {type(widget).__name__}."
    raise TypeError(msg)


def _choice_label(value: object) -> str:
    """Return a readable label for one choice value."""
    if isinstance(value, Enum):
        return str(value.value)
    text = str(value)
    return _CUSTOM_CHOICE_LABELS.get(text, text)


def _parameter_placeholder(spec: CustomParameterSpec) -> str:
    """Return a short placeholder for a text-backed workflow parameter."""
    if spec.role == "output_name":
        return "Output path"
    if spec.description:
        return spec.description
    return spec.label


def _line_plot_widget(
    plot: CustomLinePlot,
    *,
    y_bounds: tuple[float, float] | None = None,
) -> QWidget:
    """Create a matplotlib widget for one custom line plot."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(3.8, 3.35), dpi=100, facecolor=_PLOT_BACKGROUND)
    axes = figure.add_subplot(111)
    figure.patch.set_facecolor(_PLOT_BACKGROUND)
    axes.set_facecolor(_PLOT_BACKGROUND)
    y_values = plot.y.reshape(1, -1) if plot.y.ndim == 1 else plot.y
    labels = plot.labels or tuple(
        f"series {index + 1}" for index in range(y_values.shape[0])
    )
    for band in plot.bands:
        _draw_line_plot_band(axes, plot.x, band, colors=plot.colors)
    for index, values in enumerate(y_values):
        axes.plot(
            plot.x,
            values,
            label=labels[index],
            color=_line_plot_color(index, plot.colors),
            linewidth=1.8,
        )
    if y_bounds is not None:
        axes.set_ylim(*y_bounds)
    _style_line_plot_axes(axes, title=plot.title, y_label=plot.y_label)
    if y_values.shape[0] <= 12:
        _style_line_plot_legend(axes)
    figure.tight_layout()
    return _line_plot_canvas(figure)


def _draw_line_plot_band(
    axes: Axes,
    x_values: np.ndarray,
    band: CustomLineBand,
    *,
    colors: tuple[str, ...],
) -> None:
    """Draw one filled uncertainty band behind its owning line series."""
    axes.fill_between(
        x_values,
        band.lower,
        band.upper,
        color=_line_plot_color(band.series_index, colors),
        alpha=0.2,
        linewidth=0.0,
        label=band.label or None,
    )


def _line_plot_color(index: int, colors: tuple[str, ...]) -> str:
    """Return the explicit or fallback color for one custom line series."""
    if index < len(colors):
        return colors[index]
    return _PLOT_TRACE_COLORS[index % len(_PLOT_TRACE_COLORS)]


def _line_plot_y_bounds(
    plots: tuple[CustomLinePlot, ...],
) -> tuple[float, float] | None:
    """Return shared y-axis bounds for all visible custom line plots."""
    finite_values: list[np.ndarray] = []
    for plot in plots:
        values = np.asarray(plot.y, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size > 0:
            finite_values.append(finite)
        for band in plot.bands:
            band_values = np.asarray((band.lower, band.upper), dtype=np.float64)
            finite_band = band_values[np.isfinite(band_values)]
            if finite_band.size > 0:
                finite_values.append(finite_band)
    if not finite_values:
        return None

    combined = np.concatenate(finite_values)
    value_min = float(np.min(combined))
    value_max = float(np.max(combined))
    if value_min == value_max:
        padding = max(abs(value_min) * 0.1, 1.0)
        return value_min - padding, value_max + padding
    value_span = value_max - value_min
    padding = 0.05 * value_span
    return value_min - padding, value_max + padding


def _line_plot_canvas(figure: Figure) -> QWidget:
    """Return a matplotlib canvas that lets wheel scrolling reach the parent tab."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    class WheelPassthroughCanvas(FigureCanvasQTAgg):
        """Matplotlib canvas that does not consume Custom-tab wheel scrolling."""

        def wheelEvent(self, event: QWheelEvent) -> None:
            """Ignore wheel events so the surrounding scroll area can handle them."""
            event.ignore()

    canvas = WheelPassthroughCanvas(figure)
    canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    canvas.setMinimumWidth(1)
    return canvas


def _style_line_plot_axes(axes: Axes, *, title: str, y_label: str) -> None:
    """Apply the response-widget plot palette to a matplotlib axes."""
    axes.set_title(title, color=_PLOT_AXIS_COLOR, fontsize=10)
    axes.set_xlabel("Time (s)", color=_PLOT_AXIS_COLOR)
    axes.set_ylabel(y_label, color=_PLOT_AXIS_COLOR)
    axes.tick_params(colors=_PLOT_AXIS_COLOR, labelsize=8)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(_PLOT_AXIS_COLOR)
        axes.spines[side].set_linewidth(1.0)
    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()
    if x_min <= 0.0 <= x_max:
        _draw_zero_reference_line(axes, axis="x")
    if y_min <= 0.0 <= y_max:
        _draw_zero_reference_line(axes, axis="y")


def _style_line_plot_legend(axes: Axes) -> None:
    """Apply the response-widget palette to a small matplotlib legend."""
    legend = axes.legend(loc="best", fontsize=7)
    legend.get_frame().set_facecolor(_PLOT_BACKGROUND)
    legend.get_frame().set_edgecolor(_PLOT_AXIS_COLOR)
    for text in legend.get_texts():
        text.set_color(_PLOT_AXIS_COLOR)


def _draw_zero_reference_line(axes: Axes, *, axis: Literal["x", "y"]) -> None:
    """Draw one dashed zero-reference line with response-plot styling."""
    draw = axes.axvline if axis == "x" else axes.axhline
    draw(
        0.0,
        color=_PLOT_GRID_COLOR,
        linewidth=1.4,
        linestyle=(0, (5, 4)),
        zorder=0,
    )


def _wrapped_label(text: str) -> QLabel:
    """Return a word-wrapped label."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setMinimumWidth(1)
    return label


def _table_widget(table_result: CustomTable) -> QWidget:
    """Return a Qt table preview for one CSV or TSV table path."""
    path = table_result.path
    try:
        header, rows = _read_table_rows(path)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        return _wrapped_label(f"Could not preview table: {error}")
    if len(header) == 0:
        return _wrapped_label("Table is empty.")
    table = QTableWidget(len(rows), len(header))
    table.setHorizontalHeaderLabels(header)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    table.setMinimumWidth(1)
    header_view = table.horizontalHeader()
    if header_view is not None:
        header_view.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.setWordWrap(True)
    highlighted_rows = {
        row_index
        for row_index in table_result.highlighted_rows
        if 0 <= row_index < len(rows)
    }
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            item = QTableWidgetItem(value)
            if row_index in highlighted_rows:
                _apply_table_highlight(item, table)
            table.setItem(row_index, column_index, item)
    table.setMinimumHeight(min(360, 44 + 28 * max(1, len(rows))))
    return table


def _apply_table_highlight(item: QTableWidgetItem, table: QTableWidget) -> None:
    """Apply theme-aware highlight colors to one table item."""
    palette = table.palette()
    item.setBackground(palette.color(QPalette.ColorRole.Highlight))
    item.setForeground(palette.color(QPalette.ColorRole.HighlightedText))


def _table_display_path(table: CustomTable) -> Path:
    """Return the user-facing path for one custom table."""
    return table.display_path if table.display_path is not None else table.path


def _read_table_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read CSV or TSV rows for direct Custom tab display."""
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as table_file:
        rows = list(csv.reader(table_file, delimiter=delimiter))
    if len(rows) == 0:
        return [], []
    header = rows[0]
    width = len(header)
    return header, [row[:width] + [""] * max(0, width - len(row)) for row in rows[1:]]


def _int_default(spec: CustomParameterSpec) -> int:
    """Return an integer default for an integer parameter spec."""
    if isinstance(spec.default, str | int | float):
        return int(spec.default)
    msg = f"Parameter {spec.name!r} default is not integer-compatible."
    raise ValueError(msg)


def _float_default(spec: CustomParameterSpec) -> float:
    """Return a float default for a float parameter spec."""
    if isinstance(spec.default, str | int | float):
        return float(spec.default)
    msg = f"Parameter {spec.name!r} default is not float-compatible."
    raise ValueError(msg)


def _float_decimals(spec: CustomParameterSpec) -> int:
    """Return decimal precision for one float parameter widget."""
    if spec.decimals is not None:
        return spec.decimals
    if spec.role in {
        "epoch_window_start",
        "epoch_window_stop",
        "response_window_start",
        "response_window_stop",
        "table_highlight_threshold",
    }:
        return 3
    return 6


def _clear_layout(layout: QLayout) -> None:
    """Remove all widgets from a Qt layout."""
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        if widget is None:
            child_layout = item.layout()
            if child_layout is not None:
                _clear_layout(child_layout)
            continue
        widget.hide()
        widget.setParent(None)
        widget.deleteLater()
