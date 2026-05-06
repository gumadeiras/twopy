"""Qt widgets and drawing helpers for twopy response plots.

Inputs: plot-ready response data, ROI visibility state, and napari Labels
colors.
Outputs: Qt widgets that draw response traces and expose simple plot options.

This module owns drawing and small option controls only. It does not load
recordings or compute analysis outputs.
"""

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPaintEvent, QPen
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from twopy.napari.plot_data import EpochResponsePlotData, ResponsePlotData

__all__ = [
    "EpochPlotWidget",
    "axis_options_widget",
    "clear_layout",
    "epoch_key_label",
    "global_time_bounds",
    "global_value_bounds",
    "ordered_bounds",
    "roi_colors_from_layer",
    "visibility_options_widget",
]


class _LabelsColorLayer(Protocol):
    """Small protocol for napari Labels layers that can report label colors."""

    def get_color(self, label: int) -> object:
        """Return an RGBA color for one integer label.

        Args:
            label: Positive Labels-layer value.

        Returns:
            RGBA-like color object.
        """
        ...


type CallableAxisBounds = Callable[..., None]
type CallableVisibility = Callable[[str, bool], None]


class EpochPlotWidget(QWidget):
    """Small custom Qt plot for one stimulus epoch type."""

    def __init__(
        self,
        data: EpochResponsePlotData,
        *,
        show_sem: bool,
        roi_indices: tuple[int, ...],
        roi_colors: tuple[QColor, ...],
        time_min: float,
        time_max: float,
        value_min: float,
        value_max: float,
    ) -> None:
        """Create a plot widget.

        Args:
            data: Plot data for one epoch.
            show_sem: Whether SEM bands should be drawn.
            roi_indices: Zero-based ROI indices to draw.
            roi_colors: Colors matching the viewer Labels layer by ROI index.
            time_min: Shared x-axis minimum.
            time_max: Shared x-axis maximum.
            value_min: Shared y-axis minimum.
            value_max: Shared y-axis maximum.
        """
        super().__init__()
        self._data = data
        self._show_sem = show_sem
        self._roi_indices = roi_indices
        self._roi_colors = roi_colors
        self._time_min = time_min
        self._time_max = time_max
        self._value_min = value_min
        self._value_max = value_max
        self.setMinimumHeight(220)

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """Draw response traces.

        Args:
            a0: Qt paint event.

        Returns:
            None.
        """
        del a0
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        plot_rect = QRectF(self.rect().adjusted(60, 12, -18, -46))
        painter.fillRect(self.rect(), QColor("#20252d"))
        _draw_axes(
            painter=painter,
            rect=plot_rect,
            time_min=self._time_min,
            time_max=self._time_max,
            value_min=self._value_min,
            value_max=self._value_max,
        )
        for roi_index in self._roi_indices:
            color = self._roi_colors[roi_index]
            if self._show_sem:
                _draw_sem_band(
                    painter=painter,
                    rect=plot_rect,
                    time_values=self._data.time_seconds,
                    mean_values=self._data.mean_values[roi_index],
                    sem_values=self._data.sem_values[roi_index],
                    time_min=self._time_min,
                    time_max=self._time_max,
                    value_min=self._value_min,
                    value_max=self._value_max,
                    color=color,
                )
            _draw_trace(
                painter=painter,
                rect=plot_rect,
                time_values=self._data.time_seconds,
                values=self._data.mean_values[roi_index],
                time_min=self._time_min,
                time_max=self._time_max,
                value_min=self._value_min,
                value_max=self._value_max,
                color=color,
            )
        painter.end()


def clear_layout(layout: QVBoxLayout) -> None:
    """Remove all widgets from a Qt layout.

    Args:
        layout: Layout to clear.

    Returns:
        None.
    """
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def axis_options_widget(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    on_change: object,
) -> QWidget:
    """Create manual axis-bound controls.

    Args:
        x_min: Initial x-axis minimum.
        x_max: Initial x-axis maximum.
        y_min: Initial y-axis minimum.
        y_max: Initial y-axis maximum.
        on_change: Callback receiving named axis bounds.

    Returns:
        Qt widget containing four numeric inputs.
    """
    callback = cast(CallableAxisBounds, on_change)
    widget = QWidget()
    layout = QGridLayout()
    layout.addWidget(QLabel("Axis limits"), 0, 0, 1, 2)
    x_min_spin = _axis_spin_box(x_min)
    x_max_spin = _axis_spin_box(x_max)
    y_min_spin = _axis_spin_box(y_min)
    y_max_spin = _axis_spin_box(y_max)
    layout.addWidget(QLabel("x min"), 1, 0)
    layout.addWidget(x_min_spin, 1, 1)
    layout.addWidget(QLabel("x max"), 2, 0)
    layout.addWidget(x_max_spin, 2, 1)
    layout.addWidget(QLabel("y min"), 3, 0)
    layout.addWidget(y_min_spin, 3, 1)
    layout.addWidget(QLabel("y max"), 4, 0)
    layout.addWidget(y_max_spin, 4, 1)

    def update_bounds() -> None:
        """Pass current spin-box values back to the tabs widget."""
        callback(
            x_min=x_min_spin.value(),
            x_max=x_max_spin.value(),
            y_min=y_min_spin.value(),
            y_max=y_max_spin.value(),
        )

    x_min_spin.valueChanged.connect(update_bounds)
    x_max_spin.valueChanged.connect(update_bounds)
    y_min_spin.valueChanged.connect(update_bounds)
    y_max_spin.valueChanged.connect(update_bounds)
    widget.setLayout(layout)
    return widget


def visibility_options_widget(
    *,
    title: str,
    labels: tuple[str, ...],
    visibility: dict[str, bool],
    on_change: object,
) -> QWidget:
    """Create select-all/select-none checkboxes for plot visibility.

    Args:
        title: Section label.
        labels: Checkbox labels.
        visibility: Current visibility by label.
        on_change: Callback receiving ``(label, visible)``.

    Returns:
        Qt widget containing controls.
    """
    callback = cast(CallableVisibility, on_change)
    widget = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(QLabel(title))
    button_row = QHBoxLayout()
    all_button = QPushButton("Select all")
    none_button = QPushButton("None")
    button_row.addWidget(all_button)
    button_row.addWidget(none_button)
    layout.addLayout(button_row)
    list_widget = QWidget()
    list_layout = QVBoxLayout()
    list_layout.setContentsMargins(0, 0, 0, 0)
    checkboxes: list[QCheckBox] = []
    for label in labels:
        checkbox = QCheckBox(label)
        checkbox.setChecked(visibility.get(label, True))
        checkbox.stateChanged.connect(
            lambda _state, item=label, box=checkbox: callback(item, box.isChecked())
        )
        checkboxes.append(checkbox)
        list_layout.addWidget(checkbox)
    list_widget.setLayout(list_layout)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(list_widget)
    scroll.setMaximumHeight(_visibility_list_height(checkboxes, visible_rows=5))
    layout.addWidget(scroll)

    def set_all(checked: bool) -> None:
        """Set all checkboxes and visibility flags together."""
        for checkbox in checkboxes:
            checkbox.setChecked(checked)

    all_button.clicked.connect(lambda: set_all(True))
    none_button.clicked.connect(lambda: set_all(False))
    widget.setLayout(layout)
    return widget


def _visibility_list_height(
    checkboxes: list[QCheckBox],
    *,
    visible_rows: int,
) -> int:
    """Return a compact maximum height for a checkbox list.

    Args:
        checkboxes: Checkbox widgets in one visibility section.
        visible_rows: Number of rows to show before scrolling.

    Returns:
        Pixel height for the scroll area.
    """
    if len(checkboxes) == 0:
        return 0
    row_height = max(checkbox.sizeHint().height() for checkbox in checkboxes)
    rows = min(len(checkboxes), visible_rows)
    return row_height * rows + 8


def global_time_bounds(
    data: ResponsePlotData,
    epoch_keys: tuple[tuple[int, str], ...] | None = None,
) -> tuple[float, float]:
    """Return shared x-axis bounds across selected epoch plots.

    Args:
        data: Full response plot data.
        epoch_keys: Optional selected epoch keys.

    Returns:
        ``(min, max)`` response time in seconds.
    """
    epochs = _selected_epochs(data, epoch_keys)
    time_min = min(float(epoch.time_seconds[0]) for epoch in epochs)
    time_max = max(float(epoch.time_seconds[-1]) for epoch in epochs)
    if time_min == time_max:
        return time_min - 1.0, time_max + 1.0
    return time_min, time_max


def global_value_bounds(
    data: ResponsePlotData,
    roi_indices: tuple[int, ...] | None = None,
    epoch_keys: tuple[tuple[int, str], ...] | None = None,
) -> tuple[float, float]:
    """Return shared y-axis bounds across all epoch plots.

    Args:
        data: Full response plot data.
        roi_indices: Optional selected ROI indices.
        epoch_keys: Optional selected epoch keys.

    Returns:
        ``(min, max)`` bounds expanded by 20 percent and rounded to one decimal.
    """
    epochs = _selected_epochs(data, epoch_keys)
    selected_roi_indices = roi_indices or tuple(range(len(epochs[0].roi_labels)))
    lower = np.concatenate(
        tuple(
            (epoch.mean_values - epoch.sem_values)[list(selected_roi_indices), :]
            for epoch in epochs
        ),
        axis=None,
    )
    upper = np.concatenate(
        tuple(
            (epoch.mean_values + epoch.sem_values)[list(selected_roi_indices), :]
            for epoch in epochs
        ),
        axis=None,
    )
    value_min = float(np.nanmin(lower))
    value_max = float(np.nanmax(upper))
    if value_min == value_max:
        return value_min - 1.0, value_max + 1.0
    lower_bound = value_min * 1.2 if value_min < 0 else value_min * 0.8
    upper_bound = value_max * 1.2 if value_max > 0 else value_max * 0.8
    rounded_min = round(lower_bound, 1)
    rounded_max = round(upper_bound, 1)
    if rounded_min >= rounded_max:
        return rounded_min - 0.1, rounded_max + 0.1
    return rounded_min, rounded_max


def roi_colors_from_layer(layer: object | None, count: int) -> tuple[QColor, ...]:
    """Return ROI colors that match the napari Labels layer when possible.

    Args:
        layer: Current napari Labels layer.
        count: Number of ROI traces.

    Returns:
        Tuple of Qt colors.
    """
    if layer is not None and hasattr(layer, "get_color"):
        labels_layer = cast(_LabelsColorLayer, layer)
        return tuple(
            _qcolor_from_rgba(labels_layer.get_color(roi_index + 1))
            for roi_index in range(count)
        )
    base = (
        QColor("#4cc9f0"),
        QColor("#f72585"),
        QColor("#f8961e"),
        QColor("#90be6d"),
        QColor("#b5179e"),
        QColor("#43aa8b"),
    )
    return tuple(base[index % len(base)] for index in range(count))


def epoch_key_label(epoch_number: int, epoch_name: str) -> str:
    """Return a stable checkbox label for one epoch type.

    Args:
        epoch_number: Epoch number.
        epoch_name: Epoch name.

    Returns:
        Human-readable label.
    """
    return f"{epoch_number}: {epoch_name}"


def ordered_bounds(min_value: float, max_value: float) -> tuple[float, float]:
    """Return usable bounds even when manual inputs cross.

    Args:
        min_value: Requested minimum.
        max_value: Requested maximum.

    Returns:
        Ordered bounds with nonzero span.
    """
    if min_value < max_value:
        return min_value, max_value
    if min_value > max_value:
        return max_value, min_value
    return min_value - 0.5, max_value + 0.5


def _axis_spin_box(value: float) -> QDoubleSpinBox:
    """Create one numeric axis-limit input.

    Args:
        value: Initial value.

    Returns:
        Configured spin box.
    """
    spin_box = QDoubleSpinBox()
    spin_box.setRange(-1_000_000.0, 1_000_000.0)
    spin_box.setDecimals(3)
    spin_box.setSingleStep(0.1)
    spin_box.setValue(value)
    return spin_box


def _selected_epochs(
    data: ResponsePlotData,
    epoch_keys: tuple[tuple[int, str], ...] | None,
) -> tuple[EpochResponsePlotData, ...]:
    """Return epochs selected for plotting.

    Args:
        data: Full response plot data.
        epoch_keys: Optional selected epoch keys.

    Returns:
        Tuple of selected epochs.
    """
    if epoch_keys is None:
        return data.epochs
    selected = tuple(
        epoch
        for epoch in data.epochs
        if (epoch.epoch_number, epoch.epoch_name) in epoch_keys
    )
    return selected or data.epochs


def _qcolor_from_rgba(value: object) -> QColor:
    """Convert a napari RGBA color into ``QColor``.

    Args:
        value: RGBA-like sequence, usually floats in ``[0, 1]``.

    Returns:
        Qt color.
    """
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 3:
        return QColor("#4cc9f0")
    if np.nanmax(array[:3]) <= 1.0:
        array = array * 255.0
    alpha = int(array[3]) if array.size > 3 else 255
    return QColor(int(array[0]), int(array[1]), int(array[2]), alpha)


def _draw_axes(
    *,
    painter: QPainter,
    rect: QRectF,
    time_min: float,
    time_max: float,
    value_min: float,
    value_max: float,
) -> None:
    """Draw x/y axes, ticks, labels, and the zero-response line.

    Args:
        painter: Active Qt painter.
        rect: Plot rectangle.
        time_min: X-axis minimum.
        time_max: X-axis maximum.
        value_min: Y-axis minimum.
        value_max: Y-axis maximum.

    Returns:
        None.
    """
    axis_color = QColor("#cfd6df")
    grid_color = QColor("#6d7683")
    painter.setPen(QPen(axis_color, 1))
    painter.drawLine(rect.bottomLeft(), rect.bottomRight())
    painter.drawLine(rect.bottomLeft(), rect.topLeft())

    if value_min <= 0 <= value_max:
        zero_y = _plot_point(
            rect,
            time_min,
            0.0,
            time_min,
            time_max,
            value_min,
            value_max,
        ).y()
        painter.setPen(_zero_line_pen(grid_color))
        painter.drawLine(QPointF(rect.left(), zero_y), QPointF(rect.right(), zero_y))
    if time_min <= 0 <= time_max:
        zero_x = _plot_point(
            rect,
            0.0,
            value_min,
            time_min,
            time_max,
            value_min,
            value_max,
        ).x()
        painter.setPen(_zero_line_pen(grid_color))
        painter.drawLine(QPointF(zero_x, rect.bottom()), QPointF(zero_x, rect.top()))

    painter.setPen(QPen(axis_color, 1))
    for tick in _axis_ticks(time_min, time_max):
        point = _plot_point(
            rect,
            tick,
            value_min,
            time_min,
            time_max,
            value_min,
            value_max,
        )
        painter.drawLine(point, QPointF(point.x(), point.y() + 4.0))
        painter.drawText(QPointF(point.x() - 12.0, point.y() + 18.0), f"{tick:.1f}")
    for tick in _axis_ticks(value_min, value_max):
        point = _plot_point(
            rect,
            time_min,
            tick,
            time_min,
            time_max,
            value_min,
            value_max,
        )
        painter.drawLine(QPointF(point.x() - 4.0, point.y()), point)
        painter.drawText(QPointF(point.x() - 48.0, point.y() + 4.0), f"{tick:.1f}")

    painter.drawText(
        QPointF(rect.center().x() - 26.0, rect.bottom() + 36.0),
        "Time (s)",
    )
    painter.save()
    painter.translate(14.0, rect.center().y() + 18.0)
    painter.rotate(-90.0)
    painter.drawText(QPointF(0.0, 0.0), "dF/F")
    painter.restore()


def _zero_line_pen(color: QColor) -> QPen:
    """Create the shared dashed pen for zero-reference plot lines.

    Args:
        color: Zero-line color.

    Returns:
        Thick dashed pen used for both ``x=0`` and ``y=0``.
    """
    pen = QPen(color, 2)
    pen.setDashPattern([10.0, 8.0])
    return pen


def _axis_ticks(min_value: float, max_value: float) -> tuple[float, ...]:
    """Return six tick values including axis endpoints.

    Args:
        min_value: Axis minimum.
        max_value: Axis maximum.

    Returns:
        Tick values with endpoints included.
    """
    if min_value == max_value:
        return (min_value,)
    return tuple(float(value) for value in np.linspace(min_value, max_value, 6))


def _draw_trace(
    *,
    painter: QPainter,
    rect: QRectF,
    time_values: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    time_min: float,
    time_max: float,
    value_min: float,
    value_max: float,
    color: QColor,
) -> None:
    """Draw one ROI mean trace.

    Args:
        painter: Active Qt painter.
        rect: Plot rectangle.
        time_values: X-axis values in seconds.
        values: Mean response values.
        time_min: X-axis minimum.
        time_max: X-axis maximum.
        value_min: Y-axis minimum.
        value_max: Y-axis maximum.
        color: Trace color.

    Returns:
        None.
    """
    path = _trace_path(
        rect,
        time_values,
        values,
        time_min,
        time_max,
        value_min,
        value_max,
    )
    painter.setPen(QPen(color, 2))
    painter.drawPath(path)


def _draw_sem_band(
    *,
    painter: QPainter,
    rect: QRectF,
    time_values: npt.NDArray[np.float64],
    mean_values: npt.NDArray[np.float64],
    sem_values: npt.NDArray[np.float64],
    time_min: float,
    time_max: float,
    value_min: float,
    value_max: float,
    color: QColor,
) -> None:
    """Draw one ROI SEM band.

    Args:
        painter: Active Qt painter.
        rect: Plot rectangle.
        time_values: X-axis values in seconds.
        mean_values: Mean response values.
        sem_values: SEM values.
        time_min: X-axis minimum.
        time_max: X-axis maximum.
        value_min: Y-axis minimum.
        value_max: Y-axis maximum.
        color: Band color.

    Returns:
        None.
    """
    upper = mean_values + sem_values
    lower = mean_values - sem_values
    path = _trace_path(
        rect,
        time_values,
        upper,
        time_min,
        time_max,
        value_min,
        value_max,
    )
    for index in range(len(time_values) - 1, -1, -1):
        point = _plot_point(
            rect,
            time_values[index],
            lower[index],
            time_min,
            time_max,
            value_min,
            value_max,
        )
        path.lineTo(point)
    band_color = QColor(color)
    band_color.setAlpha(55)
    painter.fillPath(path, band_color)


def _trace_path(
    rect: QRectF,
    time_values: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    time_min: float,
    time_max: float,
    value_min: float,
    value_max: float,
) -> QPainterPath:
    """Build a Qt path for one trace.

    Args:
        rect: Plot rectangle.
        time_values: X-axis values in seconds.
        values: Y-axis values.
        time_min: X-axis minimum.
        time_max: X-axis maximum.
        value_min: Y-axis minimum.
        value_max: Y-axis maximum.

    Returns:
        Painter path for valid values.
    """
    path = QPainterPath()
    first = True
    for time_value, value in zip(time_values, values, strict=True):
        if np.isnan(value) or time_value < time_min or time_value > time_max:
            first = True
            continue
        point = _plot_point(
            rect,
            time_value,
            value,
            time_min,
            time_max,
            value_min,
            value_max,
        )
        if first:
            path.moveTo(point)
            first = False
        else:
            path.lineTo(point)
    return path


def _plot_point(
    rect: QRectF,
    time_value: float,
    value: float,
    time_min: float,
    time_max: float,
    value_min: float,
    value_max: float,
) -> QPointF:
    """Map data coordinates into plot coordinates.

    Args:
        rect: Plot rectangle.
        time_value: X value in seconds.
        value: Y value.
        time_min: Minimum time value.
        time_max: Maximum time value.
        value_min: Minimum y value.
        value_max: Maximum y value.

    Returns:
        Qt point in widget coordinates.
    """
    time_span = time_max - time_min
    x_fraction = 0.0 if time_span == 0 else (time_value - time_min) / time_span
    y_fraction = (value - value_min) / (value_max - value_min)
    x = rect.left() + x_fraction * rect.width()
    y = rect.bottom() - y_fraction * rect.height()
    return QPointF(x, y)
