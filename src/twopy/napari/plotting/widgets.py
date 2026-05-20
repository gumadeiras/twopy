"""Qt widgets and drawing helpers for twopy response plots.

Inputs: plot-ready response data, ROI visibility state, and napari Labels
colors.
Outputs: Qt widgets that draw response traces.

This module owns drawing helpers only. It does not load recordings, compute
analysis outputs, or build option panels.
"""

from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from qtpy.QtCore import QPointF, QRectF, QSize, Qt
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPaintEvent, QPen
from qtpy.QtWidgets import QLayout, QSizePolicy, QWidget

from twopy.napari.plotting.data import EpochResponsePlotData, ResponsePlotData

DEFAULT_VALUE_BOUNDS = (-1.0, 1.0)
_LEFT_MARGIN = 56.0
_TOP_MARGIN = 12.0
_RIGHT_MARGIN = 18.0
_BOTTOM_MARGIN = 46.0
_STIMULUS_MARKER_Y_OFFSET = 8.0

__all__ = [
    "EpochPlotWidget",
    "clear_layout",
    "global_time_bounds",
    "global_value_bounds",
    "ordered_bounds",
    "opaque_colors",
    "resolved_time_bounds",
    "resolved_value_bounds",
    "roi_colors_from_label_values",
    "roi_colors_from_layer",
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
        plot_size: int = 300,
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
            plot_size: Plot width in screen pixels.
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
        self._plot_size = int(plot_size)
        self.setFixedSize(self._plot_size, _plot_widget_height(self._plot_size))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        """Return the preferred size for one epoch plot.

        Args:
            None.

        Returns:
            Qt size with a square data area plus axis label margins.
        """
        return QSize(self._plot_size, _plot_widget_height(self._plot_size))

    def update_display(
        self,
        *,
        show_sem: bool,
        roi_indices: tuple[int, ...],
        roi_colors: tuple[QColor, ...],
        time_min: float,
        time_max: float,
        value_min: float,
        value_max: float,
        plot_size: int,
    ) -> None:
        """Update visible plot options without rebuilding the widget.

        Args:
            show_sem: Whether SEM bands should be drawn.
            roi_indices: Zero-based ROI indices to draw.
            roi_colors: Colors matching the viewer Labels layer by ROI index.
            time_min: Shared x-axis minimum.
            time_max: Shared x-axis maximum.
            value_min: Shared y-axis minimum.
            value_max: Shared y-axis maximum.
            plot_size: Plot width in screen pixels.

        Returns:
            None.

        Visibility toggles do not change the computed response arrays. Updating
        the existing widget keeps checkbox interactions responsive because Qt
        only repaints the cached plot instead of destroying and recreating the
        full plot strip.
        """
        self._show_sem = show_sem
        self._roi_indices = roi_indices
        self._roi_colors = roi_colors
        self._time_min = time_min
        self._time_max = time_max
        self._value_min = value_min
        self._value_max = value_max
        self._plot_size = int(plot_size)
        self.setFixedSize(self._plot_size, _plot_widget_height(self._plot_size))
        self.updateGeometry()
        self.update()

    def update_data(self, data: EpochResponsePlotData) -> None:
        """Replace computed response data without rebuilding the widget.

        Args:
            data: New plot data for the same epoch panel.

        Returns:
            None.
        """
        self._data = data
        self.update()

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
        font = painter.font()
        font.setPixelSize(_plot_text_pixel_size(self._plot_size))
        painter.setFont(font)
        plot_rect = _square_plot_rect(QRectF(self.rect()))
        painter.fillRect(self.rect(), QColor("#20252d"))
        _draw_axes(
            painter=painter,
            rect=plot_rect,
            time_min=self._time_min,
            time_max=self._time_max,
            value_min=self._value_min,
            value_max=self._value_max,
        )
        _draw_epoch_time_spans(
            painter=painter,
            rect=plot_rect,
            spans=self._data.epoch_time_spans,
            time_min=self._time_min,
            time_max=self._time_max,
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


def _square_plot_rect(bounds: QRectF) -> QRectF:
    """Return a square plot rectangle with room for axis labels.

    Args:
        bounds: Full widget rectangle.

    Returns:
        Square drawing rectangle inside the widget.
    """
    side = max(
        1.0,
        min(
            bounds.width() - _LEFT_MARGIN - _RIGHT_MARGIN,
            bounds.height() - _TOP_MARGIN - _BOTTOM_MARGIN,
        ),
    )
    return QRectF(_LEFT_MARGIN, _TOP_MARGIN, side, side)


def _plot_widget_height(plot_width: int) -> int:
    """Return the compact widget height that preserves a square data area.

    Args:
        plot_width: Requested plot widget width in screen pixels.

    Returns:
        Widget height in screen pixels.
    """
    axis_side = max(1.0, float(plot_width) - _LEFT_MARGIN - _RIGHT_MARGIN)
    return int(_TOP_MARGIN + axis_side + _BOTTOM_MARGIN)


def _plot_text_pixel_size(plot_width: int) -> int:
    """Return axis text size scaled to the drawn plot width."""
    return max(8, min(14, round(plot_width * 0.055)))


def clear_layout(layout: QLayout) -> None:
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
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()


def global_time_bounds(
    data: ResponsePlotData,
    epoch_indices: tuple[int, ...] | None = None,
) -> tuple[float, float]:
    """Return shared x-axis bounds across selected epoch plots.

    Args:
        data: Full response plot data.
        epoch_indices: Optional selected epoch row indices.

    Returns:
        ``(min, max)`` response time in seconds.
    """
    epochs = _selected_epochs(data, epoch_indices)
    time_min = min(float(epoch.time_seconds[0]) for epoch in epochs)
    time_max = max(float(epoch.time_seconds[-1]) for epoch in epochs)
    if time_min == time_max:
        return time_min - 1.0, time_max + 1.0
    return time_min, time_max


def global_value_bounds(
    data: ResponsePlotData,
    roi_indices: tuple[int, ...] | None = None,
    epoch_indices: tuple[int, ...] | None = None,
) -> tuple[float, float]:
    """Return shared y-axis bounds across all epoch plots.

    Args:
        data: Full response plot data.
        roi_indices: Optional selected ROI indices.
        epoch_indices: Optional selected epoch row indices.

    Returns:
        ``(min, max)`` bounds expanded by 20 percent and rounded to one decimal.
    """
    epochs = _selected_epochs(data, epoch_indices)
    if len(epochs) == 0:
        return DEFAULT_VALUE_BOUNDS
    selected_roi_indices = (
        roi_indices
        if roi_indices is not None
        else tuple(range(len(epochs[0].roi_labels)))
    )
    if len(selected_roi_indices) == 0:
        return DEFAULT_VALUE_BOUNDS
    selected_roi_rows = list(selected_roi_indices)
    lower_values = np.concatenate(
        tuple(
            (epoch.mean_values - epoch.sem_values)[selected_roi_rows, :]
            for epoch in epochs
        ),
        axis=None,
    )
    upper_values = np.concatenate(
        tuple(
            (epoch.mean_values + epoch.sem_values)[selected_roi_rows, :]
            for epoch in epochs
        ),
        axis=None,
    )
    finite_lower = lower_values[np.isfinite(lower_values)]
    finite_upper = upper_values[np.isfinite(upper_values)]
    if finite_lower.size == 0 or finite_upper.size == 0:
        return DEFAULT_VALUE_BOUNDS
    value_min = float(np.min(finite_lower))
    value_max = float(np.max(finite_upper))
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
    return roi_colors_from_label_values(
        layer,
        tuple(range(1, count + 1)),
        fallback_count=count,
    )


def roi_colors_from_label_values(
    layer: object | None,
    label_values: tuple[int, ...],
    *,
    fallback_count: int,
) -> tuple[QColor, ...]:
    """Return ROI colors for explicit napari Labels-layer values.

    Args:
        layer: Current napari Labels layer.
        label_values: Integer label values to sample.
        fallback_count: Number of fallback colors when no layer is available.

    Returns:
        Tuple of Qt colors.
    """
    if layer is not None and hasattr(layer, "get_color"):
        labels_layer = cast(_LabelsColorLayer, layer)
        return tuple(
            _qcolor_from_rgba(labels_layer.get_color(label_value))
            for label_value in label_values
        )
    base = (
        QColor("#4cc9f0"),
        QColor("#f72585"),
        QColor("#f8961e"),
        QColor("#90be6d"),
        QColor("#b5179e"),
        QColor("#43aa8b"),
    )
    return tuple(base[index % len(base)] for index in range(fallback_count))


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


def resolved_time_bounds(
    data: ResponsePlotData,
    epoch_indices: tuple[int, ...],
    *,
    manual_min: float | None,
    manual_max: float | None,
) -> tuple[float, float]:
    """Return plot x-axis bounds after applying optional manual overrides.

    Args:
        data: Full response plot data.
        epoch_indices: Visible epoch row indices.
        manual_min: Optional user-entered minimum.
        manual_max: Optional user-entered maximum.

    Returns:
        Ordered nonzero x-axis bounds.
    """
    auto_min, auto_max = global_time_bounds(data, epoch_indices)
    time_min = manual_min if manual_min is not None else auto_min
    time_max = manual_max if manual_max is not None else auto_max
    return ordered_bounds(time_min, time_max)


def resolved_value_bounds(
    data: ResponsePlotData,
    roi_indices: tuple[int, ...],
    epoch_indices: tuple[int, ...],
    *,
    manual_min: float | None,
    manual_max: float | None,
) -> tuple[float, float]:
    """Return plot y-axis bounds after applying optional manual overrides.

    Args:
        data: Full response plot data.
        roi_indices: Visible ROI indices.
        epoch_indices: Visible epoch row indices.
        manual_min: Optional user-entered minimum.
        manual_max: Optional user-entered maximum.

    Returns:
        Ordered nonzero y-axis bounds.
    """
    auto_min, auto_max = global_value_bounds(data, roi_indices, epoch_indices)
    value_min = manual_min if manual_min is not None else auto_min
    value_max = manual_max if manual_max is not None else auto_max
    return ordered_bounds(value_min, value_max)


def opaque_colors(colors: tuple[QColor, ...]) -> tuple[QColor, ...]:
    """Return ROI colors with full alpha for plotting and visibility restore.

    Args:
        colors: Colors read from the Labels layer.

    Returns:
        Same RGB colors with alpha set to fully opaque.
    """
    opaque: list[QColor] = []
    for color in colors:
        updated = QColor(color)
        updated.setAlpha(255)
        opaque.append(updated)
    return tuple(opaque)


def _selected_epochs(
    data: ResponsePlotData,
    epoch_indices: tuple[int, ...] | None,
) -> tuple[EpochResponsePlotData, ...]:
    """Return epochs selected for plotting.

    Args:
        data: Full response plot data.
        epoch_indices: Optional selected epoch row indices.

    Returns:
        Tuple of selected epochs.
    """
    if epoch_indices is None:
        return data.epochs
    return tuple(
        data.epochs[index] for index in epoch_indices if 0 <= index < len(data.epochs)
    )


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
        painter.drawText(QPointF(point.x() - 10.0, point.y() + 18.0), f"{tick:.1f}")
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
        painter.drawText(
            QRectF(point.x() - 46.0, point.y() - 10.0, 38.0, 20.0),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            f"{tick:.1f}",
        )

    painter.drawText(
        QPointF(rect.center().x() - 26.0, rect.bottom() + 36.0),
        "Time (s)",
    )
    painter.save()
    painter.translate(12.0, rect.center().y() + 18.0)
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
    """Return three tick values including axis endpoints.

    Args:
        min_value: Axis minimum.
        max_value: Axis maximum.

    Returns:
        Tick values with endpoints included.
    """
    if min_value == max_value:
        return (min_value,)
    return tuple(float(value) for value in np.linspace(min_value, max_value, 3))


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


def _draw_epoch_time_spans(
    *,
    painter: QPainter,
    rect: QRectF,
    spans: tuple[tuple[float, float], ...],
    time_min: float,
    time_max: float,
) -> None:
    """Draw a quiet top rail showing coarse epoch spans.

    Args:
        painter: Active Qt painter.
        rect: Plot rectangle.
        spans: Epoch-relative intervals to mark.
        time_min: X-axis minimum.
        time_max: X-axis maximum.

    Returns:
        None.
    """
    if len(spans) == 0:
        return
    color = QColor("#f2c14e")
    color.setAlpha(170)
    painter.setPen(QPen(color, 3))
    y = rect.top() + _STIMULUS_MARKER_Y_OFFSET
    for start, stop in spans:
        clipped_start = max(float(start), time_min)
        clipped_stop = min(float(stop), time_max)
        if clipped_stop <= clipped_start:
            continue
        painter.drawLine(
            QPointF(_time_position(rect, clipped_start, time_min, time_max), y),
            QPointF(_time_position(rect, clipped_stop, time_min, time_max), y),
        )


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


def _time_position(
    rect: QRectF,
    time_value: float,
    time_min: float,
    time_max: float,
) -> float:
    """Map one time value into the plot rectangle's x coordinate.

    Args:
        rect: Plot rectangle.
        time_value: X value in seconds.
        time_min: Minimum time value.
        time_max: Maximum time value.

    Returns:
        Widget x coordinate.
    """
    time_span = time_max - time_min
    x_fraction = 0.0 if time_span == 0 else (time_value - time_min) / time_span
    return rect.left() + x_fraction * rect.width()
