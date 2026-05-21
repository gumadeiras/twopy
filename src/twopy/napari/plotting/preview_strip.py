"""Reusable cached strip of epoch response preview plots.

Inputs: plot-ready response data, visible epoch rows, colors, and plot size.
Outputs: one Qt widget containing cached epoch plot panels.

Small workflow views such as Group Matching need the same update behavior as
the main response dock without its scroll-area wrapper. This module owns that
lighter plot strip.
"""

from collections.abc import Callable

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from twopy.napari.plotting.data import EpochResponsePlotData, ResponsePlotData
from twopy.napari.plotting.widgets import (
    EpochPlotWidget,
    global_time_bounds,
    global_value_bounds,
)

__all__ = ["ResponsePreviewStrip"]


class ResponsePreviewStrip:
    """Cache epoch plot widgets inside one horizontal preview strip."""

    def __init__(
        self,
        object_name: str,
        *,
        trace_alpha: int = 255,
        title_pixel_size: Callable[[int], int] | None = None,
    ) -> None:
        """Create an empty preview strip widget."""
        self.widget = QWidget()
        self.widget.setObjectName(object_name)
        self._layout = QHBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.widget.setLayout(self._layout)
        self._trace_alpha = trace_alpha
        self._title_pixel_size = title_pixel_size
        self._epoch_plot_widgets: dict[int, EpochPlotWidget] = {}
        self._epoch_plot_panels: dict[int, QWidget] = {}
        self._epoch_title_labels: dict[int, QLabel] = {}
        self._layout_epoch_indices: tuple[int, ...] = ()

    def clear(self) -> None:
        """Hide cached plots and remove them from the visible strip."""
        _clear_layout_preserving_widgets(self._layout)
        self._layout_epoch_indices = ()
        for panel in self._epoch_plot_panels.values():
            panel.hide()

    def visible_height_hint(self) -> int:
        """Return the height needed to show the visible epoch panels."""
        heights: list[int] = []
        for epoch_index in self._layout_epoch_indices:
            panel = self._epoch_plot_panels[epoch_index]
            panel_layout = panel.layout()
            if panel_layout is not None:
                panel_layout.invalidate()
                panel_layout.activate()
                heights.append(panel_layout.sizeHint().height())
            else:
                heights.append(panel.sizeHint().height())
        return max(heights, default=self.widget.sizeHint().height())

    def visible_width_hint(self) -> int:
        """Return the width needed to show the visible epoch panels."""
        widths: list[int] = []
        spacing = self._layout.spacing()
        for epoch_index in self._layout_epoch_indices:
            panel = self._epoch_plot_panels[epoch_index]
            widths.append(panel.sizeHint().width())
        if len(widths) == 0:
            return self.widget.sizeHint().width()
        return sum(widths) + spacing * max(0, len(widths) - 1)

    def render(
        self,
        plot_data: ResponsePlotData,
        *,
        epoch_indices: tuple[int, ...],
        roi_colors: tuple[QColor, ...],
        plot_size: int,
    ) -> None:
        """Render visible epochs while reusing cached plot widgets."""
        if len(epoch_indices) == 0:
            self.clear()
            return
        removed_stale_panels = self._ensure_cache(
            plot_data,
            roi_colors=roi_colors,
            plot_size=plot_size,
        )
        time_min, time_max = global_time_bounds(plot_data, epoch_indices)
        roi_indices = tuple(range(len(roi_colors)))
        value_min, value_max = global_value_bounds(
            plot_data,
            roi_indices,
            epoch_indices,
        )
        plot_colors = _colors_with_opacity(roi_colors, alpha=self._trace_alpha)

        visible = set(epoch_indices)
        for epoch_index, panel in self._epoch_plot_panels.items():
            panel.setVisible(epoch_index in visible)
        for epoch_index in epoch_indices:
            plot = self._epoch_plot_widgets[epoch_index]
            plot.update_display(
                show_sem=True,
                roi_indices=roi_indices,
                roi_colors=plot_colors,
                time_min=time_min,
                time_max=time_max,
                value_min=value_min,
                value_max=value_max,
                plot_size=plot_size,
            )
        if removed_stale_panels or epoch_indices != self._layout_epoch_indices:
            _clear_layout_preserving_widgets(self._layout)
            for epoch_index in epoch_indices:
                self._layout.addWidget(self._epoch_plot_panels[epoch_index])
            self._layout.addStretch(1)
            self._layout_epoch_indices = epoch_indices

    def _ensure_cache(
        self,
        plot_data: ResponsePlotData,
        *,
        roi_colors: tuple[QColor, ...],
        plot_size: int,
    ) -> bool:
        """Create or update cached epoch panels for the current data."""
        expected_indices = set(range(len(plot_data.epochs)))
        removed_stale_panels = False
        for epoch_index in tuple(self._epoch_plot_widgets):
            if epoch_index not in expected_indices:
                self._epoch_plot_panels[epoch_index].hide()
                self._epoch_plot_panels[epoch_index].deleteLater()
                del self._epoch_plot_panels[epoch_index]
                del self._epoch_plot_widgets[epoch_index]
                del self._epoch_title_labels[epoch_index]
                removed_stale_panels = True

        plot_colors = _colors_with_opacity(roi_colors, alpha=self._trace_alpha)
        roi_indices = tuple(range(len(roi_colors)))
        for epoch_index, epoch in enumerate(plot_data.epochs):
            if epoch_index in self._epoch_plot_widgets:
                self._epoch_plot_widgets[epoch_index].update_data(epoch)
                self._update_title(epoch_index, _epoch_title(epoch), plot_size)
                continue
            panel = QWidget(self.widget)
            label = QLabel(_epoch_title(epoch), panel)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._epoch_title_labels[epoch_index] = label
            self._update_title(epoch_index, _epoch_title(epoch), plot_size)
            plot = EpochPlotWidget(
                epoch,
                show_sem=True,
                roi_indices=roi_indices,
                roi_colors=plot_colors,
                time_min=0.0,
                time_max=1.0,
                value_min=-1.0,
                value_max=1.0,
                plot_size=plot_size,
            )
            plot.setParent(panel)
            panel_layout = QVBoxLayout()
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.addWidget(label)
            panel_layout.addWidget(plot)
            panel.setLayout(panel_layout)
            self._epoch_plot_widgets[epoch_index] = plot
            self._epoch_plot_panels[epoch_index] = panel
        return removed_stale_panels

    def _update_title(self, epoch_index: int, text: str, plot_size: int) -> None:
        """Update cached title text and size."""
        label = self._epoch_title_labels[epoch_index]
        label.setText(text)
        font = label.font()
        if self._title_pixel_size is not None:
            font.setPixelSize(self._title_pixel_size(plot_size))
        font.setBold(False)
        label.setFont(font)
        label.setStyleSheet("font-weight: normal;")


def _clear_layout_preserving_widgets(layout: QHBoxLayout) -> None:
    """Remove layout items without deleting cached child widgets."""
    while layout.count() > 0:
        layout.takeAt(0)


def _colors_with_opacity(
    colors: tuple[QColor, ...],
    *,
    alpha: int,
) -> tuple[QColor, ...]:
    """Return plot colors with one shared alpha value."""
    faded: list[QColor] = []
    for color in colors:
        trace_color = QColor(color)
        trace_color.setAlpha(alpha)
        faded.append(trace_color)
    return tuple(faded)


def _epoch_title(epoch: EpochResponsePlotData) -> str:
    """Return the compact title shown above one epoch plot."""
    epoch_number = epoch.epoch_number
    epoch_name = epoch.epoch_name
    return f"{epoch_number}: {epoch_name}"
