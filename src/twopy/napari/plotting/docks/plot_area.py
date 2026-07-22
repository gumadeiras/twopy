"""Plot-strip rendering and cache management for response plot docks.

Inputs: plot-ready epoch response data, visibility selections, and shared axis
bounds.
Outputs: Qt widgets laid out as one response plot panel per visible epoch.

This module owns the mutable Qt plot cache so the dock widget can coordinate
analysis state without also managing layout internals.
"""

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QWidget

from twopy.analysis.response_plotting import EpochResponsePlotData, ResponsePlotData
from twopy.napari.plotting.panels import epoch_plot_panel
from twopy.napari.plotting.widgets import EpochPlotWidget, clear_layout


class ResponsePlotArea:
    """Manage the scrollable response-plot strip and cached epoch widgets.

    Args:
        initial_status: Status message shown before any plot data is loaded.

    Returns:
        A plot area with a scroll widget plus methods for status, cache, and
        epoch panel rendering.
    """

    def __init__(self, initial_status: str) -> None:
        """Create a scrollable plot area with one initial status label."""
        self.epoch_plot_widgets: dict[int, EpochPlotWidget] = {}
        self.epoch_plot_panels: dict[int, QWidget] = {}
        self._epoch_keys: dict[int, tuple[int, str]] = {}
        self._render_cache_queue: list[int] = []
        self._render_cache_timer = QTimer()
        self._render_cache_timer.setInterval(10)
        self._render_cache_timer.timeout.connect(self._prepare_next_render_cache)
        self._status_label = QLabel(initial_status)
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout = QHBoxLayout()
        self._layout.addWidget(self._status_label, 1)

        plot_content = QWidget()
        plot_content.setLayout(self._layout)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(plot_content)

    def set_status(self, text: str) -> None:
        """Replace plots with one status message.

        Args:
            text: User-facing status message.

        Returns:
            None.
        """
        self.clear_epoch_cache()
        clear_layout(self._layout)
        self._status_label = QLabel(text)
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._status_label, 1)

    def clear(self) -> None:
        """Delete cached plots and clear all widgets from the plot strip."""
        self.clear_epoch_cache()
        clear_layout(self._layout)

    def render(
        self,
        *,
        plot_data: ResponsePlotData,
        roi_indices: tuple[int, ...],
        epoch_indices: tuple[int, ...],
        show_sem: bool,
        roi_colors: tuple[QColor, ...],
        time_min: float,
        time_max: float,
        value_min: float,
        value_max: float,
        plot_size: int,
    ) -> None:
        """Render visible epoch panels and update their plots.

        Args:
            plot_data: Plot-ready response data.
            roi_indices: ROI rows to draw.
            epoch_indices: Epoch rows to display.
            show_sem: Whether SEM bands should be drawn.
            roi_colors: ROI colors in plot ROI order.
            time_min: Shared x-axis minimum.
            time_max: Shared x-axis maximum.
            value_min: Shared y-axis minimum.
            value_max: Shared y-axis maximum.
            plot_size: Pixel width used for each response plot.

        Returns:
            None.
        """
        self.clear_layout_preserving_epoch_cache()
        self.ensure_epoch_plot_cache(
            plot_data=plot_data,
            show_sem=show_sem,
            roi_indices=roi_indices,
            roi_colors=roi_colors,
            plot_size=plot_size,
        )
        self.render_cached_epoch_widgets(
            roi_indices=roi_indices,
            epoch_indices=epoch_indices,
            show_sem=show_sem,
            roi_colors=roi_colors,
            time_min=time_min,
            time_max=time_max,
            value_min=value_min,
            value_max=value_max,
            plot_size=plot_size,
        )

    def render_cached_epoch_widgets(
        self,
        *,
        roi_indices: tuple[int, ...],
        epoch_indices: tuple[int, ...],
        show_sem: bool,
        roi_colors: tuple[QColor, ...],
        time_min: float,
        time_max: float,
        value_min: float,
        value_max: float,
        plot_size: int,
    ) -> None:
        """Show selected cached epoch panels and repaint them.

        Args:
            roi_indices: ROI rows to draw.
            epoch_indices: Epoch plot rows to display.
            show_sem: Whether SEM bands should be drawn.
            roi_colors: ROI plot colors.
            time_min: Shared x-axis minimum.
            time_max: Shared x-axis maximum.
            value_min: Shared y-axis minimum.
            value_max: Shared y-axis maximum.
            plot_size: Pixel width used for each response plot.

        Returns:
            None.

        Epoch visibility changes only alter which already-computed panels are
        visible. Keeping this path separate avoids refreshing every cached plot
        row just to hide or show one epoch.
        """
        self.clear_layout_preserving_epoch_cache()
        for epoch_index in epoch_indices:
            self._layout.addWidget(self.epoch_plot_panels[epoch_index])
            self.epoch_plot_panels[epoch_index].show()
        self._layout.addStretch(1)
        self.update_epoch_plot_widgets(
            roi_indices=roi_indices,
            epoch_indices=epoch_indices,
            show_sem=show_sem,
            roi_colors=roi_colors,
            time_min=time_min,
            time_max=time_max,
            value_min=value_min,
            value_max=value_max,
            plot_size=plot_size,
        )

    def update_epoch_plot_widgets(
        self,
        *,
        roi_indices: tuple[int, ...],
        epoch_indices: tuple[int, ...],
        show_sem: bool,
        roi_colors: tuple[QColor, ...],
        time_min: float,
        time_max: float,
        value_min: float,
        value_max: float,
        plot_size: int,
    ) -> None:
        """Update cached epoch widgets from already-computed plot data.

        Args:
            roi_indices: ROI rows to draw.
            epoch_indices: Epoch plot rows to update.
            show_sem: Whether SEM bands should be drawn.
            roi_colors: ROI plot colors.
            time_min: Shared x-axis minimum.
            time_max: Shared x-axis maximum.
            value_min: Shared y-axis minimum.
            value_max: Shared y-axis maximum.
            plot_size: Pixel width used for each response plot.

        Returns:
            None.
        """
        for epoch_index in epoch_indices:
            plot = self.epoch_plot_widgets[epoch_index]
            plot.update_display(
                show_sem=show_sem,
                roi_indices=roi_indices,
                roi_colors=roi_colors,
                time_min=time_min,
                time_max=time_max,
                value_min=value_min,
                value_max=value_max,
                plot_size=plot_size,
            )
        self.schedule_epoch_render_cache(epoch_indices)

    def has_epoch_widgets(self, epoch_indices: tuple[int, ...]) -> bool:
        """Return whether all requested epoch plots already exist.

        Args:
            epoch_indices: Epoch plot rows expected in the cache.

        Returns:
            True when every requested row has a cached plot widget.
        """
        return all(index in self.epoch_plot_widgets for index in epoch_indices)

    def ensure_epoch_plot_cache(
        self,
        *,
        plot_data: ResponsePlotData | None,
        show_sem: bool,
        roi_indices: tuple[int, ...],
        roi_colors: tuple[QColor, ...],
        plot_size: int,
    ) -> None:
        """Create one persistent plot widget per loaded epoch.

        Args:
            plot_data: Current plot data, or None before data is loaded.
            show_sem: Whether newly created plots should draw SEM bands.
            roi_indices: ROI rows used for initial plot construction.
            roi_colors: ROI colors in plot ROI order.
            plot_size: Pixel width used for each response plot.

        Returns:
            None.
        """
        if plot_data is None:
            self.clear_epoch_cache()
            return
        expected_indices = set(range(len(plot_data.epochs)))
        for epoch_index in tuple(self.epoch_plot_widgets):
            expected_key = (
                _epoch_key(plot_data.epochs[epoch_index])
                if epoch_index in expected_indices
                else None
            )
            if (
                epoch_index not in expected_indices
                or self._epoch_keys.get(epoch_index) != expected_key
            ):
                self.epoch_plot_panels[epoch_index].deleteLater()
                del self.epoch_plot_panels[epoch_index]
                del self.epoch_plot_widgets[epoch_index]
                self._epoch_keys.pop(epoch_index, None)
        for epoch_index, epoch in enumerate(plot_data.epochs):
            if epoch_index in self.epoch_plot_widgets:
                self.epoch_plot_widgets[epoch_index].update_data(epoch)
                continue
            plot = EpochPlotWidget(
                epoch,
                show_sem=show_sem,
                roi_indices=roi_indices,
                roi_colors=roi_colors,
                time_min=0.0,
                time_max=1.0,
                value_min=0.0,
                value_max=1.0,
                plot_size=plot_size,
            )
            self.epoch_plot_widgets[epoch_index] = plot
            self._epoch_keys[epoch_index] = _epoch_key(epoch)
            panel = epoch_plot_panel(
                title=f"Epoch {epoch.epoch_number}: {epoch.epoch_name}",
                plot=plot,
            )
            panel.hide()
            self.epoch_plot_panels[epoch_index] = panel

    def clear_epoch_cache(self) -> None:
        """Delete cached epoch plot panels and widgets."""
        self._render_cache_timer.stop()
        self._render_cache_queue.clear()
        for panel in self.epoch_plot_panels.values():
            panel.deleteLater()
        self.epoch_plot_widgets.clear()
        self.epoch_plot_panels.clear()
        self._epoch_keys.clear()

    def schedule_epoch_render_cache(self, epoch_indices: tuple[int, ...]) -> None:
        """Prepare visible plot rasters incrementally after display updates.

        Args:
            epoch_indices: Epoch plot rows that may be exposed by scrolling.

        Returns:
            None.

        Plot rasterization can be expensive with many ROIs. Preparing one epoch
        per timer tick avoids moving that whole cost into checkbox and axis
        interactions while still warming the scroll path during idle time.
        """
        self._render_cache_queue = [
            index for index in epoch_indices if index in self.epoch_plot_widgets
        ]
        if len(self._render_cache_queue) == 0:
            self._render_cache_timer.stop()
            return
        if not self._render_cache_timer.isActive():
            self._render_cache_timer.start()

    def _prepare_next_render_cache(self) -> None:
        """Render one queued plot pixmap and yield back to Qt."""
        while self._render_cache_queue:
            epoch_index = self._render_cache_queue.pop(0)
            plot = self.epoch_plot_widgets.get(epoch_index)
            if plot is None:
                continue
            plot.prepare_render_cache()
            break
        if len(self._render_cache_queue) == 0:
            self._render_cache_timer.stop()

    def clear_layout_preserving_epoch_cache(self) -> None:
        """Clear the plot strip while keeping cached epoch panels alive."""
        cached_panels = set(self.epoch_plot_panels.values())
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is None:
                continue
            if widget in cached_panels:
                # Qt keeps a widget visible after it is removed from a layout.
                # Hidden epoch plots must be hidden explicitly so stale panels
                # cannot remain on screen after checkbox changes.
                widget.hide()
            else:
                widget.deleteLater()


def _epoch_key(epoch: EpochResponsePlotData) -> tuple[int, str]:
    """Return the stable identity of one epoch plot row."""
    return (epoch.epoch_number, epoch.epoch_name)
