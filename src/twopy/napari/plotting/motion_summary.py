"""Inline Motion metadata widget for the twopy napari sidebar.

Inputs: converted recording alignment crop, per-frame signed movement offsets,
movement magnitudes, and motion-artifact masks.
Outputs: a compact Metadata-tab section with crop metrics and a movement plot.
"""

import numpy as np
import numpy.typing as npt
from qtpy.QtCore import QRect, Qt
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPaintEvent, QPen
from qtpy.QtWidgets import QGroupBox, QSizePolicy, QVBoxLayout, QWidget

from twopy.converted import RecordingData
from twopy.napari.plotting.panels import SidebarTextLabel
from twopy.napari.theme import active_twopy_theme_colors

__all__ = ["MotionSummaryWidget"]


class MotionSummaryWidget(QGroupBox):
    """Metadata-tab section that shows alignment movement over recording frames.

    Inputs: a loaded converted recording.
    Outputs: a compact crop/motion text summary plus a movement plot.
    """

    def __init__(self) -> None:
        """Create an empty motion summary section."""
        super().__init__("Motion")
        self._summary_label = SidebarTextLabel("No recording loaded.")
        self._plot = _MotionPlotWidget()
        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.addWidget(self._summary_label)
        layout.addWidget(self._plot)
        self.setLayout(layout)

    def set_recording(self, recording: RecordingData | None) -> None:
        """Show movement metadata for one recording.

        Args:
            recording: Loaded converted recording, or ``None`` for empty state.

        Returns:
            None.
        """
        if recording is None:
            self._summary_label.setText("No recording loaded.")
            self._plot.set_motion(None, None, None)
            return
        self._summary_label.setText(_motion_summary_text(recording))
        self._plot.set_motion(
            recording.alignment_offset_pixels,
            recording.alignment_shift_pixels,
            recording.motion_artifact_mask,
        )


class _MotionPlotWidget(QWidget):
    """Small inline plot for per-frame left-right and up-down movement."""

    def __init__(self) -> None:
        """Create an empty plot widget sized for the metadata sidebar."""
        super().__init__()
        self._values = np.empty((0, 0), dtype=np.float64)
        self._has_xy_offsets = False
        self._motion_mask = np.array([], dtype=np.bool_)
        self.setFixedHeight(120)
        self.setMinimumWidth(1)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def set_motion(
        self,
        offset_pixels: npt.NDArray[np.float64] | None,
        shift_pixels: npt.NDArray[np.float64] | None,
        motion_mask: npt.NDArray[np.bool_] | None,
    ) -> None:
        """Replace the plotted movement arrays.

        Args:
            offset_pixels: Optional signed x/y movement offsets per movie frame.
            shift_pixels: Total movement magnitude in pixels per movie frame.
            motion_mask: Boolean mask marking high-motion frames.

        Returns:
            None.
        """
        values: npt.NDArray[np.float64]
        self._has_xy_offsets = offset_pixels is not None
        if offset_pixels is not None:
            values = np.asarray(offset_pixels, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != 2:
                values = np.empty((0, 0), dtype=np.float64)
        elif shift_pixels is not None:
            values = np.ravel(np.asarray(shift_pixels, dtype=np.float64))[:, None]
        else:
            self._values = np.empty((0, 0), dtype=np.float64)
            self._motion_mask = np.array([], dtype=np.bool_)
            self.update()
            return
        self._values = (
            values
            if bool(np.all(np.isfinite(values)))
            else np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        )
        if motion_mask is None:
            self._motion_mask = np.zeros(self._values.shape[0], dtype=np.bool_)
        else:
            mask = np.ravel(np.asarray(motion_mask, dtype=np.bool_))
            self._motion_mask = (
                mask
                if mask.shape == (self._values.shape[0],)
                else np.zeros(self._values.shape[0], dtype=np.bool_)
            )
        self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """Draw the movement plot.

        Args:
            a0: Qt paint event.

        Returns:
            None.
        """
        del a0
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        theme = active_twopy_theme_colors(self.palette())
        painter.fillRect(self.rect(), QColor(theme.window))
        if self._values.size == 0:
            painter.setPen(QColor(theme.muted_text))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No motion data",
            )
            painter.end()
            return
        _draw_motion_plot(
            painter,
            self.rect(),
            self._values,
            self._motion_mask,
            has_xy_offsets=self._has_xy_offsets,
            axis_color=QColor(theme.text),
            grid_color=QColor(theme.divider),
        )
        painter.end()


def _motion_summary_text(recording: RecordingData) -> str:
    """Return compact crop and motion metrics for the Metadata tab."""
    crop = recording.alignment_valid_crop
    top = crop.axis0_start
    bottom = crop.original_shape[0] - crop.axis0_stop
    left = crop.axis1_start
    right = crop.original_shape[1] - crop.axis1_stop
    shifts = np.ravel(recording.alignment_shift_pixels)
    finite_shifts = shifts[np.isfinite(shifts)]
    bad_count = int(np.count_nonzero(recording.motion_artifact_mask))
    frame_count = int(shifts.shape[0])
    p95 = float(np.percentile(finite_shifts, 95)) if finite_shifts.size else 0.0
    max_shift = float(np.max(finite_shifts)) if finite_shifts.size else 0.0
    return (
        f"Visible crop: {crop.shape[1]} x {crop.shape[0]} px "
        f"from a {crop.original_shape[1]} x {crop.original_shape[0]} px frame\n"
        "Removed border: "
        f"left {left} px, right {right} px, top {top} px, bottom {bottom} px\n"
        f"Total movement: 95% of frames moved <= {p95:.2f} px. "
        f"Max {max_shift:.2f} px. High-motion {bad_count}/{frame_count} frames\n"
        + _plot_description(recording)
    )


def _plot_description(recording: RecordingData) -> str:
    """Return the plot legend text for available movement data."""
    if recording.alignment_offset_pixels is None:
        return (
            "Plot: total movement only. Reconvert this recording to see "
            "independent x/y pixel movement."
        )
    return "Plot: left-right movement in blue, up-down movement in yellow"


def _draw_motion_plot(
    painter: QPainter,
    widget_rect: QRect,
    values: npt.NDArray[np.float64],
    motion_mask: npt.NDArray[np.bool_],
    *,
    has_xy_offsets: bool,
    axis_color: QColor,
    grid_color: QColor,
) -> None:
    """Draw axes, movement trace, and high-motion markers."""
    if not has_xy_offsets:
        _draw_magnitude_plot(
            painter,
            widget_rect,
            values[:, 0],
            motion_mask,
            axis_color=axis_color,
            grid_color=grid_color,
        )
        return
    rect = widget_rect
    plot_left = float(rect.left()) + 42.0
    plot_top = float(rect.top()) + 10.0
    plot_width = max(1.0, float(rect.width()) - 54.0)
    plot_height = max(1.0, float(rect.height()) - 32.0)
    plot_bottom = plot_top + plot_height
    y_mid = plot_top + plot_height / 2.0
    y_abs_max = max(1.0, float(np.max(np.abs(values))))

    painter.setPen(QPen(grid_color, 1))
    painter.drawLine(
        int(plot_left),
        int(plot_bottom),
        int(plot_left + plot_width),
        int(plot_bottom),
    )
    painter.drawLine(int(plot_left), int(plot_top), int(plot_left), int(plot_bottom))
    painter.setPen(axis_color)
    painter.drawText(2, int(plot_top + 8), f"+{y_abs_max:.1f}")
    painter.drawText(2, int(y_mid + 4), "0")
    painter.drawText(2, int(plot_bottom), f"-{y_abs_max:.1f}")

    frame_count = int(values.shape[0])
    if frame_count == 1:
        x_values = np.array([plot_left], dtype=np.float64)
    else:
        x_values = np.linspace(plot_left, plot_left + plot_width, frame_count)

    _draw_offset_line(
        painter,
        x_values=x_values,
        offsets=values[:, 0],
        y_mid=y_mid,
        y_scale=plot_height / (2.0 * y_abs_max),
        color=QColor("#6ec6ff"),
    )
    _draw_offset_line(
        painter,
        x_values=x_values,
        offsets=values[:, 1],
        y_mid=y_mid,
        y_scale=plot_height / (2.0 * y_abs_max),
        color=QColor("#f2c14e"),
    )
    painter.setPen(QColor("#6ec6ff"))
    painter.drawText(int(plot_left + plot_width - 34.0), int(plot_top + 12.0), "x")
    painter.setPen(QColor("#f2c14e"))
    painter.drawText(int(plot_left + plot_width - 18.0), int(plot_top + 12.0), "y")

    if motion_mask.shape == (frame_count,) and bool(np.any(motion_mask)):
        painter.setPen(QPen(QColor("#ff6b6b"), 1))
        for x_value in x_values[motion_mask]:
            painter.drawLine(
                int(x_value),
                int(plot_top),
                int(x_value),
                int(plot_bottom),
            )


def _draw_magnitude_plot(
    painter: QPainter,
    widget_rect: QRect,
    shift_pixels: npt.NDArray[np.float64],
    motion_mask: npt.NDArray[np.bool_],
    *,
    axis_color: QColor,
    grid_color: QColor,
) -> None:
    """Draw one total movement magnitude trace."""
    rect = widget_rect
    plot_left = float(rect.left()) + 42.0
    plot_top = float(rect.top()) + 10.0
    plot_width = max(1.0, float(rect.width()) - 54.0)
    plot_height = max(1.0, float(rect.height()) - 32.0)
    plot_bottom = plot_top + plot_height
    y_max = max(1.0, float(np.max(shift_pixels)))

    painter.setPen(QPen(grid_color, 1))
    painter.drawLine(
        int(plot_left),
        int(plot_bottom),
        int(plot_left + plot_width),
        int(plot_bottom),
    )
    painter.drawLine(int(plot_left), int(plot_top), int(plot_left), int(plot_bottom))
    painter.setPen(axis_color)
    painter.drawText(2, int(plot_top + 8), f"{y_max:.1f}")
    painter.drawText(2, int(plot_bottom), "0")

    frame_count = int(shift_pixels.shape[0])
    if frame_count == 1:
        x_values = np.array([plot_left], dtype=np.float64)
    else:
        x_values = np.linspace(plot_left, plot_left + plot_width, frame_count)

    y_values = plot_bottom - shift_pixels * (plot_height / y_max)
    path = QPainterPath()
    path.moveTo(float(x_values[0]), float(y_values[0]))
    for x_value, y_value in zip(x_values[1:], y_values[1:], strict=False):
        path.lineTo(float(x_value), float(y_value))
    painter.setPen(QPen(QColor("#6ec6ff"), 1.5))
    painter.drawPath(path)
    painter.drawText(int(plot_left + plot_width - 48.0), int(plot_top + 12.0), "total")

    if motion_mask.shape == (frame_count,) and bool(np.any(motion_mask)):
        painter.setPen(QPen(QColor("#ff6b6b"), 1))
        for x_value in x_values[motion_mask]:
            painter.drawLine(
                int(x_value),
                int(plot_top),
                int(x_value),
                int(plot_bottom),
            )


def _draw_offset_line(
    painter: QPainter,
    *,
    x_values: npt.NDArray[np.float64],
    offsets: npt.NDArray[np.float64],
    y_mid: float,
    y_scale: float,
    color: QColor,
) -> None:
    """Draw one signed alignment-offset line."""
    y_values = y_mid - offsets * y_scale
    path = QPainterPath()
    path.moveTo(float(x_values[0]), float(y_values[0]))
    for x_value, y_value in zip(x_values[1:], y_values[1:], strict=False):
        path.lineTo(float(x_value), float(y_value))
    painter.setPen(QPen(color, 1.5))
    painter.drawPath(path)
