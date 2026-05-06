"""Response plotting widgets for the twopy napari adapter.

Inputs: loaded twopy analysis outputs or the current napari Labels ROI layer.
Outputs: a Qt tab widget showing one response plot per stimulus epoch and a
small options tab.

The plotting code only owns GUI state and drawing. When the user asks to update
responses from the current Labels layer, this module converts labels to a
``RoiSet`` and calls the normal analysis workflow.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPaintEvent, QPen
from qtpy.QtWidgets import (
    QCheckBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.persistence import load_analysis_outputs
from twopy.analysis.responses import (
    GroupedRoiResponses,
    RoiResponseTrial,
    group_delta_f_over_f_by_epoch,
)
from twopy.analysis.workflow import (
    DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
    analyze_recording_responses,
)
from twopy.converted import RecordingData
from twopy.napari.protocols import NapariViewer
from twopy.napari.roi import roi_label_image_from_layer
from twopy.roi import make_roi_set_from_label_image

__all__ = [
    "EpochResponsePlotData",
    "ResponsePlotData",
    "add_twopy_response_plot_widget",
    "refresh_response_plot_widget",
    "response_plot_data_from_grouped",
]


@dataclass(frozen=True)
class EpochResponsePlotData:
    """Mean and SEM traces for one stimulus epoch type.

    Inputs: grouped response trials for one epoch.
    Outputs: one mean and SEM trace per ROI on a shared time axis.
    """

    epoch_name: str
    epoch_number: int
    roi_labels: tuple[str, ...]
    time_seconds: npt.NDArray[np.float64]
    mean_values: npt.NDArray[np.float64]
    sem_values: npt.NDArray[np.float64]


@dataclass(frozen=True)
class ResponsePlotData:
    """Plot-ready ROI responses grouped by stimulus epoch.

    Inputs: grouped response analysis output.
    Outputs: epoch plot data plus the source analysis path.
    """

    source_path: Path | None
    epochs: tuple[EpochResponsePlotData, ...]


def add_twopy_response_plot_widget(
    viewer: NapariViewer,
    *,
    recording: RecordingData | None,
    roi_labels_layer: object | None = None,
    dock_name: str = "twopy responses",
    dock_area: str = "right",
) -> tuple[object, object]:
    """Add the response plotting dock widget to a napari viewer.

    Args:
        viewer: Napari viewer that should receive the dock widget.
        recording: Optional loaded converted recording.
        roi_labels_layer: Optional napari Labels layer used for analysis
            previews from current ROIs.
        dock_name: Dock widget title.
        dock_area: Napari dock area.

    Returns:
        ``(widget, dock_widget)`` created by Qt and napari.
    """
    widget = create_response_plot_widget(
        recording,
        roi_labels_layer=roi_labels_layer,
    )
    dock_widget = viewer.window.add_dock_widget(
        widget,
        name=dock_name,
        area=dock_area,
    )
    return widget, dock_widget


def create_response_plot_widget(
    recording: RecordingData | None,
    *,
    roi_labels_layer: object | None = None,
) -> object:
    """Create a response plotting widget.

    Args:
        recording: Optional loaded converted recording.
        roi_labels_layer: Optional napari Labels layer used when the user asks
            to compute responses from the current ROIs.

    Returns:
        Qt tab widget as a plain object for napari docking.
    """
    _ensure_qapplication()
    widget = _ResponsePlotTabs()
    refresh_response_plot_widget(
        widget,
        recording=recording,
        roi_labels_layer=roi_labels_layer,
    )
    return widget


def refresh_response_plot_widget(
    widget: object | None,
    *,
    recording: RecordingData | None,
    roi_labels_layer: object | None = None,
) -> None:
    """Refresh a response plotting widget after a recording is loaded.

    Args:
        widget: Optional widget returned by ``create_response_plot_widget``.
        recording: Optional loaded converted recording.
        roi_labels_layer: Optional current ROI Labels layer.

    Returns:
        None.
    """
    if not isinstance(widget, _ResponsePlotTabs):
        return
    widget.set_roi_labels_layer(roi_labels_layer)
    if recording is None:
        widget.clear_recording()
        return
    widget.load_recording(recording)


def response_plot_data_from_grouped(
    grouped: GroupedRoiResponses,
    *,
    source_path: Path | None = None,
) -> ResponsePlotData:
    """Summarize grouped responses into mean and SEM traces for plotting.

    Args:
        grouped: Grouped response object from analysis.
        source_path: Optional analysis output path used for display.

    Returns:
        Plot-ready data with one item per stimulus epoch type.
    """
    epoch_keys: list[tuple[int, str]] = []
    trials_by_epoch: dict[tuple[int, str], list[RoiResponseTrial]] = {}
    for trial in grouped.trials:
        key = (trial.epoch_number, trial.epoch_name)
        if key not in trials_by_epoch:
            epoch_keys.append(key)
            trials_by_epoch[key] = []
        trials_by_epoch[key].append(trial)

    epochs = tuple(
        _epoch_plot_data(
            epoch_number=epoch_number,
            epoch_name=epoch_name,
            trials=tuple(trials_by_epoch[(epoch_number, epoch_name)]),
            roi_labels=grouped.roi_labels,
            data_rate_hz=grouped.data_rate_hz,
        )
        for epoch_number, epoch_name in epoch_keys
    )
    return ResponsePlotData(source_path=source_path, epochs=epochs)


def _epoch_plot_data(
    *,
    epoch_number: int,
    epoch_name: str,
    trials: tuple[RoiResponseTrial, ...],
    roi_labels: tuple[str, ...],
    data_rate_hz: float,
) -> EpochResponsePlotData:
    """Build plot data for one epoch type.

    Args:
        epoch_number: Stimulus epoch number.
        epoch_name: Stimulus epoch name.
        trials: Trial responses for this epoch type.
        roi_labels: ROI labels in column order.
        data_rate_hz: Imaging frame rate in hertz.

    Returns:
        Mean and SEM traces with shape ``(rois, frames)``.
    """
    time_seconds = _shared_time_axis(trials, data_rate_hz)
    first_time = float(time_seconds[0])
    max_frames = len(time_seconds)
    trial_count = len(trials)
    roi_count = len(roi_labels)
    values = np.full((trial_count, max_frames, roi_count), np.nan, dtype=np.float64)
    for trial_index, trial in enumerate(trials):
        offsets = np.rint((trial.time_seconds - first_time) * data_rate_hz).astype(
            np.int64,
            copy=False,
        )
        values[trial_index, offsets, :] = trial.values

    mean_values = np.nanmean(values, axis=0).T
    valid_counts = np.sum(~np.isnan(values), axis=0).T
    sem_values = np.zeros((roi_count, max_frames), dtype=np.float64)
    variable_mask = valid_counts > 1
    if np.any(variable_mask):
        std_values = np.nanstd(values, axis=0, ddof=1).T
        sem_values[variable_mask] = std_values[variable_mask] / np.sqrt(
            valid_counts[variable_mask]
        )
    return EpochResponsePlotData(
        epoch_name=epoch_name,
        epoch_number=epoch_number,
        roi_labels=roi_labels,
        time_seconds=time_seconds,
        mean_values=mean_values,
        sem_values=sem_values,
    )


def _shared_time_axis(
    trials: tuple[RoiResponseTrial, ...],
    data_rate_hz: float,
) -> npt.NDArray[np.float64]:
    """Return one relative time axis that covers all trials in an epoch.

    Args:
        trials: Trial responses for one epoch type.
        data_rate_hz: Imaging frame rate in hertz.

    Returns:
        Relative time in seconds, usually starting at ``-2`` when grouped
        responses include two prestimulus seconds.
    """
    first_time = min(float(trial.time_seconds[0]) for trial in trials)
    last_time = max(float(trial.time_seconds[-1]) for trial in trials)
    frame_count = int(round((last_time - first_time) * data_rate_hz)) + 1
    return first_time + (np.arange(frame_count, dtype=np.float64) / data_rate_hz)


def _default_analysis_output_path(recording: RecordingData) -> Path:
    """Return the default analysis output path for one recording.

    Args:
        recording: Loaded converted recording.

    Returns:
        Path to ``analysis_outputs.h5`` beside ``recording_data.h5``.
    """
    return recording.path.expanduser().parent / "analysis_outputs.h5"


def _ensure_qapplication() -> None:
    """Create a Qt application when tests instantiate widgets without napari.

    Args:
        None.

    Returns:
        None.
    """
    from qtpy.QtWidgets import QApplication

    if QApplication.instance() is None:
        QApplication([])


def _load_plot_data(path: Path) -> ResponsePlotData | str:
    """Load response plot data from an analysis output file.

    Args:
        path: Candidate ``analysis_outputs.h5`` path.

    Returns:
        Plot data, or a status message for the user.
    """
    if not path.is_file():
        return f"No analysis output found: {path}"
    outputs = load_analysis_outputs(path)
    if outputs.dff is not None and len(outputs.epoch_windows) > 0:
        data_rate_hz = _analysis_output_data_rate_hz(
            grouped=outputs.grouped_responses,
            dff=outputs.dff,
        )
        if data_rate_hz is not None:
            grouped = group_delta_f_over_f_by_epoch(
                outputs.dff,
                outputs.epoch_windows,
                data_rate_hz=data_rate_hz,
                pre_window_seconds=DEFAULT_RESPONSE_PRE_WINDOW_SECONDS,
            )
            return response_plot_data_from_grouped(grouped, source_path=outputs.path)
    if outputs.grouped_responses is None:
        return f"No grouped responses in: {path}"
    return response_plot_data_from_grouped(
        outputs.grouped_responses,
        source_path=outputs.path,
    )


def _analysis_output_data_rate_hz(
    *,
    grouped: GroupedRoiResponses | None,
    dff: RoiDeltaFOverF,
) -> float | None:
    """Return the saved response frame rate when available.

    Args:
        grouped: Optional grouped responses loaded from analysis outputs.
        dff: dF/F result with audit metadata.

    Returns:
        Data rate in hertz, or ``None`` when unavailable.
    """
    if grouped is not None:
        return grouped.data_rate_hz
    value = dff.metadata.get("data_rate_hz")
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


class _ResponsePlotTabs(QTabWidget):
    """Qt tabs for response plots and plot options."""

    def __init__(self) -> None:
        """Create empty response plotting tabs."""
        super().__init__()
        self._recording: RecordingData | None = None
        self._roi_labels_layer: object | None = None
        self._analysis_path: Path | None = None
        self._plot_data: ResponsePlotData | None = None
        self._show_sem = True
        self._status_label = QLabel("No recording loaded.")
        self._plot_layout = QVBoxLayout()
        self._plot_layout.addWidget(self._status_label)
        plot_content = QWidget()
        plot_content.setLayout(self._plot_layout)
        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setWidget(plot_content)

        options = QWidget()
        options_layout = QVBoxLayout()
        self._show_sem_checkbox = QCheckBox("Show SEM")
        self._show_sem_checkbox.setChecked(True)
        self._show_sem_checkbox.stateChanged.connect(self._set_show_sem)
        refresh_button = QPushButton("Refresh responses")
        refresh_button.clicked.connect(self.reload)
        update_from_rois_button = QPushButton("Update from current ROIs")
        update_from_rois_button.clicked.connect(self.update_from_current_rois)
        self._analysis_path_label = QLabel("Analysis file: default")
        self._analysis_path_label.setWordWrap(True)
        options_layout.addWidget(self._show_sem_checkbox)
        options_layout.addWidget(refresh_button)
        options_layout.addWidget(update_from_rois_button)
        options_layout.addWidget(self._analysis_path_label)
        options_layout.addStretch(1)
        options.setLayout(options_layout)

        self.addTab(plot_scroll, "Responses")
        self.addTab(options, "Options")

    def set_roi_labels_layer(self, roi_labels_layer: object | None) -> None:
        """Store the current editable ROI Labels layer.

        Args:
            roi_labels_layer: Optional napari Labels layer.

        Returns:
            None.
        """
        self._roi_labels_layer = roi_labels_layer

    def clear_recording(self) -> None:
        """Reset the widget to the empty-launch state.

        Args:
            None.

        Returns:
            None.
        """
        self._recording = None
        self._analysis_path = None
        self._plot_data = None
        self._analysis_path_label.setText("Analysis file: default")
        self._set_status("No recording loaded.")

    def load_recording(self, recording: RecordingData) -> None:
        """Load response plots for one recording.

        Args:
            recording: Loaded converted recording.

        Returns:
            None.
        """
        self._recording = recording
        self._analysis_path = _default_analysis_output_path(recording)
        self._analysis_path_label.setText(f"Analysis file: {self._analysis_path}")
        self.reload()

    def update_from_current_rois(self) -> None:
        """Compute and plot responses from the current ROI Labels layer.

        Args:
            None.

        Returns:
            None.

        The analysis workflow streams the converted movie in chunks, so this
        action avoids loading a large full movie only to update plots.
        """
        if self._recording is None:
            self._set_status("No recording loaded.")
            return
        if self._roi_labels_layer is None:
            self._set_status("No ROI Labels layer is available.")
            return
        label_image = roi_label_image_from_layer(self._roi_labels_layer)
        if not np.any(label_image > 0):
            self._set_status("No ROI labels to analyze.")
            return

        try:
            roi_set = make_roi_set_from_label_image(label_image)
            run = analyze_recording_responses(
                self._recording.path,
                roi_set,
                output_path=self._analysis_path,
                write_summary_csv=False,
            )
        except ValueError as error:
            self._plot_data = None
            self._set_status(str(error))
            return

        self._plot_data = response_plot_data_from_grouped(
            run.grouped_responses,
            source_path=run.output_path,
        )
        self._render_plots()

    def reload(self) -> None:
        """Reload response plots from the current analysis output file."""
        if self._analysis_path is None:
            self._set_status("No recording loaded.")
            return
        result = _load_plot_data(self._analysis_path)
        if isinstance(result, str):
            self._plot_data = None
            self._set_status(result)
            return
        self._plot_data = result
        self._render_plots()

    def _set_show_sem(self, state: int) -> None:
        """Update whether SEM bands are drawn.

        Args:
            state: Qt checkbox state.

        Returns:
            None.
        """
        del state
        self._show_sem = self._show_sem_checkbox.isChecked()
        self._render_plots()

    def _set_status(self, text: str) -> None:
        """Replace plots with one status message.

        Args:
            text: User-facing status message.

        Returns:
            None.
        """
        _clear_layout(self._plot_layout)
        self._status_label = QLabel(text)
        self._status_label.setWordWrap(True)
        self._plot_layout.addWidget(self._status_label)
        self._plot_layout.addStretch(1)

    def _render_plots(self) -> None:
        """Render one plot widget per epoch type."""
        _clear_layout(self._plot_layout)
        if self._plot_data is None or len(self._plot_data.epochs) == 0:
            self._set_status("No responses available.")
            return
        value_min, value_max = _global_value_bounds(self._plot_data)
        for epoch in self._plot_data.epochs:
            title = f"Epoch {epoch.epoch_number}: {epoch.epoch_name}"
            self._plot_layout.addWidget(QLabel(title))
            self._plot_layout.addWidget(
                _EpochPlotWidget(
                    epoch,
                    show_sem=self._show_sem,
                    value_min=value_min,
                    value_max=value_max,
                ),
            )
        self._plot_layout.addStretch(1)


class _EpochPlotWidget(QWidget):
    """Small custom Qt plot for one stimulus epoch type."""

    def __init__(
        self,
        data: EpochResponsePlotData,
        *,
        show_sem: bool,
        value_min: float,
        value_max: float,
    ) -> None:
        """Create a plot widget.

        Args:
            data: Plot data for one epoch.
            show_sem: Whether SEM bands should be drawn.
            value_min: Shared y-axis minimum.
            value_max: Shared y-axis maximum.
        """
        super().__init__()
        self._data = data
        self._show_sem = show_sem
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
        plot_rect = QRectF(self.rect().adjusted(42, 12, -12, -32))
        painter.fillRect(self.rect(), QColor("#20252d"))
        painter.setPen(QPen(QColor("#8c96a3"), 1))
        painter.drawRect(plot_rect)
        colors = _roi_colors(len(self._data.roi_labels))
        for roi_index, color in enumerate(colors):
            if self._show_sem:
                _draw_sem_band(
                    painter=painter,
                    rect=plot_rect,
                    time_values=self._data.time_seconds,
                    mean_values=self._data.mean_values[roi_index],
                    sem_values=self._data.sem_values[roi_index],
                    value_min=self._value_min,
                    value_max=self._value_max,
                    color=color,
                )
            _draw_trace(
                painter=painter,
                rect=plot_rect,
                time_values=self._data.time_seconds,
                values=self._data.mean_values[roi_index],
                value_min=self._value_min,
                value_max=self._value_max,
                color=color,
            )
        painter.setPen(QPen(QColor("#cfd6df"), 1))
        painter.drawText(10, 20, f"{self._value_max:.1f}")
        painter.drawText(QPointF(10.0, plot_rect.bottom()), f"{self._value_min:.1f}")
        painter.drawText(
            QPointF(plot_rect.left(), float(self.rect().bottom() - 8)),
            f"{self._data.time_seconds[0]:.1f} s",
        )
        painter.drawText(
            QPointF(
                plot_rect.right() - 45.0,
                float(self.rect().bottom() - 8),
            ),
            f"{self._data.time_seconds[-1]:.1f} s",
        )
        painter.end()


def _clear_layout(layout: QVBoxLayout) -> None:
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


def _global_value_bounds(data: ResponsePlotData) -> tuple[float, float]:
    """Return shared y-axis bounds across all epoch plots.

    Args:
        data: Full response plot data.

    Returns:
        ``(min, max)`` bounds expanded by 20 percent and rounded to one decimal.
    """
    lower = np.concatenate(
        tuple(epoch.mean_values - epoch.sem_values for epoch in data.epochs),
        axis=None,
    )
    upper = np.concatenate(
        tuple(epoch.mean_values + epoch.sem_values for epoch in data.epochs),
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


def _roi_colors(count: int) -> tuple[QColor, ...]:
    """Return visually distinct ROI colors.

    Args:
        count: Number of ROI traces.

    Returns:
        Tuple of Qt colors.
    """
    base = (
        QColor("#4cc9f0"),
        QColor("#f72585"),
        QColor("#f8961e"),
        QColor("#90be6d"),
        QColor("#b5179e"),
        QColor("#43aa8b"),
    )
    return tuple(base[index % len(base)] for index in range(count))


def _draw_trace(
    *,
    painter: QPainter,
    rect: QRectF,
    time_values: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
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
        value_min: Y-axis minimum.
        value_max: Y-axis maximum.
        color: Trace color.

    Returns:
        None.
    """
    path = _trace_path(rect, time_values, values, value_min, value_max)
    painter.setPen(QPen(color, 2))
    painter.drawPath(path)


def _draw_sem_band(
    *,
    painter: QPainter,
    rect: QRectF,
    time_values: npt.NDArray[np.float64],
    mean_values: npt.NDArray[np.float64],
    sem_values: npt.NDArray[np.float64],
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
        value_min: Y-axis minimum.
        value_max: Y-axis maximum.
        color: Band color.

    Returns:
        None.
    """
    upper = mean_values + sem_values
    lower = mean_values - sem_values
    path = _trace_path(rect, time_values, upper, value_min, value_max)
    for index in range(len(time_values) - 1, -1, -1):
        point = _plot_point(
            rect,
            time_values[index],
            lower[index],
            time_values[0],
            time_values[-1],
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
    value_min: float,
    value_max: float,
) -> QPainterPath:
    """Build a Qt path for one trace.

    Args:
        rect: Plot rectangle.
        time_values: X-axis values in seconds.
        values: Y-axis values.
        value_min: Y-axis minimum.
        value_max: Y-axis maximum.

    Returns:
        Painter path for valid values.
    """
    path = QPainterPath()
    first = True
    time_min = float(time_values[0])
    time_max = float(time_values[-1])
    for time_value, value in zip(time_values, values, strict=True):
        if np.isnan(value):
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
