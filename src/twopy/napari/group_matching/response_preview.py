"""Selected-ROI response preview helpers for Group Matching.

Inputs: selected ROI labels, loaded recordings, processing options, and cached
response data.
Outputs: plot-ready response data, legend chips, epoch defaults, and status text.

Keeping this math outside the ROI assignment widget makes the view responsible for
routing user actions while these helpers own trace combination and display data.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QPushButton, QSizePolicy, QWidget

from twopy.analysis.response_plotting import EpochResponsePlotData, ResponsePlotData
from twopy.analysis.response_processing import (
    NormalizationOptions,
    ResponseProcessingOptions,
    SmoothingOptions,
)
from twopy.analysis.responses import finite_mean_and_sem
from twopy.analysis.trials import is_baseline_epoch_name
from twopy.napari.display_paths import format_recording_minute_label
from twopy.napari.group_matching.responses import SelectedRoiResponseCache
from twopy.napari.group_matching.roi_cards import (
    roi_label_display_id,
    roi_label_value,
)
from twopy.napari.roi import roi_label_image_from_layer_for_recording
from twopy.napari.session import LoadedNapariRecording
from twopy.napari.text import counted_noun

__all__ = [
    "SelectedRoiResponse",
    "add_response_legend",
    "combined_response_plot_data",
    "mean_response_plot_data",
    "response_epoch_title",
    "response_epoch_title_size",
    "selected_response_visibility_status",
    "selected_roi_response_plot_data",
    "visible_response_epoch_indices",
]


@dataclass(frozen=True)
class SelectedRoiResponse:
    """Response data for one selected ROI in one loaded recording."""

    recording_path: Path
    roi_label: str
    plot_data: ResponsePlotData
    color: QColor


def selected_roi_response_plot_data(
    recording: LoadedNapariRecording,
    roi_label: str,
    *,
    cache: SelectedRoiResponseCache,
    normalization_options: NormalizationOptions,
    smoothing_options: SmoothingOptions,
) -> ResponsePlotData:
    """Compute response plot data for one selected ROI label."""
    label_value = roi_label_value(roi_label)
    if label_value is None or recording.roi_labels_layer is None:
        msg = "Selected ROI is missing."
        raise ValueError(msg)
    label_image = roi_label_image_from_layer_for_recording(
        recording.roi_labels_layer,
        recording.recording,
    )
    return cache.compute_selected_response(
        recording.recording,
        label_image,
        roi_label=roi_label,
        label_value=label_value,
        source_path=recording.recording.source_session_dir.expanduser(),
        response_processing_options=ResponseProcessingOptions(
            smoothing=smoothing_options,
            normalization=normalization_options,
        ),
    )


def combined_response_plot_data(
    selected_responses: tuple[SelectedRoiResponse, ...],
) -> ResponsePlotData | None:
    """Return one plot data object with selected recordings overlaid as ROIs."""
    if len(selected_responses) == 0:
        return None
    reference_epochs = selected_responses[0].plot_data.epochs
    if len(reference_epochs) == 0:
        return None
    roi_labels = tuple(
        f"{format_recording_minute_label(response.recording_path)} {response.roi_label}"
        for response in selected_responses
    )
    combined_epochs: list[EpochResponsePlotData] = []
    for reference_epoch in reference_epochs:
        mean_values, sem_values = combined_epoch_values(
            reference_epoch,
            selected_responses,
        )
        combined_epochs.append(
            EpochResponsePlotData(
                epoch_name=reference_epoch.epoch_name,
                epoch_number=reference_epoch.epoch_number,
                roi_labels=roi_labels,
                time_seconds=reference_epoch.time_seconds,
                mean_values=mean_values,
                sem_values=sem_values,
                epoch_time_spans=reference_epoch.epoch_time_spans,
            ),
        )
    return ResponsePlotData(source_path=None, epochs=tuple(combined_epochs))


def combined_epoch_values(
    reference_epoch: EpochResponsePlotData,
    selected_responses: tuple[SelectedRoiResponse, ...],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return stacked mean and SEM rows for one shared comparison epoch."""
    key = (reference_epoch.epoch_number, reference_epoch.epoch_name)
    mean_rows: list[npt.NDArray[np.float64]] = []
    sem_rows: list[npt.NDArray[np.float64]] = []
    for response in selected_responses:
        epoch = _matching_epoch(response.plot_data, key=key)
        if epoch is None:
            missing_row = np.full_like(reference_epoch.time_seconds, np.nan)
            mean_rows.append(missing_row)
            sem_rows.append(missing_row)
        elif _same_time_axis(reference_epoch.time_seconds, epoch.time_seconds):
            mean_rows.append(epoch.mean_values[0])
            sem_rows.append(epoch.sem_values[0])
        else:
            mean_rows.append(
                _interpolated_row(
                    epoch.time_seconds,
                    epoch.mean_values[0],
                    reference_epoch.time_seconds,
                ),
            )
            sem_rows.append(
                _interpolated_row(
                    epoch.time_seconds,
                    epoch.sem_values[0],
                    reference_epoch.time_seconds,
                ),
            )
    return np.vstack(mean_rows), np.vstack(sem_rows)


def mean_response_plot_data(plot_data: ResponsePlotData) -> ResponsePlotData:
    """Return one mean trace with SEM across visible recording traces."""
    epochs: list[EpochResponsePlotData] = []
    for epoch in plot_data.epochs:
        means, sems = finite_mean_and_sem(epoch.mean_values, axis=0)
        epochs.append(
            EpochResponsePlotData(
                epoch_name=epoch.epoch_name,
                epoch_number=epoch.epoch_number,
                roi_labels=("mean",),
                time_seconds=epoch.time_seconds,
                mean_values=means[np.newaxis, :],
                sem_values=sems[np.newaxis, :],
                epoch_time_spans=epoch.epoch_time_spans,
            ),
        )
    return ResponsePlotData(source_path=plot_data.source_path, epochs=tuple(epochs))


def selected_response_visibility_status(
    *,
    total_count: int,
    visible_count: int,
) -> str:
    """Return the Selected ROIs summary for the current trace visibility."""
    if visible_count == total_count:
        if total_count == 1:
            return "Showing the selected recording trace."
        selected_traces = counted_noun(total_count, "selected recording trace")
        return f"Showing all {selected_traces}."
    total_traces = counted_noun(total_count, "selected recording trace")
    hidden_traces = counted_noun(total_count - visible_count, "recording trace")
    return f"Showing {visible_count} of {total_traces}; {hidden_traces} hidden."


def _matching_epoch(
    plot_data: ResponsePlotData,
    *,
    key: tuple[int, str],
) -> EpochResponsePlotData | None:
    """Return the epoch matching one reference epoch key."""
    for epoch in plot_data.epochs:
        if (epoch.epoch_number, epoch.epoch_name) == key:
            return epoch
    return None


def _same_time_axis(
    first: npt.NDArray[np.float64],
    second: npt.NDArray[np.float64],
) -> bool:
    """Return whether two epoch traces can be overlaid directly."""
    return first.shape == second.shape and bool(np.allclose(first, second))


def _interpolated_row(
    source_time: npt.NDArray[np.float64],
    source_values: npt.NDArray[np.float64],
    target_time: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return one response row sampled onto the shared plot time axis.

    ROI matching overlays recordings that can have different frame timing. The
    plot uses the first visible recording as the reference axis and samples
    later recordings onto that axis instead of dropping mismatched traces.
    """
    finite = np.isfinite(source_time) & np.isfinite(source_values)
    if int(np.sum(finite)) < 2:
        return np.full_like(target_time, np.nan)
    sorted_indices = np.argsort(source_time[finite])
    sorted_time = source_time[finite][sorted_indices]
    sorted_values = source_values[finite][sorted_indices]
    return np.interp(
        target_time,
        sorted_time,
        sorted_values,
        left=np.nan,
        right=np.nan,
    )


def response_epoch_title(epoch: EpochResponsePlotData) -> str:
    """Return the compact title shown above one ROI comparison plot."""
    return f"{epoch.epoch_number}: {epoch.epoch_name}"


def visible_response_epoch_indices(plot_data: ResponsePlotData) -> tuple[int, ...]:
    """Return default visible ROI-comparison epochs."""
    return tuple(
        index
        for index, epoch in enumerate(plot_data.epochs)
        if not is_baseline_epoch_name(epoch.epoch_name)
    )


def response_epoch_title_size(plot_size: int) -> int:
    """Return a plot-title font size that follows the preview size."""
    return max(10, min(16, round(plot_size * 0.07)))


def add_response_legend(
    layout: QGridLayout,
    selected_responses: tuple[SelectedRoiResponse, ...],
    *,
    hidden_recordings: set[Path],
    set_visible: Callable[[Path, bool], None],
    max_width: int,
) -> None:
    """Add clickable color toggles for overlaid recording traces."""
    row_index = 0
    row_width = 0
    spacing = max(0, layout.spacing())
    available_width = max(120, max_width)
    row_layout = _new_response_legend_row(layout, row_index, spacing)
    for response in selected_responses:
        visible = response.recording_path not in hidden_recordings
        recording_label = format_recording_minute_label(response.recording_path)
        button = QPushButton(
            f"{recording_label} - ROI {roi_label_display_id(response.roi_label)}",
        )
        button.setCheckable(True)
        button.setChecked(visible)
        button.setStyleSheet(trace_button_style(response.color, visible=visible))
        button.clicked.connect(
            lambda checked, path=response.recording_path: set_visible(path, checked),
        )
        button_width = button.sizeHint().width()
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        button.setFixedWidth(button_width)
        next_width = (
            button_width
            if row_layout.count() == 0
            else row_width + spacing + button_width
        )
        if row_layout.count() > 0 and next_width > available_width:
            row_index += 1
            row_width = 0
            row_layout = _new_response_legend_row(layout, row_index, spacing)
        row_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignLeft)
        row_width = (
            button_width
            if row_layout.count() == 1
            else row_width + spacing + button_width
        )


def _new_response_legend_row(
    layout: QGridLayout,
    row_index: int,
    spacing: int,
) -> QHBoxLayout:
    """Add one packed left-aligned chip row to the legend grid."""
    row_widget = QWidget()
    row_layout = QHBoxLayout()
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(spacing)
    row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
    row_widget.setLayout(row_layout)
    layout.addWidget(row_widget, row_index, 0)
    return row_layout


def trace_button_style(color: QColor, *, visible: bool) -> str:
    """Return stylesheet text for one clickable trace visibility chip."""
    background = color.name() if visible else "#555555"
    return (
        "QPushButton {"
        f"background-color: {background};"
        "color: white;"
        "font-weight: bold;"
        "padding: 2px 6px;"
        "border-radius: 3px;"
        "border: 0;"
        "}"
    )
