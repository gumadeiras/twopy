"""Qt controls that trigger response-plot exports.

Inputs: a callable that returns the current response export state.
Outputs: one Export tab with buttons for recording images, ROI overlays, and
response plots.

This module owns export button wiring only. It keeps the response dock focused
on live plot state while the pure export functions write the actual figures.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from twopy.analysis.response_maps import ResponseMapData
from twopy.analysis_cache import (
    AnalysisSyncResult,
    build_analysis_sync_plan,
    copy_analysis_sync_plan,
)
from twopy.converted import RecordingData
from twopy.napari.display_paths import format_output_folder
from twopy.napari.plotting.data import ResponsePlotData
from twopy.napari.plotting.export import (
    export_epoch_plots,
    export_epoch_roi_overlay_plots,
    export_recording_roi_overlay,
    export_recording_view,
    export_response_heatmaps,
    export_roi_view,
)
from twopy.napari.plotting.panels import scrolling_tab
from twopy.napari.text import counted_noun

__all__ = ["ResponseExportState", "create_response_export_tab"]


@dataclass(frozen=True)
class ResponseExportState:
    """Current napari and response-plot state needed for export actions.

    Inputs: viewer, loaded recording, ROI layer, plot data, visible selections,
    axis bounds, and output folder.
    Outputs: immutable state object consumed by export buttons.
    """

    viewer: object | None
    recording: RecordingData | None
    roi_labels_layer: object | None
    plot_data: ResponsePlotData | None
    output_dir: Path
    roi_label_values: tuple[int, ...]
    roi_colors: tuple[str, ...]
    epoch_indices: tuple[int, ...]
    roi_indices: tuple[int, ...]
    show_sem: bool
    time_bounds: tuple[float, float]
    value_bounds: tuple[float, float]
    response_map_data: ResponseMapData | None = None
    response_map_shared_limits: bool = True


def create_response_export_tab(
    get_state: Callable[[], ResponseExportState],
    *,
    save_analysis_button: QPushButton,
) -> QWidget:
    """Create the response export tab.

    Args:
        get_state: Callback returning current response and napari state.
        save_analysis_button: Button that saves ROIs and persisted analysis.

    Returns:
        Qt widget with export buttons and status text.
    """
    status_label = QLabel("Exports save beside the recording.")
    status_label.setWordWrap(True)
    layout = QVBoxLayout()
    layout.addWidget(save_analysis_button)
    for text, callback in (
        ("Save recording view", _save_recording_view),
        ("Save ROI view", _save_roi_view),
        ("Save recording ROI overlay", _save_recording_roi_overlay),
        ("Save plots", _save_plots),
        ("Save plots with ROIs", _save_plots_with_rois),
        ("Save heatmaps", _save_heatmaps),
    ):
        layout.addWidget(_button(text, get_state, status_label, callback))
    layout.addWidget(status_label)
    layout.addStretch(1)
    return scrolling_tab(layout)


def _save_recording_view(state: ResponseExportState) -> tuple[Path, ...] | str:
    """Export the current movie frame or mean-image fallback."""
    if state.recording is None:
        return "No recording loaded."
    return export_recording_view(
        viewer=state.viewer,
        recording=state.recording,
        output_dir=state.output_dir,
    )


def _save_roi_view(state: ResponseExportState) -> tuple[Path, ...] | str:
    """Export current ROI contours without a recording image."""
    if state.recording is None:
        return "No recording loaded."
    return export_roi_view(
        roi_labels_layer=state.roi_labels_layer,
        recording=state.recording,
        output_dir=state.output_dir,
        roi_label_values=state.roi_label_values,
        roi_colors=state.roi_colors,
    )


def _save_recording_roi_overlay(
    state: ResponseExportState,
) -> tuple[Path, ...] | str:
    """Export current recording image with ROI contours overlaid."""
    if state.recording is None:
        return "No recording loaded."
    return export_recording_roi_overlay(
        viewer=state.viewer,
        recording=state.recording,
        roi_labels_layer=state.roi_labels_layer,
        output_dir=state.output_dir,
        roi_label_values=state.roi_label_values,
        roi_colors=state.roi_colors,
    )


def _save_plots(state: ResponseExportState) -> tuple[Path, ...] | str:
    """Export each visible epoch response plot."""
    if state.plot_data is None:
        return "No responses available."
    return export_epoch_plots(
        plot_data=state.plot_data,
        output_dir=state.output_dir,
        epoch_indices=state.epoch_indices,
        roi_indices=state.roi_indices,
        roi_colors=state.roi_colors,
        show_sem=state.show_sem,
        time_bounds=state.time_bounds,
        value_bounds=state.value_bounds,
    )


def _save_plots_with_rois(state: ResponseExportState) -> tuple[Path, ...] | str:
    """Export two-column ROI overlay plus response plot figures."""
    if state.recording is None:
        return "No recording loaded."
    if state.plot_data is None:
        return "No responses available."
    return export_epoch_roi_overlay_plots(
        viewer=state.viewer,
        recording=state.recording,
        roi_labels_layer=state.roi_labels_layer,
        plot_data=state.plot_data,
        output_dir=state.output_dir,
        epoch_indices=state.epoch_indices,
        roi_indices=state.roi_indices,
        roi_label_values=state.roi_label_values,
        roi_colors=state.roi_colors,
        show_sem=state.show_sem,
        time_bounds=state.time_bounds,
        value_bounds=state.value_bounds,
    )


def _save_heatmaps(state: ResponseExportState) -> tuple[Path, ...] | str:
    """Export each visible epoch response heatmap."""
    if state.response_map_data is None:
        return "No heatmaps available."
    return export_response_heatmaps(
        map_data=state.response_map_data,
        output_dir=state.output_dir,
        epoch_indices=state.epoch_indices,
        shared_limits=state.response_map_shared_limits,
    )


def _button(
    text: str,
    get_state: Callable[[], ResponseExportState],
    status_label: QLabel,
    callback: Callable[[ResponseExportState], tuple[Path, ...] | str],
) -> QPushButton:
    """Create one export button.

    Args:
        text: Button label.
        get_state: Callback returning current export state.
        status_label: Label updated after the button runs.
        callback: Export action for this button.

    Returns:
        Configured Qt button.
    """
    button = QPushButton(text)

    def run_export() -> None:
        """Run the export action and show a short status."""
        state = get_state()
        result = callback(state)
        sync_result = _sync_exported_paths(result, state.recording)
        status_label.setText(_export_status(result, state.recording, sync_result))

    button.clicked.connect(run_export)
    return button


def _sync_exported_paths(
    result: tuple[Path, ...] | str,
    recording: RecordingData | None,
) -> AnalysisSyncResult | str | None:
    """Copy just-written export files from local cache to publish storage.

    Args:
        result: Export callback result.
        recording: Loaded recording that defines local and publish roots.

    Returns:
        Sync result, error text, or ``None`` when sync does not apply.
    """
    if isinstance(result, str) or recording is None:
        return None
    sync_plan = build_analysis_sync_plan(recording=recording, local_paths=result)
    if sync_plan is None:
        return None
    try:
        return copy_analysis_sync_plan(sync_plan)
    except Exception as error:
        return f"sync failed: {error}"


def _export_status(
    result: tuple[Path, ...] | str,
    recording: RecordingData | None,
    sync_result: AnalysisSyncResult | str | None = None,
) -> str:
    """Return a compact export status message.

    Args:
        result: Either an error/status string or written files.
        recording: Optional selected recording for compact folder display.
        sync_result: Optional publish sync result or sync error text.

    Returns:
        User-facing status string.
    """
    if isinstance(result, str):
        return result
    if len(result) == 0:
        return "No files exported."
    folder = format_output_folder(result[0].parent, recording)
    status = f"Saved {counted_noun(len(result), 'file')} to {folder}"
    if isinstance(sync_result, str):
        return f"{status}; {sync_result}"
    if sync_result is None:
        return status
    if len(sync_result.copied_paths) == 0:
        return f"{status}; sync already current"
    return (
        f"{status}; synced {counted_noun(len(sync_result.copied_paths), 'file')} "
        f"to {sync_result.publish_root}"
    )
