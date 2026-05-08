"""Dock entry points for response plotting in the twopy napari adapter.

Inputs: a napari viewer, optional converted recording, and optional ROI layer
state.
Outputs: response plotting dock widgets and the response-options widget.

This module keeps napari-facing factory functions separate from the larger Qt
widget implementation so import callers see a small, stable API.
"""

from pathlib import Path

from qtpy.QtWidgets import QApplication

from twopy.converted import RecordingData
from twopy.napari.plotting.docks.response_plot_widget import _ResponsePlotWidget
from twopy.napari.protocols import NapariViewer

_QT_APPLICATION: object | None = None
DEFAULT_RESPONSE_DOCK_HEIGHT = 260


def add_twopy_response_plot_widget(
    viewer: NapariViewer,
    *,
    recording: RecordingData | None,
    roi_labels_layer: object | None = None,
    roi_save_file: Path | None = None,
    dock_name: str = "twopy responses",
    dock_area: str = "top",
    initial_height: int = DEFAULT_RESPONSE_DOCK_HEIGHT,
) -> tuple[object, object]:
    """Add the response plotting dock widget to a napari viewer.

    Args:
        viewer: Napari viewer that should receive the dock widget.
        recording: Optional loaded converted recording.
        roi_labels_layer: Optional napari Labels layer used for analysis
            previews from current ROIs.
        roi_save_file: Optional ROI HDF5 path used by the Save Analysis action.
        dock_name: Dock widget title.
        dock_area: Napari dock area.
        initial_height: Preferred height, in screen pixels, for the top response
            dock when it first opens.

    Returns:
        ``(widget, dock_widget)`` created by Qt and napari.
    """
    widget = create_response_plot_widget(
        recording,
        viewer=viewer,
        roi_labels_layer=roi_labels_layer,
        roi_save_file=roi_save_file,
    )
    dock_widget = viewer.window.add_dock_widget(
        widget,
        name=dock_name,
        area=dock_area,
    )
    set_top_dock_initial_height(viewer, dock_widget, height=initial_height)
    return widget, dock_widget


def set_top_dock_initial_height(
    viewer: NapariViewer,
    dock_widget: object,
    *,
    height: int,
) -> None:
    """Set a best-effort opening height for a top napari dock.

    Args:
        viewer: Napari viewer that owns the dock.
        dock_widget: Dock widget returned by napari.
        height: Preferred dock height in screen pixels.

    Returns:
        None.

    Napari exposes dock creation but not a stable high-level opening-size API.
    Qt's ``resizeDocks`` is the direct layout operation for this job. The helper
    quietly skips non-Qt test doubles or future napari versions that do not
    expose the underlying Qt window.
    """
    if height <= 0:
        return
    try:
        from qtpy.QtCore import Qt
    except ImportError:
        return

    qt_window = getattr(viewer.window, "_qt_window", None)
    if qt_window is None or not hasattr(qt_window, "resizeDocks"):
        return
    qt_window.resizeDocks(
        [dock_widget],
        [height],
        Qt.Orientation.Vertical,
    )


def create_twopy_response_options_widget(
    response_plot_widget: object,
) -> object | None:
    """Return the response plot options widget.

    Args:
        response_plot_widget: Widget returned by ``create_response_plot_widget``.

    Returns:
        Tabbed response-options widget when the plot widget owns options,
        otherwise ``None``.

    The options widget is not docked here. The app prepends the Load tab to it
    and docks that single twopy tab stack on the right side.
    """
    if not isinstance(response_plot_widget, _ResponsePlotWidget):
        return None
    return response_plot_widget.options_widget()


def create_response_plot_widget(
    recording: RecordingData | None,
    *,
    viewer: object | None = None,
    roi_labels_layer: object | None = None,
    roi_save_file: Path | None = None,
) -> object:
    """Create a response plotting widget.

    Args:
        recording: Optional loaded converted recording.
        viewer: Optional napari viewer used for figure export.
        roi_labels_layer: Optional napari Labels layer used when the user asks
            to compute responses from the current ROIs.
        roi_save_file: Optional ROI HDF5 path used by the Save Analysis action.

    Returns:
        Qt widget as a plain object for napari docking.
    """
    _ensure_qapplication()
    widget = _ResponsePlotWidget(viewer=viewer)
    refresh_response_plot_widget(
        widget,
        recording=recording,
        roi_labels_layer=roi_labels_layer,
        roi_save_file=roi_save_file,
    )
    return widget


def refresh_response_plot_widget(
    widget: object | None,
    *,
    recording: RecordingData | None,
    roi_labels_layer: object | None = None,
    roi_save_file: Path | None = None,
) -> None:
    """Refresh a response plotting widget after a recording is loaded.

    Args:
        widget: Optional widget returned by ``create_response_plot_widget``.
        recording: Optional loaded converted recording.
        roi_labels_layer: Optional current ROI Labels layer.
        roi_save_file: Optional ROI HDF5 path used by the Save Analysis action.

    Returns:
        None.
    """
    if not isinstance(widget, _ResponsePlotWidget):
        return
    widget.set_roi_labels_layer(roi_labels_layer)
    widget.set_roi_save_file(roi_save_file)
    if recording is None:
        widget.clear_recording()
        return
    widget.load_recording(recording)


def _ensure_qapplication() -> None:
    """Create a Qt application when tests instantiate widgets without napari."""
    global _QT_APPLICATION
    if QApplication.instance() is None:
        _QT_APPLICATION = QApplication([])
