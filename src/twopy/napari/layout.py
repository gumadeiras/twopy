"""Napari dock-layout helpers for twopy widgets.

Inputs: napari viewer and dock-widget objects.
Outputs: best-effort Qt dock placement.

These helpers own visual placement only. If napari or Qt internals change, the
widgets still exist and continue to work; only the preferred layout is skipped.
"""

from twopy.napari.protocols import NapariViewer

__all__ = [
    "place_loaded_recordings_dock_after_load",
    "place_response_options_after_recording_list",
]


def place_loaded_recordings_dock_after_load(
    viewer: NapariViewer,
    load_dock_widget: object,
    loaded_recordings_dock_widget: object,
) -> None:
    """Stack Loaded Recordings under the Load Recording dock.

    Args:
        viewer: Napari viewer that owns the dock widgets.
        load_dock_widget: Dock widget returned for load controls.
        loaded_recordings_dock_widget: Dock widget returned for the loaded
            recordings panel.

    Returns:
        None.
    """
    try:
        from qtpy.QtCore import Qt
    except ImportError:
        return

    qt_window = getattr(viewer.window, "_qt_window", None)
    if qt_window is None:
        return
    if not hasattr(qt_window, "splitDockWidget") or not hasattr(
        qt_window,
        "resizeDocks",
    ):
        return
    qt_window.splitDockWidget(
        load_dock_widget,
        loaded_recordings_dock_widget,
        Qt.Orientation.Vertical,
    )
    qt_window.resizeDocks(
        [load_dock_widget, loaded_recordings_dock_widget],
        [260, 180],
        Qt.Orientation.Vertical,
    )


def place_response_options_after_recording_list(
    viewer: NapariViewer,
    loaded_recordings_dock_widget: object | None,
    response_options_dock_widget: object | None,
) -> None:
    """Stack response options under the Loaded Recordings dock.

    Args:
        viewer: Napari viewer that owns the dock widgets.
        loaded_recordings_dock_widget: Dock widget returned for the loaded
            recordings panel.
        response_options_dock_widget: Dock widget returned for response
            options.

    Returns:
        None.
    """
    if loaded_recordings_dock_widget is None or response_options_dock_widget is None:
        return
    try:
        from qtpy.QtCore import Qt
    except ImportError:
        return

    qt_window = getattr(viewer.window, "_qt_window", None)
    if qt_window is None:
        return
    if not hasattr(qt_window, "splitDockWidget") or not hasattr(
        qt_window,
        "resizeDocks",
    ):
        return
    qt_window.splitDockWidget(
        loaded_recordings_dock_widget,
        response_options_dock_widget,
        Qt.Orientation.Vertical,
    )
    qt_window.resizeDocks(
        [loaded_recordings_dock_widget, response_options_dock_widget],
        [180, 10000],
        Qt.Orientation.Vertical,
    )
