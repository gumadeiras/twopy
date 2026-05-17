"""Tabbed right-sidebar widget for the twopy napari adapter.

Inputs: small Qt or magicgui widgets for loading recordings, listing loaded
recordings, and controlling response plots.
Outputs: one tabbed Qt widget that napari docks on the right side.

The right side is one dock on purpose. Napari docks do not share one native
scroll area, so stacking separate twopy docks wastes vertical space on smaller
screens. A single tabbed twopy dock keeps resizing predictable while leaving
napari's built-in layer controls alone.
"""

from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

__all__ = [
    "GroupMatchingWindowButton",
    "TWOPY_SIDEBAR_MINIMUM_WIDTH",
    "create_twopy_sidebar_widget",
]

TWOPY_SIDEBAR_MINIMUM_WIDTH = 420


def create_twopy_sidebar_widget(
    *,
    load_widget: object,
    loaded_recordings_widget: object,
    save_recording_list_button: object | None,
    group_matching_button: object | None,
    response_load_button: object | None,
    response_options_widget: object | None,
) -> QTabWidget:
    """Create the single tabbed right-side twopy widget.

    Args:
        load_widget: Magicgui recording-loader widget.
        loaded_recordings_widget: Qt widget listing loaded recordings.
        save_recording_list_button: Optional Qt button that writes the loaded
            recording list to CSV.
        group_matching_button: Optional Qt button that opens manual group ROI
            matching in a separate window.
        response_load_button: Optional response action button for the Load tab.
        response_options_widget: Optional Qt tab widget for response controls.

    Returns:
        Tab widget containing the supplied widgets in workflow order.

    The widgets are supplied by their owners. This helper only controls visual
    grouping, so loading, session state, and plotting logic remain scoped to
    their existing modules.
    """
    sidebar_tabs = _response_options_tabs(response_options_widget)
    sidebar_tabs.setMinimumWidth(TWOPY_SIDEBAR_MINIMUM_WIDTH)
    sidebar_tabs.insertTab(
        0,
        _load_tab(
            load_widget=load_widget,
            loaded_recordings_widget=loaded_recordings_widget,
            save_recording_list_button=save_recording_list_button,
            group_matching_button=group_matching_button,
            response_load_button=response_load_button,
        ),
        "Load",
    )
    sidebar_tabs.setCurrentIndex(0)
    return sidebar_tabs


def _load_tab(
    *,
    load_widget: object,
    loaded_recordings_widget: object,
    save_recording_list_button: object | None,
    group_matching_button: object | None,
    response_load_button: object | None,
) -> QScrollArea:
    """Create the scrollable Load tab.

    Args:
        load_widget: Magicgui recording-loader widget.
        loaded_recordings_widget: Qt widget listing loaded recordings.
        save_recording_list_button: Optional Qt button that writes the loaded
            recording list to CSV.
        group_matching_button: Optional Qt button opened from the Load tab.
        response_load_button: Optional response action button for the Load tab.

    Returns:
        Scroll area containing recording loading and loaded-recording controls.
    """
    content = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(12)
    layout.addWidget(_qt_widget(load_widget))
    layout.addWidget(_qt_widget(loaded_recordings_widget))
    if save_recording_list_button is not None:
        layout.addWidget(_qt_widget(save_recording_list_button))
    if response_load_button is not None:
        layout.addWidget(_qt_widget(response_load_button))
    if group_matching_button is not None:
        layout.addWidget(_qt_widget(group_matching_button))
    layout.addStretch(1)
    content.setLayout(layout)

    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setWidget(content)
    return scroll_area


class GroupMatchingWindowButton(QPushButton):
    """Button that opens the group-matching workflow in a separate window."""

    def __init__(self, group_matching_widget: QWidget) -> None:
        """Create a button that owns one reusable non-modal dialog.

        Args:
            group_matching_widget: Widget to place inside the popup window.
        """
        super().__init__("Open Group Matching")
        self._group_matching_widget = group_matching_widget
        self._dialog = QDialog()
        self._dialog.setWindowTitle("twopy Group Matching")
        layout = QVBoxLayout()
        layout.addWidget(group_matching_widget)
        self._dialog.setLayout(layout)
        self._dialog.resize(980, 720)
        self.clicked.connect(self.open_group_matching_window)

    def open_group_matching_window(self) -> None:
        """Show the group-matching dialog and refresh its loaded-recording list."""
        refresh = getattr(self._group_matching_widget, "refresh", None)
        if callable(refresh):
            refresh()
        self._dialog.show()
        _maximize_dialog_height(self._dialog)
        self._dialog.raise_()
        self._dialog.activateWindow()


def _response_options_tabs(response_options_widget: object | None) -> QTabWidget:
    """Return the response-options tab widget or an empty tab widget.

    Args:
        response_options_widget: Optional Qt tab widget for response controls.

    Returns:
        Tab widget that can receive the Load tab.

    Response options are already a tab widget. Reusing it avoids an extra nested
    tab stack and keeps the response plot widget's tab references valid.
    """
    if response_options_widget is None:
        return QTabWidget()
    if not isinstance(response_options_widget, QTabWidget):
        msg = (
            "Twopy sidebar expected response options to be a QTabWidget, got "
            f"{type(response_options_widget).__name__}."
        )
        raise TypeError(msg)
    return response_options_widget


def _qt_widget(widget: object) -> QWidget:
    """Return a Qt widget for a Qt object or magicgui widget.

    Args:
        widget: Qt widget or magicgui widget exposing a ``native`` Qt widget.

    Returns:
        QWidget that can be inserted into a Qt layout.

    Magicgui widgets wrap real Qt widgets in their ``native`` attribute. Keeping
    the unwrap here makes the sidebar layout code explicit and avoids teaching
    the recording-loader code about Qt containers.
    """
    native_widget = getattr(widget, "native", widget)
    if not isinstance(native_widget, QWidget):
        msg = f"Twopy sidebar expected a QWidget, got {type(native_widget).__name__}."
        raise TypeError(msg)
    return native_widget


def _maximize_dialog_height(dialog: QDialog) -> None:
    """Resize a popup to available screen height while preserving width.

    Args:
        dialog: Popup dialog that should open tall enough for dense workflows.
    """
    screen = dialog.screen() or QApplication.primaryScreen()
    if screen is None:
        return
    available = screen.availableGeometry()
    geometry = dialog.geometry()
    width = geometry.width()
    max_x = available.left() + max(0, available.width() - width)
    x_position = min(max(geometry.x(), available.left()), max_x)
    height = available.height()
    dialog.setMaximumHeight(height)
    dialog.setGeometry(x_position, available.top(), width, height)
