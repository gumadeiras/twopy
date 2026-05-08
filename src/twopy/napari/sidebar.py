"""Tabbed right-sidebar widget for the twopy napari adapter.

Inputs: small Qt or magicgui widgets for loading recordings, listing loaded
recordings, and controlling response plots.
Outputs: one tabbed Qt widget that napari docks on the right side.

The right side is one dock on purpose. Napari docks do not share one native
scroll area, so stacking separate twopy docks wastes vertical space on smaller
screens. A single tabbed twopy dock keeps resizing predictable while leaving
napari's built-in layer controls alone.
"""

from qtpy.QtWidgets import QScrollArea, QTabWidget, QVBoxLayout, QWidget

__all__ = ["TWOPY_SIDEBAR_MINIMUM_WIDTH", "create_twopy_sidebar_widget"]

TWOPY_SIDEBAR_MINIMUM_WIDTH = 420


def create_twopy_sidebar_widget(
    *,
    load_widget: object,
    loaded_recordings_widget: object,
    response_options_widget: object | None,
) -> QTabWidget:
    """Create the single tabbed right-side twopy widget.

    Args:
        load_widget: Magicgui recording-loader widget.
        loaded_recordings_widget: Qt widget listing loaded recordings.
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
        ),
        "Load",
    )
    sidebar_tabs.setCurrentIndex(0)
    return sidebar_tabs


def _load_tab(
    *,
    load_widget: object,
    loaded_recordings_widget: object,
) -> QScrollArea:
    """Create the scrollable Load tab.

    Args:
        load_widget: Magicgui recording-loader widget.
        loaded_recordings_widget: Qt widget listing loaded recordings.

    Returns:
        Scroll area containing recording loading and loaded-recording controls.
    """
    content = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(12)
    layout.addWidget(_qt_widget(load_widget))
    layout.addWidget(_qt_widget(loaded_recordings_widget))
    layout.addStretch(1)
    content.setLayout(layout)

    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setWidget(content)
    return scroll_area


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
