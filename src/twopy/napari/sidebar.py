"""Scrollable right-sidebar widget for the twopy napari adapter.

Inputs: small Qt or magicgui widgets for loading recordings, listing loaded
recordings, and controlling response plots.
Outputs: one scrollable Qt widget that napari docks on the right side.

The right side is one dock on purpose. Napari docks do not share one native
scroll area, so stacking separate twopy docks wastes vertical space on smaller
screens. A single scrollable twopy dock keeps resizing predictable while leaving
napari's built-in layer controls alone.
"""

from qtpy.QtWidgets import QScrollArea, QVBoxLayout, QWidget

__all__ = ["create_twopy_sidebar_widget"]


def create_twopy_sidebar_widget(
    *,
    load_widget: object,
    loaded_recordings_widget: object,
    response_options_widget: object | None,
) -> QScrollArea:
    """Create the single scrollable right-side twopy widget.

    Args:
        load_widget: Magicgui recording-loader widget.
        loaded_recordings_widget: Qt widget listing loaded recordings.
        response_options_widget: Optional Qt tab widget for response controls.

    Returns:
        Scroll area containing the supplied widgets in workflow order.

    The widgets are supplied by their owners. This helper only controls visual
    grouping, so loading, session state, and plotting logic remain scoped to
    their existing modules.
    """
    content = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(12)
    layout.addWidget(_qt_widget(load_widget))
    layout.addWidget(_qt_widget(loaded_recordings_widget))
    if response_options_widget is not None:
        layout.addWidget(_qt_widget(response_options_widget))
    layout.addStretch(1)
    content.setLayout(layout)

    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setWidget(content)
    return scroll_area


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
