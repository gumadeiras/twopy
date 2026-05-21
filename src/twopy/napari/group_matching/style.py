"""Shared visual styling helpers for the napari group-matching popup.

Inputs: Qt widgets used by the FOV and ROI matching views.
Outputs: consistent buttons, section boxes, and panel styling.

The helpers keep display choices close to the napari UI layer. They do not
change FOV or ROI matching data, and they do not read or write CSV decisions.
"""

from string import Template

from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QGroupBox, QLayout, QPushButton, QWidget

__all__ = [
    "group_matching_button",
    "group_matching_section",
    "style_group_matching_panel",
]

_STYLE_TEMPLATE = Template(
    """
QWidget#group_matching_panel {
    background: $window;
    color: $text;
}
QLabel#group_matching_title {
    color: $text;
    font-size: 18px;
    font-weight: 700;
}
QLabel#group_matching_subtitle {
    color: $muted_text;
}
QLabel#group_matching_caption {
    color: $muted_text;
    font-size: 11px;
}
QLabel#fov_id_label {
    color: $text;
    font-size: 15px;
    font-weight: 700;
}
QLabel#fov_id_value {
    background: $input_base;
    color: $text;
    border: 1px solid $border;
    border-radius: 4px;
    font-size: 15px;
    font-weight: 700;
    padding: 3px 8px;
    min-width: 28px;
}
QGroupBox#group_matching_section {
    background: $base;
    border: 1px solid $border;
    border-radius: 6px;
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
}
QGroupBox#group_matching_section::title {
    color: $text;
    font-weight: 700;
    left: 10px;
    background: $title_pill;
    border: 1px solid $border;
    border-radius: 4px;
    padding: 2px 8px;
    subcontrol-origin: margin;
}
QFrame#fov_recording_card,
QFrame#roi_assignment_card {
    background: $base;
    border: 1px solid $border;
    border-radius: 6px;
}
QFrame#fov_recording_card[selected="true"] {
    border: 2px solid $highlight;
    background: $selected_base;
}
QLabel#fov_card_overlay {
    background: rgba(0, 0, 0, 110);
    color: white;
    font-weight: 700;
    padding: 4px 6px;
}
QPushButton#group_matching_primary_button {
    background: $highlight;
    color: $highlighted_text;
    border: 1px solid $highlight_border;
    border-radius: 4px;
    font-weight: 700;
    padding: 6px 12px;
}
QPushButton#group_matching_primary_button:hover {
    background: $highlight_hover;
}
QPushButton#group_matching_secondary_button {
    background: $button;
    color: $button_text;
    border: 1px solid $border;
    border-radius: 4px;
    padding: 5px 10px;
}
QPushButton#group_matching_secondary_button:hover,
QPushButton#group_matching_quiet_button:hover {
    background: $button_hover;
}
QPushButton#group_matching_quiet_button {
    background: $button;
    color: $button_text;
    border: 1px solid $border;
    border-radius: 4px;
    padding: 5px 10px;
}
QPushButton#group_matching_danger_button {
    background: $danger_base;
    color: $danger_text;
    border: 1px solid $danger_border;
    border-radius: 4px;
    padding: 5px 10px;
}
QPushButton#group_matching_danger_button:hover {
    background: $danger_hover;
}
QToolButton#fov_id_step_button {
    background: $button;
    color: $button_text;
    border: 1px solid $border;
    border-radius: 4px;
    padding: 0px;
}
QToolButton#fov_id_step_button:hover {
    background: $button_hover;
}
QPushButton#group_matching_primary_button:disabled,
QPushButton#group_matching_secondary_button:disabled,
QPushButton#group_matching_quiet_button:disabled,
QPushButton#group_matching_danger_button:disabled,
QToolButton#fov_id_step_button:disabled {
    background: $disabled_base;
    color: $disabled_text;
    border-color: $border;
}
QLineEdit, QSpinBox {
    background: $input_base;
    color: $text;
    border: 1px solid $border;
    border-radius: 4px;
    padding: 3px 5px;
}
QLineEdit:disabled, QSpinBox:disabled {
    background: $disabled_base;
    color: $disabled_text;
    border-color: $border;
}
QTableWidget {
    background: $input_base;
    color: $text;
    border: 1px solid $border;
    gridline-color: $border;
}
"""
)


def style_group_matching_panel(widget: QWidget) -> None:
    """Apply the shared group-matching visual theme to a widget tree.

    Args:
        widget: Popup root or staged view that should inherit the stylesheet.

    Returns:
        None.
    """
    widget.setObjectName("group_matching_panel")
    widget.setStyleSheet(_style_from_palette(widget.palette()))


def group_matching_button(text: str, *, role: str = "secondary") -> QPushButton:
    """Return one consistently sized group-matching action button.

    Args:
        text: Visible button label.
        role: Visual emphasis: ``primary``, ``secondary``, ``quiet``, or
            ``danger``.

    Returns:
        A styled ``QPushButton`` ready for signal wiring.
    """
    button = QPushButton(text)
    button.setObjectName(f"group_matching_{role}_button")
    button.setMinimumHeight(30)
    return button


def group_matching_section(title: str, layout: QLayout) -> QGroupBox:
    """Wrap one layout in a titled group-matching section.

    Args:
        title: Section title shown to the user.
        layout: Layout containing the controls for this section.

    Returns:
        A styled ``QGroupBox`` containing ``layout``.
    """
    group = QGroupBox(title)
    group.setObjectName("group_matching_section")
    group.setLayout(layout)
    return group


def _style_from_palette(palette: QPalette) -> str:
    """Return a stylesheet whose colors come from the active Qt palette."""
    window = palette.window().color()
    base = palette.base().color()
    alternate_base = palette.alternateBase().color()
    text = palette.windowText().color()
    button = palette.button().color()
    button_text = palette.buttonText().color()
    highlight = palette.highlight().color()
    highlighted_text = palette.highlightedText().color()
    border = palette.mid().color()
    is_dark = window.lightness() < 128
    return _STYLE_TEMPLATE.substitute(
        window=_css_color(window),
        base=_css_color(base),
        text=_css_color(text),
        muted_text=_css_color(_blend(text, window, 0.42)),
        border=_css_color(border),
        title_pill=_css_color(_blend(base, highlight, 0.10)),
        selected_base=_css_color(_blend(base, highlight, 0.18)),
        highlight=_css_color(highlight),
        highlight_hover=_css_color(
            highlight.lighter(118) if is_dark else highlight.darker(110),
        ),
        highlighted_text=_css_color(highlighted_text),
        highlight_border=_css_color(
            highlight.darker(125) if is_dark else highlight.darker(145),
        ),
        button=_css_color(button),
        button_hover=_css_color(button.lighter(125) if is_dark else button.darker(108)),
        button_text=_css_color(button_text),
        danger_base=_css_color(_blend(alternate_base, QColor("#d95f02"), 0.18)),
        danger_hover=_css_color(_blend(alternate_base, QColor("#d95f02"), 0.32)),
        danger_text=_css_color(QColor("#ffb199") if is_dark else QColor("#7f2d1c")),
        danger_border=_css_color(QColor("#bf6b55") if is_dark else QColor("#c58d7d")),
        disabled_base=_css_color(_blend(button, window, 0.35)),
        disabled_text=_css_color(_blend(button_text, window, 0.45)),
        input_base=_css_color(base),
    )


def _blend(first: QColor, second: QColor, second_weight: float) -> QColor:
    """Return a weighted blend of two colors."""
    first_weight = 1.0 - second_weight
    return QColor(
        round(first.red() * first_weight + second.red() * second_weight),
        round(first.green() * first_weight + second.green() * second_weight),
        round(first.blue() * first_weight + second.blue() * second_weight),
    )


def _css_color(color: QColor) -> str:
    """Return one color formatted for Qt stylesheets."""
    return color.name()
