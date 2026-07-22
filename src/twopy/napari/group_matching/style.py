"""Shared visual styling helpers for the napari group-matching popup.

Inputs: Qt widgets used by the FOV and ROI matching views.
Outputs: consistent buttons, section boxes, and panel styling.

The helpers keep display choices close to the napari UI layer. They do not
change FOV or ROI matching data, and they do not read or write CSV decisions.
"""

from string import Template

from qtpy.QtWidgets import QGroupBox, QLayout, QPushButton, QWidget

from twopy.napari.theme import (
    TwopyThemeColors,
    apply_twopy_theme,
    style_action_button,
)

__all__ = [
    "GROUP_MATCHING_OUTER_MARGIN",
    "group_matching_button",
    "group_matching_section",
    "style_group_matching_panel",
]

GROUP_MATCHING_OUTER_MARGIN = 8

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
    background: $window;
    border: 1px solid $border;
    border-radius: 10px;
    margin-top: 14px;
    padding: 14px 10px 10px 10px;
}
QGroupBox#group_matching_section::title {
    color: $text;
    font-weight: 700;
    left: 10px;
    background: $window;
    border: none;
    padding: 2px 7px;
    subcontrol-origin: margin;
}
QFrame#fov_recording_card,
QFrame#roi_assignment_card {
    background: $base;
    border: 1px solid $border;
    border-radius: 8px;
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
    border-radius: 7px;
    font-weight: 700;
    padding: 0px 14px;
}
QPushButton#group_matching_primary_button:hover {
    background: $highlight_hover;
}
QPushButton#group_matching_secondary_button {
    background: $button;
    color: $button_text;
    border: 1px solid $button_border;
    border-radius: 7px;
    padding: 0px 12px;
}
QPushButton#group_matching_secondary_button:hover,
QPushButton#group_matching_quiet_button:hover {
    background: $button_hover;
}
QPushButton#group_matching_quiet_button {
    background: $button;
    color: $button_text;
    border: 1px solid $button_border;
    border-radius: 7px;
    padding: 0px 12px;
}
QToolButton#fov_id_step_button {
    background: $button;
    color: $button_text;
    border: 1px solid $button_border;
    border-radius: 7px;
    font-size: 16px;
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
    border-color: $button_border;
    font-weight: 400;
}
QLineEdit, QSpinBox {
    background: $input_base;
    color: $text;
    border: 1px solid $border;
    border-radius: 7px;
    padding: 4px 7px;
}
QLineEdit:disabled, QSpinBox:disabled {
    background: $disabled_base;
    color: $disabled_text;
    border-color: $border;
}
QTableWidget {
    background: $input_base;
    color: $text;
    border: 1px solid $button_border;
    gridline-color: $button_border;
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
    apply_twopy_theme(
        widget,
        name="group_matching_panel",
        additional_style=_style_from_theme,
    )


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
    style_action_button(button, role=role)
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


def _style_from_theme(theme: TwopyThemeColors) -> str:
    """Return group-matching styles from the active napari theme."""
    return _STYLE_TEMPLATE.substitute(
        window=theme.window,
        base=theme.surface,
        text=theme.text,
        muted_text=theme.muted_text,
        border=theme.control_outline,
        selected_base=theme.selected_surface,
        highlight=theme.accent,
        highlight_hover=theme.accent_hover,
        highlighted_text=theme.accent_text,
        highlight_border=theme.primary_outline,
        button=theme.button,
        button_hover=theme.button_hover,
        button_text=theme.text,
        button_border=theme.control_outline,
        disabled_base=theme.disabled_surface,
        disabled_text=theme.disabled_text,
        input_base=theme.field,
    )
