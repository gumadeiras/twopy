"""Shared Qt form-control helpers for response Plot-tab option groups.

Inputs: Qt widgets that appear as fields inside Plot-tab form rows.
Outputs: consistent field widths and form-layout spacing across response
plot controls.

This module owns only visual control sizing. Scientific option widgets keep
their own value translation and analysis-specific validation.
"""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QSizePolicy,
    QWidget,
)

__all__ = [
    "PLOT_DROPDOWN_WIDTH",
    "PLOT_CONTROL_WIDTH",
    "plot_form_layout",
    "set_plot_control_width",
    "set_plot_dropdown_width",
    "show_all_plot_dropdown_items",
]

PLOT_CONTROL_WIDTH = 100
PLOT_DROPDOWN_WIDTH = 160


def plot_form_layout() -> QFormLayout:
    """Return the shared form layout used by Plot-tab option groups.

    Args:
        None.

    Returns:
        Form layout with the same alignment and spacing across Plot controls.
    """
    layout = QFormLayout()
    layout.setFormAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
    layout.setVerticalSpacing(4)
    return layout


def set_plot_control_width(control: QWidget, *, width: int | None = None) -> None:
    """Give one Plot-tab non-dropdown field widget the shared fixed width.

    Args:
        control: Field widget in a Plot-tab form row.
        width: Optional width for controls whose visible text needs more room.

    Returns:
        None.
    """
    if width is not None:
        _set_plot_field_width(control, width)
        return
    if isinstance(control, QCheckBox):
        _set_plot_field_width(control, PLOT_DROPDOWN_WIDTH)
        return
    _set_plot_field_width(control, PLOT_CONTROL_WIDTH)


def set_plot_dropdown_width(control: QComboBox) -> None:
    """Give one Plot-tab dropdown the shared wider fixed width.

    Args:
        control: Dropdown field widget in a Plot-tab form row.

    Returns:
        None.
    """
    _set_plot_field_width(control, PLOT_DROPDOWN_WIDTH)
    view = control.view()
    if view is None:
        return
    view.setFixedWidth(PLOT_DROPDOWN_WIDTH)
    popup = view.window()
    if popup is not None:
        popup.setFixedWidth(PLOT_DROPDOWN_WIDTH)


def show_all_plot_dropdown_items(control: QComboBox) -> None:
    """Make one Plot-tab dropdown popup show every current item.

    Args:
        control: Dropdown whose current choices should fit without a popup
            scrollbar.

    Returns:
        None.
    """
    control.setMaxVisibleItems(max(1, control.count()))
    control.setStyleSheet(f"width: {PLOT_DROPDOWN_WIDTH}px; combobox-popup: 0;")
    view = control.view()
    if view is None:
        return
    fallback_row_height = control.sizeHint().height()
    row_heights = [view.sizeHintForRow(index) for index in range(control.count())]
    popup_height = sum(
        row_height if row_height > 0 else max(1, fallback_row_height)
        for row_height in row_heights
    )
    if popup_height == 0:
        popup_height = max(1, fallback_row_height)
    popup_height += 2 * view.frameWidth()
    view.setMinimumHeight(popup_height)
    view.setFixedHeight(popup_height)
    popup = view.window()
    if popup is not None:
        popup.setMinimumHeight(popup_height)
        popup.setFixedHeight(popup_height)


def _set_plot_field_width(control: QWidget, width: int) -> None:
    """Apply one fixed outer field width to a Qt widget."""
    control.setStyleSheet(f"width: {width}px;")
    control.setFixedWidth(width)
    control.setSizePolicy(
        QSizePolicy.Policy.Fixed,
        control.sizePolicy().verticalPolicy(),
    )
