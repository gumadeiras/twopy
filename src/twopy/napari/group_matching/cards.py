"""Shared card-layout helpers for Group Matching views.

Inputs: Qt labels, grid layouts, and fixed card dimensions used by FOV and ROI
assignment cards.
Outputs: consistent image-overlay widgets, responsive card columns, and grid
cleanup behavior.

The FOV and ROI cards keep their own controls because they represent different
workflow decisions. This module only owns repeated card chrome and layout math
that must stay visually consistent across both stages.
"""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QGridLayout, QLabel, QSizePolicy, QWidget

__all__ = [
    "GROUP_MATCHING_CARD_GRID_SIDE_PADDING",
    "GROUP_MATCHING_CARD_SPACING",
    "card_columns_for_width",
    "clear_card_grid_layout",
    "fov_group_display_id",
    "image_overlay_stack",
]

GROUP_MATCHING_CARD_GRID_SIDE_PADDING = 48
GROUP_MATCHING_CARD_SPACING = 12


def card_columns_for_width(width: int, *, card_width: int, spacing: int) -> int:
    """Return how many fixed-width cards fit in one row.

    Args:
        width: Current viewport width available to the card grid.
        card_width: Fixed width of one card.
        spacing: Horizontal spacing between grid columns.

    Returns:
        At least one column, capped only by the available viewport width.

    Group Matching uses fixed-size image cards so controls do not jump as a
    user changes selections. The column count is the only responsive part of
    the grid.
    """
    available_width = max(card_width, width - GROUP_MATCHING_CARD_GRID_SIDE_PADDING)
    return max(1, (available_width + spacing) // (card_width + spacing))


def clear_card_grid_layout(layout: QGridLayout, *, delete_widgets: bool) -> None:
    """Remove all child widgets from one card grid.

    Args:
        layout: Grid that owns FOV or ROI card widgets.
        delete_widgets: Whether removed widgets should be hidden, detached, and
            scheduled for deletion.

    Returns:
        None.

    Re-layouts keep existing card widgets parented so Qt does not briefly show
    them as top-level windows. Full refreshes delete widgets because cards are
    rebuilt from current loaded-recording state.
    """
    while layout.count() > 0:
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        if widget is not None and delete_widgets:
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()


def image_overlay_stack(
    image_label: QLabel,
    overlay_label: QLabel,
    *,
    size: int,
) -> QWidget:
    """Return a fixed image stack with a bottom-left overlay label.

    Args:
        image_label: Label displaying the mean image or ROI overlay pixmap.
        overlay_label: Label containing recording and FOV text.
        size: Square pixel size shared by the image and stack.

    Returns:
        A fixed-size widget containing the image and overlay labels.
    """
    image_label.setFixedSize(size, size)
    image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    overlay_label.setObjectName("fov_card_overlay")
    overlay_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    image_stack = QWidget()
    image_stack.setFixedSize(size, size)
    image_stack.setSizePolicy(
        QSizePolicy.Policy.Fixed,
        QSizePolicy.Policy.Fixed,
    )
    image_layout = QGridLayout()
    image_layout.setContentsMargins(0, 0, 0, 0)
    image_layout.setSpacing(0)
    image_layout.addWidget(image_label, 0, 0)
    image_layout.addWidget(
        overlay_label,
        0,
        0,
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
    )
    image_stack.setLayout(image_layout)
    return image_stack


def fov_group_display_id(fov_group_id: str) -> str:
    """Return compact FOV text for card overlays and selectors.

    Args:
        fov_group_id: Stored FOV id such as ``"fov_3"`` or display text such
            as ``"none"``.

    Returns:
        Numeric suffix when the stored id follows the normal ``fov_N`` shape;
        otherwise the original text.
    """
    suffix = fov_group_id.removeprefix("fov_")
    return suffix if suffix.isdecimal() else fov_group_id
