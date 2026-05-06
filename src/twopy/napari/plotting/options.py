"""Qt option widgets for twopy response plots.

Inputs: axis limits, ROI or epoch labels, visibility state, and optional ROI
colors.
Outputs: compact Qt controls for plot customization.

This module owns option-panel widgets only. It does not draw response traces,
load analysis files, or compute responses.
"""

from collections.abc import Callable
from typing import cast

from qtpy.QtCore import QSignalBlocker
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

__all__ = [
    "axis_options_widget",
    "visibility_options_widget",
]

type CallableAxisBounds = Callable[..., None]
type CallableVisibility = Callable[[str, bool], None]
type CallableVisibilityBatch = Callable[[dict[str, bool]], None]


def axis_options_widget(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    on_change: object,
) -> QWidget:
    """Create manual axis-bound controls.

    Args:
        x_min: Initial x-axis minimum.
        x_max: Initial x-axis maximum.
        y_min: Initial y-axis minimum.
        y_max: Initial y-axis maximum.
        on_change: Callback receiving named axis bounds.

    Returns:
        Qt widget containing four numeric inputs.
    """
    callback = cast(CallableAxisBounds, on_change)
    widget = QWidget()
    layout = QGridLayout()
    layout.addWidget(QLabel("Axis limits"), 0, 0, 1, 2)
    x_min_spin = _axis_spin_box(x_min)
    x_max_spin = _axis_spin_box(x_max)
    y_min_spin = _axis_spin_box(y_min)
    y_max_spin = _axis_spin_box(y_max)
    layout.addWidget(QLabel("x min"), 1, 0)
    layout.addWidget(x_min_spin, 1, 1)
    layout.addWidget(QLabel("x max"), 2, 0)
    layout.addWidget(x_max_spin, 2, 1)
    layout.addWidget(QLabel("y min"), 3, 0)
    layout.addWidget(y_min_spin, 3, 1)
    layout.addWidget(QLabel("y max"), 4, 0)
    layout.addWidget(y_max_spin, 4, 1)

    def update_bounds() -> None:
        """Pass current spin-box values back to the response widget."""
        callback(
            x_min=x_min_spin.value(),
            x_max=x_max_spin.value(),
            y_min=y_min_spin.value(),
            y_max=y_max_spin.value(),
        )

    x_min_spin.valueChanged.connect(update_bounds)
    x_max_spin.valueChanged.connect(update_bounds)
    y_min_spin.valueChanged.connect(update_bounds)
    y_max_spin.valueChanged.connect(update_bounds)
    widget.setLayout(layout)
    return widget


def visibility_options_widget(
    *,
    title: str,
    labels: tuple[str, ...],
    visibility: dict[str, bool],
    on_change: object,
    on_change_batch: object | None = None,
    colors: tuple[QColor, ...] | None = None,
) -> QWidget:
    """Create select-all/select-none checkboxes for plot visibility.

    Args:
        title: Section label.
        labels: Checkbox labels.
        visibility: Current visibility by label.
        on_change: Callback receiving ``(label, visible)``.
        on_change_batch: Optional callback receiving all labels changed by
            Select all or None. This prevents one bulk click from redrawing the
            plots once per checkbox.
        colors: Optional swatch colors matching ``labels`` by position.

    Returns:
        Qt widget containing controls.
    """
    callback = cast(CallableVisibility, on_change)
    batch_callback = (
        cast(CallableVisibilityBatch, on_change_batch)
        if on_change_batch is not None
        else None
    )
    widget = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(QLabel(title))
    button_row = QHBoxLayout()
    all_button = QPushButton("Select all")
    none_button = QPushButton("None")
    button_row.addWidget(all_button)
    button_row.addWidget(none_button)
    layout.addLayout(button_row)

    list_widget = QWidget()
    list_layout = QVBoxLayout()
    list_layout.setContentsMargins(0, 0, 0, 0)
    checkboxes: list[QCheckBox] = []
    for index, label in enumerate(labels):
        checkbox = QCheckBox(label)
        checkbox.setChecked(visibility.get(label, True))
        checkbox.stateChanged.connect(
            lambda _state, item=label, box=checkbox: callback(item, box.isChecked())
        )
        checkboxes.append(checkbox)
        list_layout.addWidget(_visibility_row(checkbox, _color_at(colors, index)))
    list_widget.setLayout(list_layout)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(list_widget)
    scroll.setMaximumHeight(_visibility_list_height(checkboxes, visible_rows=8))
    layout.addWidget(scroll)

    def set_all(checked: bool) -> None:
        """Set all checkboxes and visibility flags together."""
        blockers = [QSignalBlocker(checkbox) for checkbox in checkboxes]
        for checkbox in checkboxes:
            checkbox.setChecked(checked)
        del blockers
        if batch_callback is not None:
            batch_callback({checkbox.text(): checked for checkbox in checkboxes})
            return
        for checkbox in checkboxes:
            callback(checkbox.text(), checked)

    all_button.clicked.connect(lambda: set_all(True))
    none_button.clicked.connect(lambda: set_all(False))
    widget.setLayout(layout)
    return widget


def _axis_spin_box(value: float) -> QDoubleSpinBox:
    """Create one numeric axis-limit input.

    Args:
        value: Initial value.

    Returns:
        Configured spin box.
    """
    spin_box = QDoubleSpinBox()
    spin_box.setRange(-1_000_000.0, 1_000_000.0)
    spin_box.setDecimals(3)
    spin_box.setSingleStep(0.1)
    spin_box.setValue(value)
    return spin_box


def _visibility_row(checkbox: QCheckBox, color: QColor | None) -> QWidget:
    """Create one compact checkbox row, with an optional color swatch.

    Args:
        checkbox: Visibility checkbox.
        color: Optional ROI color.

    Returns:
        Row widget for the visibility list.
    """
    row = QWidget()
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    if color is not None:
        layout.addWidget(_color_swatch(color))
    layout.addWidget(checkbox)
    layout.addStretch(1)
    row.setLayout(layout)
    return row


def _color_swatch(color: QColor) -> QWidget:
    """Create the small color square shown beside one ROI label.

    Args:
        color: ROI color.

    Returns:
        Fixed-size Qt widget styled as a color square.
    """
    swatch = QWidget()
    swatch.setFixedSize(12, 12)
    swatch.setStyleSheet(
        f"background-color: {color.name()}; border: 1px solid #cfd6df;"
    )
    return swatch


def _color_at(colors: tuple[QColor, ...] | None, index: int) -> QColor | None:
    """Return one optional swatch color.

    Args:
        colors: Optional colors.
        index: Requested label index.

    Returns:
        Matching color, or ``None``.
    """
    if colors is None or index >= len(colors):
        return None
    return colors[index]


def _visibility_list_height(
    checkboxes: list[QCheckBox],
    *,
    visible_rows: int,
) -> int:
    """Return a compact maximum height for a checkbox list.

    Args:
        checkboxes: Checkbox widgets in one visibility section.
        visible_rows: Number of rows to show before scrolling.

    Returns:
        Pixel height for the scroll area.
    """
    if len(checkboxes) == 0:
        return 0
    row_height = max(checkbox.sizeHint().height() for checkbox in checkboxes)
    rows = min(len(checkboxes), visible_rows)
    return row_height * rows + 8
