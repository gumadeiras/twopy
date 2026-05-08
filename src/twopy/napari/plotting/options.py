"""Qt option widgets for twopy response plots.

Inputs: axis limits, ROI or epoch labels, visibility state, and optional ROI
colors.
Outputs: compact Qt controls for plot customization.

This module owns option-panel widgets only. It does not draw response traces,
load analysis files, or compute responses.
"""

from collections.abc import Callable, Hashable, Mapping
from typing import cast

from qtpy.QtCore import QSignalBlocker
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from twopy.napari.plotting.form_controls import (
    plot_form_layout,
    set_plot_control_width,
)

__all__ = [
    "plot_display_options_group",
    "visibility_options_widget",
]

type CallableAxisBounds = Callable[..., None]
type CallableVisibility = Callable[[Hashable, bool], None]
type CallableVisibilityBatch = Callable[[dict[Hashable, bool]], None]
type VisibilityKey = int | str
type VisibilityState = Mapping[int, bool] | Mapping[str, bool]


def plot_display_options_group(
    *,
    show_sem_checkbox: QCheckBox,
    plot_size_spin: QSpinBox,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    on_change: object,
) -> QGroupBox:
    """Create the plot option group using the same form layout as other groups.

    Args:
        show_sem_checkbox: Toggle for drawing SEM bands.
        plot_size_spin: Numeric plot-size input.
        x_min: Initial x-axis minimum.
        x_max: Initial x-axis maximum.
        y_min: Initial y-axis minimum.
        y_max: Initial y-axis maximum.
        on_change: Callback receiving named axis bounds.

    Returns:
        Qt group box with plot display controls.
    """
    callback = cast(CallableAxisBounds, on_change)
    group = QGroupBox("Plot")
    layout = plot_form_layout()
    x_min_spin = _axis_spin_box(x_min)
    x_max_spin = _axis_spin_box(x_max)
    y_min_spin = _axis_spin_box(y_min)
    y_max_spin = _axis_spin_box(y_max)
    set_plot_control_width(show_sem_checkbox)
    set_plot_control_width(plot_size_spin)
    layout.addRow("", show_sem_checkbox)
    layout.addRow("Size", plot_size_spin)
    layout.addRow("X min", x_min_spin)
    layout.addRow("X max", x_max_spin)
    layout.addRow("Y min", y_min_spin)
    layout.addRow("Y max", y_max_spin)

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
    group.setLayout(layout)
    return group


def visibility_options_widget(
    *,
    title: str,
    labels: tuple[str, ...],
    visibility: VisibilityState,
    on_change: object,
    on_change_batch: object | None = None,
    keys: tuple[VisibilityKey, ...] | None = None,
    colors: tuple[QColor, ...] | None = None,
) -> QWidget:
    """Create select-all/select-none checkboxes for plot visibility.

    Args:
        title: Section label.
        labels: Checkbox labels.
        visibility: Current visibility by label or by ``keys`` when keys are
            provided.
        on_change: Callback receiving ``(label, visible)``.
        on_change_batch: Optional callback receiving all labels changed by
            Select all or None. This prevents one bulk click from redrawing the
            plots once per checkbox.
        keys: Optional stable callback keys matching ``labels``. Labels stay
            human-readable; keys prevent duplicate display names from toggling
            the wrong underlying item.
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
    visibility_lookup = dict(visibility.items())
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
    checkbox_keys: list[VisibilityKey] = []
    for index, label in enumerate(labels):
        key = keys[index] if keys is not None and index < len(keys) else label
        checkbox = QCheckBox(label)
        checkbox.setChecked(visibility_lookup.get(key, True))
        checkbox.stateChanged.connect(
            lambda _state, item=key, box=checkbox: callback(item, box.isChecked())
        )
        checkboxes.append(checkbox)
        checkbox_keys.append(key)
        list_layout.addWidget(_visibility_row(checkbox, _color_at(colors, index)))
    list_widget.setLayout(list_layout)

    layout.addWidget(list_widget)

    def set_all(checked: bool) -> None:
        """Set all checkboxes and visibility flags together."""
        blockers = [QSignalBlocker(checkbox) for checkbox in checkboxes]
        for checkbox in checkboxes:
            checkbox.setChecked(checked)
        del blockers
        if batch_callback is not None:
            batch_callback({key: checked for key in checkbox_keys})
            return
        for key in checkbox_keys:
            callback(key, checked)

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
    spin_box.setDecimals(2)
    spin_box.setSingleStep(0.1)
    spin_box.setValue(value)
    set_plot_control_width(spin_box)
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
