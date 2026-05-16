"""Qt controls for response heatmap settings.

Inputs: typed ``ResponseMapOptions`` and an optional recording crop shape.
Outputs: a compact Plot-tab widget that returns GUI-independent heatmap options.

The widget owns only GUI state and validation ranges. The analysis package owns
the response-map math.
"""

from collections.abc import Callable

from qtpy.QtCore import QSignalBlocker
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.response_maps import ResponseMapMode, ResponseMapOptions
from twopy.napari.plotting.form_controls import (
    plot_form_layout,
    set_plot_control_width,
    set_plot_dropdown_width,
)
from twopy.typing_guards import require_string_choice

__all__ = ["ResponseMapOptionsWidget"]

_MODE_LABELS: tuple[tuple[str, ResponseMapMode], ...] = (
    ("pixel", "pixel"),
    ("window", "window"),
)
_MODES = tuple(value for _label, value in _MODE_LABELS)
_WINDOW_PRESETS = (
    ("3x3 stride 1", 3, 1),
    ("4x4 stride 2", 4, 2),
    ("5x5 stride 2", 5, 2),
    ("custom", 0, 0),
)


class ResponseMapOptionsWidget(QWidget):
    """Widget that exposes response heatmap settings.

    Inputs: initial response-map options, optional recording crop shape, and an
    optional change callback.
    Outputs: Qt controls plus ``options()`` for reading typed computation
    settings, and ``shared_limits()`` for display-only color scaling.

    The widget intentionally exposes only usable controls. Pixel mode shows
    Gaussian sigma; window mode shows preset, square size, and stride. The
    shared-limits checkbox is display-only and does not trigger heatmap
    recomputation.
    """

    def __init__(
        self,
        options: ResponseMapOptions,
        *,
        spatial_shape: tuple[int, int] | None = None,
        on_change: Callable[[ResponseMapOptions], None] | None = None,
        on_shared_limits_change: Callable[[bool], None] | None = None,
    ) -> None:
        """Create response-map option controls.

        Args:
            options: Initial response-map settings.
            spatial_shape: Optional crop shape used to cap window size.
            on_change: Optional callback receiving typed settings whenever a
                control changes.
            on_shared_limits_change: Optional callback receiving the display
                scaling choice. This does not change heatmap computation.
        """
        super().__init__()
        self._on_change = on_change
        self._on_shared_limits_change = on_shared_limits_change
        self._mode = _combo_box(_MODE_LABELS)
        self._shared_limits = QCheckBox("")
        self._shared_limits.setChecked(True)
        set_plot_control_width(self._shared_limits)
        self._pixel_smoothing_sigma = _double_spin_box(
            minimum=0.0,
            maximum=100.0,
            value=options.pixel_smoothing_sigma,
        )
        self._window_preset = _preset_combo_box()
        self._window_size = _spin_box(
            minimum=1,
            maximum=_max_window_size(spatial_shape),
            value=min(options.window_size_pixels, _max_window_size(spatial_shape)),
        )
        self._window_stride = _spin_box(
            minimum=1,
            maximum=max(1, self._window_size.value()),
            value=min(options.window_stride_pixels, max(1, self._window_size.value())),
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._response_map_group())
        self.setLayout(layout)

        self.set_options(options)
        self.set_spatial_shape(spatial_shape)
        self._connect_changes()
        self._refresh_enabled_state()

    def options(self) -> ResponseMapOptions:
        """Return the current typed response-map options.

        Args:
            None.

        Returns:
            ``ResponseMapOptions`` built from current controls.
        """
        return ResponseMapOptions(
            mode=require_string_choice(
                str(self._mode.currentData()),
                name="response map mode",
                allowed=_MODES,
            ),
            pixel_smoothing_sigma=self._pixel_smoothing_sigma.value(),
            window_size_pixels=self._window_size.value(),
            window_stride_pixels=self._window_stride.value(),
        )

    def shared_limits(self) -> bool:
        """Return whether heatmap colors should use shared epoch limits.

        Args:
            None.

        Returns:
            ``True`` when all epochs share one robust display color limit.
        """
        return self._shared_limits.isChecked()

    def set_options(self, options: ResponseMapOptions) -> None:
        """Update controls from typed options without emitting changes.

        Args:
            options: Response-map settings to show.

        Returns:
            None.
        """
        blockers = [
            QSignalBlocker(self._mode),
            QSignalBlocker(self._pixel_smoothing_sigma),
            QSignalBlocker(self._window_preset),
            QSignalBlocker(self._window_size),
            QSignalBlocker(self._window_stride),
        ]
        self._set_combo_data(self._mode, options.mode)
        self._pixel_smoothing_sigma.setValue(options.pixel_smoothing_sigma)
        max_window_size = self._window_size.maximum()
        window_size = min(options.window_size_pixels, max_window_size)
        self._window_size.setValue(window_size)
        self._window_stride.setMaximum(window_size)
        self._window_stride.setValue(min(options.window_stride_pixels, window_size))
        self._set_preset_from_values()
        del blockers
        self._refresh_enabled_state()

    def set_spatial_shape(self, spatial_shape: tuple[int, int] | None) -> None:
        """Set the spatial shape used to cap valid window sizes.

        Args:
            spatial_shape: Recording crop shape, or ``None`` before recording
                load.

        Returns:
            None.
        """
        max_window_size = _max_window_size(spatial_shape)
        blockers = [
            QSignalBlocker(self._window_size),
            QSignalBlocker(self._window_stride),
        ]
        self._window_size.setMaximum(max_window_size)
        if self._window_size.value() > max_window_size:
            self._window_size.setValue(max_window_size)
        self._window_stride.setMaximum(max(1, self._window_size.value()))
        if self._window_stride.value() > self._window_stride.maximum():
            self._window_stride.setValue(self._window_stride.maximum())
        del blockers
        self._set_preset_from_values()
        self._refresh_enabled_state()

    def _response_map_group(self) -> QGroupBox:
        """Create the response-map option group.

        Shared limits comes first because it is independent from the computation
        mode. Mode-specific rows are hidden by ``_refresh_enabled_state`` so the
        user sees only parameters that affect the selected method.
        """
        group = QGroupBox("Heatmap")
        self._form_layout = plot_form_layout()
        layout = self._form_layout
        layout.addRow("Shared limits", self._shared_limits)
        layout.addRow("Mode", self._mode)
        layout.addRow("Sigma", self._pixel_smoothing_sigma)
        layout.addRow("Preset", self._window_preset)
        layout.addRow("Size", self._window_size)
        layout.addRow("Stride", self._window_stride)
        group.setLayout(layout)
        return group

    def _connect_changes(self) -> None:
        """Connect controls to callbacks."""
        self._mode.currentIndexChanged.connect(self._emit_change)
        self._shared_limits.stateChanged.connect(self._emit_shared_limits_change)
        self._pixel_smoothing_sigma.valueChanged.connect(self._emit_change)
        self._window_preset.currentIndexChanged.connect(self._set_preset)
        self._window_size.valueChanged.connect(self._set_window_size)
        self._window_stride.valueChanged.connect(self._set_custom_stride)

    def _set_preset(self, *_args: object) -> None:
        """Apply a selected preset when it fits the current crop."""
        preset = self._window_preset.currentData()
        size, stride = preset if isinstance(preset, tuple) else (0, 0)
        if size <= 0 or stride <= 0:
            self._emit_change()
            return
        blockers = [
            QSignalBlocker(self._window_size),
            QSignalBlocker(self._window_stride),
        ]
        size = min(size, self._window_size.maximum())
        stride = min(stride, size)
        self._window_size.setValue(size)
        self._window_stride.setMaximum(size)
        self._window_stride.setValue(stride)
        del blockers
        self._emit_change()

    def _set_window_size(self, value: int) -> None:
        """Keep stride valid after a window-size edit."""
        blockers = [QSignalBlocker(self._window_stride)]
        self._window_stride.setMaximum(max(1, int(value)))
        if self._window_stride.value() > self._window_stride.maximum():
            self._window_stride.setValue(self._window_stride.maximum())
        del blockers
        self._set_preset_from_values()
        self._emit_change()

    def _set_custom_stride(self, *_args: object) -> None:
        """Mark preset custom after a stride edit."""
        self._set_preset_from_values()
        self._emit_change()

    def _emit_change(self, *_args: object) -> None:
        """Emit typed options after a GUI value changes."""
        self._refresh_enabled_state()
        if self._on_change is not None:
            self._on_change(self.options())

    def _emit_shared_limits_change(self, *_args: object) -> None:
        """Emit display scaling changes without recomputing heatmaps."""
        if self._on_shared_limits_change is not None:
            self._on_shared_limits_change(self.shared_limits())

    def _refresh_enabled_state(self) -> None:
        """Show controls relevant to the selected mode."""
        pixel_mode = self._mode.currentData() == "pixel"
        _set_form_row_visible(
            self._form_layout, self._pixel_smoothing_sigma, pixel_mode
        )
        for widget in (self._window_preset, self._window_size, self._window_stride):
            _set_form_row_visible(self._form_layout, widget, not pixel_mode)

    def _set_preset_from_values(self) -> None:
        """Select the preset matching current size/stride or custom."""
        size = self._window_size.value()
        stride = self._window_stride.value()
        for index in range(self._window_preset.count()):
            if self._window_preset.itemData(index) == (size, stride):
                self._window_preset.setCurrentIndex(index)
                return
        self._window_preset.setCurrentIndex(self._window_preset.count() - 1)

    def _set_combo_data(self, combo_box: QComboBox, value: str) -> None:
        """Select a combo item by its data value."""
        index = combo_box.findData(value)
        if index >= 0:
            combo_box.setCurrentIndex(index)


def _combo_box(labels: tuple[tuple[str, str], ...]) -> QComboBox:
    """Return a fixed-width combo box with labels and data values."""
    combo_box = QComboBox()
    for label, value in labels:
        combo_box.addItem(label, value)
    set_plot_dropdown_width(combo_box)
    return combo_box


def _preset_combo_box() -> QComboBox:
    """Return the fixed response-map window preset menu."""
    combo_box = QComboBox()
    for label, size, stride in _WINDOW_PRESETS:
        combo_box.addItem(label, (size, stride))
    set_plot_dropdown_width(combo_box)
    return combo_box


def _spin_box(*, minimum: int, maximum: int, value: int) -> QSpinBox:
    """Create one integer heatmap input."""
    spin_box = QSpinBox()
    spin_box.setRange(minimum, maximum)
    spin_box.setSingleStep(1)
    spin_box.setValue(value)
    set_plot_control_width(spin_box)
    return spin_box


def _double_spin_box(*, minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
    """Create one floating-point heatmap input."""
    spin_box = QDoubleSpinBox()
    spin_box.setRange(minimum, maximum)
    spin_box.setDecimals(1)
    spin_box.setSingleStep(0.5)
    spin_box.setValue(value)
    set_plot_control_width(spin_box)
    return spin_box


def _max_window_size(spatial_shape: tuple[int, int] | None) -> int:
    """Return the maximum valid square window size for one crop."""
    if spatial_shape is None:
        return 512
    return max(1, min(spatial_shape))


def _set_form_row_visible(
    layout: QFormLayout,
    field: QWidget,
    visible: bool,
) -> None:
    """Show or hide one form row by its field widget."""
    label = layout.labelForField(field)
    if label is not None:
        label.setVisible(visible)
    field.setVisible(visible)
