"""Qt controls for selecting response-processing options.

Inputs: a current ``ResponseProcessingOptions`` object and an optional callback.
Outputs: a compact widget that returns typed GUI-independent processing options.

This module owns only GUI controls and value translation. The analysis package
owns validation, math, and persistence of the selected settings.
"""

import math
from collections.abc import Callable

from qtpy.QtCore import QSignalBlocker
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.response_processing import (
    CorrelationFilterOptions,
    CorrelationFilterReference,
    LowPassFilterMethod,
    LowPassFilterOptions,
    NormalizationOptions,
    ResponseProcessingOptions,
    SmoothingMethod,
    SmoothingOptions,
)
from twopy.napari.plotting.form_controls import (
    plot_form_layout,
    set_plot_control_width,
    set_plot_dropdown_width,
)
from twopy.typing_guards import require_string_choice

__all__ = ["ResponseProcessingOptionsWidget", "SmoothingOptionsWidget"]

_SMOOTHING_METHOD_LABELS: tuple[tuple[str, SmoothingMethod], ...] = (
    ("none", "none"),
    ("moving average", "moving_average"),
    ("savitzky-golay", "savgol"),
)
_SMOOTHING_METHODS = tuple(value for _label, value in _SMOOTHING_METHOD_LABELS)
_LOW_PASS_METHOD_LABELS: tuple[tuple[str, LowPassFilterMethod], ...] = (
    ("none", "none"),
    ("butterworth", "butterworth"),
)
_LOW_PASS_METHODS = tuple(value for _label, value in _LOW_PASS_METHOD_LABELS)
_CORRELATION_REFERENCE_LABELS: tuple[
    tuple[str, CorrelationFilterReference],
    ...,
] = (
    ("none", "none"),
    ("epoch mean", "epoch_mean"),
    ("epoch peak", "epoch_peak"),
)
_CORRELATION_REFERENCES = tuple(
    value for _label, value in _CORRELATION_REFERENCE_LABELS
)


class SmoothingOptionsWidget(QWidget):
    """Widget that exposes the Plot-tab smoothing controls on their own.

    Inputs: initial smoothing options and an optional change callback.
    Outputs: a Qt widget plus ``options()`` for reading typed smoothing
    settings before preview actions.
    """

    def __init__(
        self,
        options: SmoothingOptions,
        *,
        on_change: Callable[[SmoothingOptions], None] | None = None,
    ) -> None:
        """Create standalone smoothing controls.

        Args:
            options: Initial smoothing settings.
            on_change: Optional callback receiving new typed settings whenever
                a GUI control changes.
        """
        super().__init__()
        self._on_change = on_change
        self._smoothing_method = _combo_box(_SMOOTHING_METHOD_LABELS)
        self._smoothing_window_frames = _spin_box(
            minimum=1,
            maximum=1001,
            value=options.window_frames,
        )
        self._smoothing_polynomial_order = _spin_box(
            minimum=0,
            maximum=16,
            value=options.polynomial_order,
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._smoothing_group())
        self.setLayout(layout)

        _set_combo_data(self._smoothing_method, options.method)
        self._connect_changes()
        self._refresh_enabled_state()

    def options(self) -> SmoothingOptions:
        """Return the current typed smoothing options.

        Args:
            None.

        Returns:
            ``SmoothingOptions`` built from the current controls.
        """
        return SmoothingOptions(
            method=require_string_choice(
                str(self._smoothing_method.currentData()),
                name="smoothing method",
                allowed=_SMOOTHING_METHODS,
            ),
            window_frames=self._smoothing_window_frames.value(),
            polynomial_order=self._smoothing_polynomial_order.value(),
        )

    def set_options(self, options: SmoothingOptions) -> None:
        """Update controls from typed options without emitting changes.

        Args:
            options: Smoothing settings loaded from another source.

        Returns:
            None.
        """
        blockers = [
            QSignalBlocker(self._smoothing_method),
            QSignalBlocker(self._smoothing_window_frames),
            QSignalBlocker(self._smoothing_polynomial_order),
        ]
        _set_combo_data(self._smoothing_method, options.method)
        self._smoothing_window_frames.setValue(options.window_frames)
        self._smoothing_polynomial_order.setValue(options.polynomial_order)
        del blockers
        self._refresh_enabled_state()

    def _smoothing_group(self) -> QGroupBox:
        """Create the smoothing control group."""
        group = QGroupBox("Smoothing")
        layout = plot_form_layout()
        layout.addRow("Method", self._smoothing_method)
        layout.addRow("Window", self._smoothing_window_frames)
        layout.addRow("Order", self._smoothing_polynomial_order)
        group.setLayout(layout)
        return group

    def _connect_changes(self) -> None:
        """Connect control changes to state refresh and callback dispatch."""
        self._smoothing_method.currentIndexChanged.connect(self._emit_change)
        self._smoothing_window_frames.valueChanged.connect(self._emit_change)
        self._smoothing_polynomial_order.valueChanged.connect(self._emit_change)

    def _emit_change(self, *_args: object) -> None:
        """Emit typed options after a GUI value changes."""
        self._refresh_enabled_state()
        if self._on_change is not None:
            self._on_change(self.options())

    def _refresh_enabled_state(self) -> None:
        """Enable parameters only when their smoothing method is active."""
        self._refresh_smoothing_window_constraints()
        smoothing_method = self._smoothing_method.currentData()
        self._smoothing_window_frames.setEnabled(
            smoothing_method in {"moving_average", "savgol"},
        )
        self._smoothing_polynomial_order.setEnabled(
            smoothing_method == "savgol",
        )

    def _refresh_smoothing_window_constraints(self) -> None:
        """Keep the smoothing window spinbox aligned with the selected method."""
        if self._smoothing_method.currentData() != "savgol":
            self._smoothing_window_frames.setMinimum(1)
            self._smoothing_window_frames.setSingleStep(1)
            return

        blocker = QSignalBlocker(self._smoothing_window_frames)
        self._smoothing_window_frames.setMinimum(
            _minimum_savgol_window(self._smoothing_polynomial_order.value()),
        )
        self._smoothing_window_frames.setSingleStep(2)
        self._smoothing_window_frames.setValue(
            self._nearest_valid_savgol_window(),
        )
        del blocker

    def _nearest_valid_savgol_window(self) -> int:
        """Return the nearest valid Savitzky-Golay smoothing window."""
        value = self._smoothing_window_frames.value()
        minimum = self._smoothing_window_frames.minimum()
        maximum = self._smoothing_window_frames.maximum()
        if value < minimum:
            return minimum
        if value % 2 == 1:
            return value
        if value < maximum:
            return value + 1
        return value - 1


class ResponseProcessingOptionsWidget(QWidget):
    """Widget that exposes typed response-processing settings.

    Inputs: initial processing options and an optional change callback.
    Outputs: a Qt widget plus ``options()`` for reading the current typed
    settings before preview or persistence actions.
    """

    def __init__(
        self,
        options: ResponseProcessingOptions,
        *,
        on_change: Callable[[ResponseProcessingOptions], None] | None = None,
    ) -> None:
        """Create the response-processing option controls.

        Args:
            options: Initial processing settings.
            on_change: Optional callback receiving new typed settings whenever
                a GUI control changes.
        """
        super().__init__()
        self._on_change = on_change
        self._normalization_options = options.normalization
        self._smoothing_widget = SmoothingOptionsWidget(
            options.smoothing,
            on_change=lambda _options: self._emit_change(),
        )
        self._smoothing_method = self._smoothing_widget._smoothing_method
        self._smoothing_window_frames = self._smoothing_widget._smoothing_window_frames
        self._smoothing_polynomial_order = (
            self._smoothing_widget._smoothing_polynomial_order
        )
        self._low_pass_method = _combo_box(_LOW_PASS_METHOD_LABELS)
        self._low_pass_cutoff_hz = _double_spin_box(
            minimum=0.001,
            maximum=1_000_000.0,
            value=options.low_pass.cutoff_hz or 1.0,
            suffix=" Hz",
        )
        self._low_pass_order = _spin_box(
            minimum=1,
            maximum=16,
            value=options.low_pass.order,
        )
        self._correlation_reference = _combo_box(_CORRELATION_REFERENCE_LABELS)
        self._minimum_correlation = _double_spin_box(
            minimum=-1.0,
            maximum=1.0,
            value=options.correlation_filter.minimum_correlation,
            single_step=0.05,
        )
        start_seconds, stop_seconds = options.correlation_filter.window_seconds
        self._correlation_window_start = _double_spin_box(
            minimum=-1_000_000.0,
            maximum=1_000_000.0,
            value=0.0 if start_seconds is None else start_seconds,
            suffix=" s",
        )
        self._correlation_window_stop = _double_spin_box(
            minimum=-1_000_000.0,
            maximum=1_000_000.0,
            value=0.0 if stop_seconds is None else stop_seconds,
            suffix=" s",
        )
        self._correlation_window_has_stop = QCheckBox("Use stop")
        set_plot_control_width(self._correlation_window_has_stop)
        self._correlation_window_has_stop.setChecked(stop_seconds is not None)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._smoothing_widget)
        layout.addWidget(self._low_pass_group())
        layout.addWidget(self._correlation_group())
        self.setLayout(layout)

        _set_combo_data(self._low_pass_method, options.low_pass.method)
        _set_combo_data(
            self._correlation_reference,
            options.correlation_filter.reference,
        )
        self._connect_changes()
        self._refresh_enabled_state()

    def options(self) -> ResponseProcessingOptions:
        """Return the current typed response-processing options.

        Args:
            None.

        Returns:
            ``ResponseProcessingOptions`` built from the current controls.
        """
        low_pass_method = require_string_choice(
            str(self._low_pass_method.currentData()),
            name="low-pass method",
            allowed=_LOW_PASS_METHODS,
        )
        return ResponseProcessingOptions(
            smoothing=self._smoothing_widget.options(),
            low_pass=LowPassFilterOptions(
                method=low_pass_method,
                cutoff_hz=(
                    self._low_pass_cutoff_hz.value()
                    if low_pass_method == "butterworth"
                    else None
                ),
                order=self._low_pass_order.value(),
            ),
            normalization=self._normalization_options,
            correlation_filter=CorrelationFilterOptions(
                reference=require_string_choice(
                    str(self._correlation_reference.currentData()),
                    name="correlation reference",
                    allowed=_CORRELATION_REFERENCES,
                ),
                minimum_correlation=self._minimum_correlation.value(),
                window_seconds=(
                    self._correlation_window_start.value(),
                    self._correlation_window_stop.value()
                    if self._correlation_window_has_stop.isChecked()
                    else None,
                ),
            ),
        )

    def set_options(self, options: ResponseProcessingOptions) -> None:
        """Update controls from typed options without emitting changes.

        Args:
            options: Processing settings loaded from saved analysis output.

        Returns:
            None.

        Saved analysis reloads should update the visible controls, but they
        must not trigger a new preview computation. The loaded plot already
        reflects these settings.
        """
        blockers = [
            QSignalBlocker(self._low_pass_method),
            QSignalBlocker(self._low_pass_cutoff_hz),
            QSignalBlocker(self._low_pass_order),
            QSignalBlocker(self._correlation_reference),
            QSignalBlocker(self._minimum_correlation),
            QSignalBlocker(self._correlation_window_start),
            QSignalBlocker(self._correlation_window_stop),
            QSignalBlocker(self._correlation_window_has_stop),
        ]
        self._normalization_options = options.normalization
        self._smoothing_widget.set_options(options.smoothing)
        _set_combo_data(self._low_pass_method, options.low_pass.method)
        if options.low_pass.cutoff_hz is not None:
            self._low_pass_cutoff_hz.setValue(options.low_pass.cutoff_hz)
        self._low_pass_order.setValue(options.low_pass.order)
        _set_combo_data(
            self._correlation_reference,
            options.correlation_filter.reference,
        )
        self._minimum_correlation.setValue(
            options.correlation_filter.minimum_correlation,
        )
        start_seconds, stop_seconds = options.correlation_filter.window_seconds
        self._correlation_window_start.setValue(
            0.0 if start_seconds is None else start_seconds,
        )
        self._correlation_window_has_stop.setChecked(stop_seconds is not None)
        if stop_seconds is not None:
            self._correlation_window_stop.setValue(stop_seconds)
        del blockers
        self._refresh_enabled_state()

    def set_normalization_options(self, options: NormalizationOptions) -> None:
        """Store normalization settings controlled by the separate subsection.

        Args:
            options: ``NormalizationOptions`` carried by the parent response
                processing object.

        Returns:
            None.
        """
        self._normalization_options = options

    def set_correlation_window_stop_default(self, stop_seconds: float | None) -> None:
        """Set the inactive correlation-window stop value used when enabled.

        Args:
            stop_seconds: Default stop time in seconds, or ``None`` when no
                epoch-duration default is available.

        Returns:
            None.

        The checkbox still owns whether a stop is active. This only changes
        the latent value shown in the spin box, so enabling the stop starts
        from the shortest epoch duration instead of the invalid ``0 s`` value.
        """
        if self._correlation_window_has_stop.isChecked():
            return
        resolved_stop = 0.0 if stop_seconds is None else stop_seconds
        if not math.isfinite(resolved_stop):
            return
        start_seconds = self._correlation_window_start.value()
        if stop_seconds is not None and resolved_stop <= start_seconds:
            return
        blocker = QSignalBlocker(self._correlation_window_stop)
        self._correlation_window_stop.setValue(resolved_stop)
        del blocker

    def _low_pass_group(self) -> QGroupBox:
        """Create the low-pass control group."""
        group = QGroupBox("Low-pass filter")
        layout = plot_form_layout()
        layout.addRow("Method", self._low_pass_method)
        layout.addRow("Cutoff", self._low_pass_cutoff_hz)
        layout.addRow("Order", self._low_pass_order)
        group.setLayout(layout)
        return group

    def _correlation_group(self) -> QGroupBox:
        """Create the correlation-filter control group."""
        group = QGroupBox("Correlation filter")
        layout = plot_form_layout()
        layout.addRow("Reference", self._correlation_reference)
        layout.addRow("Minimum r", self._minimum_correlation)
        layout.addRow("Window start", self._correlation_window_start)
        layout.addRow("Window stop", self._correlation_window_stop)
        layout.addRow("", self._correlation_window_has_stop)
        group.setLayout(layout)
        return group

    def _connect_changes(self) -> None:
        """Connect control changes to state refresh and callback dispatch."""
        for combo in (self._low_pass_method, self._correlation_reference):
            combo.currentIndexChanged.connect(self._emit_change)
        for spin_box in (
            self._low_pass_cutoff_hz,
            self._low_pass_order,
            self._minimum_correlation,
            self._correlation_window_start,
            self._correlation_window_stop,
        ):
            spin_box.valueChanged.connect(self._emit_change)
        self._correlation_window_has_stop.stateChanged.connect(self._emit_change)

    def _emit_change(self, *_args: object) -> None:
        """Emit typed options after a GUI value changes."""
        self._refresh_enabled_state()
        if self._on_change is not None:
            self._on_change(self.options())

    def _refresh_enabled_state(self) -> None:
        """Enable parameters only when their processing method is active."""
        low_pass_enabled = self._low_pass_method.currentData() == "butterworth"
        self._low_pass_cutoff_hz.setEnabled(low_pass_enabled)
        self._low_pass_order.setEnabled(low_pass_enabled)
        correlation_enabled = self._correlation_reference.currentData() != "none"
        self._minimum_correlation.setEnabled(correlation_enabled)
        self._correlation_window_start.setEnabled(correlation_enabled)
        self._correlation_window_stop.setEnabled(
            correlation_enabled and self._correlation_window_has_stop.isChecked(),
        )
        self._correlation_window_has_stop.setEnabled(correlation_enabled)


def _combo_box(values: tuple[tuple[str, str], ...]) -> QComboBox:
    """Create one combo box with fixed text options."""
    combo_box = QComboBox()
    for label, value in values:
        combo_box.addItem(label, value)
    combo_box.setMaxVisibleItems(len(values))
    set_plot_dropdown_width(combo_box)
    return combo_box


def _set_combo_data(combo_box: QComboBox, value: str) -> None:
    """Set a combo box by stored item data when the item exists."""
    index = combo_box.findData(value)
    if index >= 0:
        combo_box.setCurrentIndex(index)


def _spin_box(*, minimum: int, maximum: int, value: int) -> QSpinBox:
    """Create one integer spin box."""
    spin_box = QSpinBox()
    spin_box.setRange(minimum, maximum)
    spin_box.setValue(value)
    set_plot_control_width(spin_box)
    return spin_box


def _minimum_savgol_window(polynomial_order: int) -> int:
    """Return the smallest odd Savitzky-Golay window for one polynomial order."""
    minimum = max(1, polynomial_order + 1)
    if minimum % 2 == 0:
        minimum += 1
    return minimum


def _double_spin_box(
    *,
    minimum: float,
    maximum: float,
    value: float,
    suffix: str = "",
    single_step: float = 0.1,
) -> QDoubleSpinBox:
    """Create one floating-point spin box."""
    spin_box = QDoubleSpinBox()
    spin_box.setRange(minimum, maximum)
    spin_box.setDecimals(2)
    spin_box.setSingleStep(single_step)
    spin_box.setValue(value)
    if suffix:
        spin_box.setSuffix(suffix)
    set_plot_control_width(spin_box)
    return spin_box
