"""Qt controls for selecting response-processing options.

Inputs: a current ``ResponseProcessingOptions`` object and an optional callback.
Outputs: a compact widget that returns typed GUI-independent processing options.

This module owns only GUI controls and value translation. The analysis package
owns validation, math, and persistence of the selected settings.
"""

from collections.abc import Callable

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

from twopy.analysis.response_processing import (
    CorrelationFilterOptions,
    CorrelationFilterReference,
    LowPassFilterMethod,
    LowPassFilterOptions,
    ResponseProcessingOptions,
    SmoothingMethod,
    SmoothingOptions,
)
from twopy.typing_guards import require_string_choice

__all__ = ["ResponseProcessingOptionsWidget"]

_SMOOTHING_METHODS: tuple[SmoothingMethod, ...] = (
    "none",
    "moving_average",
    "savgol",
)
_LOW_PASS_METHODS: tuple[LowPassFilterMethod, ...] = ("none", "butterworth")
_CORRELATION_REFERENCES: tuple[CorrelationFilterReference, ...] = (
    "none",
    "epoch_mean",
    "epoch_peak",
)


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
        self._smoothing_method = _combo_box(_SMOOTHING_METHODS)
        self._smoothing_window_frames = _spin_box(
            minimum=1,
            maximum=1001,
            value=options.smoothing.window_frames,
        )
        self._smoothing_polynomial_order = _spin_box(
            minimum=0,
            maximum=16,
            value=options.smoothing.polynomial_order,
        )
        self._low_pass_method = _combo_box(_LOW_PASS_METHODS)
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
        self._correlation_reference = _combo_box(_CORRELATION_REFERENCES)
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
        self._correlation_window_has_stop.setChecked(stop_seconds is not None)

        layout = QVBoxLayout()
        layout.addWidget(self._smoothing_group())
        layout.addWidget(self._low_pass_group())
        layout.addWidget(self._correlation_group())
        layout.addStretch(1)
        self.setLayout(layout)

        self._set_combo_value(self._smoothing_method, options.smoothing.method)
        self._set_combo_value(self._low_pass_method, options.low_pass.method)
        self._set_combo_value(
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
            self._low_pass_method.currentText(),
            name="low-pass method",
            allowed=_LOW_PASS_METHODS,
        )
        return ResponseProcessingOptions(
            smoothing=SmoothingOptions(
                method=require_string_choice(
                    self._smoothing_method.currentText(),
                    name="smoothing method",
                    allowed=_SMOOTHING_METHODS,
                ),
                window_frames=self._smoothing_window_frames.value(),
                polynomial_order=self._smoothing_polynomial_order.value(),
            ),
            low_pass=LowPassFilterOptions(
                method=low_pass_method,
                cutoff_hz=(
                    self._low_pass_cutoff_hz.value()
                    if low_pass_method == "butterworth"
                    else None
                ),
                order=self._low_pass_order.value(),
            ),
            correlation_filter=CorrelationFilterOptions(
                reference=require_string_choice(
                    self._correlation_reference.currentText(),
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

    def _smoothing_group(self) -> QGroupBox:
        """Create the smoothing control group."""
        group = QGroupBox("Smoothing")
        layout = QFormLayout()
        layout.addRow("Method", self._smoothing_method)
        layout.addRow("Window", self._smoothing_window_frames)
        layout.addRow("Order", self._smoothing_polynomial_order)
        group.setLayout(layout)
        return group

    def _low_pass_group(self) -> QGroupBox:
        """Create the low-pass control group."""
        group = QGroupBox("Low-pass filter")
        layout = QFormLayout()
        layout.addRow("Method", self._low_pass_method)
        layout.addRow("Cutoff", self._low_pass_cutoff_hz)
        layout.addRow("Order", self._low_pass_order)
        group.setLayout(layout)
        return group

    def _correlation_group(self) -> QGroupBox:
        """Create the correlation-filter control group."""
        group = QGroupBox("Correlation filter")
        layout = QFormLayout()
        layout.addRow("Reference", self._correlation_reference)
        layout.addRow("Minimum r", self._minimum_correlation)
        layout.addRow("Window start", self._correlation_window_start)
        layout.addRow("Window stop", self._correlation_window_stop)
        layout.addRow("", self._correlation_window_has_stop)
        group.setLayout(layout)
        return group

    def _connect_changes(self) -> None:
        """Connect control changes to state refresh and callback dispatch."""
        for combo in (
            self._smoothing_method,
            self._low_pass_method,
            self._correlation_reference,
        ):
            combo.currentTextChanged.connect(self._emit_change)
        for spin_box in (
            self._smoothing_window_frames,
            self._smoothing_polynomial_order,
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
        if (
            self._smoothing_method.currentText() == "savgol"
            and self._smoothing_window_frames.value() % 2 == 0
        ):
            self._smoothing_window_frames.setValue(
                self._nearest_odd_smoothing_window(),
            )
        self._smoothing_window_frames.setEnabled(
            self._smoothing_method.currentText() in {"moving_average", "savgol"},
        )
        self._smoothing_polynomial_order.setEnabled(
            self._smoothing_method.currentText() == "savgol",
        )
        low_pass_enabled = self._low_pass_method.currentText() == "butterworth"
        self._low_pass_cutoff_hz.setEnabled(low_pass_enabled)
        self._low_pass_order.setEnabled(low_pass_enabled)
        correlation_enabled = self._correlation_reference.currentText() != "none"
        self._minimum_correlation.setEnabled(correlation_enabled)
        self._correlation_window_start.setEnabled(correlation_enabled)
        self._correlation_window_stop.setEnabled(
            correlation_enabled and self._correlation_window_has_stop.isChecked(),
        )
        self._correlation_window_has_stop.setEnabled(correlation_enabled)

    def _nearest_odd_smoothing_window(self) -> int:
        """Return the nearest valid odd smoothing window."""
        value = self._smoothing_window_frames.value()
        if value < self._smoothing_window_frames.maximum():
            return value + 1
        return value - 1

    def _set_combo_value(self, combo_box: QComboBox, value: str) -> None:
        """Set a combo box by text when the item exists."""
        index = combo_box.findText(value)
        if index >= 0:
            combo_box.setCurrentIndex(index)


def _combo_box(values: tuple[str, ...]) -> QComboBox:
    """Create one combo box with fixed text options."""
    combo_box = QComboBox()
    combo_box.addItems(values)
    return combo_box


def _spin_box(*, minimum: int, maximum: int, value: int) -> QSpinBox:
    """Create one integer spin box."""
    spin_box = QSpinBox()
    spin_box.setRange(minimum, maximum)
    spin_box.setValue(value)
    return spin_box


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
    spin_box.setDecimals(3)
    spin_box.setSingleStep(single_step)
    spin_box.setValue(value)
    if suffix:
        spin_box.setSuffix(suffix)
    return spin_box
