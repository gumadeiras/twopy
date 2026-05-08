"""Qt controls for response-trial time-window settings.

Inputs: typed ``ResponseWindowOptions`` and an optional recording-derived
interleave-duration limit.
Outputs: a compact Plot-tab widget that returns GUI-independent options.

The widget owns only GUI state. Analysis code resolves the final pre/post
seconds from these options and the selected recording.
"""

from collections.abc import Callable

from qtpy.QtCore import QSignalBlocker
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.response_window_options import ResponseWindowOptions
from twopy.napari.plotting.form_controls import (
    plot_form_layout,
    set_plot_control_width,
)

__all__ = ["ResponseWindowOptionsWidget"]

_UNCAPPED_WINDOW_SECONDS = 1_000_000.0


class ResponseWindowOptionsWidget(QWidget):
    """Widget that exposes response-window settings.

    Inputs: initial response-window options, an optional max duration, and an
    optional change callback.
    Outputs: a Qt widget plus ``options()`` for reading typed settings before
    preview or persistence actions.
    """

    def __init__(
        self,
        options: ResponseWindowOptions,
        *,
        max_window_seconds: float | None = None,
        on_change: Callable[[ResponseWindowOptions], None] | None = None,
    ) -> None:
        """Create the response-window controls.

        Args:
            options: Initial response-window settings.
            max_window_seconds: Optional gray/interleave duration used as the
                maximum manual pre/post value.
            on_change: Optional callback receiving typed settings whenever a
                GUI control changes.
        """
        super().__init__()
        self._on_change = on_change
        self._auto = QCheckBox("Auto")
        set_plot_control_width(self._auto)
        self._auto.setChecked(options.auto)
        self._pre_seconds = _window_spin_box(options.pre_window_seconds)
        self._post_seconds = _window_spin_box(options.post_window_seconds)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._response_window_group())
        self.setLayout(layout)

        self.set_max_window_seconds(max_window_seconds)
        self._connect_changes()
        self._refresh_enabled_state()

    def options(self) -> ResponseWindowOptions:
        """Return the current typed response-window options.

        Args:
            None.

        Returns:
            ``ResponseWindowOptions`` built from the current controls.
        """
        return ResponseWindowOptions(
            auto=self._auto.isChecked(),
            pre_window_seconds=self._pre_seconds.value(),
            post_window_seconds=self._post_seconds.value(),
        )

    def set_options(self, options: ResponseWindowOptions) -> None:
        """Update controls from typed options without emitting changes.

        Args:
            options: Response-window settings to show.

        Returns:
            None.
        """
        blockers = [
            QSignalBlocker(self._auto),
            QSignalBlocker(self._pre_seconds),
            QSignalBlocker(self._post_seconds),
        ]
        self._auto.setChecked(options.auto)
        self._pre_seconds.setValue(options.pre_window_seconds)
        self._post_seconds.setValue(options.post_window_seconds)
        del blockers
        self._refresh_enabled_state()

    def set_max_window_seconds(self, max_window_seconds: float | None) -> None:
        """Set the maximum manual pre/post duration.

        Args:
            max_window_seconds: Gray/interleave duration in seconds, or
                ``None`` when no named interleave duration is available.

        Returns:
            None.
        """
        maximum = (
            _UNCAPPED_WINDOW_SECONDS
            if max_window_seconds is None
            else max(0.0, float(max_window_seconds))
        )
        blockers = [
            QSignalBlocker(self._pre_seconds),
            QSignalBlocker(self._post_seconds),
        ]
        self._pre_seconds.setMaximum(maximum)
        self._post_seconds.setMaximum(maximum)
        del blockers

    def _response_window_group(self) -> QGroupBox:
        """Create the response-window control group."""
        group = QGroupBox("Response window")
        layout = plot_form_layout()
        layout.addRow("", self._auto)
        layout.addRow("Pre stim.", self._pre_seconds)
        layout.addRow("Post stim.", self._post_seconds)
        group.setLayout(layout)
        return group

    def _connect_changes(self) -> None:
        """Connect controls to state refresh and callback dispatch."""
        self._auto.stateChanged.connect(self._emit_change)
        self._pre_seconds.valueChanged.connect(self._emit_change)
        self._post_seconds.valueChanged.connect(self._emit_change)

    def _emit_change(self, *_args: object) -> None:
        """Emit typed options after a GUI value changes."""
        self._refresh_enabled_state()
        if self._on_change is not None:
            self._on_change(self.options())

    def _refresh_enabled_state(self) -> None:
        """Enable manual controls only when automatic windowing is off."""
        manual = not self._auto.isChecked()
        self._pre_seconds.setEnabled(manual)
        self._post_seconds.setEnabled(manual)


def _window_spin_box(value: float) -> QDoubleSpinBox:
    """Create one response-window seconds input."""
    spin_box = QDoubleSpinBox()
    spin_box.setRange(0.0, _UNCAPPED_WINDOW_SECONDS)
    spin_box.setDecimals(1)
    spin_box.setSingleStep(0.1)
    spin_box.setSuffix(" s")
    set_plot_control_width(spin_box)
    spin_box.setValue(value)
    return spin_box
