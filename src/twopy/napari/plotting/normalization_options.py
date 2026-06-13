"""Qt controls for selecting response normalization options.

Inputs: a current ``NormalizationOptions`` object and optional epoch choices.
Outputs: a compact widget that returns typed GUI-independent normalization
settings.

This module owns only GUI controls and value translation. Core analysis owns
the response normalization math and validation.
"""

from collections.abc import Callable

from qtpy.QtCore import QSignalBlocker
from qtpy.QtWidgets import QCheckBox, QComboBox, QGroupBox, QVBoxLayout, QWidget

from twopy.analysis.response_processing import NormalizationOptions
from twopy.analysis.trials import is_baseline_epoch_name
from twopy.napari.plotting.form_controls import (
    plot_form_layout,
    set_plot_control_width,
    set_plot_dropdown_width,
)

__all__ = ["NormalizationOptionsWidget", "default_normalization_epoch_number"]

_NORMALIZE_TO_EPOCH_ABS_PEAK_WIDTH = 230


class NormalizationOptionsWidget(QWidget):
    """Widget that exposes typed response normalization settings.

    Inputs: initial normalization options and an optional change callback.
    Outputs: a Qt widget plus ``options()`` for reading current typed settings
    before preview or persistence actions.
    """

    def __init__(
        self,
        options: NormalizationOptions,
        *,
        on_change: Callable[[NormalizationOptions], None] | None = None,
    ) -> None:
        """Create the response normalization controls.

        Args:
            options: Initial normalization settings.
            on_change: Optional callback receiving new typed settings whenever
                a GUI control changes.
        """
        super().__init__()
        self._on_change = on_change
        self._normalize_to_epoch_abs_peak = QCheckBox(
            "Normalize to strongest response",
        )
        set_plot_control_width(
            self._normalize_to_epoch_abs_peak,
            width=_NORMALIZE_TO_EPOCH_ABS_PEAK_WIDTH,
        )
        self._normalize_to_epoch_abs_peak.setChecked(
            options.method == "epoch_abs_peak",
        )
        self._epoch = QComboBox()
        set_plot_dropdown_width(self._epoch)
        self._epoch_name_values: dict[int, str | None] = {}

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._normalization_group())
        self.setLayout(layout)

        self.set_epoch_choices(
            {options.epoch_number: options.epoch_name}
            if options.epoch_number is not None and options.epoch_name is not None
            else {},
            selected_epoch_number=options.epoch_number,
        )
        self._connect_changes()
        self._refresh_enabled_state()

    def options(self) -> NormalizationOptions:
        """Return the current typed normalization options.

        Args:
            None.

        Returns:
            ``NormalizationOptions`` built from the current controls.
        """
        epoch_number = self._selected_epoch_number()
        epoch_name = (
            self._epoch_name_values.get(epoch_number)
            if epoch_number is not None
            else None
        )
        return NormalizationOptions(
            method=(
                "epoch_abs_peak"
                if self._normalize_to_epoch_abs_peak.isChecked()
                else "none"
            ),
            epoch_number=epoch_number,
            epoch_name=epoch_name,
        )

    def set_epoch_choices(
        self,
        epoch_names: dict[int, str],
        *,
        selected_epoch_number: int | None = None,
    ) -> None:
        """Update the normalization epoch dropdown from recording metadata.

        Args:
            epoch_names: Mapping from stimulus epoch numbers to display names.
            selected_epoch_number: Optional epoch number to select after
                rebuilding the dropdown.

        Returns:
            None.
        """
        selected = selected_epoch_number or default_normalization_epoch_number(
            epoch_names,
        )
        self._epoch_name_values = dict(sorted(epoch_names.items()))
        if selected is not None and selected not in self._epoch_name_values:
            self._epoch_name_values[selected] = None

        blocker = QSignalBlocker(self._epoch)
        self._epoch.clear()
        for epoch_number, epoch_name in sorted(self._epoch_name_values.items()):
            self._epoch.addItem(
                _epoch_choice_label(epoch_number, epoch_name),
                epoch_number,
            )
        if selected is not None:
            self._set_combo_data(self._epoch, selected)
        del blocker

    def set_options(self, options: NormalizationOptions) -> None:
        """Update controls from typed options without emitting changes.

        Args:
            options: Normalization settings loaded from saved analysis output.

        Returns:
            None.
        """
        blockers = [
            QSignalBlocker(self._normalize_to_epoch_abs_peak),
            QSignalBlocker(self._epoch),
        ]
        epoch_names = {
            epoch_number: epoch_name
            for epoch_number, epoch_name in self._epoch_name_values.items()
            if epoch_name is not None
        }
        if options.epoch_number is not None and options.epoch_name is not None:
            epoch_names[options.epoch_number] = options.epoch_name
        self.set_epoch_choices(
            epoch_names,
            selected_epoch_number=options.epoch_number,
        )
        self._normalize_to_epoch_abs_peak.setChecked(
            options.method == "epoch_abs_peak",
        )
        del blockers
        self._refresh_enabled_state()

    def _normalization_group(self) -> QGroupBox:
        """Create the normalization control group."""
        group = QGroupBox("Normalization")
        layout = plot_form_layout()
        layout.setVerticalSpacing(8)
        layout.addRow("", self._normalize_to_epoch_abs_peak)
        layout.addRow("Epoch", self._epoch)
        group.setLayout(layout)
        return group

    def _connect_changes(self) -> None:
        """Connect control changes to state refresh and callback dispatch."""
        self._normalize_to_epoch_abs_peak.stateChanged.connect(self._emit_change)
        self._epoch.currentIndexChanged.connect(self._emit_change)

    def _emit_change(self, *_args: object) -> None:
        """Emit typed options after a GUI value changes."""
        self._refresh_enabled_state()
        if self._on_change is not None:
            self._on_change(self.options())

    def _refresh_enabled_state(self) -> None:
        """Enable epoch selection only when normalization is active."""
        self._epoch.setEnabled(self._normalize_to_epoch_abs_peak.isChecked())

    def _selected_epoch_number(self) -> int | None:
        """Return the selected epoch number, if any."""
        value = self._epoch.currentData()
        if isinstance(value, int):
            return value
        return None

    def _set_combo_data(self, combo_box: QComboBox, value: int) -> None:
        """Set a combo box by stored item data when the item exists."""
        index = combo_box.findData(value)
        if index >= 0:
            combo_box.setCurrentIndex(index)


def default_normalization_epoch_number(epoch_names: dict[int, str]) -> int | None:
    """Return the first non-baseline epoch number for response normalization.

    Args:
        epoch_names: Mapping from stimulus epoch numbers to display names.

    Returns:
        First non-gray/interleave epoch number, or the first epoch number when
        every named epoch looks like baseline.
    """
    if len(epoch_names) == 0:
        return None
    for epoch_number, epoch_name in sorted(epoch_names.items()):
        if not is_baseline_epoch_name(epoch_name):
            return epoch_number
    return min(epoch_names)


def _epoch_choice_label(epoch_number: int, epoch_name: str | None) -> str:
    """Return a compact epoch dropdown label."""
    if epoch_name is None:
        return f"Epoch {epoch_number}"
    return f"{epoch_number}: {epoch_name}"
