"""ROI generation controls for the twopy napari ROIs tab.

Inputs: loaded recording metadata, calibration rows, and user grid settings.
Outputs: small Qt controls that report requested generated ROI templates.

This module owns only option-panel widgets. Core ROI creation and calibration
resolution stay in ``twopy.roi_extraction`` and ``twopy.pixel_calibration``.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
)

from twopy.converted import RecordingData
from twopy.pixel_calibration import PixelCalibrationRow

__all__ = [
    "RoiGenerationControls",
    "RoiGenerationOptions",
    "RoiGenerationUnits",
]

RoiGenerationUnits = Literal["pixels", "microns"]


@dataclass(frozen=True)
class RoiGenerationOptions:
    """Requested ROI grid-generation settings from the ROIs tab.

    Args:
        units: Whether the grid size is specified in pixels or microns.
        grid_size_pixels: Pixel grid width used when ``units`` is ``pixels``.
        micron_grid_size: Physical grid width used when ``units`` is
            ``microns``.
        rig: Calibration rig key.
        mode: Calibration scanner mode key.
        scanner: Calibration scanner family key.
        zoom: Requested zoom setting.
        allow_extrapolation: Whether calibration may extrapolate outside the
            measured zoom range.
    """

    units: RoiGenerationUnits
    grid_size_pixels: int
    micron_grid_size: float
    rig: str
    mode: int
    scanner: str
    zoom: float
    allow_extrapolation: bool


class RoiGenerationControls(QGroupBox):
    """Create ROIs-tab controls for generated grid templates.

    Args:
        calibrations: Calibration rows used to populate rig/mode/scanner
            choices.
        on_generate: Callback invoked when the user clicks Create grid.

    The widget auto-fills zoom from converted acquisition metadata, but
    calibration rig/mode/scanner remain explicit choices because converted
    files do not yet persist those keys as typed calibration fields.
    """

    def __init__(
        self,
        calibrations: tuple[PixelCalibrationRow, ...],
        *,
        on_generate: Callable[[RoiGenerationOptions], None],
    ) -> None:
        """Create the grid-generation control group.

        Args:
            calibrations: Calibration rows used to populate dropdowns.
            on_generate: Callback invoked with current options.

        Returns:
            None.
        """
        super().__init__("Create grid ROIs")
        self._calibrations = calibrations
        self._on_generate = on_generate

        self._units = QComboBox()
        self._units.addItem("pixels", "pixels")
        self._units.addItem("microns", "microns")
        self._pixel_grid_size = QSpinBox()
        self._pixel_grid_size.setRange(1, 2048)
        self._pixel_grid_size.setValue(16)
        self._micron_grid_size = QDoubleSpinBox()
        self._micron_grid_size.setRange(0.001, 10000.0)
        self._micron_grid_size.setDecimals(3)
        self._micron_grid_size.setValue(10.0)
        self._rig = QComboBox()
        self._mode = QComboBox()
        self._scanner = QComboBox()
        self._zoom = QDoubleSpinBox()
        self._zoom.setRange(0.001, 10000.0)
        self._zoom.setDecimals(3)
        self._allow_extrapolation = QCheckBox("Allow extrapolation")
        self._status = QLabel("No recording loaded.")
        self._status.setWordWrap(True)
        self._create_button = QPushButton("Create grid")

        self._populate_calibration_choices()
        self._units.currentIndexChanged.connect(self._sync_units)
        self._create_button.clicked.connect(self._generate)

        layout = QFormLayout()
        layout.addRow("Units", self._units)
        layout.addRow("Pixels", self._pixel_grid_size)
        layout.addRow("Microns", self._micron_grid_size)
        layout.addRow("Rig", self._rig)
        layout.addRow("Mode", self._mode)
        layout.addRow("Scanner", self._scanner)
        layout.addRow("Zoom", self._zoom)
        layout.addRow("", self._allow_extrapolation)
        layout.addRow("", self._create_button)
        layout.addRow("", self._status)
        self.setLayout(layout)
        self._sync_units()

    def set_recording(self, recording: RecordingData | None) -> None:
        """Update controls from the selected recording.

        Args:
            recording: Loaded converted recording, or ``None`` when no
                recording is active.

        Returns:
            None.
        """
        if recording is None:
            self._status.setText("No recording loaded.")
            return
        zoom = _metadata_float(recording.acquisition_metadata, "acq.zoomFactor")
        if zoom is None:
            self._status.setText("Zoom missing from converted metadata.")
            return
        self._zoom.setValue(zoom)
        self._status.setText(f"Zoom from metadata: {zoom:g}")

    def options(self) -> RoiGenerationOptions:
        """Return the current grid-generation options.

        Returns:
            Plain generation options.
        """
        return RoiGenerationOptions(
            units=_units_value(self._units),
            grid_size_pixels=self._pixel_grid_size.value(),
            micron_grid_size=self._micron_grid_size.value(),
            rig=self._rig.currentText(),
            mode=int(self._mode.currentData()),
            scanner=self._scanner.currentText(),
            zoom=self._zoom.value(),
            allow_extrapolation=self._allow_extrapolation.isChecked(),
        )

    def set_status(self, text: str) -> None:
        """Show generation status in the ROIs tab.

        Args:
            text: User-facing status text.

        Returns:
            None.
        """
        self._status.setText(text)

    def _populate_calibration_choices(self) -> None:
        """Populate calibration dropdowns from available rows."""
        for rig in _unique_text(row.rig for row in self._calibrations):
            self._rig.addItem(rig)
        for mode in sorted({row.mode for row in self._calibrations}):
            self._mode.addItem(str(mode), mode)
        for scanner in _unique_text(row.scanner for row in self._calibrations):
            self._scanner.addItem(scanner)

    def _sync_units(self) -> None:
        """Enable inputs that apply to the selected grid-size unit."""
        uses_microns = _units_value(self._units) == "microns"
        self._pixel_grid_size.setEnabled(not uses_microns)
        self._micron_grid_size.setEnabled(uses_microns)
        self._rig.setEnabled(uses_microns)
        self._mode.setEnabled(uses_microns)
        self._scanner.setEnabled(uses_microns)
        self._zoom.setEnabled(uses_microns)
        self._allow_extrapolation.setEnabled(uses_microns)

    def _generate(self) -> None:
        """Call the owner with the current generation options."""
        self._on_generate(self.options())


def _metadata_float(metadata: dict[str, object], key: str) -> float | None:
    """Read one optional numeric metadata value.

    Args:
        metadata: Converted metadata mapping.
        key: Metadata key to read.

    Returns:
        Float value, or ``None`` when absent or nonnumeric.
    """
    value = metadata.get(key)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _unique_text(values: Iterable[object]) -> tuple[str, ...]:
    """Return sorted unique text values from an iterable.

    Args:
        values: Iterable object containing text-like values.

    Returns:
        Sorted unique string values.
    """
    return tuple(sorted({str(value) for value in values}))


def _units_value(combo: QComboBox) -> RoiGenerationUnits:
    """Return selected unit data from a combo box.

    Args:
        combo: Units combo box.

    Returns:
        ``pixels`` or ``microns``.
    """
    value = combo.currentData()
    if value == "microns":
        return "microns"
    return "pixels"
