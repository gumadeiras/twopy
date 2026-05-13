"""ROI generation controls for the twopy napari ROIs tab.

Inputs: loaded recording metadata, calibration rows, and user ROI mode settings.
Outputs: small Qt controls that report requested generated ROI templates.

This module owns only option-panel widgets. Core ROI creation and calibration
resolution stay in ``twopy.roi_extraction`` and ``twopy.pixel_calibration``.
"""

from collections.abc import Callable, Iterable

from qtpy.QtCore import QSignalBlocker
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from twopy.converted import RecordingData
from twopy.napari.plotting.roi_generation.options import (
    RoiGenerationMode,
    RoiGenerationOptions,
    RoiGenerationUnits,
)
from twopy.pixel_calibration import PixelCalibrationRow
from twopy.pixel_calibration_profiles import (
    DEFAULT_PIXEL_CALIBRATION_PROFILE_PATH,
    PixelCalibrationGroup,
    PixelCalibrationProfile,
    load_pixel_calibration_profile_mappings,
    resolve_pixel_calibration_profile,
    select_pixel_calibration_group,
)

__all__ = [
    "RoiGenerationControls",
    "RoiGenerationMode",
    "RoiGenerationOptions",
    "RoiGenerationUnits",
]


class RoiGenerationControls(QGroupBox):
    """Create ROIs-tab controls for manual and generated ROI modes.

    Args:
        calibrations: Calibration rows used to populate rig/mode/scanner
            choices.
        on_generate: Callback invoked when the user clicks a generated-ROI
            action.

    Manual mode is the default and leaves napari Labels editing unchanged.
    Generated modes collect only typed options and delegate creation to the
    owner callback.
    """

    def __init__(
        self,
        calibrations: tuple[PixelCalibrationRow, ...],
        *,
        on_generate: Callable[[RoiGenerationOptions], None],
    ) -> None:
        """Create the ROI mode control group.

        Args:
            calibrations: Calibration rows used to populate dropdowns.
            on_generate: Callback invoked with current options.

        Returns:
            None.
        """
        super().__init__("ROI mode")
        self._calibrations = calibrations
        self._profile_mappings = load_pixel_calibration_profile_mappings(
            DEFAULT_PIXEL_CALIBRATION_PROFILE_PATH,
        )
        self._on_generate = on_generate
        self._recording_loaded = False
        self._loaded_zoom: float | None = None
        self._calibration_profile_status: str | None = None

        self._roi_mode = QComboBox()
        self._roi_mode.addItem("manual", "manual")
        self._roi_mode.addItem("grid", "grid")
        self._roi_mode.addItem("watershed", "watershed")
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
        self._watershed_min_pixels = QSpinBox()
        self._watershed_min_pixels.setRange(1, 1_000_000)
        self._watershed_min_pixels.setValue(1)
        self._watershed_smoothing_sigma = QDoubleSpinBox()
        self._watershed_smoothing_sigma.setRange(0.0, 100.0)
        self._watershed_smoothing_sigma.setDecimals(3)
        self._watershed_smoothing_sigma.setValue(0.0)
        self._status = QLabel("No recording loaded.")
        self._status.setWordWrap(True)
        self._create_button = QPushButton("Create ROIs")

        self._populate_calibration_choices()
        self._roi_mode.currentIndexChanged.connect(self._sync_mode)
        self._units.currentIndexChanged.connect(self._sync_units)
        self._rig.currentIndexChanged.connect(self._sync_calibration_mode_choices)
        self._mode.currentIndexChanged.connect(self._sync_calibration_scanner_choices)
        self._create_button.clicked.connect(self._generate)

        self._form_layout = QFormLayout()
        self._form_layout.addRow("Mode", self._roi_mode)
        self._form_layout.addRow("Units", self._units)
        self._form_layout.addRow("Pixels", self._pixel_grid_size)
        self._form_layout.addRow("Microns", self._micron_grid_size)
        self._form_layout.addRow("Rig", self._rig)
        self._form_layout.addRow("Cal mode", self._mode)
        self._form_layout.addRow("Scanner", self._scanner)
        self._form_layout.addRow("Zoom", self._zoom)
        self._form_layout.addRow("", self._allow_extrapolation)
        self._form_layout.addRow("Min pixels", self._watershed_min_pixels)
        self._form_layout.addRow("Smoothing", self._watershed_smoothing_sigma)
        self._form_layout.addRow("", self._create_button)
        self._form_layout.addRow("", self._status)
        self.setLayout(self._form_layout)
        self._sync_mode()

    def set_recording(self, recording: RecordingData | None) -> None:
        """Update controls from the selected recording.

        Args:
            recording: Loaded converted recording, or ``None`` when no
                recording is active.

        Returns:
            None.
        """
        if recording is None:
            self._recording_loaded = False
            self._loaded_zoom = None
            self._calibration_profile_status = None
            self._status.setText("No recording loaded.")
            self._sync_mode()
            return
        self._recording_loaded = True
        self._apply_calibration_profile(recording)
        zoom = _metadata_float(recording.acquisition_metadata, "acq.zoomFactor")
        if zoom is None:
            self._loaded_zoom = None
            self._sync_mode()
            return
        self._loaded_zoom = zoom
        self._zoom.setValue(zoom)
        self._sync_mode()

    def options(self) -> RoiGenerationOptions:
        """Return the current ROI-generation options.

        Returns:
            Plain generation options.
        """
        return RoiGenerationOptions(
            roi_mode=_roi_mode_value(self._roi_mode),
            units=_units_value(self._units),
            grid_size_pixels=self._pixel_grid_size.value(),
            micron_grid_size=self._micron_grid_size.value(),
            rig=self._rig.currentText(),
            calibration_mode=int(self._mode.currentData()),
            scanner=self._scanner.currentText(),
            zoom=self._zoom.value(),
            allow_extrapolation=self._allow_extrapolation.isChecked(),
            watershed_min_pixels=self._watershed_min_pixels.value(),
            watershed_smoothing_sigma=self._watershed_smoothing_sigma.value(),
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
        self._sync_calibration_mode_choices()

    def _sync_calibration_mode_choices(self) -> None:
        """Keep mode choices valid for the selected calibration rig."""
        rig = self._rig.currentText()
        modes = tuple(
            (str(mode), mode)
            for mode in sorted(
                {row.mode for row in self._calibrations if row.rig == rig},
            )
        )
        _replace_combo_items(
            self._mode,
            modes,
            previous=self._mode.currentData(),
        )
        self._sync_calibration_scanner_choices()

    def _sync_calibration_scanner_choices(self) -> None:
        """Keep scanner choices valid for the selected rig and mode."""
        rig = self._rig.currentText()
        mode = self._mode.currentData()
        scanners = tuple(
            (scanner, scanner)
            for scanner in _unique_text(
                row.scanner
                for row in self._calibrations
                if row.rig == rig and row.mode == mode
            )
        )
        _replace_combo_items(
            self._scanner,
            scanners,
            previous=self._scanner.currentText(),
        )

    def _apply_calibration_profile(self, recording: RecordingData) -> None:
        """Prefill calibration choices when metadata identifies one group.

        Args:
            recording: Loaded converted recording.

        Returns:
            None.
        """
        profile = resolve_pixel_calibration_profile(
            recording.acquisition_metadata,
            recording.run_metadata,
            mappings=self._profile_mappings,
        )
        group = select_pixel_calibration_group(profile, self._calibrations)
        if group is not None:
            self._select_calibration_group(group)
            self._calibration_profile_status = (
                "Auto-selected calibration: "
                f"{group.rig} mode {group.mode} {group.scanner} "
                f"({_profile_evidence_text(profile)})."
            )
            return

        self._calibration_profile_status = _partial_profile_status(profile)

    def _select_calibration_group(self, group: PixelCalibrationGroup) -> None:
        """Select one measured calibration group in dependent dropdowns.

        Args:
            group: Unique measured calibration group.

        Returns:
            None.
        """
        if _set_combo_text(self._rig, group.rig):
            self._sync_calibration_mode_choices()
        if _set_combo_data(self._mode, group.mode):
            self._sync_calibration_scanner_choices()
        _set_combo_text(self._scanner, group.scanner)

    def _sync_mode(self) -> None:
        """Show and enable controls for the selected ROI mode."""
        roi_mode = _roi_mode_value(self._roi_mode)
        uses_grid = roi_mode == "grid"
        uses_watershed = roi_mode == "watershed"
        for widget in (
            self._units,
            self._pixel_grid_size,
            self._micron_grid_size,
            self._rig,
            self._mode,
            self._scanner,
            self._zoom,
            self._allow_extrapolation,
        ):
            _set_form_row_visible(self._form_layout, widget, uses_grid)
        for widget in (self._watershed_min_pixels, self._watershed_smoothing_sigma):
            _set_form_row_visible(self._form_layout, widget, uses_watershed)
        self._create_button.setVisible(roi_mode != "manual")
        self._create_button.setEnabled(roi_mode != "manual")
        if roi_mode == "grid":
            self._create_button.setText("Create grid")
        elif roi_mode == "watershed":
            self._create_button.setText("Create watershed")
        else:
            self._create_button.setText("Create ROIs")
        self._sync_units()
        self._set_mode_status()

    def _sync_units(self) -> None:
        """Enable inputs that apply to the selected grid-size unit."""
        uses_grid = _roi_mode_value(self._roi_mode) == "grid"
        uses_microns = uses_grid and _units_value(self._units) == "microns"
        self._pixel_grid_size.setEnabled(uses_grid and not uses_microns)
        self._micron_grid_size.setEnabled(uses_microns)
        self._rig.setEnabled(uses_microns)
        self._mode.setEnabled(uses_microns)
        self._scanner.setEnabled(uses_microns)
        self._zoom.setEnabled(uses_microns)
        self._allow_extrapolation.setEnabled(uses_microns)

    def _set_mode_status(self) -> None:
        """Show the default status text for the selected mode."""
        if not self._recording_loaded:
            self._status.setText("No recording loaded.")
            return
        roi_mode = _roi_mode_value(self._roi_mode)
        if roi_mode == "manual":
            self._status.setText("Manual mode: draw or edit Labels ROIs.")
        elif roi_mode == "grid" and self._loaded_zoom is None:
            self._status.setText("Zoom missing from converted metadata.")
        elif roi_mode == "grid" and self._calibration_profile_status is not None:
            self._status.setText(
                f"Zoom from metadata: {self._loaded_zoom:g}. "
                f"{self._calibration_profile_status}",
            )
        elif roi_mode == "grid":
            self._status.setText(f"Zoom from metadata: {self._loaded_zoom:g}")
        else:
            self._status.setText("Watershed mode uses the converted mean image.")

    def _generate(self) -> None:
        """Call the owner with the current generation options."""
        if _roi_mode_value(self._roi_mode) == "manual":
            self._set_mode_status()
            return
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


def _set_form_row_visible(layout: QFormLayout, field: QWidget, visible: bool) -> None:
    """Set visibility for a form field and its label.

    Args:
        layout: Form layout containing the field.
        field: Field widget whose row should be shown or hidden.
        visible: Desired visibility state.

    Returns:
        None.
    """
    label = layout.labelForField(field)
    if label is not None:
        label.setVisible(visible)
    field.setVisible(visible)


def _replace_combo_items(
    combo: QComboBox,
    items: tuple[tuple[str, object], ...],
    *,
    previous: object,
) -> None:
    """Replace combo items while preserving a still-valid selection.

    Args:
        combo: Combo box to update.
        items: ``(label, data)`` pairs to display.
        previous: Previously selected item data or text.

    Returns:
        None.
    """
    blocker = QSignalBlocker(combo)
    try:
        combo.clear()
        selected_index = 0
        for index, (label, data) in enumerate(items):
            combo.addItem(label, data)
            if data == previous or label == previous:
                selected_index = index
        if len(items) > 0:
            combo.setCurrentIndex(selected_index)
    finally:
        del blocker


def _set_combo_text(combo: QComboBox, text: str) -> bool:
    """Select a combo item by display text.

    Args:
        combo: Combo box to update.
        text: Desired display text.

    Returns:
        Whether the selection changed.
    """
    index = combo.findText(text)
    if index < 0 or index == combo.currentIndex():
        return False
    combo.setCurrentIndex(index)
    return True


def _set_combo_data(combo: QComboBox, data: object) -> bool:
    """Select a combo item by user data.

    Args:
        combo: Combo box to update.
        data: Desired item data.

    Returns:
        Whether the selection changed.
    """
    index = combo.findData(data)
    if index < 0 or index == combo.currentIndex():
        return False
    combo.setCurrentIndex(index)
    return True


def _partial_profile_status(profile: PixelCalibrationProfile) -> str | None:
    """Return status text for profile evidence that cannot select a group.

    Args:
        profile: Partial calibration profile.

    Returns:
        User-facing status text, or ``None`` when no profile fields were found.
    """
    fields = _profile_fields_text(profile)
    if fields == "":
        return None
    if (
        profile.rig is not None
        and profile.mode is not None
        and profile.scanner is not None
    ):
        return (
            f"Detected calibration profile {fields} "
            f"({_profile_evidence_text(profile)}), but no measured calibration rows "
            "match it; choose a measured group manually."
        )
    return (
        f"Detected calibration profile {fields} "
        f"({_profile_evidence_text(profile)}); choose remaining fields manually."
    )


def _profile_fields_text(profile: PixelCalibrationProfile) -> str:
    """Format known profile fields for status text.

    Args:
        profile: Calibration profile.

    Returns:
        Comma-separated known fields.
    """
    fields: list[str] = []
    if profile.rig is not None:
        fields.append(f"rig {profile.rig}")
    if profile.mode is not None:
        fields.append(f"mode {profile.mode}")
    if profile.scanner is not None:
        fields.append(f"scanner {profile.scanner}")
    return ", ".join(fields)


def _profile_evidence_text(profile: PixelCalibrationProfile) -> str:
    """Format profile evidence for compact UI status.

    Args:
        profile: Calibration profile.

    Returns:
        Human-readable evidence summary.
    """
    if len(profile.evidence) == 0:
        return "no metadata evidence"
    return ", ".join(profile.evidence)


def _roi_mode_value(combo: QComboBox) -> RoiGenerationMode:
    """Return selected ROI mode data from a combo box.

    Args:
        combo: ROI mode combo box.

    Returns:
        Selected ROI mode.
    """
    value = combo.currentData()
    if value == "grid":
        return "grid"
    if value == "watershed":
        return "watershed"
    return "manual"


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
