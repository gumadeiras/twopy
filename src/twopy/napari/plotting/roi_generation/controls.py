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
    DEFAULT_GRID_SIZE_PIXELS,
    DEFAULT_MICRON_GRID_SIZE,
    DEFAULT_RESPONSE_WATERSHED_CLOSING_RADIUS,
    DEFAULT_RESPONSE_WATERSHED_FILL_HOLES,
    DEFAULT_RESPONSE_WATERSHED_MIN_PIXELS,
    DEFAULT_RESPONSE_WATERSHED_SMOOTHING_SIGMA,
    DEFAULT_WATERSHED_MIN_PIXELS,
    DEFAULT_WATERSHED_SMOOTHING_SIGMA,
    RoiGenerationMode,
    RoiGenerationOptions,
    RoiGenerationUnits,
)
from twopy.napari.theme import style_action_button
from twopy.pixel_calibration import PixelCalibrationRow
from twopy.pixel_calibration_profiles import (
    PixelCalibrationGroup,
    PixelCalibrationProfile,
    PixelCalibrationProfileMapping,
    resolve_pixel_calibration_profile,
    select_pixel_calibration_group,
)

__all__ = [
    "RoiGenerationControls",
    "RoiGenerationMode",
    "RoiGenerationOptions",
    "RoiGenerationUnits",
]

_RIG_PLACEHOLDER = "Select rig"
_MODE_PLACEHOLDER = "Select mode"
_SCANNER_PLACEHOLDER = "Select scanner"


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
        profile_mappings: tuple[PixelCalibrationProfileMapping, ...],
        *,
        on_generate: Callable[[RoiGenerationOptions], None],
    ) -> None:
        """Create the ROI mode control group.

        Args:
            calibrations: Calibration rows used to populate dropdowns.
            profile_mappings: ScanImage config mappings used to prefill
                calibration choices when recording metadata is incomplete.
            on_generate: Callback invoked with current options.

        Returns:
            None.
        """
        super().__init__("ROI mode")
        self._calibrations = calibrations
        self._profile_mappings = profile_mappings
        self._mode_labels = _mode_labels_by_mode(profile_mappings)
        self._on_generate = on_generate
        self._recording_loaded = False
        self._loaded_zoom: float | None = None
        self._active_profile: PixelCalibrationProfile | None = None

        self._roi_mode = QComboBox()
        self._roi_mode.addItem("manual", "manual")
        self._roi_mode.addItem("grid", "grid")
        self._roi_mode.addItem("watershed", "watershed")
        self._roi_mode.addItem("response watershed", "response_watershed")
        self._units = QComboBox()
        self._units.addItem("pixels", "pixels")
        self._units.addItem("microns", "microns")
        self._pixel_grid_size = QSpinBox()
        self._pixel_grid_size.setRange(1, 2048)
        self._pixel_grid_size.setValue(DEFAULT_GRID_SIZE_PIXELS)
        self._micron_grid_size = QDoubleSpinBox()
        self._micron_grid_size.setRange(0.001, 10000.0)
        self._micron_grid_size.setDecimals(3)
        self._micron_grid_size.setValue(DEFAULT_MICRON_GRID_SIZE)
        self._rig = QComboBox()
        self._mode = QComboBox()
        self._scanner = QComboBox()
        self._zoom = QDoubleSpinBox()
        self._zoom.setRange(0.001, 10000.0)
        self._zoom.setDecimals(3)
        self._allow_extrapolation = QCheckBox("Allow extrapolation")
        self._allow_extrapolation.setChecked(True)
        self._watershed_min_pixels = QSpinBox()
        self._watershed_min_pixels.setRange(1, 1_000_000)
        self._watershed_min_pixels.setValue(DEFAULT_WATERSHED_MIN_PIXELS)
        self._watershed_smoothing_sigma = QDoubleSpinBox()
        self._watershed_smoothing_sigma.setRange(0.0, 100.0)
        self._watershed_smoothing_sigma.setDecimals(3)
        self._watershed_smoothing_sigma.setValue(DEFAULT_WATERSHED_SMOOTHING_SIGMA)
        self._response_watershed_min_pixels = QSpinBox()
        self._response_watershed_min_pixels.setRange(1, 1_000_000)
        self._response_watershed_min_pixels.setValue(
            DEFAULT_RESPONSE_WATERSHED_MIN_PIXELS
        )
        self._response_watershed_smoothing_sigma = QDoubleSpinBox()
        self._response_watershed_smoothing_sigma.setRange(0.0, 100.0)
        self._response_watershed_smoothing_sigma.setDecimals(3)
        self._response_watershed_smoothing_sigma.setValue(
            DEFAULT_RESPONSE_WATERSHED_SMOOTHING_SIGMA
        )
        self._response_watershed_fill_holes = QCheckBox("Fill response holes")
        self._response_watershed_fill_holes.setChecked(
            DEFAULT_RESPONSE_WATERSHED_FILL_HOLES
        )
        self._response_watershed_closing_radius = QSpinBox()
        self._response_watershed_closing_radius.setRange(0, 100)
        self._response_watershed_closing_radius.setValue(
            DEFAULT_RESPONSE_WATERSHED_CLOSING_RADIUS
        )
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._create_button = QPushButton("Create ROIs")
        style_action_button(self._create_button, role="primary")

        self._populate_calibration_choices()
        self._roi_mode.currentIndexChanged.connect(self._sync_mode)
        self._units.currentIndexChanged.connect(self._sync_units)
        self._rig.currentIndexChanged.connect(self._sync_calibration_mode_choices)
        self._mode.currentIndexChanged.connect(self._sync_calibration_scanner_choices)
        self._scanner.currentIndexChanged.connect(self._sync_create_button)
        self._create_button.clicked.connect(self._generate)

        self._form_layout = QFormLayout()
        self._form_layout.addRow("Mode", self._roi_mode)
        self._form_layout.addRow("Units", self._units)
        self._form_layout.addRow("Pixels", self._pixel_grid_size)
        self._form_layout.addRow("Microns", self._micron_grid_size)
        self._form_layout.addRow("Rig", self._rig)
        self._form_layout.addRow("Mode", self._mode)
        self._form_layout.addRow("Scanner", self._scanner)
        self._form_layout.addRow("Zoom", self._zoom)
        self._form_layout.addRow("", self._allow_extrapolation)
        self._form_layout.addRow("Min pixels", self._watershed_min_pixels)
        self._form_layout.addRow("Smoothing", self._watershed_smoothing_sigma)
        self._form_layout.addRow(
            "Response min pixels", self._response_watershed_min_pixels
        )
        self._form_layout.addRow(
            "Response smoothing", self._response_watershed_smoothing_sigma
        )
        self._form_layout.addRow("", self._response_watershed_fill_holes)
        self._form_layout.addRow(
            "Response closing", self._response_watershed_closing_radius
        )
        self._form_layout.addRow("", self._create_button)
        self._form_layout.addRow("", self._status)
        self.setLayout(self._form_layout)
        _set_form_row_visible(self._form_layout, self._status, False)
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
            self._active_profile = None
            self._select_no_calibration_group()
            self._clear_status()
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
        units = _units_value(self._units)
        if units == "microns":
            rig = _required_combo_text(self._rig, "calibration rig")
            calibration_mode = _required_combo_int(self._mode, "calibration mode")
            scanner = _required_combo_text(self._scanner, "calibration scanner")
        else:
            rig = ""
            calibration_mode = 0
            scanner = ""
        return RoiGenerationOptions(
            roi_mode=_roi_mode_value(self._roi_mode),
            units=units,
            grid_size_pixels=self._pixel_grid_size.value(),
            micron_grid_size=self._micron_grid_size.value(),
            rig=rig,
            calibration_mode=calibration_mode,
            scanner=scanner,
            zoom=self._zoom.value(),
            allow_extrapolation=self._allow_extrapolation.isChecked(),
            watershed_min_pixels=self._watershed_min_pixels.value(),
            watershed_smoothing_sigma=self._watershed_smoothing_sigma.value(),
            response_watershed_min_pixels=self._response_watershed_min_pixels.value(),
            response_watershed_smoothing_sigma=(
                self._response_watershed_smoothing_sigma.value()
            ),
            response_watershed_fill_holes=(
                self._response_watershed_fill_holes.isChecked()
            ),
            response_watershed_closing_radius=(
                self._response_watershed_closing_radius.value()
            ),
        )

    def set_options(self, options: RoiGenerationOptions) -> None:
        """Update controls from saved ROI-generation settings.

        Args:
            options: Generation settings read from a saved ROI file.

        Returns:
            None.

        Reloading a saved analysis should show the mode and settings that made
        the current ROI masks. The controls remain normal editable widgets
        after this method applies the saved values.
        """
        _set_combo_data(self._roi_mode, options.roi_mode)
        if options.roi_mode == "grid":
            _set_combo_data(self._units, options.units)
            if options.units == "pixels":
                self._pixel_grid_size.setValue(options.grid_size_pixels)
                self._select_no_calibration_group()
            else:
                self._micron_grid_size.setValue(options.micron_grid_size)
                self._zoom.setValue(options.zoom)
                self._allow_extrapolation.setChecked(options.allow_extrapolation)
                self._select_saved_calibration_fields(options)
        elif options.roi_mode == "watershed":
            self._watershed_min_pixels.setValue(options.watershed_min_pixels)
            self._watershed_smoothing_sigma.setValue(
                options.watershed_smoothing_sigma,
            )
        elif options.roi_mode == "response_watershed":
            self._response_watershed_min_pixels.setValue(
                options.response_watershed_min_pixels,
            )
            self._response_watershed_smoothing_sigma.setValue(
                options.response_watershed_smoothing_sigma,
            )
            self._response_watershed_fill_holes.setChecked(
                options.response_watershed_fill_holes,
            )
            self._response_watershed_closing_radius.setValue(
                options.response_watershed_closing_radius,
            )
        self._sync_mode()
        if (
            options.roi_mode == "grid"
            and options.units == "microns"
            and not self._can_generate()
        ):
            self.set_status("Saved ROI generation calibration is not available.")
        else:
            self._clear_status()

    def set_status(self, text: str) -> None:
        """Show generation status in the ROIs tab.

        Args:
            text: User-facing status text.

        Returns:
            None.
        """
        self._status.setText(text)
        _set_form_row_visible(self._form_layout, self._status, text != "")

    def _populate_calibration_choices(self) -> None:
        """Populate calibration dropdowns from available rows."""
        self._rig.addItem(_RIG_PLACEHOLDER, None)
        for rig in _unique_text(row.rig for row in self._calibrations):
            self._rig.addItem(rig, rig)
        self._sync_calibration_mode_choices()

    def _sync_calibration_mode_choices(self) -> None:
        """Keep mode choices valid for the selected calibration rig."""
        self._replace_calibration_mode_choices(
            previous=self._mode.currentData(),
            use_profile_fallback=True,
        )

    def _replace_calibration_mode_choices(
        self,
        *,
        previous: object,
        use_profile_fallback: bool,
    ) -> None:
        """Replace mode choices for the selected rig."""
        rig = self._rig.currentData()
        modes = tuple(
            (_mode_label(mode, self._mode_labels), mode)
            for mode in sorted(
                {row.mode for row in self._calibrations if row.rig == rig},
            )
        )
        selected = previous
        if (
            selected is None
            and use_profile_fallback
            and self._active_profile is not None
        ):
            selected = self._active_profile.mode
        _replace_combo_items(
            self._mode,
            ((_MODE_PLACEHOLDER, None), *modes),
            previous=selected,
        )
        self._sync_calibration_scanner_choices()
        self._sync_create_button()

    def _sync_calibration_scanner_choices(self) -> None:
        """Keep scanner choices valid for the selected rig and mode."""
        self._replace_calibration_scanner_choices(
            previous=self._scanner.currentData(),
            use_profile_fallback=True,
        )

    def _replace_calibration_scanner_choices(
        self,
        *,
        previous: object,
        use_profile_fallback: bool,
    ) -> None:
        """Replace scanner choices for the selected rig and mode."""
        rig = self._rig.currentData()
        mode = self._mode.currentData()
        scanners = tuple(
            (scanner, scanner)
            for scanner in _unique_text(
                row.scanner
                for row in self._calibrations
                if row.rig == rig and row.mode == mode
            )
        )
        selected = previous
        if (
            selected is None
            and use_profile_fallback
            and self._active_profile is not None
        ):
            selected = self._active_profile.scanner
        _replace_combo_items(
            self._scanner,
            ((_SCANNER_PLACEHOLDER, None), *scanners),
            previous=selected,
        )
        self._sync_create_button()

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
        self._active_profile = profile
        self._select_no_calibration_group()
        group = select_pixel_calibration_group(profile, self._calibrations)
        if group is not None:
            self._select_calibration_group(group)
        else:
            self._select_available_calibration_profile_fields(profile)

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
        self._sync_create_button()

    def _select_no_calibration_group(self) -> None:
        """Reset calibration dropdowns to explicit user-selection placeholders."""
        if _set_combo_data(self._rig, None):
            self._sync_calibration_mode_choices()
        else:
            self._sync_calibration_mode_choices()

    def _select_available_calibration_profile_fields(
        self,
        profile: PixelCalibrationProfile,
    ) -> None:
        """Select known profile fields only when the exact option exists.

        Args:
            profile: Partial metadata-derived calibration profile.

        Returns:
            None.

        This preserves useful metadata, such as OdorRig -> night, without
        inventing a measured mode/scanner when calibration rows are absent.
        """
        if profile.rig is not None and _set_combo_text(self._rig, profile.rig):
            self._sync_calibration_mode_choices()
        if profile.mode is not None and _set_combo_data(self._mode, profile.mode):
            self._sync_calibration_scanner_choices()
        if profile.scanner is not None:
            _set_combo_text(self._scanner, profile.scanner)
        self._sync_create_button()

    def _select_saved_calibration_fields(self, options: RoiGenerationOptions) -> None:
        """Select calibration fields from saved micron-grid settings."""
        _set_combo_data(self._rig, None)
        self._replace_calibration_mode_choices(
            previous=None,
            use_profile_fallback=False,
        )
        if not _set_combo_text(self._rig, options.rig):
            self._sync_create_button()
            return
        self._replace_calibration_mode_choices(
            previous=options.calibration_mode,
            use_profile_fallback=False,
        )
        self._replace_calibration_scanner_choices(
            previous=options.scanner,
            use_profile_fallback=False,
        )
        _set_combo_text(self._scanner, options.scanner)
        self._sync_create_button()

    def _sync_mode(self) -> None:
        """Show and enable controls for the selected ROI mode."""
        roi_mode = _roi_mode_value(self._roi_mode)
        uses_grid = roi_mode == "grid"
        uses_watershed = roi_mode == "watershed"
        uses_response_watershed = roi_mode == "response_watershed"
        _set_form_row_visible(self._form_layout, self._units, uses_grid)
        for widget in (self._watershed_min_pixels, self._watershed_smoothing_sigma):
            _set_form_row_visible(self._form_layout, widget, uses_watershed)
        for widget in (
            self._response_watershed_min_pixels,
            self._response_watershed_smoothing_sigma,
            self._response_watershed_fill_holes,
            self._response_watershed_closing_radius,
        ):
            _set_form_row_visible(
                self._form_layout,
                widget,
                uses_response_watershed,
            )
        self._create_button.setVisible(roi_mode != "manual")
        if roi_mode == "grid":
            self._create_button.setText("Create grid")
        elif roi_mode == "watershed":
            self._create_button.setText("Create watershed")
        elif roi_mode == "response_watershed":
            self._create_button.setText("Create response watershed")
        else:
            self._create_button.setText("Create ROIs")
        self._sync_units()
        self._sync_create_button()
        self._set_mode_status()

    def _sync_units(self) -> None:
        """Enable inputs that apply to the selected grid-size unit."""
        uses_grid = _roi_mode_value(self._roi_mode) == "grid"
        uses_pixels = uses_grid and _units_value(self._units) == "pixels"
        uses_microns = uses_grid and _units_value(self._units) == "microns"
        self._pixel_grid_size.setEnabled(uses_pixels)
        self._micron_grid_size.setEnabled(uses_microns)
        self._rig.setEnabled(uses_microns)
        self._mode.setEnabled(uses_microns)
        self._scanner.setEnabled(uses_microns)
        self._zoom.setEnabled(uses_microns)
        self._allow_extrapolation.setEnabled(uses_microns)
        _set_form_row_visible(self._form_layout, self._pixel_grid_size, uses_pixels)
        for widget in (
            self._micron_grid_size,
            self._rig,
            self._mode,
            self._scanner,
            self._zoom,
            self._allow_extrapolation,
        ):
            _set_form_row_visible(self._form_layout, widget, uses_microns)
        self._sync_create_button()

    def _sync_create_button(self) -> None:
        """Enable generated-ROI creation only when required options are selected."""
        self._create_button.setEnabled(self._can_generate())

    def _can_generate(self) -> bool:
        """Return whether current controls can generate ROIs.

        Returns:
            ``True`` when the current ROI mode has all required user choices.
        """
        roi_mode = _roi_mode_value(self._roi_mode)
        if roi_mode == "manual":
            return False
        if roi_mode in {"watershed", "response_watershed"}:
            return True
        if _units_value(self._units) == "pixels":
            return True
        return (
            self._rig.currentData() is not None
            and self._mode.currentData() is not None
            and self._scanner.currentData() is not None
        )

    def _set_mode_status(self) -> None:
        """Clear passive mode status text."""
        if not self._recording_loaded or _roi_mode_value(self._roi_mode) == "manual":
            self._clear_status()

    def _generate(self) -> None:
        """Call the owner with the current generation options."""
        if _roi_mode_value(self._roi_mode) == "manual":
            self._set_mode_status()
            return
        try:
            options = self.options()
        except ValueError as error:
            self.set_status(str(error))
            return
        self._on_generate(options)

    def _clear_status(self) -> None:
        """Hide passive ROI-generation status text."""
        self.set_status("")


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


def _mode_labels_by_mode(
    mappings: Iterable[PixelCalibrationProfileMapping],
) -> dict[int, tuple[str, ...]]:
    """Return display config names grouped by calibration mode.

    Args:
        mappings: ScanImage config-to-profile mappings.

    Returns:
        Mapping from mode number to sorted config display names.
    """
    labels: dict[int, set[str]] = {}
    for mapping in mappings:
        labels.setdefault(mapping.mode, set()).add(_display_config_name(mapping))
    return {mode: tuple(sorted(names)) for mode, names in labels.items()}


def _mode_label(mode: int, labels_by_mode: dict[int, tuple[str, ...]]) -> str:
    """Return the calibration mode dropdown label.

    Args:
        mode: Calibration mode number.
        labels_by_mode: Display config names keyed by mode.

    Returns:
        Label including known ScanImage config names when available.
    """
    labels = labels_by_mode.get(mode, ())
    if len(labels) == 0:
        return str(mode)
    return f"{mode}: {', '.join(labels)}"


def _display_config_name(mapping: PixelCalibrationProfileMapping) -> str:
    """Return a compact config name for a mode dropdown item.

    Args:
        mapping: Profile mapping with a ScanImage config name.

    Returns:
        Config basename without a ``.cfg`` suffix.
    """
    name = mapping.config_name.strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    if name.lower().endswith(".cfg"):
        return name[:-4]
    return name


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


def _required_combo_text(combo: QComboBox, field_name: str) -> str:
    """Return selected text from a combo box with a placeholder item.

    Args:
        combo: Combo box to read.
        field_name: User-facing field name for validation errors.

    Returns:
        Selected display text.

    Raises:
        ValueError: If no concrete item is selected.
    """
    if combo.currentData() is None:
        msg = f"Select {field_name}."
        raise ValueError(msg)
    return combo.currentText()


def _required_combo_int(combo: QComboBox, field_name: str) -> int:
    """Return selected integer data from a combo box.

    Args:
        combo: Combo box to read.
        field_name: User-facing field name for validation errors.

    Returns:
        Selected integer value.

    Raises:
        ValueError: If no integer item is selected.
    """
    value = combo.currentData()
    if not isinstance(value, int):
        msg = f"Select {field_name}."
        raise ValueError(msg)
    return value


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
    if value == "response_watershed":
        return "response_watershed"
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
