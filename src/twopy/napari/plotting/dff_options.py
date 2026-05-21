"""Qt controls for selecting dF/F analysis options.

Inputs: a current ``DeltaFOverFOptions`` object and an optional callback.
Outputs: a compact widget that returns typed GUI-independent dF/F settings.

The widget owns only GUI controls and value translation. Core analysis owns the
background subtraction, baseline fitting, and validation rules.
"""

from collections.abc import Callable

from qtpy.QtCore import QModelIndex, QRectF, QSignalBlocker, QSize, Qt
from qtpy.QtGui import QPainter, QTextDocument
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.background_subtraction import BackgroundCorrectionMethod
from twopy.analysis.dff import DeltaFOverFFitMode
from twopy.analysis.dff_options import DeltaFOverFBaselineMode, DeltaFOverFOptions
from twopy.napari.plotting.form_controls import (
    plot_form_layout,
    set_plot_control_width,
    set_plot_dropdown_width,
)
from twopy.typing_guards import require_string_choice

__all__ = ["DeltaFOverFOptionsWidget"]

_RICH_TEXT_ROLE = int(Qt.ItemDataRole.UserRole) + 1
_BASELINE_MODE_LABELS: tuple[
    tuple[str, DeltaFOverFBaselineMode, str | None],
    ...,
] = (
    ("baseline epoch", "epoch", None),
    ("no baseline epoch", "no_baseline_epoch", None),
)
_BASELINE_MODES = tuple(value for _label, value, _rich in _BASELINE_MODE_LABELS)
_BACKGROUND_METHOD_LABELS: tuple[
    tuple[str, BackgroundCorrectionMethod, str | None],
    ...,
] = (
    ("none", "none", None),
    ("global percentile", "movie_global_percentile", None),
    (
        "shared y-stripe P%",
        "movie_y_stripe_percentile",
        "shared y-stripe P<sub>%</sub>",
    ),
    ("ROI y-stripe P%", "roi_y_stripe_percentile", "ROI y-stripe P<sub>%</sub>"),
)
_BACKGROUND_METHODS = tuple(value for _label, value, _rich in _BACKGROUND_METHOD_LABELS)
_FIT_MODE_LABELS: tuple[tuple[str, DeltaFOverFFitMode, str | None], ...] = (
    ("bounded tau", "direct_bounded_tau", None),
    ("log-linear", "log_linear", None),
    (
        "bounded tau/amplitude",
        "direct_bounded_tau_and_log_amplitude",
        None,
    ),
)
_FIT_MODES = tuple(value for _label, value, _rich in _FIT_MODE_LABELS)


class DeltaFOverFOptionsWidget(QWidget):
    """Widget that exposes typed dF/F analysis settings.

    Inputs: initial dF/F options and an optional change callback.
    Outputs: a Qt widget plus ``options()`` for reading the current typed
    settings before preview or persistence actions.
    """

    def __init__(
        self,
        options: DeltaFOverFOptions,
        *,
        on_change: Callable[[DeltaFOverFOptions], None] | None = None,
    ) -> None:
        """Create the dF/F option controls.

        Args:
            options: Initial dF/F analysis settings.
            on_change: Optional callback receiving new typed settings whenever
                a GUI control changes.
        """
        super().__init__()
        self._on_change = on_change
        self._baseline_mode = _combo_box(_BASELINE_MODE_LABELS)
        self._baseline_epoch_label = QLabel("Baseline epoch")
        self._background_method = _combo_box(_BACKGROUND_METHOD_LABELS)
        self._baseline_epoch = QComboBox()
        set_plot_dropdown_width(self._baseline_epoch)
        self._baseline_epoch_names: dict[int, str] = {}
        self._baseline_epoch_name_values: dict[int, str | None] = {}
        self._baseline_seconds = _double_spin_box(
            minimum=0.001,
            maximum=1_000_000.0,
            value=options.baseline_sample_seconds or 1.0,
            suffix=" s",
        )
        self._use_full_baseline = QCheckBox("Use full baseline")
        set_plot_control_width(self._use_full_baseline)
        self._use_full_baseline.setChecked(options.baseline_sample_seconds is None)
        self._fit_mode = _combo_box(_FIT_MODE_LABELS)
        self._apply_motion_mask = QCheckBox("Mask motion artifacts")
        set_plot_control_width(self._apply_motion_mask)
        self._apply_motion_mask.setChecked(options.apply_motion_mask)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._dff_group())
        self.setLayout(layout)

        initial_baseline_epoch = options.baseline_epoch_number or 1
        self.set_epoch_choices(
            {initial_baseline_epoch: options.baseline_epoch_name}
            if options.baseline_epoch_name is not None
            else {},
            selected_epoch_number=initial_baseline_epoch,
        )
        self._set_combo_data(self._baseline_mode, options.baseline_mode)
        self._set_combo_data(self._background_method, options.background_method)
        self._set_combo_data(self._fit_mode, options.fit_mode)
        self._connect_changes()
        self._refresh_enabled_state()

    def options(self) -> DeltaFOverFOptions:
        """Return the current typed dF/F options.

        Args:
            None.

        Returns:
            ``DeltaFOverFOptions`` built from the current controls.
        """
        baseline_epoch_number = self._selected_baseline_epoch_number()
        return DeltaFOverFOptions(
            baseline_mode=self._selected_baseline_mode(),
            baseline_epoch_number=baseline_epoch_number,
            baseline_epoch_name=self._baseline_epoch_name_values.get(
                baseline_epoch_number,
            ),
            background_method=require_string_choice(
                str(self._background_method.currentData()),
                name="background",
                allowed=_BACKGROUND_METHODS,
            ),
            baseline_sample_seconds=(
                None
                if self._use_full_baseline.isChecked()
                else self._baseline_seconds.value()
            ),
            fit_mode=require_string_choice(
                str(self._fit_mode.currentData()),
                name="dF/F fit mode",
                allowed=_FIT_MODES,
            ),
            apply_motion_mask=self._apply_motion_mask.isChecked(),
        )

    def set_epoch_choices(
        self,
        epoch_names: dict[int, str],
        *,
        selected_epoch_number: int | None = None,
    ) -> None:
        """Update the baseline epoch dropdown from recording metadata.

        Args:
            epoch_names: Mapping from stimulus epoch numbers to display names.
            selected_epoch_number: Optional epoch number to select after
                rebuilding the dropdown.

        Returns:
            None.
        """
        selected = selected_epoch_number or self._selected_baseline_epoch_number()
        self._baseline_epoch_names = dict(sorted(epoch_names.items()))
        self._baseline_epoch_name_values = dict(sorted(epoch_names.items()))
        if selected not in self._baseline_epoch_names:
            self._baseline_epoch_names[selected] = f"Epoch {selected}"
            self._baseline_epoch_name_values[selected] = None

        blocker = QSignalBlocker(self._baseline_epoch)
        self._baseline_epoch.clear()
        for epoch_number, epoch_name in sorted(self._baseline_epoch_names.items()):
            self._baseline_epoch.addItem(
                _epoch_choice_label(epoch_number, epoch_name),
                epoch_number,
            )
        self._set_combo_data(self._baseline_epoch, selected)
        del blocker

    def set_options(self, options: DeltaFOverFOptions) -> None:
        """Update controls from typed options without emitting changes.

        Args:
            options: dF/F settings loaded from saved analysis output.

        Returns:
            None.

        Saved analysis reloads should update visible controls without triggering
        a new preview computation. The loaded plot already reflects the saved
        settings.
        """
        blockers = [
            QSignalBlocker(self._baseline_mode),
            QSignalBlocker(self._background_method),
            QSignalBlocker(self._baseline_epoch),
            QSignalBlocker(self._baseline_seconds),
            QSignalBlocker(self._use_full_baseline),
            QSignalBlocker(self._fit_mode),
            QSignalBlocker(self._apply_motion_mask),
        ]
        self._set_combo_data(self._baseline_mode, options.baseline_mode)
        epoch_names = {
            epoch_number: epoch_name
            for epoch_number, epoch_name in self._baseline_epoch_name_values.items()
            if epoch_name is not None
        }
        baseline_epoch_number = options.baseline_epoch_number or 1
        if options.baseline_epoch_name is not None:
            epoch_names[baseline_epoch_number] = options.baseline_epoch_name
        self.set_epoch_choices(
            epoch_names,
            selected_epoch_number=baseline_epoch_number,
        )
        self._set_combo_data(self._background_method, options.background_method)
        if options.baseline_sample_seconds is not None:
            self._baseline_seconds.setValue(options.baseline_sample_seconds)
        self._use_full_baseline.setChecked(options.baseline_sample_seconds is None)
        self._set_combo_data(self._fit_mode, options.fit_mode)
        self._apply_motion_mask.setChecked(options.apply_motion_mask)
        del blockers
        self._refresh_enabled_state()

    def _dff_group(self) -> QGroupBox:
        """Create the dF/F control group."""
        group = QGroupBox("dF/F")
        layout = plot_form_layout()
        layout.addRow("Background", self._background_method)
        layout.addRow("Baseline mode", self._baseline_mode)
        layout.addRow(self._baseline_epoch_label, self._baseline_epoch)
        layout.addRow("Baseline span", self._baseline_seconds)
        layout.addRow("", self._use_full_baseline)
        layout.addRow("Fit mode", self._fit_mode)
        layout.addRow("", self._apply_motion_mask)
        group.setLayout(layout)
        return group

    def _connect_changes(self) -> None:
        """Connect control changes to state refresh and callback dispatch."""
        for combo in (
            self._baseline_mode,
            self._background_method,
            self._baseline_epoch,
            self._fit_mode,
        ):
            combo.currentIndexChanged.connect(self._emit_change)
        self._baseline_seconds.valueChanged.connect(self._emit_change)
        self._use_full_baseline.stateChanged.connect(self._emit_change)
        self._apply_motion_mask.stateChanged.connect(self._emit_change)

    def _emit_change(self, *_args: object) -> None:
        """Emit typed options after a GUI value changes."""
        self._refresh_enabled_state()
        if self._on_change is not None:
            self._on_change(self.options())

    def _refresh_enabled_state(self) -> None:
        """Enable interleave seconds only when partial-window sampling is used."""
        self._baseline_seconds.setEnabled(not self._use_full_baseline.isChecked())
        if self._selected_baseline_mode() == "no_baseline_epoch":
            self._baseline_epoch_label.setText("First epoch")
        else:
            self._baseline_epoch_label.setText("Baseline epoch")

    def _selected_baseline_mode(self) -> DeltaFOverFBaselineMode:
        """Return the selected baseline interpretation mode."""
        return require_string_choice(
            str(self._baseline_mode.currentData()),
            name="dF/F baseline mode",
            allowed=_BASELINE_MODES,
        )

    def _selected_baseline_epoch_number(self) -> int:
        """Return the selected baseline epoch number."""
        data = self._baseline_epoch.currentData()
        if isinstance(data, int):
            return data
        try:
            return int(str(data))
        except ValueError:
            return 1

    def _set_combo_data(self, combo_box: QComboBox, value: object) -> None:
        """Set a combo box by stored item data when the item exists."""
        index = combo_box.findData(value)
        if index >= 0:
            combo_box.setCurrentIndex(index)


class _RichTextComboDelegate(QStyledItemDelegate):
    """Delegate that renders selected combo labels with simple HTML markup."""

    def paint(
        self,
        painter: QPainter | None,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        """Draw one combo item, using rich text when a rich label is available."""
        rich_text = index.data(_RICH_TEXT_ROLE)
        if painter is None or not isinstance(rich_text, str):
            super().paint(painter, option, index)
            return

        option_copy = QStyleOptionViewItem(option)
        self.initStyleOption(option_copy, index)
        option_copy.text = ""
        style = (
            option_copy.widget.style()
            if option_copy.widget is not None
            else QApplication.style()
        )
        if style is None:
            super().paint(painter, option, index)
            return

        style.drawControl(
            QStyle.ControlElement.CE_ItemViewItem,
            option_copy,
            painter,
            option_copy.widget,
        )

        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText,
            option_copy,
            option_copy.widget,
        )
        document = _rich_text_document(rich_text, option_copy)
        painter.save()
        painter.translate(text_rect.topLeft())
        document.drawContents(
            painter,
            QRectF(0.0, 0.0, float(text_rect.width()), float(text_rect.height())),
        )
        painter.restore()

    def sizeHint(
        self,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> QSize:
        """Return a size hint large enough for rich combo text."""
        size = super().sizeHint(option, index)
        rich_text = index.data(_RICH_TEXT_ROLE)
        if not isinstance(rich_text, str):
            return size

        option_copy = QStyleOptionViewItem(option)
        self.initStyleOption(option_copy, index)
        document = _rich_text_document(rich_text, option_copy)
        return QSize(
            max(size.width(), int(document.idealWidth()) + 4),
            max(size.height(), int(document.size().height()) + 2),
        )


def _combo_box(values: tuple[tuple[str, object, str | None], ...]) -> QComboBox:
    """Create one combo box with fixed text options."""
    combo_box = QComboBox()
    has_rich_text = False
    for label, value, rich_text in values:
        combo_box.addItem(label, value)
        if rich_text is not None:
            combo_box.setItemData(combo_box.count() - 1, rich_text, _RICH_TEXT_ROLE)
            has_rich_text = True
    if has_rich_text:
        combo_box.setItemDelegate(_RichTextComboDelegate(combo_box))
        combo_box.setLabelDrawingMode(QComboBox.LabelDrawingMode.UseDelegate)
    set_plot_dropdown_width(combo_box)
    return combo_box


def _rich_text_document(
    rich_text: str,
    option: QStyleOptionViewItem,
) -> QTextDocument:
    """Build a text document matching the combo item's current font."""
    document = QTextDocument()
    document.setDefaultFont(option.font)
    document.setDocumentMargin(0.0)
    document.setHtml(rich_text)
    return document


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


def _epoch_choice_label(epoch_number: int, epoch_name: str) -> str:
    """Return one readable dropdown label for a stimulus epoch."""
    return f"{epoch_number}: {epoch_name}"
