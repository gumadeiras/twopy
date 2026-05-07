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
    QFormLayout,
    QGroupBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.background_subtraction import BackgroundCorrectionMethod
from twopy.analysis.dff import DeltaFOverFFitMode
from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.typing_guards import require_string_choice

__all__ = ["DeltaFOverFOptionsWidget"]

_RICH_TEXT_ROLE = int(Qt.ItemDataRole.UserRole) + 1
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
    ("robust", "robust", None),
    ("log-amplitude bounded", "source_bounds", None),
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
        self._background_method = _combo_box(_BACKGROUND_METHOD_LABELS)
        self._interleave_epoch = QComboBox()
        self._interleave_epoch_names: dict[int, str] = {}
        self._interleave_epoch_name_values: dict[int, str | None] = {}
        self._interleave_seconds = _double_spin_box(
            minimum=0.001,
            maximum=1_000_000.0,
            value=options.seconds_interleave_use or 1.0,
            suffix=" s",
        )
        self._use_full_interleave = QCheckBox("Use full interleave")
        self._use_full_interleave.setChecked(options.seconds_interleave_use is None)
        self._fit_mode = _combo_box(_FIT_MODE_LABELS)
        self._apply_motion_mask = QCheckBox("Mask motion artifacts")
        self._apply_motion_mask.setChecked(options.apply_motion_mask)

        layout = QVBoxLayout()
        layout.addWidget(self._dff_group())
        layout.addStretch(1)
        self.setLayout(layout)

        self.set_epoch_choices(
            {options.interleave_epoch_number: options.interleave_epoch_name}
            if options.interleave_epoch_name is not None
            else {},
            selected_epoch_number=options.interleave_epoch_number,
        )
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
        interleave_epoch_number = self._selected_interleave_epoch_number()
        return DeltaFOverFOptions(
            interleave_epoch_number=interleave_epoch_number,
            interleave_epoch_name=self._interleave_epoch_name_values.get(
                interleave_epoch_number,
            ),
            background_method=require_string_choice(
                str(self._background_method.currentData()),
                name="background",
                allowed=_BACKGROUND_METHODS,
            ),
            seconds_interleave_use=(
                None
                if self._use_full_interleave.isChecked()
                else self._interleave_seconds.value()
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
        """Update the interleave epoch dropdown from recording metadata.

        Args:
            epoch_names: Mapping from stimulus epoch numbers to display names.
            selected_epoch_number: Optional epoch number to select after
                rebuilding the dropdown.

        Returns:
            None.
        """
        selected = selected_epoch_number or self._selected_interleave_epoch_number()
        self._interleave_epoch_names = dict(sorted(epoch_names.items()))
        self._interleave_epoch_name_values = dict(sorted(epoch_names.items()))
        if selected not in self._interleave_epoch_names:
            self._interleave_epoch_names[selected] = f"Epoch {selected}"
            self._interleave_epoch_name_values[selected] = None

        blocker = QSignalBlocker(self._interleave_epoch)
        self._interleave_epoch.clear()
        for epoch_number, epoch_name in sorted(self._interleave_epoch_names.items()):
            self._interleave_epoch.addItem(
                _epoch_choice_label(epoch_number, epoch_name),
                epoch_number,
            )
        self._set_combo_data(self._interleave_epoch, selected)
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
            QSignalBlocker(self._background_method),
            QSignalBlocker(self._interleave_epoch),
            QSignalBlocker(self._interleave_seconds),
            QSignalBlocker(self._use_full_interleave),
            QSignalBlocker(self._fit_mode),
            QSignalBlocker(self._apply_motion_mask),
        ]
        epoch_names = {
            epoch_number: epoch_name
            for epoch_number, epoch_name in self._interleave_epoch_name_values.items()
            if epoch_name is not None
        }
        if options.interleave_epoch_name is not None:
            epoch_names[options.interleave_epoch_number] = options.interleave_epoch_name
        self.set_epoch_choices(
            epoch_names,
            selected_epoch_number=options.interleave_epoch_number,
        )
        self._set_combo_data(self._background_method, options.background_method)
        if options.seconds_interleave_use is not None:
            self._interleave_seconds.setValue(options.seconds_interleave_use)
        self._use_full_interleave.setChecked(options.seconds_interleave_use is None)
        self._set_combo_data(self._fit_mode, options.fit_mode)
        self._apply_motion_mask.setChecked(options.apply_motion_mask)
        del blockers
        self._refresh_enabled_state()

    def _dff_group(self) -> QGroupBox:
        """Create the dF/F control group."""
        group = QGroupBox("dF/F")
        layout = QFormLayout()
        layout.addRow("Background", self._background_method)
        layout.addRow("Interleave epoch", self._interleave_epoch)
        layout.addRow("Interleave span", self._interleave_seconds)
        layout.addRow("", self._use_full_interleave)
        layout.addRow("Fit mode", self._fit_mode)
        layout.addRow("", self._apply_motion_mask)
        group.setLayout(layout)
        return group

    def _connect_changes(self) -> None:
        """Connect control changes to state refresh and callback dispatch."""
        for combo in (self._background_method, self._interleave_epoch, self._fit_mode):
            combo.currentIndexChanged.connect(self._emit_change)
        self._interleave_seconds.valueChanged.connect(self._emit_change)
        self._use_full_interleave.stateChanged.connect(self._emit_change)
        self._apply_motion_mask.stateChanged.connect(self._emit_change)

    def _emit_change(self, *_args: object) -> None:
        """Emit typed options after a GUI value changes."""
        self._refresh_enabled_state()
        if self._on_change is not None:
            self._on_change(self.options())

    def _refresh_enabled_state(self) -> None:
        """Enable interleave seconds only when partial-window sampling is used."""
        self._interleave_seconds.setEnabled(not self._use_full_interleave.isChecked())

    def _selected_interleave_epoch_number(self) -> int:
        """Return the selected interleave epoch number."""
        data = self._interleave_epoch.currentData()
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


def _combo_box(values: tuple[tuple[str, str, str | None], ...]) -> QComboBox:
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
    return combo_box


def _rich_text_document(
    rich_text: str,
    option: QStyleOptionViewItem,
) -> QTextDocument:
    """Build a text document matching the combo item's current font."""
    document = QTextDocument()
    document.setDefaultFont(option.font)
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
    return spin_box


def _epoch_choice_label(epoch_number: int, epoch_name: str) -> str:
    """Return one readable dropdown label for a stimulus epoch."""
    return f"{epoch_number}: {epoch_name}"
