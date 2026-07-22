"""Progress dialog for asynchronous napari recording loads.

Inputs: selected recording paths, batch counts, current path text, and a cancel
callback.
Outputs: a modeless Qt dialog that reports many-recording load progress without
owning the recording-load workflow.
"""

from collections.abc import Callable
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from twopy.napari.display_paths import (
    format_recording_queue_path,
    format_recording_read_folder,
    recording_path_display_summary,
)
from twopy.napari.theme import TwopyThemeColors, apply_twopy_theme, style_action_button

__all__ = ["RecordingLoadProgressDialog"]

_MINIMUM_DIALOG_HEIGHT = 460
_QUEUE_WIDTH = 500
_ACTIVE_WIDTH = 560
_HORIZONTAL_MARGIN = 18
_PANEL_SPACING = 16
_MINIMUM_DIALOG_WIDTH = (
    _QUEUE_WIDTH + _ACTIVE_WIDTH + 2 * _HORIZONTAL_MARGIN + _PANEL_SPACING
)
_ROW_HEIGHT = 34
_QUEUE_SCROLLBAR_GUTTER = 8
_SELECTED_COUNT_WIDTH = 104


class RecordingLoadProgressDialog(QDialog):
    """Modeless progress dialog for a recording-load batch.

    Args:
        paths: Recording folders or converted recording paths requested by the
            user.
        on_cancel_pending: Callback invoked when the user cancels queued work.
        parent: Optional Qt parent widget.

    Outputs:
        A non-modal dialog with queue state, active path fields, progress, and
        cancel state.
    """

    def __init__(
        self,
        *,
        paths: tuple[Path, ...],
        on_cancel_pending: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        """Create the progress dialog."""
        super().__init__(parent)
        self._paths = paths
        self._on_cancel_pending = on_cancel_pending
        self._cancel_pending = False
        self._current_path: Path | None = None
        self._active_index: int | None = None
        self._completed_count = 0
        self._failed_indexes: set[int] = set()
        self.setWindowTitle("Loading recordings")
        self.setModal(False)
        self.setMinimumSize(_MINIMUM_DIALOG_WIDTH, _MINIMUM_DIALOG_HEIGHT)

        self._queue_rows: tuple[_QueueRowWidgets, ...] = tuple(
            _QueueRowWidgets.create(index=index, path=path)
            for index, path in enumerate(paths, start=1)
        )
        self._selected_count = QLabel(f"{len(paths)} selected")
        self._selected_count.setObjectName("recording_load_muted")
        self._selected_count.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._selected_count.setMinimumWidth(_SELECTED_COUNT_WIDTH)
        self._selected_count.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self._root_value = self._metadata_value_label()
        self._genotype_value = self._metadata_value_label()
        self._stimulus_value = self._metadata_value_label()
        self._date_value = self._metadata_value_label()
        self._reading_from_value = self._metadata_value_label()
        self._reading_route_value = self._metadata_value_label()
        self._progress = QProgressBar()
        self._progress.setRange(0, max(len(paths), 1))
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._queue_scroll = QScrollArea()
        self._hint = QLabel(
            "Cancel keeps this recording running and skips the remaining queue.",
        )
        self._hint.setObjectName("recording_load_muted")
        self._hint.setWordWrap(True)
        self._cancel_button = QPushButton("Cancel pending")
        self._cancel_button.setObjectName("recording_load_danger_button")
        style_action_button(self._cancel_button, role="danger")
        self._cancel_button.clicked.connect(self._cancel_pending_loads)

        apply_twopy_theme(
            self,
            name="twopy_recording_load_progress",
            additional_style=_style_sheet,
        )
        self.setLayout(self._layout())
        self.update_progress(
            completed_count=0,
            total_count=len(paths),
            current_path=None,
            reading_from_path=None,
            reading_route=None,
            active_index=None,
            phase="Preparing load queue...",
        )
        self.resize(_MINIMUM_DIALOG_WIDTH, _MINIMUM_DIALOG_HEIGHT)

    def update_progress(
        self,
        *,
        completed_count: int,
        total_count: int,
        current_path: Path | None,
        reading_from_path: Path | None,
        reading_route: str | None,
        active_index: int | None,
        phase: str,
        failed_indexes: tuple[int, ...] = (),
    ) -> None:
        """Update the visible load progress.

        Args:
            completed_count: Number of recordings finished or failed.
            total_count: Total requested recording count.
            current_path: Source-like path that identifies the active recording.
            reading_from_path: Converted HDF5 folder being read, or ``None``
                before resolution.
            reading_route: Short route label for ``reading_from_path``.
            active_index: Zero-based queue row currently being prepared or
                applied.
            phase: Short user-facing phase text.
            failed_indexes: Zero-based queue rows that finished with errors.

        Returns:
            None.
        """
        bounded_completed = min(max(completed_count, 0), max(total_count, 0))
        self._completed_count = bounded_completed
        self._current_path = current_path
        self._active_index = active_index
        self._failed_indexes = {
            index for index in failed_indexes if 0 <= index < len(self._queue_rows)
        }
        row_index = self._active_index_for_progress(active_index, current_path)
        self._refresh_queue_rows(
            completed_count=bounded_completed,
            active_index=row_index,
            phase=phase,
            failed_indexes=self._failed_indexes,
        )
        self._refresh_active_path(current_path)
        self._refresh_reading_path(reading_from_path, reading_route)
        self._progress.setRange(0, max(total_count, 1))
        self._progress.setValue(bounded_completed)
        self._progress.setFormat(
            _progress_text(
                completed_count=bounded_completed,
                total_count=total_count,
                active_index=row_index,
                phase=phase,
            ),
        )

    def mark_finishing(self, *, phase: str) -> None:
        """Show final status and disable cancellation.

        Args:
            phase: Final user-facing phase text.

        Returns:
            None.
        """
        self._cancel_button.setEnabled(False)
        self._hint.setText(phase)
        self._progress.setFormat(_plain_phase_text(phase))

    def _layout(self) -> QHBoxLayout:
        """Build the two-panel progress dialog layout."""
        layout = QHBoxLayout()
        layout.setContentsMargins(_HORIZONTAL_MARGIN, 16, _HORIZONTAL_MARGIN, 16)
        layout.setSpacing(_PANEL_SPACING)
        layout.addWidget(self._queue_panel(), 2)
        layout.addWidget(self._active_panel(), 3)
        return layout

    def _queue_panel(self) -> QWidget:
        """Return the scrollable queue panel."""
        panel = QFrame()
        panel.setObjectName("recording_load_panel")
        panel.setMinimumWidth(_QUEUE_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(_section_title("Queue"))
        header.addStretch(1)
        header.addWidget(self._selected_count)
        layout.addLayout(header)
        layout.addWidget(_queue_header_row())

        scroll = self._queue_scroll
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, _QUEUE_SCROLLBAR_GUTTER, 0)
        body_layout.setSpacing(5)
        for row in self._queue_rows:
            body_layout.addWidget(row.frame)
        body_layout.addStretch(1)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        return panel

    def _active_panel(self) -> QWidget:
        """Return the active recording panel."""
        panel = QFrame()
        panel.setObjectName("recording_load_panel")
        panel.setMinimumWidth(_ACTIVE_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(14)
        layout.addWidget(_section_title("Now loading"))
        layout.addWidget(_subsection_title("Source recording"))
        layout.addWidget(self._path_metadata_panel())
        layout.addWidget(_subsection_title("Reading from"))
        layout.addWidget(self._reading_metadata_panel())
        layout.addWidget(self._progress)
        layout.addWidget(self._hint)
        layout.addStretch(1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._cancel_button)
        layout.addLayout(buttons)
        return panel

    def _path_metadata_panel(self) -> QWidget:
        """Return Metadata-tab style fields for the active path."""
        panel = QFrame()
        panel.setObjectName("recording_load_path_panel")
        layout = QGridLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)
        fields = (
            ("Source root", self._root_value),
            ("Genotype", self._genotype_value),
            ("Stimulus", self._stimulus_value),
            ("Date", self._date_value),
        )
        for row_index, (label, value) in enumerate(fields):
            key = QLabel(label)
            key.setObjectName("recording_load_path_key")
            layout.addWidget(key, row_index, 0)
            layout.addWidget(value, row_index, 1)
        layout.setColumnStretch(1, 1)
        return panel

    def _reading_metadata_panel(self) -> QWidget:
        """Return compact fields for the converted data path being read."""
        panel = QFrame()
        panel.setObjectName("recording_load_path_panel")
        layout = QGridLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)
        fields = (
            ("Folder", self._reading_from_value),
            ("Route", self._reading_route_value),
        )
        for row_index, (label, value) in enumerate(fields):
            key = QLabel(label)
            key.setObjectName("recording_load_path_key")
            layout.addWidget(key, row_index, 0)
            layout.addWidget(value, row_index, 1)
        layout.setColumnStretch(1, 1)
        return panel

    def _metadata_value_label(self) -> QLabel:
        """Return one selectable path-metadata value label."""
        label = QLabel("")
        label.setObjectName("recording_load_path_value")
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        return label

    def _refresh_queue_rows(
        self,
        *,
        completed_count: int,
        active_index: int | None,
        phase: str,
        failed_indexes: set[int],
    ) -> None:
        """Apply current state text and highlighting to the queue rows."""
        for row_index, row in enumerate(self._queue_rows):
            if row_index in failed_indexes:
                state = "Failed"
            elif active_index == row_index:
                state = _queue_active_state(phase)
            elif row_index < completed_count:
                state = "Loaded"
            elif self._cancel_pending:
                state = "Skipped"
            else:
                state = "Queued"
            row.set_state(state, active=active_index == row_index)

    def _refresh_active_path(self, current_path: Path | None) -> None:
        """Show the active path as root, genotype, stimulus, and date fields."""
        if current_path is None:
            values = ("", "", "", "")
        else:
            summary = recording_path_display_summary(current_path)
            values = (
                str(summary.root),
                summary.genotype,
                summary.stimulus,
                summary.date,
            )
        for label, value in zip(
            (
                self._root_value,
                self._genotype_value,
                self._stimulus_value,
                self._date_value,
            ),
            values,
            strict=True,
        ):
            label.setText(value)
            label.setToolTip(value)

    def _refresh_reading_path(
        self,
        reading_from_path: Path | None,
        reading_route: str | None,
    ) -> None:
        """Show the actual converted data folder being read."""
        path_text = (
            ""
            if reading_from_path is None
            else format_recording_read_folder(reading_from_path)
        )
        path_tooltip = "" if reading_from_path is None else str(reading_from_path)
        route_text = "" if reading_route is None else reading_route
        for label, value, tooltip in (
            (self._reading_from_value, path_text, path_tooltip),
            (self._reading_route_value, route_text, route_text),
        ):
            label.setText(value)
            label.setToolTip(tooltip)

    def _active_index_for_progress(
        self,
        active_index: int | None,
        current_path: Path | None,
    ) -> int | None:
        """Return the queue index for the active path when it is known."""
        if active_index is not None:
            if 0 <= active_index < len(self._queue_rows):
                return active_index
            return None
        if current_path is None:
            return None
        for index, path in enumerate(self._paths[self._completed_count :]):
            if path == current_path:
                return self._completed_count + index
        for index, path in enumerate(self._paths):
            if path == current_path:
                return index
        return None

    def _cancel_pending_loads(self) -> None:
        """Ask the controller to skip queued recordings after the active load."""
        self._cancel_pending = True
        self._cancel_button.setEnabled(False)
        self._hint.setText("Finishing current recording. Pending loads canceled.")
        self._refresh_queue_rows(
            completed_count=self._completed_count,
            active_index=self._active_index_for_progress(
                self._active_index,
                self._current_path,
            ),
            phase="Finishing current recording.",
            failed_indexes=self._failed_indexes,
        )
        self._on_cancel_pending()


class _QueueRowWidgets:
    """Widgets that render one recording in the load queue."""

    def __init__(
        self,
        *,
        frame: QFrame,
        state: QLabel,
        recording: QLabel,
    ) -> None:
        """Store one queue row's mutable widgets."""
        self.frame = frame
        self._state = state
        self._recording = recording

    @classmethod
    def create(cls, *, index: int, path: Path) -> "_QueueRowWidgets":
        """Create one compact queue row.

        Args:
            index: One-based row number in the selected batch.
            path: Full selected recording path.

        Returns:
            Mutable row widgets for later state updates.
        """
        frame = QFrame()
        frame.setObjectName("recording_load_queue_row")
        frame.setFixedHeight(_ROW_HEIGHT)
        layout = QGridLayout(frame)
        layout.setContentsMargins(10, 3, 10, 3)
        layout.setHorizontalSpacing(8)

        number = QLabel(str(index))
        number.setObjectName("recording_load_number")
        layout.addWidget(number, 0, 0)
        state = _state_chip("Queued")
        layout.addWidget(state, 0, 1)
        recording = QLabel(format_recording_queue_path(path))
        recording.setObjectName("recording_load_recording")
        recording.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        recording.setToolTip(str(path))
        layout.addWidget(recording, 0, 2)
        layout.setColumnStretch(2, 1)
        return cls(frame=frame, state=state, recording=recording)

    def set_state(self, state: str, *, active: bool) -> None:
        """Update row state text and active-row highlighting."""
        self._state.setText(state)
        self._state.setProperty("load_state", _state_property(state))
        _repolish(self._state)
        self.frame.setObjectName(
            "recording_load_queue_row_active" if active else "recording_load_queue_row"
        )
        _repolish(self.frame)


def _queue_header_row() -> QWidget:
    """Return dense queue column labels."""
    row = QFrame()
    row.setObjectName("recording_load_queue_header")
    layout = QGridLayout(row)
    layout.setContentsMargins(10, 2, 10, 2)
    layout.setHorizontalSpacing(8)
    for column, text in enumerate(("#", "State", "Recording")):
        label = QLabel(text)
        label.setObjectName("recording_load_header_text")
        layout.addWidget(label, 0, column)
    layout.setColumnStretch(2, 1)
    return row


def _section_title(text: str) -> QLabel:
    """Return one section heading label."""
    label = QLabel(text)
    label.setObjectName("recording_load_section_title")
    return label


def _subsection_title(text: str) -> QLabel:
    """Return one compact subsection heading label."""
    label = QLabel(text)
    label.setObjectName("recording_load_subsection_title")
    return label


def _state_chip(text: str) -> QLabel:
    """Return one queue state label."""
    label = QLabel(text)
    label.setObjectName("recording_load_state")
    label.setProperty("load_state", _state_property(text))
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return label


def _queue_active_state(phase: str) -> str:
    """Return the compact queue state for one active phase."""
    normalized = phase.lower()
    if "prepar" in normalized or "read" in normalized:
        return "Reading"
    if "adding" in normalized:
        return "Adding"
    if "finished" in normalized or "complete" in normalized:
        return "Loaded"
    return "Loading"


def _progress_text(
    *,
    completed_count: int,
    total_count: int,
    active_index: int | None,
    phase: str,
) -> str:
    """Return progress-bar text for the current batch state."""
    if total_count <= 0:
        return _plain_phase_text(phase)
    current_number = completed_count if active_index is None else active_index + 1
    current_number = min(max(current_number, 0), total_count)
    return f"{current_number} of {total_count} - {_plain_phase_text(phase)}"


def _plain_phase_text(phase: str) -> str:
    """Return short lowercase phase text without trailing punctuation."""
    return phase.strip().rstrip(".").lower()


def _state_property(state: str) -> str:
    """Return stylesheet state name for one queue label."""
    if state in {"Loaded", "Reading", "Adding", "Loading"}:
        return "active"
    if state == "Failed":
        return "failed"
    if state == "Skipped":
        return "skipped"
    return "queued"


def _repolish(widget: QWidget) -> None:
    """Refresh stylesheet-dependent properties on one widget."""
    style = widget.style()
    if style is None:
        return
    style.unpolish(widget)
    style.polish(widget)


def _style_sheet(theme: TwopyThemeColors) -> str:
    """Return progress-dialog styles from the active napari theme."""
    window = theme.window
    panel = theme.window
    surface = theme.field
    active_surface = theme.selected_surface
    text = theme.text
    muted = theme.muted_text
    border = theme.control_outline
    accent = theme.accent
    highlighted_text = theme.accent_text
    queued = theme.disabled_surface
    failed_fill = theme.raised_surface
    skipped_fill = theme.surface
    return f"""
QDialog {{
    background: {window};
    color: {text};
}}
QFrame#recording_load_panel {{
    background: {panel};
    border: 1px solid {border};
    border-radius: 10px;
}}
QLabel#recording_load_section_title {{
    color: {text};
    font-size: 16px;
    font-weight: 700;
}}
QLabel#recording_load_subsection_title {{
    color: {muted};
    font-weight: 700;
}}
QLabel#recording_load_muted,
QLabel#recording_load_header_text,
QLabel#recording_load_number {{
    color: {muted};
}}
QFrame#recording_load_queue_header {{
    background: transparent;
}}
QFrame#recording_load_queue_row {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QFrame#recording_load_queue_row_active {{
    background: {active_surface};
    border: 1px solid {accent};
    border-radius: 8px;
}}
QLabel#recording_load_recording {{
    color: {text};
    font-weight: 700;
}}
QLabel#recording_load_state {{
    border-radius: 10px;
    padding: 3px 10px;
    font-weight: 700;
}}
QLabel#recording_load_state[load_state="active"] {{
    background: {accent};
    color: {highlighted_text};
}}
QLabel#recording_load_state[load_state="queued"] {{
    background: {queued};
    color: {text};
}}
QLabel#recording_load_state[load_state="failed"] {{
    background: {failed_fill};
    color: {text};
}}
QLabel#recording_load_state[load_state="skipped"] {{
    background: {skipped_fill};
    color: {text};
}}
QFrame#recording_load_path_panel {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 8px;
}}
QLabel#recording_load_path_key {{
    color: {muted};
    font-weight: 700;
}}
QLabel#recording_load_path_value {{
    color: {text};
    font-family: "Menlo";
    font-size: 12px;
}}
QProgressBar {{
    background: {surface};
    border: 1px solid {border};
    border-radius: 5px;
    color: {text};
    min-height: 16px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {accent};
    border-radius: 4px;
}}
QScrollArea {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: {surface};
    border: none;
    margin: 0px;
    width: 12px;
}}
QScrollBar::handle:vertical {{
    background: {muted};
    border-radius: 6px;
    min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{
    background: {accent};
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: transparent;
}}
"""
