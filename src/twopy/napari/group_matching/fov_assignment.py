"""FOV-assignment controls for manual group matching.

Inputs: loaded napari recordings, mutable FOV assignment state, and a FOV CSV path.
Outputs: mean-image cards plus plain CSV FOV group rows.

This module owns the first stage of Group Matching. Users compare mean images,
select recordings that share a field of view, and persist those assignments before
moving into ROI matching.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.group_matching import (
    load_manual_fov_group_rows,
    make_manual_fov_group_rows,
    save_manual_fov_group_rows,
)
from twopy.napari.display_paths import format_recording_minute_label
from twopy.napari.group_matching.cards import (
    GROUP_MATCHING_CARD_SPACING,
    card_columns_for_width,
    clear_card_grid_layout,
)
from twopy.napari.group_matching.fov_cards import (
    FOV_CARD_WIDTH,
    FovRecordingCard,
    mean_image_thumbnail_pixmap,
)
from twopy.napari.group_matching.style import (
    group_matching_button,
    group_matching_section,
)
from twopy.napari.group_matching.tables import ToggleSelectionTable
from twopy.napari.session import LoadedNapariRecording
from twopy.napari.text import configure_placeholder

__all__ = [
    "FOV_GROUP_TABLE_FILENAME",
    "FovAssignmentView",
    "FovRecordingCard",
    "mean_image_thumbnail_pixmap",
]

FOV_GROUP_TABLE_FILENAME = "fov_groups.csv"
_FOV_TABLE_VISIBLE_ROWS = 4
_FOV_TABLE_ROW_HEIGHT = 24
_FOV_ID_STEP_BUTTON_SIZE = 26


class GroupMatchingState(Protocol):
    """Napari control-state fields needed by the FOV assignment view."""

    loaded_recordings: list[LoadedNapariRecording]


class FovAssignmentView(QWidget):
    """Visual FOV assignment view with mean-image cards."""

    def __init__(
        self,
        *,
        state: GroupMatchingState,
        fov_groups: dict[Path, str],
        fov_notes: dict[Path, str],
        output_path: Path,
        on_output_path_changed: Callable[[Path], None],
        on_finalize: Callable[[], None],
    ) -> None:
        """Create the first-stage FOV assignment view.

        Args:
            state: Shared loaded-recording state.
            fov_groups: Mutable recording-to-FOV assignment mapping.
            fov_notes: Mutable recording-to-FOV-note mapping.
            output_path: Initial FOV CSV path owned by the parent panel.
            on_output_path_changed: Callback for user-visible FOV path changes.
            on_finalize: Callback invoked by Finalize and assign ROIs.
        """
        super().__init__()
        self._state = state
        self._fov_groups = fov_groups
        self._fov_notes = fov_notes
        self._cards: list[FovRecordingCard] = []
        self._card_grid_rows = 0
        self._card_grid_columns = 0
        self._on_output_path_changed = on_output_path_changed
        self._on_finalize = on_finalize
        self._path_edit = QLineEdit(str(output_path.expanduser()))
        self._path_edit.setObjectName("fov_group_path")
        self._path_edit.setMinimumWidth(0)
        self._path_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        configure_placeholder(self._path_edit, "FOV CSV path")
        self._path_edit.editingFinished.connect(self._path_editing_finished)
        self._instruction_label = QLabel(
            "Compare mean images, select matching fields, then save FOV groups.",
        )
        self._instruction_label.setObjectName("group_matching_subtitle")
        self._instruction_label.setWordWrap(True)
        self._status_label = QLabel("")
        self._status_label.setObjectName("group_matching_caption")
        self._status_label.setMinimumWidth(0)
        self._status_label.setWordWrap(True)
        self._status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse,
        )
        self._fov_group_value = 1
        self._fov_group_value_label = QLabel(str(self._fov_group_value))
        self._fov_group_value_label.setObjectName("fov_id_value")
        self._fov_group_label = QLabel("FOV ID")
        self._fov_group_label.setObjectName("fov_id_label")
        self._fov_table = self._create_fov_table()
        self._grid_widget = QWidget()
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(GROUP_MATCHING_CARD_SPACING)
        self._grid_widget.setLayout(self._grid)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setWidget(self._grid_widget)
        workspace_viewport = scroll_area.viewport()
        assert workspace_viewport is not None
        self._workspace_viewport = workspace_viewport
        self._workspace_viewport.installEventFilter(self)

        assign_button = group_matching_button("Assign FOV ID", role="primary")
        assign_button.clicked.connect(self.assign_selected_to_fov)
        new_fov_button = group_matching_button("Assign new FOV ID")
        new_fov_button.clicked.connect(self.new_fov_from_selected)
        increment_fov_button = _fov_id_step_button(Qt.ArrowType.UpArrow)
        increment_fov_button.clicked.connect(
            lambda _checked=False: self._step_fov_group_number(1),
        )
        decrement_fov_button = _fov_id_step_button(Qt.ArrowType.DownArrow)
        decrement_fov_button.clicked.connect(
            lambda _checked=False: self._step_fov_group_number(-1),
        )
        select_all_button = group_matching_button("Select all", role="quiet")
        select_all_button.clicked.connect(self.select_all)
        select_none_button = group_matching_button("Select none", role="quiet")
        select_none_button.clicked.connect(self.select_none)
        clear_button = group_matching_button("Remove assigned FOV", role="danger")
        clear_button.clicked.connect(self.clear_selected_fov)
        load_button = group_matching_button("Load FOV CSV")
        load_button.clicked.connect(self.load_fov_group_path)
        browse_button = group_matching_button("Browse save path")
        browse_button.clicked.connect(self.browse_fov_group_save_path)
        save_button = group_matching_button("Save FOV groups")
        save_button.clicked.connect(self.save_fov_groups)
        finalize_button = group_matching_button(
            "Save and continue to ROI assignment",
            role="primary",
        )
        finalize_button.clicked.connect(self._finalize)

        path_controls = QVBoxLayout()
        path_controls.addWidget(self._path_edit)
        path_button_row = QHBoxLayout()
        path_button_row.addWidget(load_button)
        path_button_row.addWidget(browse_button)
        path_controls.addLayout(path_button_row)

        assignment_controls = QVBoxLayout()
        fov_target_row = QHBoxLayout()
        fov_target_row.addStretch(1)
        fov_target_row.addWidget(
            self._fov_group_label,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        fov_target_row.addWidget(
            self._fov_group_value_label,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        fov_target_row.addWidget(
            increment_fov_button,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        fov_target_row.addWidget(
            decrement_fov_button,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        fov_target_row.addStretch(1)
        assignment_controls.addLayout(fov_target_row)
        assignment_controls.addWidget(assign_button)
        assignment_controls.addWidget(clear_button)
        assignment_controls.addWidget(new_fov_button)

        cleanup_controls = QHBoxLayout()
        cleanup_controls.addWidget(select_all_button)
        cleanup_controls.addWidget(select_none_button)

        output_controls = QVBoxLayout()
        output_controls.addWidget(save_button)
        output_controls.addWidget(finalize_button)

        title = QLabel("FOV assignment")
        title.setObjectName("group_matching_title")
        assignment_section = _section_with_layout(
            "Assign selected recordings",
            assignment_controls,
        )
        selection_section = _section_with_layout(
            "Selection",
            cleanup_controls,
        )
        output_section = _section_with_layout("Finish", output_controls)
        group_table_layout = QVBoxLayout()
        group_table_layout.addWidget(self._fov_table)
        group_table_section = group_matching_section(
            "Current FOV groups",
            group_table_layout,
        )
        card_section_layout = QVBoxLayout()
        card_section_layout.addWidget(scroll_area)

        left_column = QVBoxLayout()
        left_column.setSpacing(10)
        left_column.addWidget(title)
        left_column.addWidget(self._instruction_label)
        left_column.addWidget(group_matching_section("FOV file", path_controls))
        left_column.addWidget(self._status_label)
        left_column.addWidget(assignment_section)
        left_column.addWidget(selection_section)
        left_column.addWidget(group_table_section)
        left_column.addWidget(output_section)
        left_column.addStretch(1)
        left_content = QWidget()
        left_content.setMinimumWidth(0)
        left_content.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        left_content.setLayout(left_column)
        left_scroll = QScrollArea()
        left_scroll.setObjectName("fov_assignment_left_scroll")
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_content)
        left_scroll.setMinimumWidth(360)
        left_scroll.setMaximumWidth(380)

        layout = QHBoxLayout()
        layout.setSpacing(12)
        layout.addWidget(left_scroll)
        layout.addWidget(
            group_matching_section("Mean-image cards", card_section_layout),
            1,
        )
        self.setLayout(layout)

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """Relayout fixed-size FOV cards when the workspace width changes."""
        if (
            a0 is getattr(self, "_workspace_viewport", None)
            and a1 is not None
            and a1.type() == QEvent.Type.Resize
        ):
            self._layout_cards()
        return super().eventFilter(a0, a1)

    def refresh(self) -> None:
        """Refresh mean-image cards from currently loaded recordings."""
        clear_card_grid_layout(self._grid, delete_widgets=True)
        self._cards.clear()
        for recording in self._state.loaded_recordings:
            recording_path = recording.recording.source_session_dir.expanduser()
            card = FovRecordingCard(
                recording=recording,
                fov_group_id=self._fov_groups.get(recording_path, ""),
                note=self._fov_notes.get(recording_path, ""),
            )
            self._cards.append(card)
        self._layout_cards()
        self._refresh_fov_table()

    def _layout_cards(self) -> None:
        """Wrap fixed-size FOV cards to fit the current workspace width."""
        clear_card_grid_layout(self._grid, delete_widgets=False)
        for row_index in range(self._card_grid_rows + 1):
            self._grid.setRowStretch(row_index, 0)
        for column_index in range(self._card_grid_columns):
            self._grid.setColumnStretch(column_index, 0)
        columns = card_columns_for_width(
            self._workspace_viewport.width(),
            card_width=FOV_CARD_WIDTH,
            spacing=self._grid.spacing(),
        )
        for index, card in enumerate(self._cards):
            self._grid.addWidget(card, index // columns, index % columns)
        self._card_grid_columns = columns
        self._card_grid_rows = (len(self._cards) + columns - 1) // columns
        self._grid.setRowStretch(self._card_grid_rows, 1)

    def assign_selected_to_fov(self) -> None:
        """Assign selected cards to the chosen FOV group."""
        cards = self._selected_cards()
        if len(cards) == 0:
            self._status_label.setText("Select one or more recording cards.")
            return
        fov_group_id = self._selected_fov_group_id()
        for card in cards:
            self._fov_groups[card.recording_path] = fov_group_id
            card.set_fov_group_id(fov_group_id)
        self._refresh_fov_table()
        self._status_label.setText(
            f"Assigned {len(cards)} recordings to {fov_group_id}.",
        )

    def new_fov_from_selected(self) -> None:
        """Assign selected cards to the next unused FOV group."""
        self._set_fov_group_number(self._next_fov_group_number())
        self.assign_selected_to_fov()

    def clear_selected_fov(self) -> None:
        """Clear FOV assignments for selected cards."""
        cards = self._selected_cards()
        if len(cards) == 0:
            self._status_label.setText("Select one or more recording cards.")
            return
        for card in cards:
            self._fov_groups.pop(card.recording_path, None)
            card.set_fov_group_id("")
        self._refresh_fov_table()
        self._status_label.setText(f"Removed {len(cards)} FOV assignments.")

    def select_none(self) -> None:
        """Clear the current card selection without changing FOV assignments."""
        self._fov_table.clearSelection()
        for card in self._cards:
            card.set_selected(False)
        self._status_label.setText("Cleared card selection.")

    def select_all(self) -> None:
        """Select all visible cards without changing FOV assignments."""
        for card in self._cards:
            card.set_selected(True)
        self._status_label.setText(f"Selected {len(self._cards)} recording cards.")

    def load_fov_group_path(self) -> None:
        """Choose an existing FOV CSV path and load matching rows."""
        selected_path = self._choose_existing_fov_group_path()
        if selected_path is None:
            return
        self._path_edit.setText(str(selected_path))
        self._on_output_path_changed(self.output_path())
        self.load_fov_groups_from_path()

    def browse_fov_group_save_path(self) -> None:
        """Choose where future FOV assignments will be saved."""
        selected_path = self._choose_fov_group_save_path()
        if selected_path is None:
            return
        self._path_edit.setText(str(selected_path))
        self._on_output_path_changed(self.output_path())
        if selected_path.exists():
            self.load_fov_groups_from_path()
        else:
            self._status_label.setText(f"FOV CSV will be saved to {selected_path}.")

    def load_fov_groups_from_path(self) -> None:
        """Load existing FOV assignments for currently loaded recordings."""
        path = self.output_path()
        if not path.exists():
            self._status_label.setText(f"FOV CSV will be saved to {path}.")
            return
        try:
            rows = load_manual_fov_group_rows(path)
        except (OSError, ValueError) as error:
            self._status_label.setText(f"Could not load FOV CSV: {error}")
            return
        loaded_paths = {
            recording.recording.source_session_dir.expanduser()
            for recording in self._state.loaded_recordings
        }
        for recording_path in loaded_paths:
            self._fov_groups.pop(recording_path, None)
            self._fov_notes.pop(recording_path, None)
        loaded_rows = {
            row.recording_path.expanduser(): row.fov_group_id
            for row in rows
            if row.recording_path.expanduser() in loaded_paths
        }
        loaded_notes = {
            row.recording_path.expanduser(): row.note
            for row in rows
            if row.recording_path.expanduser() in loaded_paths
        }
        self._fov_groups.update(loaded_rows)
        self._fov_notes.update(loaded_notes)
        self._sync_cards_from_state()
        self._refresh_fov_table()
        self._status_label.setText(
            f"Loaded {len(loaded_rows)} FOV assignments from {path}",
        )

    def save_fov_groups(self) -> bool:
        """Persist assigned recording-to-FOV rows to CSV.

        Returns:
            ``True`` when rows were saved, otherwise ``False``.
        """
        recording_groups = self._loaded_recording_groups()
        if len(recording_groups) == 0:
            self._status_label.setText("No FOV assignments to save.")
            return False
        recording_notes = self._loaded_recording_notes(recording_groups)
        self._fov_groups.clear()
        self._fov_groups.update(recording_groups)
        self._fov_notes.clear()
        self._fov_notes.update(recording_notes)
        rows = make_manual_fov_group_rows(
            recording_groups,
            notes=recording_notes,
        )
        save_manual_fov_group_rows(rows, self.output_path())
        self._refresh_fov_table()
        self._status_label.setText(f"Saved {len(rows)} FOV assignments.")
        return True

    def output_path(self) -> Path:
        """Return the current FOV grouping CSV path."""
        return Path(self._path_edit.text()).expanduser()

    def set_output_path(self, path: Path) -> None:
        """Set the current FOV grouping CSV path."""
        self._path_edit.setText(str(path.expanduser()))

    def _path_editing_finished(self) -> None:
        """Load the typed FOV CSV path and update parent-owned path state."""
        self._on_output_path_changed(self.output_path())
        self.load_fov_groups_from_path()

    def _choose_existing_fov_group_path(self) -> Path | None:
        """Return a user-selected existing FOV CSV path."""
        current_path = self.output_path()
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Load FOV groups CSV")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("CSV files (*.csv);;All files (*)")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setDirectory(str(current_path.parent))
        dialog.selectFile(current_path.name)
        return _selected_dialog_path(dialog)

    def _choose_fov_group_save_path(self) -> Path | None:
        """Return a user-selected path for saving FOV CSV rows."""
        current_path = self.output_path()
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Choose FOV groups save path")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter("CSV files (*.csv);;All files (*)")
        dialog.setDefaultSuffix("csv")
        dialog.setOption(QFileDialog.Option.DontConfirmOverwrite, True)
        dialog.setDirectory(str(current_path.parent))
        dialog.selectFile(current_path.name)
        return _selected_dialog_path(dialog)

    def _finalize(self) -> None:
        """Call the parent finalize callback."""
        self._on_finalize()

    def _create_fov_table(self) -> QTableWidget:
        """Return a compact table of current FOV assignments."""
        table = ToggleSelectionTable(0, 3)
        table.setObjectName("fov_assignment_table")
        table.setHorizontalHeaderLabels(("FOV", "N", "Recordings"))
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setFixedHeight(
            _FOV_TABLE_ROW_HEIGHT * _FOV_TABLE_VISIBLE_ROWS + _FOV_TABLE_ROW_HEIGHT + 4,
        )
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        horizontal_header = table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setDefaultAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            horizontal_header.setSectionResizeMode(
                0,
                QHeaderView.ResizeMode.ResizeToContents,
            )
            horizontal_header.setSectionResizeMode(
                1,
                QHeaderView.ResizeMode.ResizeToContents,
            )
            horizontal_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        vertical_header = table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
            vertical_header.setDefaultSectionSize(_FOV_TABLE_ROW_HEIGHT)
        table.itemSelectionChanged.connect(self._select_fov_table_group)
        return table

    def _refresh_fov_table(self) -> None:
        """Refresh the current FOV table from visible card assignments."""
        groups = self._visible_fov_group_summary()
        self._fov_table.blockSignals(True)
        try:
            self._fov_table.setRowCount(0)
            for row_index, (fov_group_id, recording_names) in enumerate(groups):
                self._fov_table.insertRow(row_index)
                fov_item = QTableWidgetItem(fov_group_id)
                fov_item.setData(Qt.ItemDataRole.UserRole, fov_group_id)
                self._fov_table.setItem(row_index, 0, fov_item)
                self._fov_table.setItem(
                    row_index,
                    1,
                    QTableWidgetItem(str(len(recording_names))),
                )
                self._fov_table.setItem(
                    row_index,
                    2,
                    QTableWidgetItem(", ".join(recording_names)),
                )
            self._fov_table.resizeColumnsToContents()
            horizontal_header = self._fov_table.horizontalHeader()
            if horizontal_header is not None:
                horizontal_header.setSectionResizeMode(
                    2,
                    QHeaderView.ResizeMode.Stretch,
                )
        finally:
            self._fov_table.blockSignals(False)

    def _visible_fov_group_summary(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return assigned FOV groups and visible recording names."""
        grouped: dict[str, list[str]] = {}
        for card in self._cards:
            fov_group_id = self._fov_groups.get(card.recording_path, "")
            if fov_group_id == "":
                continue
            grouped.setdefault(fov_group_id, []).append(
                format_recording_minute_label(card.recording_path),
            )
        return tuple(
            (fov_group_id, tuple(sorted(recording_names)))
            for fov_group_id, recording_names in sorted(grouped.items())
        )

    def _select_fov_table_group(self) -> None:
        """Select cards belonging to the chosen FOV table row."""
        if len(self._fov_table.selectedItems()) == 0:
            for card in self._cards:
                card.set_selected(False)
            return
        row_index = self._fov_table.currentRow()
        if row_index < 0:
            return
        item = self._fov_table.item(row_index, 0)
        if item is None:
            return
        fov_group_id = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(fov_group_id, str):
            return
        for card in self._cards:
            card.set_selected(self._fov_groups.get(card.recording_path) == fov_group_id)
        suffix = fov_group_id.rsplit("_", maxsplit=1)[-1]
        if suffix.isdecimal():
            self._set_fov_group_number(int(suffix))
        self._status_label.setText(f"Selected recordings assigned to {fov_group_id}.")

    def _selected_fov_group_id(self) -> str:
        """Return the FOV id currently shown in the compact selector."""
        return f"fov_{self._fov_group_value}"

    def _step_fov_group_number(self, step: int) -> None:
        """Increment or decrement the compact FOV id selector."""
        self._set_fov_group_number(self._fov_group_value + step)

    def _set_fov_group_number(self, value: int) -> None:
        """Set the compact FOV id selector value within the supported range."""
        self._fov_group_value = min(max(int(value), 1), 9999)
        self._fov_group_value_label.setText(str(self._fov_group_value))

    def _selected_cards(self) -> tuple["FovRecordingCard", ...]:
        """Return currently selected recording cards."""
        return tuple(card for card in self._cards if card.is_selected())

    def _next_fov_group_number(self) -> int:
        """Return the next unused numeric FOV group suffix."""
        used_numbers: list[int] = []
        for fov_group_id in self._fov_groups.values():
            suffix = fov_group_id.rsplit("_", maxsplit=1)[-1]
            if suffix.isdecimal():
                used_numbers.append(int(suffix))
        if len(used_numbers) == 0:
            return 1
        return max(used_numbers) + 1

    def _loaded_recording_groups(self) -> dict[Path, str]:
        """Return FOV assignments for currently loaded recordings only."""
        loaded_paths = {
            recording.recording.source_session_dir.expanduser()
            for recording in self._state.loaded_recordings
        }
        return {
            recording_path: fov_group_id
            for recording_path, fov_group_id in self._fov_groups.items()
            if recording_path in loaded_paths
        }

    def _loaded_recording_notes(
        self,
        recording_groups: dict[Path, str],
    ) -> dict[Path, str]:
        """Return card notes for currently assigned recordings."""
        notes = dict(self._fov_notes)
        for card in self._cards:
            notes[card.recording_path] = card.note()
        return {
            recording_path: notes.get(recording_path, "")
            for recording_path in recording_groups
        }

    def _sync_cards_from_state(self) -> None:
        """Update visible card assignments and notes without rebuilding widgets."""
        for card in self._cards:
            card.set_fov_group_id(self._fov_groups.get(card.recording_path, ""))
            card.set_note(self._fov_notes.get(card.recording_path, ""))
        self._refresh_fov_table()


def _fov_id_step_button(arrow_type: Qt.ArrowType) -> QToolButton:
    """Return a compact native-arrow button for stepping the FOV id."""
    button = QToolButton()
    button.setObjectName("fov_id_step_button")
    button.setArrowType(arrow_type)
    button.setFixedSize(_FOV_ID_STEP_BUTTON_SIZE, _FOV_ID_STEP_BUTTON_SIZE)
    tooltip = (
        "Increment FOV ID" if arrow_type == Qt.ArrowType.UpArrow else "Decrement FOV ID"
    )
    button.setToolTip(tooltip)
    return button


def _section_with_layout(title: str, layout: QLayout) -> QGroupBox:
    """Return a titled section for a horizontal control row."""
    return group_matching_section(title, layout)


def _selected_dialog_path(dialog: QFileDialog) -> Path | None:
    """Return the accepted file dialog path, or ``None`` when cancelled."""
    if dialog.exec() != QFileDialog.DialogCode.Accepted:
        return None
    selected_files = dialog.selectedFiles()
    if len(selected_files) == 0:
        return None
    return Path(selected_files[0]).expanduser()
