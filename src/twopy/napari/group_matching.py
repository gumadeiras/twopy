"""Manual group-analysis controls for loaded napari recordings.

Inputs: recordings already loaded in the shared napari viewer, their mean-image
layers, and their ROI Labels layers.
Outputs: plain CSV FOV grouping and ROI match tables that downstream group
analysis can load without depending on napari.

The popup is intentionally staged. FOV assignment comes first because ROI
matching across fields of view should not be the default path.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
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
from twopy.napari.group_matching_images import (
    THUMBNAIL_SIZE,
    mean_image_thumbnail_pixmap,
)
from twopy.napari.group_matching_roi import (
    RoiAssignmentView,
    roi_labels_from_layer_data,
)
from twopy.napari.group_matching_style import (
    group_matching_button,
    group_matching_section,
    style_group_matching_panel,
)
from twopy.napari.session import LoadedNapariRecording
from twopy.napari.text import configure_placeholder

__all__ = [
    "FovAssignmentView",
    "FovRecordingCard",
    "GroupMatchingPanel",
    "RoiAssignmentView",
    "mean_image_thumbnail_pixmap",
    "roi_labels_from_layer_data",
]

FOV_GROUP_TABLE_FILENAME = "fov_groups.csv"
_FOV_CARD_COLUMNS = 3
_FOV_CARD_MIN_WIDTH = THUMBNAIL_SIZE + 72
_FOV_CARD_GRID_MIN_WIDTH = (_FOV_CARD_MIN_WIDTH * _FOV_CARD_COLUMNS) + 28
_FOV_TABLE_VISIBLE_ROWS = 4
_FOV_TABLE_ROW_HEIGHT = 24
_FOV_ID_STEP_BUTTON_SIZE = 26


class GroupMatchingState(Protocol):
    """Napari control-state fields needed by the group-matching widget."""

    loaded_recordings: list[LoadedNapariRecording]


class GroupMatchingPanel(QWidget):
    """Two-step group-analysis popup for FOV assignment then ROI matching."""

    def __init__(self, state: GroupMatchingState) -> None:
        """Create the staged group-matching panel.

        Args:
            state: Shared napari control state. The panel reads loaded
                recordings and the active recording selection when buttons are
                pressed.
        """
        super().__init__()
        self._state = state
        self._fov_groups: dict[Path, str] = {}
        self._fov_notes: dict[Path, str] = {}
        self._current_rois: dict[Path, str] = {}
        self._theme_style_refreshing = False
        self._stack = QStackedWidget()
        self._fov_view = FovAssignmentView(
            state=state,
            fov_groups=self._fov_groups,
            fov_notes=self._fov_notes,
            on_finalize=self.finalize_fov_assignments,
        )
        self._roi_view = RoiAssignmentView(
            state=state,
            fov_groups=self._fov_groups,
            current_rois=self._current_rois,
            on_back=self.show_fov_assignment,
        )
        self._stack.addWidget(self._fov_view)
        self._stack.addWidget(self._roi_view)

        style_group_matching_panel(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self._stack)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        """Refresh both staged views from loaded recordings and saved files."""
        self._load_fov_groups_if_available()
        self._fov_view.refresh()
        self._roi_view.refresh()

    def changeEvent(self, a0: QEvent | None) -> None:  # noqa: N802
        """Refresh palette-derived styling when napari changes theme."""
        super().changeEvent(a0)
        if a0 is not None and a0.type() in (
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.PaletteChange,
        ):
            if self._theme_style_refreshing:
                return
            self._theme_style_refreshing = True
            try:
                style_group_matching_panel(self)
            finally:
                self._theme_style_refreshing = False

    def finalize_fov_assignments(self) -> None:
        """Save current FOV assignments and switch to ROI assignment."""
        if self._fov_view.save_fov_groups():
            self._roi_view.set_default_output_folder(
                self._fov_view.output_path().parent
            )
            self._roi_view.refresh_fov_filter()
            self._stack.setCurrentWidget(self._roi_view)

    def show_fov_assignment(self) -> None:
        """Return to the FOV assignment view."""
        self._stack.setCurrentWidget(self._fov_view)
        self._fov_view.refresh()

    def load_fov_groups_from_folder(self, folder: Path) -> bool:
        """Load ``fov_groups.csv`` from a recording-list folder when present.

        Args:
            folder: Folder that may contain the manual FOV grouping CSV.

        Returns:
            ``True`` when an adjacent FOV CSV was found and loaded.
        """
        candidate = folder.expanduser() / FOV_GROUP_TABLE_FILENAME
        if not candidate.exists():
            return False
        self._fov_view.set_output_path(candidate)
        self._fov_view.load_fov_groups_from_path()
        self._roi_view.refresh_fov_filter()
        return True

    def _load_fov_groups_if_available(self) -> None:
        """Populate FOV assignments from disk when the panel has none yet."""
        if len(self._fov_groups) > 0:
            return
        output_path = self._fov_view.output_path()
        if not output_path.exists():
            return
        loaded_paths = {
            recording.recording.source_session_dir.expanduser()
            for recording in self._state.loaded_recordings
        }
        rows = load_manual_fov_group_rows(output_path)
        self._fov_groups.update(
            {
                row.recording_path.expanduser(): row.fov_group_id
                for row in rows
                if row.recording_path.expanduser() in loaded_paths
            },
        )
        self._fov_notes.update(
            {
                row.recording_path.expanduser(): row.note
                for row in rows
                if row.recording_path.expanduser() in loaded_paths
            },
        )


class FovAssignmentView(QWidget):
    """Visual FOV assignment view with mean-image cards."""

    def __init__(
        self,
        *,
        state: GroupMatchingState,
        fov_groups: dict[Path, str],
        fov_notes: dict[Path, str],
        on_finalize: Callable[[], None],
    ) -> None:
        """Create the first-stage FOV assignment view.

        Args:
            state: Shared loaded-recording state.
            fov_groups: Mutable recording-to-FOV assignment mapping.
            fov_notes: Mutable recording-to-FOV-note mapping.
            on_finalize: Callback invoked by Finalize and assign ROIs.
        """
        super().__init__()
        self._state = state
        self._fov_groups = fov_groups
        self._fov_notes = fov_notes
        self._cards: list[FovRecordingCard] = []
        self._on_finalize = on_finalize
        self._path_edit = QLineEdit(str(Path.cwd() / FOV_GROUP_TABLE_FILENAME))
        self._path_edit.setObjectName("fov_group_path")
        self._path_edit.setMinimumWidth(0)
        self._path_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        configure_placeholder(self._path_edit, "FOV CSV path")
        self._path_edit.editingFinished.connect(self.load_fov_groups_from_path)
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
        self._grid_widget.setMinimumWidth(_FOV_CARD_GRID_MIN_WIDTH)
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(12)
        self._grid_widget.setLayout(self._grid)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMinimumWidth(_FOV_CARD_GRID_MIN_WIDTH + 28)
        scroll_area.setWidget(self._grid_widget)

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

    def refresh(self) -> None:
        """Refresh mean-image cards from currently loaded recordings."""
        _clear_grid_layout(self._grid)
        self._cards.clear()
        for index, recording in enumerate(self._state.loaded_recordings):
            recording_path = recording.recording.source_session_dir.expanduser()
            card = FovRecordingCard(
                recording=recording,
                fov_group_id=self._fov_groups.get(recording_path, ""),
                note=self._fov_notes.get(recording_path, ""),
            )
            self._cards.append(card)
            self._grid.addWidget(
                card,
                index // _FOV_CARD_COLUMNS,
                index % _FOV_CARD_COLUMNS,
            )
        self._grid.setRowStretch(
            (len(self._cards) + _FOV_CARD_COLUMNS - 1) // _FOV_CARD_COLUMNS,
            1,
        )
        self._refresh_fov_table()

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
        self.load_fov_groups_from_path()

    def browse_fov_group_save_path(self) -> None:
        """Choose where future FOV assignments will be saved."""
        selected_path = self._choose_fov_group_save_path()
        if selected_path is None:
            return
        self._path_edit.setText(str(selected_path))
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
        table = _ToggleSelectionTable(0, 3)
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
            grouped.setdefault(fov_group_id, []).append(card.recording_path.name)
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


class FovRecordingCard(QFrame):
    """One selectable mean-image card for FOV assignment."""

    def __init__(
        self,
        *,
        recording: LoadedNapariRecording,
        fov_group_id: str,
        note: str,
    ) -> None:
        """Create a FOV-assignment card.

        Args:
            recording: Loaded recording represented by this card.
            fov_group_id: Existing FOV assignment, if any.
            note: Existing row-specific FOV note, if any.
        """
        super().__init__()
        self.recording_path = recording.recording.source_session_dir.expanduser()
        self._mean_image_layer = recording.mean_image_layer
        self._image_label = QLabel()
        self._image_label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay_label = QLabel()
        self._overlay_label.setObjectName("fov_card_overlay")
        self._overlay_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self._note_edit = QLineEdit(note)
        self._note_edit.setObjectName("fov_recording_note")
        configure_placeholder(self._note_edit, "Optional note")
        self._select_button = group_matching_button("Select", role="quiet")
        self._select_button.setCheckable(True)
        self._select_button.toggled.connect(self._sync_selection_style)
        self._contrast_slider = QSlider(Qt.Orientation.Vertical)
        self._contrast_slider.setRange(1, 100)
        self._contrast_slider.setValue(100)
        self._contrast_slider.setFixedHeight(THUMBNAIL_SIZE)
        self._contrast_slider.setToolTip("Contrast")
        self._contrast_slider.valueChanged.connect(self._refresh_image)

        image_stack = QWidget()
        image_stack.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        image_layout = QGridLayout()
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)
        image_layout.addWidget(self._image_label, 0, 0)
        image_layout.addWidget(
            self._overlay_label,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
        )
        image_stack.setLayout(image_layout)

        preview_row = QHBoxLayout()
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(8)
        preview_row.addWidget(image_stack)
        preview_row.addWidget(self._contrast_slider)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addLayout(preview_row)
        layout.addWidget(self._note_edit)
        layout.addWidget(self._select_button)
        self.setLayout(layout)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("fov_recording_card")
        self.setMinimumWidth(_FOV_CARD_MIN_WIDTH)
        self.set_fov_group_id(fov_group_id)
        self._refresh_image()
        self._sync_selection_style()

    def is_selected(self) -> bool:
        """Return whether this card is selected for batch assignment."""
        return self._select_button.isChecked()

    def set_selected(self, selected: bool) -> None:
        """Set whether this card is selected for batch assignment."""
        self._select_button.setChecked(selected)

    def set_fov_group_id(self, fov_group_id: str) -> None:
        """Set the visual FOV assignment label."""
        fov_text = fov_group_id if fov_group_id else "none"
        self._overlay_label.setText(
            f"{self.recording_path.name} - FOV ID: {_fov_group_display_id(fov_text)}",
        )

    def note(self) -> str:
        """Return the row-specific note for this recording."""
        return self._note_edit.text()

    def set_note(self, note: str) -> None:
        """Set the row-specific note without rebuilding the card."""
        self._note_edit.setText(note)

    def _refresh_image(self) -> None:
        """Refresh the card thumbnail using the current contrast slider."""
        pixmap = mean_image_thumbnail_pixmap(
            self._mean_image_layer,
            contrast_percentile=float(self._contrast_slider.value()),
        )
        if pixmap is not None:
            self._image_label.setPixmap(pixmap)

    def _sync_selection_style(self) -> None:
        """Show selected cards with a visible border."""
        if self.is_selected():
            self._select_button.setText("Selected")
        else:
            self._select_button.setText("Select")
        self.setProperty("selected", self.is_selected())
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)


class _ToggleSelectionTable(QTableWidget):
    """Table that clears selection when the selected row is clicked again."""

    def mousePressEvent(self, e: QMouseEvent | None) -> None:  # noqa: N802
        """Toggle the current row off when users click it a second time."""
        if e is None:
            super().mousePressEvent(e)
            return
        row_index = self.rowAt(e.pos().y())
        if row_index >= 0 and row_index == self.currentRow() and self.selectedItems():
            self.clearSelection()
            self.setCurrentCell(-1, -1)
            return
        super().mousePressEvent(e)


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


def _fov_group_display_id(fov_group_id: str) -> str:
    """Return the compact numeric text shown in mean-image overlays."""
    if fov_group_id == "none":
        return "none"
    suffix = fov_group_id.removeprefix("fov_")
    return suffix if suffix.isdecimal() else fov_group_id


def _clear_grid_layout(layout: QGridLayout) -> None:
    """Remove all child widgets from a grid layout."""
    while layout.count() > 0:
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)


def _selected_dialog_path(dialog: QFileDialog) -> Path | None:
    """Return the accepted file dialog path, or ``None`` when cancelled."""
    if dialog.exec() != QFileDialog.DialogCode.Accepted:
        return None
    selected_files = dialog.selectedFiles()
    if len(selected_files) == 0:
        return None
    return Path(selected_files[0]).expanduser()
