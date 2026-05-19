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

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
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

        layout = QVBoxLayout()
        layout.addWidget(self._stack)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        """Refresh both staged views from loaded recordings and saved files."""
        self._load_fov_groups_if_available()
        self._fov_view.refresh()
        self._roi_view.refresh()

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
        configure_placeholder(self._path_edit, "FOV CSV path")
        self._path_edit.editingFinished.connect(self.load_fov_groups_from_path)
        self._instruction_label = QLabel("Select recordings that share a FOV.")
        self._status_label = QLabel("")
        self._fov_group_spin = QSpinBox()
        self._fov_group_spin.setRange(1, 9999)
        self._fov_group_spin.setMinimumWidth(108)
        spin_font = self._fov_group_spin.font()
        spin_font.setPointSize(spin_font.pointSize() + 4)
        self._fov_group_spin.setFont(spin_font)
        self._grid_widget = QWidget()
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(12)
        self._grid_widget.setLayout(self._grid)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self._grid_widget)

        assign_button = QPushButton("Assign selected to FOV")
        assign_button.clicked.connect(self.assign_selected_to_fov)
        new_fov_button = QPushButton("New FOV from selected")
        new_fov_button.clicked.connect(self.new_fov_from_selected)
        select_all_button = QPushButton("Select all")
        select_all_button.clicked.connect(self.select_all)
        select_none_button = QPushButton("Select none")
        select_none_button.clicked.connect(self.select_none)
        clear_button = QPushButton("Clear selected FOV")
        clear_button.clicked.connect(self.clear_selected_fov)
        load_button = QPushButton("Load FOV CSV")
        load_button.clicked.connect(self.load_fov_group_path)
        browse_button = QPushButton("Browse save path")
        browse_button.clicked.connect(self.browse_fov_group_save_path)
        save_button = QPushButton("Save FOV groups")
        save_button.clicked.connect(self.save_fov_groups)
        finalize_button = QPushButton("Save and continue to ROI assignment")
        finalize_button.clicked.connect(self._finalize)

        path_controls = QHBoxLayout()
        path_controls.addWidget(self._path_edit)
        path_controls.addWidget(load_button)
        path_controls.addWidget(browse_button)

        assignment_controls = QHBoxLayout()
        assignment_controls.addWidget(QLabel("Assign selection to FOV"))
        assignment_controls.addWidget(self._fov_group_spin)
        assignment_controls.addWidget(assign_button)
        assignment_controls.addWidget(new_fov_button)
        assignment_controls.addStretch(1)

        cleanup_controls = QHBoxLayout()
        cleanup_controls.addWidget(select_all_button)
        cleanup_controls.addWidget(select_none_button)
        cleanup_controls.addWidget(clear_button)
        cleanup_controls.addStretch(1)

        output_controls = QHBoxLayout()
        output_controls.addStretch(1)
        output_controls.addWidget(save_button)
        output_controls.addWidget(finalize_button)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("FOV Assignment"))
        layout.addWidget(self._instruction_label)
        layout.addLayout(path_controls)
        layout.addWidget(scroll_area)
        layout.addLayout(assignment_controls)
        layout.addLayout(cleanup_controls)
        layout.addLayout(output_controls)
        layout.addWidget(self._status_label)
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
            self._grid.addWidget(card, index // 3, index % 3)
        self._grid.setRowStretch((len(self._cards) + 2) // 3, 1)

    def assign_selected_to_fov(self) -> None:
        """Assign selected cards to the chosen FOV group."""
        cards = self._selected_cards()
        if len(cards) == 0:
            self._status_label.setText("Select one or more recording cards.")
            return
        fov_group_id = f"fov_{self._fov_group_spin.value()}"
        for card in cards:
            self._fov_groups[card.recording_path] = fov_group_id
            card.set_fov_group_id(fov_group_id)
        self._status_label.setText(
            f"Assigned {len(cards)} recordings to {fov_group_id}.",
        )

    def new_fov_from_selected(self) -> None:
        """Assign selected cards to the next unused FOV group."""
        self._fov_group_spin.setValue(self._next_fov_group_number())
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
        self._status_label.setText(f"Cleared {len(cards)} FOV assignments.")

    def select_none(self) -> None:
        """Clear the current card selection without changing FOV assignments."""
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
        self._status_label.setText(
            f"Loaded {len(loaded_rows)} FOV assignments from {path}.",
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
        self._fov_label = QLabel()
        self._path_label = QLabel(self.recording_path.name)
        self._path_label.setWordWrap(True)
        self._note_edit = QLineEdit(note)
        self._note_edit.setObjectName("fov_recording_note")
        configure_placeholder(self._note_edit, "Optional note")
        self._select_button = QPushButton("Select")
        self._select_button.setCheckable(True)
        self._select_button.toggled.connect(self._sync_selection_style)
        self._contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self._contrast_slider.setRange(1, 100)
        self._contrast_slider.setValue(100)
        self._contrast_slider.valueChanged.connect(self._refresh_image)

        layout = QVBoxLayout()
        layout.addWidget(self._image_label)
        layout.addWidget(self._path_label)
        layout.addWidget(self._fov_label)
        layout.addWidget(QLabel("Note"))
        layout.addWidget(self._note_edit)
        layout.addWidget(QLabel("Contrast"))
        layout.addWidget(self._contrast_slider)
        layout.addWidget(self._select_button)
        self.setLayout(layout)
        self.setFrameShape(QFrame.Shape.StyledPanel)
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
        self._fov_label.setText(f"FOV: {fov_group_id}" if fov_group_id else "FOV: none")

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
            self.setStyleSheet("QFrame { border: 2px solid #2d7dd2; }")
        else:
            self._select_button.setText("Select")
            self.setStyleSheet("")


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
