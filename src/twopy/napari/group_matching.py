"""Manual group-analysis controls for loaded napari recordings.

Inputs: recordings already loaded in the shared napari viewer and the user's
current selected ROI label.
Outputs: plain CSV FOV grouping and ROI match tables that downstream group
analysis can load without depending on napari.

The popup is intentionally staged. FOV assignment comes first because ROI
matching across fields of view should not be the default path.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.group_matching import (
    ManualRoiMatchRow,
    ManualRoiMatchStatus,
    append_manual_roi_match_rows,
    load_manual_fov_group_rows,
    load_manual_roi_match_rows,
    make_manual_fov_group_rows,
    make_manual_roi_match_rows,
    next_group_cell_id,
    save_manual_fov_group_rows,
)
from twopy.napari.session import LoadedNapariRecording

__all__ = [
    "FovAssignmentView",
    "FovRecordingCard",
    "GroupMatchingPanel",
    "RoiAssignmentView",
    "mean_image_thumbnail_pixmap",
    "roi_label_from_selected_layer_value",
]

MATCH_TABLE_FILENAME = "roi_matches.csv"
FOV_GROUP_TABLE_FILENAME = "fov_groups.csv"
THUMBNAIL_SIZE = 180


class GroupMatchingState(Protocol):
    """Napari control-state fields needed by the group-matching widget."""

    loaded_recordings: list[LoadedNapariRecording]
    selected_recording_index: int | None


class _LayerWithSelectedLabel(Protocol):
    """Small protocol for napari Labels layers with an active label value."""

    selected_label: int
    data: object


class _LayerWithData(Protocol):
    """Small protocol for napari layers whose image data can be previewed."""

    data: object


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
        self._current_rois: dict[Path, str] = {}
        self._stack = QStackedWidget()
        self._fov_view = FovAssignmentView(
            state=state,
            fov_groups=self._fov_groups,
            on_finalize=self.finalize_fov_assignments,
        )
        self._roi_view = RoiAssignmentView(
            state=state,
            fov_groups=self._fov_groups,
            current_rois=self._current_rois,
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
            self._roi_view.refresh_fov_filter()
            self._stack.setCurrentWidget(self._roi_view)

    def show_fov_assignment(self) -> None:
        """Return to the FOV assignment view."""
        self._stack.setCurrentWidget(self._fov_view)
        self._fov_view.refresh()

    def _load_fov_groups_if_available(self) -> None:
        """Populate FOV assignments from disk when the panel has none yet."""
        if len(self._fov_groups) > 0:
            return
        output_path = self._fov_view.output_path()
        if not output_path.exists():
            return
        self._fov_groups.update(
            {
                row.recording_path.expanduser(): row.fov_group_id
                for row in load_manual_fov_group_rows(output_path)
            },
        )


class FovAssignmentView(QWidget):
    """Visual FOV assignment view with mean-image cards."""

    def __init__(
        self,
        *,
        state: GroupMatchingState,
        fov_groups: dict[Path, str],
        on_finalize: Callable[[], None],
    ) -> None:
        """Create the first-stage FOV assignment view.

        Args:
            state: Shared loaded-recording state.
            fov_groups: Mutable recording-to-FOV assignment mapping.
            on_finalize: Callback invoked by Finalize and assign ROIs.
        """
        super().__init__()
        self._state = state
        self._fov_groups = fov_groups
        self._cards: list[FovRecordingCard] = []
        self._on_finalize = on_finalize
        self._path_edit = QLineEdit(str(Path.cwd() / FOV_GROUP_TABLE_FILENAME))
        self._path_edit.setObjectName("fov_group_path")
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
        select_none_button = QPushButton("Select none")
        select_none_button.clicked.connect(self.select_none)
        clear_button = QPushButton("Clear selected FOV")
        clear_button.clicked.connect(self.clear_selected_fov)
        save_button = QPushButton("Save FOV groups")
        save_button.clicked.connect(self.save_fov_groups)
        finalize_button = QPushButton("Save and continue to ROI assignment")
        finalize_button.clicked.connect(self._finalize)

        assignment_controls = QHBoxLayout()
        assignment_controls.addWidget(QLabel("Assign selection to FOV"))
        assignment_controls.addWidget(self._fov_group_spin)
        assignment_controls.addWidget(assign_button)
        assignment_controls.addWidget(new_fov_button)
        assignment_controls.addStretch(1)

        cleanup_controls = QHBoxLayout()
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
        layout.addWidget(self._path_edit)
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
            recording_path = recording.display_path.expanduser()
            card = FovRecordingCard(
                recording=recording,
                fov_group_id=self._fov_groups.get(recording_path, ""),
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

    def save_fov_groups(self) -> bool:
        """Persist assigned recording-to-FOV rows to CSV.

        Returns:
            ``True`` when rows were saved, otherwise ``False``.
        """
        if len(self._fov_groups) == 0:
            self._status_label.setText("No FOV assignments to save.")
            return False
        rows = make_manual_fov_group_rows(self._fov_groups)
        save_manual_fov_group_rows(rows, self.output_path())
        self._status_label.setText(f"Saved {len(rows)} FOV assignments.")
        return True

    def output_path(self) -> Path:
        """Return the current FOV grouping CSV path."""
        return Path(self._path_edit.text()).expanduser()

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


class FovRecordingCard(QFrame):
    """One selectable mean-image card for FOV assignment."""

    def __init__(
        self,
        *,
        recording: LoadedNapariRecording,
        fov_group_id: str,
    ) -> None:
        """Create a FOV-assignment card.

        Args:
            recording: Loaded recording represented by this card.
            fov_group_id: Existing FOV assignment, if any.
        """
        super().__init__()
        self.recording_path = recording.display_path.expanduser()
        self._mean_image_layer = recording.mean_image_layer
        self._image_label = QLabel()
        self._image_label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fov_label = QLabel()
        self._path_label = QLabel(self.recording_path.name)
        self._path_label.setWordWrap(True)
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


class RoiAssignmentView(QWidget):
    """ROI assignment view filtered by finalized FOV groups."""

    def __init__(
        self,
        *,
        state: GroupMatchingState,
        fov_groups: dict[Path, str],
        current_rois: dict[Path, str],
    ) -> None:
        """Create the second-stage ROI assignment view."""
        super().__init__()
        self._state = state
        self._fov_groups = fov_groups
        self._current_rois = current_rois
        self._status_label = QLabel("Filter by FOV, then add selected ROIs.")
        self._path_edit = QLineEdit(str(Path.cwd() / MATCH_TABLE_FILENAME))
        self._path_edit.setObjectName("roi_match_path")
        self._fov_filter = QComboBox()
        self._fov_filter.currentIndexChanged.connect(lambda _index: self.refresh())
        self._roi_table = QTableWidget(0, 3)
        self._roi_table.setHorizontalHeaderLabels(("Recording", "Current ROI", "Saved"))

        capture_button = QPushButton("Add selected ROI to group")
        capture_button.clicked.connect(self.capture_active_roi)
        accept_button = QPushButton("Save as same cell")
        accept_button.clicked.connect(self.accept_match_group)
        unmatched_button = QPushButton("Save selected ROI as unmatched")
        unmatched_button.clicked.connect(self.mark_active_unmatched)
        clear_button = QPushButton("Clear unsaved ROI group")
        clear_button.clicked.connect(self.clear_current_group)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("FOV"))
        controls.addWidget(self._fov_filter)
        controls.addWidget(capture_button)
        controls.addWidget(accept_button)
        controls.addWidget(unmatched_button)
        controls.addWidget(clear_button)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("ROI Assignment"))
        layout.addWidget(self._path_edit)
        layout.addLayout(controls)
        layout.addWidget(self._roi_table)
        layout.addWidget(self._status_label)
        self.setLayout(layout)
        self.refresh_fov_filter()

    def refresh_fov_filter(self) -> None:
        """Refresh the FOV filter choices from saved FOV assignments."""
        current = self._fov_filter.currentText()
        choices = ("All FOVs", *tuple(sorted(set(self._fov_groups.values()))))
        self._fov_filter.blockSignals(True)
        try:
            self._fov_filter.clear()
            self._fov_filter.addItems(choices)
            if current in choices:
                self._fov_filter.setCurrentText(current)
        finally:
            self._fov_filter.blockSignals(False)
        self.refresh()

    def refresh(self) -> None:
        """Refresh the ROI assignment table from current FOV filter state."""
        recordings = self._filtered_recordings()
        self._roi_table.setRowCount(len(recordings))
        saved_counts = self._saved_counts_by_recording()
        for row, recording in enumerate(recordings):
            recording_path = recording.display_path.expanduser()
            self._set_table_item(row, 0, str(recording_path))
            self._set_table_item(row, 1, self._current_rois.get(recording_path, ""))
            self._set_table_item(row, 2, str(saved_counts.get(recording_path, 0)))

    def capture_active_roi(self) -> None:
        """Capture the active Labels selected label for the selected recording."""
        loaded = self._active_loaded_recording()
        if loaded is None:
            self._status_label.setText("Select a loaded recording first.")
            return
        recording_path = loaded.display_path.expanduser()
        if not self._recording_matches_filter(recording_path):
            self._status_label.setText("Selected recording is outside the FOV filter.")
            return
        roi_label = roi_label_from_selected_layer_value(loaded.roi_labels_layer)
        if roi_label is None:
            self._status_label.setText("Select a positive ROI label first.")
            return
        self._current_rois[recording_path] = roi_label
        self._status_label.setText(f"Added {roi_label} to the unsaved group.")
        self.refresh()

    def accept_match_group(self) -> None:
        """Persist current captured ROIs as one matched cell group."""
        filtered_rois = self._filtered_current_rois()
        if len(filtered_rois) < 2:
            self._status_label.setText("Add ROIs from at least two recordings.")
            return
        rows = self._rows_for_recording_rois(filtered_rois, status="matched")
        self._append_rows(rows)
        group_cell_id = rows[0].group_cell_id
        for recording_path in filtered_rois:
            self._current_rois.pop(recording_path, None)
        self._status_label.setText(
            f"Saved group {group_cell_id} with {len(rows)} ROIs.",
        )
        self.refresh()

    def mark_active_unmatched(self) -> None:
        """Persist the active selected ROI as reviewed but unmatched."""
        loaded = self._active_loaded_recording()
        if loaded is None:
            self._status_label.setText("Select a loaded recording first.")
            return
        recording_path = loaded.display_path.expanduser()
        if not self._recording_matches_filter(recording_path):
            self._status_label.setText("Selected recording is outside the FOV filter.")
            return
        roi_label = roi_label_from_selected_layer_value(loaded.roi_labels_layer)
        if roi_label is None:
            self._status_label.setText("Select a positive ROI label first.")
            return
        rows = self._rows_for_recording_rois(
            {recording_path: roi_label},
            status="unmatched",
        )
        self._append_rows(rows)
        self._current_rois.pop(recording_path, None)
        self._status_label.setText(f"Saved {roi_label} as unmatched.")
        self.refresh()

    def clear_current_group(self) -> None:
        """Clear captured unsaved ROI selections in the current FOV filter."""
        for recording_path in self._filtered_current_rois():
            self._current_rois.pop(recording_path, None)
        self._status_label.setText("Cleared unsaved ROI group.")
        self.refresh()

    def output_path(self) -> Path:
        """Return the current manual ROI match CSV path."""
        return Path(self._path_edit.text()).expanduser()

    def _filtered_recordings(self) -> tuple[LoadedNapariRecording, ...]:
        """Return loaded recordings visible under the current FOV filter."""
        return tuple(
            recording
            for recording in self._state.loaded_recordings
            if self._recording_matches_filter(recording.display_path.expanduser())
        )

    def _recording_matches_filter(self, recording_path: Path) -> bool:
        """Return whether one recording path belongs in the active FOV filter."""
        selected_fov = self._fov_filter.currentText()
        if selected_fov in ("", "All FOVs"):
            return True
        return self._fov_groups.get(recording_path) == selected_fov

    def _filtered_current_rois(self) -> dict[Path, str]:
        """Return captured ROIs that belong to the current FOV filter."""
        return {
            recording_path: roi_label
            for recording_path, roi_label in self._current_rois.items()
            if self._recording_matches_filter(recording_path)
        }

    def _rows_for_recording_rois(
        self,
        recording_rois: dict[Path, str],
        *,
        status: ManualRoiMatchStatus,
    ) -> tuple[ManualRoiMatchRow, ...]:
        """Build rows for one ROI decision."""
        existing_rows = self._existing_rows()
        return make_manual_roi_match_rows(
            recording_rois,
            group_cell_id=next_group_cell_id(existing_rows),
            fov_group_id=self._current_fov_group_id(),
            status=status,
        )

    def _current_fov_group_id(self) -> str:
        """Return the selected FOV group id, or empty text for all FOVs."""
        selected_fov = self._fov_filter.currentText()
        return "" if selected_fov == "All FOVs" else selected_fov

    def _append_rows(self, rows: tuple[ManualRoiMatchRow, ...]) -> None:
        """Append rows to the current output path."""
        append_manual_roi_match_rows(self.output_path(), rows)

    def _existing_rows(self) -> tuple[ManualRoiMatchRow, ...]:
        """Load existing saved rows, treating a missing file as empty."""
        output_path = self.output_path()
        return load_manual_roi_match_rows(output_path) if output_path.exists() else ()

    def _saved_counts_by_recording(self) -> dict[Path, int]:
        """Return saved row counts keyed by recording path."""
        counts: dict[Path, int] = {}
        for row in self._existing_rows():
            recording_path = row.recording_path.expanduser()
            counts[recording_path] = counts.get(recording_path, 0) + 1
        return counts

    def _active_loaded_recording(self) -> LoadedNapariRecording | None:
        """Return the selected loaded recording from shared state."""
        index = self._state.selected_recording_index
        if index is None or index < 0 or index >= len(self._state.loaded_recordings):
            return None
        return self._state.loaded_recordings[index]

    def _set_table_item(self, row: int, column: int, text: str) -> None:
        """Set one non-editable ROI table cell."""
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._roi_table.setItem(row, column, item)


def roi_label_from_selected_layer_value(layer: object | None) -> str | None:
    """Return the twopy ROI label for the active napari Labels value.

    Args:
        layer: Napari Labels layer or a test double exposing ``selected_label``.

    Returns:
        ``roi_####`` for a positive selected label, otherwise ``None``.
    """
    if layer is None or not hasattr(layer, "selected_label"):
        return None
    labels_layer = cast(_LayerWithSelectedLabel, layer)
    label_value = int(labels_layer.selected_label)
    if label_value <= 0:
        return None
    label_image = np.asarray(labels_layer.data)
    if not np.any(label_image == label_value):
        return None
    return f"roi_{label_value:04d}"


def mean_image_thumbnail_pixmap(
    layer: object,
    *,
    contrast_percentile: float,
) -> QPixmap | None:
    """Return a Qt pixmap thumbnail for a recording mean-image layer.

    Args:
        layer: Napari Image layer or test double exposing ``data``.
        contrast_percentile: Upper percentile used as the white point.

    Returns:
        ``QPixmap`` containing a grayscale thumbnail, or ``None`` when the
        layer data cannot be represented as a two-dimensional image.
    """
    if not hasattr(layer, "data"):
        return None
    image = np.asarray(cast(_LayerWithData, layer).data)
    if image.ndim != 2:
        return None
    thumbnail = _uint8_thumbnail(image, contrast_percentile=contrast_percentile)
    height, width = thumbnail.shape
    qimage = QImage(
        thumbnail.tobytes(),
        width,
        height,
        width,
        QImage.Format.Format_Grayscale8,
    ).copy()
    return QPixmap.fromImage(qimage).scaled(
        THUMBNAIL_SIZE,
        THUMBNAIL_SIZE,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _uint8_thumbnail(
    image: np.ndarray,
    *,
    contrast_percentile: float,
) -> np.ndarray:
    """Normalize one image to a contiguous uint8 grayscale thumbnail array."""
    values = np.asarray(image, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    minimum = float(np.min(finite_values))
    clipped_percentile = min(max(contrast_percentile, 1.0), 100.0)
    maximum = float(np.percentile(finite_values, clipped_percentile))
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)
    return np.ascontiguousarray(np.round(scaled * 255.0).astype(np.uint8))


def _clear_grid_layout(layout: QGridLayout) -> None:
    """Remove all child widgets from a grid layout."""
    while layout.count() > 0:
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
