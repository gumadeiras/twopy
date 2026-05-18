"""Visual ROI assignment controls for manual group matching.

Inputs: loaded napari recordings, FOV assignments, and editable ROI Labels
layers.
Outputs: visual ROI cards plus plain CSV ROI match decisions.

The ROI assignment stage is intentionally self-contained. Each recording card
owns its ROI selector and mean-image overlay. The view owns the saved-group
table, group save action, and shared response preview so users can compare
selected cells across recordings without changing the active layer selection in
the main napari viewer.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.group_matching import (
    ManualRoiMatchGroup,
    ManualRoiMatchRow,
    add_manual_roi_match_group,
    load_manual_roi_match_rows,
    matched_manual_roi_groups,
    remove_manual_roi_match_group,
    replace_manual_roi_match_group,
    save_manual_roi_match_rows,
)
from twopy.analysis.response_processing import (
    NormalizationOptions,
    ResponseProcessingOptions,
)
from twopy.analysis.trials import is_baseline_epoch_name
from twopy.napari.group_matching_images import mean_image_roi_overlay_pixmap
from twopy.napari.plotting.data import EpochResponsePlotData, ResponsePlotData
from twopy.napari.plotting.normalization_options import NormalizationOptionsWidget
from twopy.napari.plotting.widgets import (
    EpochPlotWidget,
    clear_layout,
    global_time_bounds,
    global_value_bounds,
)
from twopy.napari.responses import (
    compute_response_preview,
    response_analysis_request_from_labels,
)
from twopy.napari.session import LoadedNapariRecording
from twopy.stimulus import stimulus_epoch_names_by_number

__all__ = [
    "RoiAssignmentView",
    "RoiRecordingCard",
    "roi_labels_from_layer_data",
]

MATCH_TABLE_FILENAME = "roi_matches.csv"
_ROI_PREVIEW_PLOT_SIZE = 180
_ROI_TRACE_ALPHA = int(round(255 * 0.75))


class GroupMatchingState(Protocol):
    """Napari control-state fields needed by the ROI assignment view."""

    loaded_recordings: list[LoadedNapariRecording]


class _LayerWithData(Protocol):
    """Small protocol for napari layers whose label data can be inspected."""

    data: object


@dataclass(frozen=True)
class _SelectedRoiLayer:
    """Minimal Labels-layer stand-in for computing one selected ROI response."""

    data: object


@dataclass(frozen=True)
class _SelectedRoiResponse:
    """Response data for one selected ROI in one loaded recording."""

    recording_path: Path
    roi_label: str
    plot_data: ResponsePlotData
    color: QColor


class RoiAssignmentView(QWidget):
    """Visual ROI assignment view filtered by finalized FOV groups."""

    def __init__(
        self,
        *,
        state: GroupMatchingState,
        fov_groups: dict[Path, str],
        current_rois: dict[Path, str],
        on_back: Callable[[], None],
    ) -> None:
        """Create the second-stage ROI assignment view.

        Args:
            state: Shared loaded-recording state.
            fov_groups: Recording-to-FOV assignments from the first stage.
            current_rois: Mutable unsaved ROI group keyed by recording path.
            on_back: Callback returning the popup to FOV assignment.
        """
        super().__init__()
        self._state = state
        self._fov_groups = fov_groups
        self._current_rois = current_rois
        self._on_back = on_back
        self._cards: list[RoiRecordingCard] = []
        self._hidden_response_recordings: set[Path] = set()
        self._selected_group_dirty = False
        self._normalization_options = NormalizationOptions()
        self._status_label = QLabel("Choose ROIs in the cards, then save a group.")
        self._response_status = QLabel("Choose ROIs to compare responses.")
        self._response_status.setWordWrap(True)
        self._normalization_widget = self._create_normalization_widget()
        self._legend_widget = QWidget()
        self._legend_layout = QHBoxLayout()
        self._legend_layout.setContentsMargins(0, 0, 0, 0)
        self._legend_widget.setLayout(self._legend_layout)
        self._response_widget = QWidget()
        self._response_widget.setObjectName("roi_response_preview")
        self._response_layout = QHBoxLayout()
        self._response_layout.setContentsMargins(0, 0, 0, 0)
        self._response_widget.setLayout(self._response_layout)
        self._path_edit = QLineEdit(str(Path.cwd() / MATCH_TABLE_FILENAME))
        self._path_edit.setObjectName("roi_match_path")
        self._path_edit.editingFinished.connect(self.load_match_rows_from_path)
        self._note_edit = QLineEdit("")
        self._note_edit.setObjectName("roi_match_note")
        self._note_edit.setPlaceholderText("Optional note saved with this ROI group")
        self._note_edit.textChanged.connect(
            lambda _text: self._mark_selected_group_dirty(),
        )
        self._fov_filter = QComboBox()
        self._fov_filter.setObjectName("fov_filter")
        self._fov_filter.currentIndexChanged.connect(lambda _index: self.refresh())
        self._grid_widget = QWidget()
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(12)
        self._grid_widget.setLayout(self._grid)
        self._group_table = self._create_group_table()
        load_button = QPushButton("Load ROI CSV")
        load_button.clicked.connect(self.load_match_row_path)
        browse_button = QPushButton("Browse save path")
        browse_button.clicked.connect(self.browse_match_row_save_path)
        add_button = QPushButton("Add new group")
        add_button.clicked.connect(self.add_match_group)
        save_button = QPushButton("Overwrite selected group")
        save_button.clicked.connect(self.save_selected_match_group)
        remove_button = QPushButton("Remove selected group")
        remove_button.clicked.connect(self.remove_selected_match_group)
        clear_button = QPushButton("Clear ROI selection")
        clear_button.clicked.connect(self.clear_current_group)
        back_button = QPushButton("Back to FOV assignment")
        back_button.clicked.connect(self.return_to_fov_assignment)
        close_button = QPushButton("Save and close")
        close_button.clicked.connect(self.close_group_matching_window)

        path_controls = QHBoxLayout()
        path_controls.addWidget(self._path_edit)
        path_controls.addWidget(load_button)
        path_controls.addWidget(browse_button)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("FOV"))
        controls.addWidget(self._fov_filter)
        controls.addStretch(1)

        action_panel = QVBoxLayout()
        action_panel.addWidget(add_button)
        action_panel.addWidget(save_button)
        action_panel.addWidget(remove_button)
        action_panel.addWidget(clear_button)
        action_panel.addWidget(back_button)
        action_panel.addWidget(close_button)
        action_panel.addWidget(QLabel("Note"))
        action_panel.addWidget(self._note_edit)
        action_panel.addWidget(self._normalization_widget)
        action_panel.addStretch(1)
        self._normalization_widget.setMinimumWidth(420)

        group_label = QLabel("Saved groups")
        group_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        group_panel = QVBoxLayout()
        group_panel.setAlignment(Qt.AlignmentFlag.AlignTop)
        group_panel.addWidget(group_label)
        group_panel.addWidget(self._group_table, 1)

        review_panel = QHBoxLayout()
        review_panel.addLayout(group_panel, 1)
        review_panel.addLayout(action_panel, 1)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("ROI Assignment"))
        layout.addLayout(path_controls)
        layout.addLayout(controls)
        layout.addWidget(self._grid_widget)
        layout.addLayout(review_panel)
        layout.addWidget(self._response_status)
        layout.addWidget(self._legend_widget)
        layout.addWidget(self._response_widget)
        layout.addWidget(self._status_label)
        content = QWidget()
        content.setLayout(layout)
        scroll_area = QScrollArea()
        scroll_area.setObjectName("roi_assignment_scroll_area")
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content)
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll_area)
        self.setLayout(outer_layout)
        self.refresh_fov_filter()

    def _create_normalization_widget(self) -> NormalizationOptionsWidget:
        """Return the ROI response normalization controls."""
        widget = NormalizationOptionsWidget(
            self._normalization_options,
            on_change=self._set_normalization_options,
        )
        for checkbox in widget.findChildren(QCheckBox):
            checkbox.setFixedWidth(240)
            checkbox.setStyleSheet("width: 240px;")
        return widget

    def _create_group_table(self) -> QTableWidget:
        """Return the saved ROI group table with selection behavior wired."""
        table = QTableWidget(0, 3)
        table.setObjectName("roi_match_group_table")
        table.setHorizontalHeaderLabels(("Group", "ROIs", "Note"))
        table.setMinimumHeight(160)
        table.setMaximumHeight(240)
        vertical_header = table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        horizontal_header = table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setSectionResizeMode(
                0,
                QHeaderView.ResizeMode.ResizeToContents,
            )
            horizontal_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            horizontal_header.setSectionResizeMode(
                2,
                QHeaderView.ResizeMode.ResizeToContents,
            )
        table.itemSelectionChanged.connect(self._restore_selected_group)
        return table

    def refresh_fov_filter(self) -> None:
        """Refresh the FOV filter choices from saved FOV assignments."""
        current = self._fov_filter.currentText()
        choices = tuple(sorted(set(self._fov_groups.values())))
        self._fov_filter.blockSignals(True)
        try:
            self._fov_filter.clear()
            self._fov_filter.addItems(choices)
            if current in choices:
                self._fov_filter.setCurrentText(current)
            elif len(choices) > 0:
                self._fov_filter.setCurrentIndex(0)
        finally:
            self._fov_filter.blockSignals(False)
        self.refresh()

    def set_default_output_folder(self, folder: Path) -> None:
        """Set the default ROI match CSV beside the finalized FOV CSV."""
        current_path = self.output_path()
        default_path = Path.cwd() / MATCH_TABLE_FILENAME
        if current_path == default_path:
            self._path_edit.setText(str(folder.expanduser() / MATCH_TABLE_FILENAME))
            self.load_match_rows_from_path()

    def refresh(self) -> None:
        """Refresh the visual ROI cards from current FOV filter state."""
        _clear_grid_layout(self._grid)
        self._cards.clear()
        recordings = self._filtered_recordings()
        self._normalization_widget.set_epoch_choices(
            _epoch_names_for_recordings(recordings),
            selected_epoch_number=self._normalization_options.epoch_number,
        )
        for index, recording in enumerate(recordings):
            recording_path = recording.recording.source_session_dir.expanduser()
            card = RoiRecordingCard(
                recording=recording,
                fov_group_id=self._fov_groups.get(recording_path, ""),
                selected_roi=self._current_rois.get(recording_path, ""),
                trace_color=_trace_color(index),
                on_selection_changed=self._handle_roi_selection_changed,
            )
            self._cards.append(card)
            self._grid.addWidget(card, index // 2, index % 2)
        self._grid.setRowStretch((len(self._cards) + 1) // 2, 1)
        if len(recordings) == 0:
            self._status_label.setText("No recordings assigned to this FOV.")
        elif self._status_label.text() == "No recordings assigned to this FOV.":
            self._status_label.setText("Choose ROIs in the cards, then save a group.")
        self._refresh_group_table()
        self.refresh_response_preview()

    def refresh_response_preview(self) -> None:
        """Refresh the shared response preview from selected card ROIs."""
        self._sync_current_rois_from_cards()
        clear_layout(self._legend_layout)
        clear_layout(self._response_layout)
        selected_responses = self._selected_response_data()
        self._hidden_response_recordings.intersection_update(
            {response.recording_path for response in selected_responses},
        )
        if len(selected_responses) == 0:
            self._response_status.setText("Choose ROIs to compare responses.")
            return
        _add_response_legend(
            self._legend_layout,
            selected_responses,
            hidden_recordings=self._hidden_response_recordings,
            set_visible=self._set_response_visible,
        )
        visible_responses = tuple(
            response
            for response in selected_responses
            if response.recording_path not in self._hidden_response_recordings
        )
        if len(visible_responses) == 0:
            self._response_status.setText("All selected recording traces are hidden.")
            return
        combined = _combined_response_plot_data(visible_responses)
        if combined is None:
            self._response_status.setText(
                "Responses unavailable for the selected ROI combination.",
            )
            return
        self._response_status.setText("")
        _add_response_preview_widgets(
            self._response_layout,
            combined,
            roi_colors=tuple(response.color for response in visible_responses),
        )

    def _set_response_visible(self, recording_path: Path, visible: bool) -> None:
        """Show or hide one selected recording in the shared response preview."""
        if visible:
            self._hidden_response_recordings.discard(recording_path)
        else:
            self._hidden_response_recordings.add(recording_path)
        self.refresh_response_preview()

    def add_match_group(self) -> None:
        """Append current selected ROIs as a new matched cell group."""
        selected_rois = self._selected_rois_by_recording()
        if len(selected_rois) == 0:
            self._status_label.setText("Choose at least one ROI before adding a group.")
            return
        existing_rows = self._existing_rows()
        rows, group_cell_id = add_manual_roi_match_group(
            existing_rows,
            selected_rois,
            fov_group_id=self._fov_filter.currentText(),
            note=self._note_edit.text(),
        )
        save_manual_roi_match_rows(rows, self.output_path())
        self._status_label.setText(
            f"Added group {group_cell_id} with {len(selected_rois)} ROIs.",
        )
        self._refresh_group_table()
        self._select_group_in_table(group_cell_id)
        self._set_note_text("")
        self._selected_group_dirty = False

    def save_selected_match_group(self) -> None:
        """Overwrite the selected saved group with current selected ROIs."""
        group_cell_id = self._selected_group_cell_id()
        if group_cell_id is None:
            self._status_label.setText("Select a saved group before overwriting.")
            return
        selected_rois = self._selected_rois_by_recording()
        if len(selected_rois) == 0:
            self._status_label.setText("Choose at least one ROI before saving a group.")
            return
        rows = replace_manual_roi_match_group(
            self._existing_rows(),
            selected_rois,
            fov_group_id=self._fov_filter.currentText(),
            group_cell_id=group_cell_id,
            note=self._note_edit.text(),
        )
        save_manual_roi_match_rows(rows, self.output_path())
        self._status_label.setText(
            f"Overwrote group {group_cell_id} with {len(selected_rois)} ROIs.",
        )
        self._refresh_group_table()
        self._select_group_in_table(group_cell_id)
        self._set_note_text("")
        self._selected_group_dirty = False

    def remove_selected_match_group(self) -> None:
        """Remove the selected saved group from the match CSV."""
        group_cell_id = self._selected_group_cell_id()
        if group_cell_id is None:
            self._status_label.setText("Select a saved group before removing.")
            return
        rows = remove_manual_roi_match_group(
            self._existing_rows(),
            fov_group_id=self._fov_filter.currentText(),
            group_cell_id=group_cell_id,
        )
        save_manual_roi_match_rows(rows, self.output_path())
        self._clear_group_table_selection()
        self._set_note_text("")
        self._selected_group_dirty = False
        self._status_label.setText(f"Removed group {group_cell_id}.")
        self._refresh_group_table()

    def close_group_matching_window(self) -> None:
        """Close the group-matching popup after saved decisions are reviewed."""
        if self._selected_group_cell_id() is not None and self._selected_group_dirty:
            self.save_selected_match_group()
        window = self.window()
        if window is not None:
            window.close()

    def return_to_fov_assignment(self) -> None:
        """Return to FOV assignment without closing the group-matching popup."""
        self._sync_current_rois_from_cards()
        self._on_back()

    def load_match_row_path(self) -> None:
        """Choose an existing ROI match CSV path and load its rows."""
        selected_path = self._choose_existing_match_path()
        if selected_path is None:
            return
        self._path_edit.setText(str(selected_path))
        self.load_match_rows_from_path()

    def browse_match_row_save_path(self) -> None:
        """Choose where future ROI match groups will be saved."""
        selected_path = self._choose_match_save_path()
        if selected_path is None:
            return
        self._path_edit.setText(str(selected_path))
        if selected_path.exists():
            self.load_match_rows_from_path()
        else:
            self._status_label.setText(f"ROI CSV will be saved to {selected_path}.")

    def load_match_rows_from_path(self) -> None:
        """Load existing ROI match rows into the saved-groups table."""
        path = self.output_path()
        if not path.exists():
            self._refresh_group_table()
            self._status_label.setText(f"ROI CSV will be saved to {path}.")
            return
        try:
            rows = load_manual_roi_match_rows(path)
        except (OSError, ValueError) as error:
            self._status_label.setText(f"Could not load ROI CSV: {error}")
            return
        self._refresh_group_table()
        self._status_label.setText(f"Loaded {len(rows)} ROI match rows from {path}.")

    def clear_current_group(self) -> None:
        """Clear selected ROIs in the current FOV filter."""
        self._clear_group_table_selection()
        for card in self._cards:
            card.set_selected_roi("")
            self._current_rois.pop(card.recording_path, None)
        self._selected_group_dirty = False
        self._status_label.setText("Cleared ROI selection.")
        self.refresh_response_preview()

    def output_path(self) -> Path:
        """Return the current manual ROI match CSV path."""
        return Path(self._path_edit.text()).expanduser()

    def _choose_existing_match_path(self) -> Path | None:
        """Return a user-selected existing ROI match CSV path."""
        current_path = self.output_path()
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Load ROI matches CSV")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("CSV files (*.csv);;All files (*)")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setDirectory(str(current_path.parent))
        dialog.selectFile(current_path.name)
        return _selected_dialog_path(dialog)

    def _choose_match_save_path(self) -> Path | None:
        """Return a user-selected path for saving ROI match rows."""
        current_path = self.output_path()
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Choose ROI matches save path")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilter("CSV files (*.csv);;All files (*)")
        dialog.setDefaultSuffix("csv")
        dialog.setOption(QFileDialog.Option.DontConfirmOverwrite, True)
        dialog.setDirectory(str(current_path.parent))
        dialog.selectFile(current_path.name)
        return _selected_dialog_path(dialog)

    def _set_normalization_options(self, options: NormalizationOptions) -> None:
        """Update response normalization options and redraw previews."""
        self._normalization_options = options
        self.refresh_response_preview()

    def _filtered_recordings(self) -> tuple[LoadedNapariRecording, ...]:
        """Return loaded recordings visible under the current FOV filter."""
        return tuple(
            recording
            for recording in self._state.loaded_recordings
            if self._recording_matches_filter(
                recording.recording.source_session_dir.expanduser(),
            )
        )

    def _recording_matches_filter(self, recording_path: Path) -> bool:
        """Return whether one recording path belongs in the active FOV filter."""
        selected_fov = self._fov_filter.currentText()
        return (
            selected_fov != "" and self._fov_groups.get(recording_path) == selected_fov
        )

    def _selected_rois_by_recording(self) -> dict[Path, str]:
        """Return selected non-empty ROI labels keyed by visible recording."""
        selected: dict[Path, str] = {}
        for card in self._cards:
            roi_label = card.selected_roi_label()
            if roi_label is not None:
                selected[card.recording_path] = roi_label
        return selected

    def _sync_current_rois_from_cards(self) -> None:
        """Mirror current visible selector state into the shared popup state."""
        visible_paths = {card.recording_path for card in self._cards}
        for recording_path in visible_paths:
            self._current_rois.pop(recording_path, None)
        self._current_rois.update(self._selected_rois_by_recording())

    def _handle_roi_selection_changed(self) -> None:
        """Refresh previews and remember selected saved groups with edits."""
        self._mark_selected_group_dirty()
        self.refresh_response_preview()

    def _mark_selected_group_dirty(self) -> None:
        """Mark the selected saved group as edited by the user."""
        if self._selected_group_cell_id() is not None:
            self._selected_group_dirty = True

    def _set_note_text(self, text: str) -> None:
        """Set the note field without marking a saved group edited."""
        self._note_edit.blockSignals(True)
        try:
            self._note_edit.setText(text)
        finally:
            self._note_edit.blockSignals(False)

    def _existing_rows(self) -> tuple[ManualRoiMatchRow, ...]:
        """Load existing saved rows, treating a missing file as empty."""
        output_path = self.output_path()
        return load_manual_roi_match_rows(output_path) if output_path.exists() else ()

    def _matched_groups_for_filter(self) -> tuple[ManualRoiMatchGroup, ...]:
        """Return saved matched groups for the active FOV filter."""
        fov_group_id = self._fov_filter.currentText()
        if fov_group_id == "":
            return ()
        return matched_manual_roi_groups(
            self._existing_rows(),
            fov_group_id=fov_group_id,
        )

    def _refresh_group_table(self) -> None:
        """Refresh the saved matched-group table for the selected FOV."""
        groups = self._matched_groups_for_filter()
        self._group_table.blockSignals(True)
        try:
            self._group_table.setRowCount(0)
            for row_index, group in enumerate(groups):
                self._group_table.insertRow(row_index)
                group_item = QTableWidgetItem(str(group.group_cell_id))
                group_item.setData(Qt.ItemDataRole.UserRole, group.group_cell_id)
                self._group_table.setItem(row_index, 0, group_item)
                self._group_table.setItem(
                    row_index,
                    1,
                    QTableWidgetItem(self._group_summary(group.rows)),
                )
                self._group_table.setItem(
                    row_index,
                    2,
                    QTableWidgetItem(group.note),
                )
        finally:
            self._group_table.blockSignals(False)

    def _group_summary(self, rows: tuple[ManualRoiMatchRow, ...]) -> str:
        """Return a compact recording-to-ROI summary for one saved group."""
        rows_by_path = {row.recording_path.expanduser(): row for row in rows}
        parts: list[str] = []
        for card in self._cards:
            row = rows_by_path.get(card.recording_path)
            roi_label = row.roi_label if row is not None else "none"
            parts.append(f"{card.recording_path.name}: {roi_label}")
        return " | ".join(parts)

    def _restore_selected_group(self) -> None:
        """Restore ROI selectors from the currently selected saved group row."""
        group_cell_id = self._selected_group_cell_id()
        if group_cell_id is None:
            return
        group = self._matched_group_by_id(group_cell_id)
        if group is None:
            return
        rois_by_path = {
            row.recording_path.expanduser(): row.roi_label for row in group.rows
        }
        for card in self._cards:
            card.set_selected_roi(rois_by_path.get(card.recording_path, ""))
        self._set_note_text(group.note)
        self._selected_group_dirty = False
        self._status_label.setText(f"Loaded saved group {group_cell_id}.")
        self.refresh_response_preview()

    def _matched_group_by_id(self, group_cell_id: int) -> ManualRoiMatchGroup | None:
        """Return the active-FOV matched group with this id, if present."""
        for group in self._matched_groups_for_filter():
            if group.group_cell_id == group_cell_id:
                return group
        return None

    def _selected_group_cell_id(self) -> int | None:
        """Return the currently selected saved group id, if any."""
        if len(self._group_table.selectedItems()) == 0:
            return None
        row_index = self._group_table.currentRow()
        if row_index < 0:
            return None
        item = self._group_table.item(row_index, 0)
        if item is None:
            return None
        group_cell_id = item.data(Qt.ItemDataRole.UserRole)
        return group_cell_id if isinstance(group_cell_id, int) else None

    def _select_group_in_table(self, group_cell_id: int) -> None:
        """Select one saved group row in the table."""
        for row_index in range(self._group_table.rowCount()):
            item = self._group_table.item(row_index, 0)
            if (
                item is not None
                and item.data(Qt.ItemDataRole.UserRole) == group_cell_id
            ):
                self._group_table.selectRow(row_index)
                return

    def _clear_group_table_selection(self) -> None:
        """Clear saved-group table selection without restoring a group."""
        self._group_table.blockSignals(True)
        try:
            self._group_table.clearSelection()
        finally:
            self._group_table.blockSignals(False)
        self._selected_group_dirty = False

    def _selected_response_data(self) -> tuple[_SelectedRoiResponse, ...]:
        """Return response data for card ROIs selected in the popup."""
        responses: list[_SelectedRoiResponse] = []
        for card in self._cards:
            roi_label = card.selected_roi_label()
            if roi_label is None:
                continue
            try:
                plot_data = _selected_roi_response_plot_data(
                    card.recording,
                    roi_label,
                    normalization_options=self._normalization_options,
                )
            except ValueError:
                continue
            responses.append(
                _SelectedRoiResponse(
                    recording_path=card.recording_path,
                    roi_label=roi_label,
                    plot_data=plot_data,
                    color=card.trace_color,
                ),
            )
        return tuple(responses)


class RoiRecordingCard(QFrame):
    """One visual ROI assignment card for a loaded recording."""

    def __init__(
        self,
        *,
        recording: LoadedNapariRecording,
        fov_group_id: str,
        selected_roi: str,
        trace_color: QColor,
        on_selection_changed: Callable[[], None],
    ) -> None:
        """Create one ROI card."""
        super().__init__()
        self.recording_path = recording.recording.source_session_dir.expanduser()
        self.recording = recording
        self.trace_color = trace_color
        self._image_label = QLabel()
        self._image_label.setObjectName("roi_preview_image")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._roi_selector = QComboBox()
        self._roi_selector.setObjectName("roi_selector")
        self._trace_label = QLabel("Trace color")
        self._trace_label.setStyleSheet(_trace_label_style(trace_color))

        self._roi_selector.addItem("No ROI", "")
        for roi_label in roi_labels_from_layer_data(recording.roi_labels_layer):
            self._roi_selector.addItem(roi_label, roi_label)
        if selected_roi:
            self._roi_selector.setCurrentText(selected_roi)
        self._roi_selector.currentIndexChanged.connect(
            lambda _index: self._refresh_after_selection_change(on_selection_changed),
        )

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QLabel("ROI"))
        controls.addWidget(self._roi_selector)

        side_panel = QVBoxLayout()
        side_panel.addWidget(QLabel(self.recording_path.name))
        side_panel.addWidget(QLabel(f"FOV: {fov_group_id}"))
        side_panel.addLayout(controls)
        side_panel.addWidget(self._trace_label)
        side_panel.addStretch(1)

        body = QHBoxLayout()
        body.addWidget(self._image_label)
        body.addLayout(side_panel, 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(body)
        self.setLayout(layout)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("roi_assignment_card")
        self.refresh_preview()

    def selected_roi_label(self) -> str | None:
        """Return the currently selected ROI label."""
        data = self._roi_selector.currentData()
        if not isinstance(data, str) or data == "":
            return None
        return data

    def set_selected_roi(self, roi_label: str) -> None:
        """Set the selected ROI label without emitting intermediate updates."""
        self._roi_selector.blockSignals(True)
        try:
            index = 0 if roi_label == "" else self._roi_selector.findData(roi_label)
            self._roi_selector.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self._roi_selector.blockSignals(False)
        self.refresh_preview()

    def refresh_preview(self) -> None:
        """Refresh selected-ROI image and response previews."""
        roi_label = self.selected_roi_label()
        pixmap = mean_image_roi_overlay_pixmap(
            mean_image_layer=self.recording.mean_image_layer,
            roi_labels_layer=self.recording.roi_labels_layer,
            roi_label=roi_label,
            selected_color=self.trace_color,
            contrast_percentile=100.0,
        )
        if pixmap is not None:
            self._image_label.setPixmap(pixmap)

    def _refresh_after_selection_change(self, callback: Callable[[], None]) -> None:
        """Refresh the ROI image and shared response plot after selection."""
        self.refresh_preview()
        callback()


def roi_labels_from_layer_data(layer: object | None) -> tuple[str, ...]:
    """Return selectable twopy ROI labels from a napari Labels layer.

    Args:
        layer: Napari Labels layer or a test double exposing integer ``data``.

    Returns:
        Sorted ``roi_####`` labels for positive values present in the label
        image. Zero is background and is omitted.
    """
    if layer is None or not hasattr(layer, "data"):
        return ()
    labels_layer = cast(_LayerWithData, layer)
    label_image = np.asarray(labels_layer.data)
    if label_image.size == 0:
        return ()
    labels = sorted(
        {
            int(label_value)
            for label_value in np.unique(label_image)
            if int(label_value) > 0
        },
    )
    return tuple(f"roi_{label_value:04d}" for label_value in labels)


def _selected_roi_response_plot_data(
    recording: LoadedNapariRecording,
    roi_label: str,
    *,
    normalization_options: NormalizationOptions,
) -> ResponsePlotData:
    """Compute response plot data for one selected ROI label."""
    label_value = _roi_label_value(roi_label)
    if label_value is None or recording.roi_labels_layer is None:
        msg = "Selected ROI is missing."
        raise ValueError(msg)
    label_image = np.asarray(cast(_LayerWithData, recording.roi_labels_layer).data)
    selected = np.where(label_image.astype(np.int64, copy=False) == label_value, 1, 0)
    request = response_analysis_request_from_labels(
        recording.recording,
        _SelectedRoiLayer(data=selected.astype(np.int64, copy=False)),
        source_path=recording.recording.source_session_dir.expanduser(),
        response_processing_options=ResponseProcessingOptions(
            normalization=normalization_options,
        ),
    )
    return compute_response_preview(request)


def _combined_response_plot_data(
    selected_responses: tuple[_SelectedRoiResponse, ...],
) -> ResponsePlotData | None:
    """Return one plot data object with selected recordings overlaid as ROIs."""
    if len(selected_responses) == 0:
        return None
    reference_epochs = selected_responses[0].plot_data.epochs
    if len(reference_epochs) == 0:
        return None
    roi_labels = tuple(
        f"{response.recording_path.name} {response.roi_label}"
        for response in selected_responses
    )
    combined_epochs: list[EpochResponsePlotData] = []
    for reference_epoch in reference_epochs:
        mean_values, sem_values = _combined_epoch_values(
            reference_epoch,
            selected_responses,
        )
        combined_epochs.append(
            EpochResponsePlotData(
                epoch_name=reference_epoch.epoch_name,
                epoch_number=reference_epoch.epoch_number,
                roi_labels=roi_labels,
                time_seconds=reference_epoch.time_seconds,
                mean_values=mean_values,
                sem_values=sem_values,
                epoch_time_spans=reference_epoch.epoch_time_spans,
            ),
        )
    return ResponsePlotData(source_path=None, epochs=tuple(combined_epochs))


def _combined_epoch_values(
    reference_epoch: EpochResponsePlotData,
    selected_responses: tuple[_SelectedRoiResponse, ...],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return stacked mean and SEM rows for one shared comparison epoch."""
    key = (reference_epoch.epoch_number, reference_epoch.epoch_name)
    mean_rows: list[npt.NDArray[np.float64]] = []
    sem_rows: list[npt.NDArray[np.float64]] = []
    for response in selected_responses:
        epoch = _matching_epoch(response.plot_data, key=key)
        if epoch is None or not _same_time_axis(
            reference_epoch.time_seconds,
            epoch.time_seconds,
        ):
            missing_row = np.full_like(reference_epoch.time_seconds, np.nan)
            mean_rows.append(missing_row)
            sem_rows.append(missing_row)
        else:
            mean_rows.append(epoch.mean_values[0])
            sem_rows.append(epoch.sem_values[0])
    return np.vstack(mean_rows), np.vstack(sem_rows)


def _matching_epoch(
    plot_data: ResponsePlotData,
    *,
    key: tuple[int, str],
) -> EpochResponsePlotData | None:
    """Return the epoch matching one reference epoch key."""
    for epoch in plot_data.epochs:
        if (epoch.epoch_number, epoch.epoch_name) == key:
            return epoch
    return None


def _same_time_axis(
    first: npt.NDArray[np.float64],
    second: npt.NDArray[np.float64],
) -> bool:
    """Return whether two epoch traces can be overlaid directly."""
    return first.shape == second.shape and bool(np.allclose(first, second))


def _add_response_preview_widgets(
    layout: QHBoxLayout,
    plot_data: ResponsePlotData,
    *,
    roi_colors: tuple[QColor, ...],
) -> None:
    """Add one compact response plot per epoch as a horizontal strip."""
    epoch_indices = _visible_response_epoch_indices(plot_data)
    if len(epoch_indices) == 0:
        return
    time_min, time_max = global_time_bounds(plot_data, epoch_indices)
    roi_indices = tuple(range(len(roi_colors)))
    value_min, value_max = global_value_bounds(plot_data, roi_indices, epoch_indices)
    plot_colors = _trace_colors_with_opacity(roi_colors)
    for epoch_index in epoch_indices:
        epoch = plot_data.epochs[epoch_index]
        epoch_panel = QVBoxLayout()
        epoch_label = QLabel(_epoch_title(epoch))
        epoch_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        epoch_font = epoch_label.font()
        epoch_font.setBold(False)
        epoch_label.setFont(epoch_font)
        epoch_label.setStyleSheet("font-weight: normal;")
        epoch_panel.addWidget(epoch_label)
        epoch_panel.addWidget(
            EpochPlotWidget(
                epoch,
                show_sem=True,
                roi_indices=roi_indices,
                roi_colors=plot_colors,
                time_min=time_min,
                time_max=time_max,
                value_min=value_min,
                value_max=value_max,
                plot_size=_ROI_PREVIEW_PLOT_SIZE,
            ),
        )
        layout.addLayout(epoch_panel)
    layout.addStretch(1)


def _visible_response_epoch_indices(plot_data: ResponsePlotData) -> tuple[int, ...]:
    """Return ROI-comparison epochs, omitting baseline-like interleave rows."""
    return tuple(
        index
        for index, epoch in enumerate(plot_data.epochs)
        if not is_baseline_epoch_name(epoch.epoch_name)
    )


def _epoch_title(epoch: EpochResponsePlotData) -> str:
    """Return the compact title shown above one ROI comparison plot."""
    return f"{epoch.epoch_number}: {epoch.epoch_name}"


def _trace_colors_with_opacity(colors: tuple[QColor, ...]) -> tuple[QColor, ...]:
    """Return response-trace colors at the ROI matching preview opacity."""
    faded: list[QColor] = []
    for color in colors:
        trace_color = QColor(color)
        trace_color.setAlpha(_ROI_TRACE_ALPHA)
        faded.append(trace_color)
    return tuple(faded)


def _add_response_legend(
    layout: QHBoxLayout,
    selected_responses: tuple[_SelectedRoiResponse, ...],
    *,
    hidden_recordings: set[Path],
    set_visible: Callable[[Path, bool], None],
) -> None:
    """Add clickable color toggles for overlaid recording traces."""
    for response in selected_responses:
        visible = response.recording_path not in hidden_recordings
        button = QPushButton(f"{response.recording_path.name} {response.roi_label}")
        button.setCheckable(True)
        button.setChecked(visible)
        button.setStyleSheet(_trace_button_style(response.color, visible=visible))
        button.clicked.connect(
            lambda checked, path=response.recording_path: set_visible(path, checked),
        )
        layout.addWidget(button)
    layout.addStretch(1)


def _trace_color(index: int) -> QColor:
    """Return the recording trace color for one visible ROI card."""
    colors = (
        QColor("#1f77b4"),
        QColor("#d95f02"),
        QColor("#2ca02c"),
        QColor("#9467bd"),
        QColor("#8c564b"),
        QColor("#17becf"),
    )
    return colors[index % len(colors)]


def _trace_label_style(color: QColor) -> str:
    """Return stylesheet text for one trace-color chip."""
    return (
        "QLabel {"
        f"background-color: {color.name()};"
        "color: white;"
        "font-weight: bold;"
        "padding: 2px 6px;"
        "border-radius: 3px;"
        "}"
    )


def _trace_button_style(color: QColor, *, visible: bool) -> str:
    """Return stylesheet text for one clickable trace visibility chip."""
    background = color.name() if visible else "#555555"
    return (
        "QPushButton {"
        f"background-color: {background};"
        "color: white;"
        "font-weight: bold;"
        "padding: 2px 6px;"
        "border-radius: 3px;"
        "border: 0;"
        "}"
    )


def _epoch_names_for_recordings(
    recordings: tuple[LoadedNapariRecording, ...],
) -> dict[int, str]:
    """Return stimulus epoch choices available to the current FOV filter."""
    epoch_names: dict[int, str] = {}
    for recording in recordings:
        for epoch_number, epoch_name in stimulus_epoch_names_by_number(
            recording.recording,
        ).items():
            epoch_names.setdefault(epoch_number, epoch_name)
    return epoch_names


def _roi_label_value(roi_label: str | None) -> int | None:
    """Return the integer Labels value encoded in ``roi_####`` text."""
    if roi_label is None:
        return None
    suffix = roi_label.removeprefix("roi_")
    return int(suffix) if suffix.isdecimal() else None


def _clear_grid_layout(layout: QGridLayout) -> None:
    """Remove all child widgets from a grid layout."""
    while layout.count() > 0:
        item = layout.takeAt(0)
        if item is None:
            continue
        widget = item.widget()
        if widget is not None:
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()


def _selected_dialog_path(dialog: QFileDialog) -> Path | None:
    """Return the accepted file dialog path, or ``None`` when cancelled."""
    if dialog.exec() != QFileDialog.DialogCode.Accepted:
        return None
    selected_files = dialog.selectedFiles()
    if len(selected_files) == 0:
        return None
    return Path(selected_files[0]).expanduser()
