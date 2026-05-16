"""Visual ROI assignment controls for manual group matching.

Inputs: loaded napari recordings, FOV assignments, and editable ROI Labels
layers.
Outputs: visual ROI cards plus plain CSV ROI match decisions.

The ROI assignment stage is intentionally self-contained. Each recording card
owns its ROI selector, mean-image overlay, response preview, and match actions
so users can compare cells without changing the active layer selection in the
main napari viewer.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.group_matching import (
    ManualRoiMatchRow,
    ManualRoiMatchStatus,
    append_manual_roi_match_rows,
    load_manual_roi_match_rows,
    make_manual_roi_match_rows,
    next_group_cell_id,
)
from twopy.napari.group_matching_images import mean_image_roi_overlay_pixmap
from twopy.napari.plotting.data import ResponsePlotData
from twopy.napari.plotting.widgets import (
    EpochPlotWidget,
    clear_layout,
    global_time_bounds,
    global_value_bounds,
)
from twopy.napari.responses import compute_response_plot_data_from_labels
from twopy.napari.session import LoadedNapariRecording

__all__ = [
    "RoiAssignmentView",
    "RoiRecordingCard",
    "roi_labels_from_layer_data",
]

MATCH_TABLE_FILENAME = "roi_matches.csv"
_ROI_PREVIEW_PLOT_SIZE = 180


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


class RoiAssignmentView(QWidget):
    """Visual ROI assignment view filtered by finalized FOV groups."""

    def __init__(
        self,
        *,
        state: GroupMatchingState,
        fov_groups: dict[Path, str],
        current_rois: dict[Path, str],
    ) -> None:
        """Create the second-stage ROI assignment view.

        Args:
            state: Shared loaded-recording state.
            fov_groups: Recording-to-FOV assignments from the first stage.
            current_rois: Mutable unsaved ROI group keyed by recording path.
        """
        super().__init__()
        self._state = state
        self._fov_groups = fov_groups
        self._current_rois = current_rois
        self._cards: list[RoiRecordingCard] = []
        self._status_label = QLabel("Choose ROIs in the cards, then save a group.")
        self._path_edit = QLineEdit(str(Path.cwd() / MATCH_TABLE_FILENAME))
        self._path_edit.setObjectName("roi_match_path")
        self._fov_filter = QComboBox()
        self._fov_filter.setObjectName("fov_filter")
        self._fov_filter.currentIndexChanged.connect(lambda _index: self.refresh())
        self._grid_widget = QWidget()
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(12)
        self._grid_widget.setLayout(self._grid)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self._grid_widget)

        accept_button = QPushButton("Save as same cell")
        accept_button.clicked.connect(self.accept_match_group)
        clear_button = QPushButton("Clear unsaved ROI group")
        clear_button.clicked.connect(self.clear_current_group)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("FOV"))
        controls.addWidget(self._fov_filter)
        controls.addStretch(1)
        controls.addWidget(accept_button)
        controls.addWidget(clear_button)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("ROI Assignment"))
        layout.addWidget(self._path_edit)
        layout.addLayout(controls)
        layout.addWidget(scroll_area)
        layout.addWidget(self._status_label)
        self.setLayout(layout)
        self.refresh_fov_filter()

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

    def refresh(self) -> None:
        """Refresh the visual ROI cards from current FOV filter state."""
        _clear_grid_layout(self._grid)
        self._cards.clear()
        recordings = self._filtered_recordings()
        saved_counts = self._saved_counts_by_recording()
        for index, recording in enumerate(recordings):
            recording_path = recording.display_path.expanduser()
            card = RoiRecordingCard(
                recording=recording,
                fov_group_id=self._fov_groups.get(recording_path, ""),
                selected_roi=self._current_rois.get(recording_path, ""),
                saved_count=saved_counts.get(recording_path, 0),
                on_add=self.add_recording_roi_to_group,
                on_unmatched=self.mark_recording_roi_unmatched,
            )
            self._cards.append(card)
            self._grid.addWidget(card, index // 2, index % 2)
        self._grid.setRowStretch((len(self._cards) + 1) // 2, 1)
        if len(recordings) == 0:
            self._status_label.setText("No recordings assigned to this FOV.")

    def add_recording_roi_to_group(self, recording: LoadedNapariRecording) -> None:
        """Add the card's selected ROI to the unsaved match group."""
        recording_path = recording.display_path.expanduser()
        roi_label = self._selected_roi_label(recording_path)
        if roi_label is None:
            self._status_label.setText("Choose an ROI in the card first.")
            return
        self._current_rois[recording_path] = roi_label
        self._status_label.setText(f"Added {roi_label} to the unsaved group.")
        for card in self._cards:
            if card.recording_path == recording_path:
                card.set_unsaved_roi(roi_label)
                return

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

    def mark_recording_roi_unmatched(self, recording: LoadedNapariRecording) -> None:
        """Persist the card's selected ROI as reviewed but unmatched."""
        recording_path = recording.display_path.expanduser()
        roi_label = self._selected_roi_label(recording_path)
        if roi_label is None:
            self._status_label.setText("Choose an ROI in the card first.")
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

    def _selected_roi_label(self, recording_path: Path) -> str | None:
        """Return the currently selected ROI label for one visible card."""
        for card in self._cards:
            if card.recording_path == recording_path:
                return card.selected_roi_label()
        return None

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
        return (
            selected_fov != "" and self._fov_groups.get(recording_path) == selected_fov
        )

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
            fov_group_id=self._fov_filter.currentText(),
            status=status,
        )

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


class RoiRecordingCard(QFrame):
    """One visual ROI assignment card for a loaded recording."""

    def __init__(
        self,
        *,
        recording: LoadedNapariRecording,
        fov_group_id: str,
        selected_roi: str,
        saved_count: int,
        on_add: Callable[[LoadedNapariRecording], None],
        on_unmatched: Callable[[LoadedNapariRecording], None],
    ) -> None:
        """Create one ROI card."""
        super().__init__()
        self.recording_path = recording.display_path.expanduser()
        self._recording = recording
        self._image_label = QLabel()
        self._image_label.setObjectName("roi_preview_image")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._roi_selector = QComboBox()
        self._roi_selector.setObjectName("roi_selector")
        self._unsaved_label = QLabel("")
        self._saved_label = QLabel(f"Saved rows: {saved_count}")
        self._response_status = QLabel("")
        self._response_status.setWordWrap(True)
        self._response_layout = QHBoxLayout()
        self._response_layout.setContentsMargins(0, 0, 0, 0)
        self._response_widget = QWidget()
        self._response_widget.setObjectName("roi_response_preview")
        self._response_widget.setLayout(self._response_layout)

        self._roi_selector.addItem("No ROI", "")
        for roi_label in roi_labels_from_layer_data(recording.roi_labels_layer):
            self._roi_selector.addItem(roi_label, roi_label)
        if selected_roi:
            self._roi_selector.setCurrentText(selected_roi)
        self._roi_selector.currentIndexChanged.connect(
            lambda _index: self.refresh_preview(),
        )

        add_button = QPushButton("Add to group")
        add_button.clicked.connect(lambda _checked=False: on_add(recording))
        unmatched_button = QPushButton("Save unmatched")
        unmatched_button.clicked.connect(lambda _checked=False: on_unmatched(recording))

        controls = QHBoxLayout()
        controls.addWidget(QLabel("ROI"))
        controls.addWidget(self._roi_selector)
        controls.addWidget(add_button)
        controls.addWidget(unmatched_button)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(self.recording_path.name))
        layout.addWidget(QLabel(f"FOV: {fov_group_id}"))
        layout.addWidget(self._image_label)
        layout.addLayout(controls)
        layout.addWidget(self._unsaved_label)
        layout.addWidget(self._saved_label)
        layout.addWidget(self._response_status)
        layout.addWidget(self._response_widget)
        self.setLayout(layout)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("roi_assignment_card")
        self.set_unsaved_roi(selected_roi)
        self.refresh_preview()

    def selected_roi_label(self) -> str | None:
        """Return the currently selected ROI label."""
        data = self._roi_selector.currentData()
        if not isinstance(data, str) or data == "":
            return None
        return data

    def set_unsaved_roi(self, roi_label: str) -> None:
        """Update the visible unsaved-group label."""
        text = f"Unsaved group: {roi_label}" if roi_label else "Unsaved group: none"
        self._unsaved_label.setText(text)

    def refresh_preview(self) -> None:
        """Refresh selected-ROI image and response previews."""
        roi_label = self.selected_roi_label()
        pixmap = mean_image_roi_overlay_pixmap(
            mean_image_layer=self._recording.mean_image_layer,
            roi_labels_layer=self._recording.roi_labels_layer,
            roi_label=roi_label,
            contrast_percentile=100.0,
        )
        if pixmap is not None:
            self._image_label.setPixmap(pixmap)
        self._render_response_preview(roi_label)

    def _render_response_preview(self, roi_label: str | None) -> None:
        """Render response plots for the currently selected ROI."""
        clear_layout(self._response_layout)
        if roi_label is None:
            self._response_status.setText("Choose an ROI to preview responses.")
            return
        try:
            plot_data = _selected_roi_response_plot_data(self._recording, roi_label)
        except ValueError as error:
            self._response_status.setText(f"Responses unavailable: {error}")
            return
        self._response_status.setText("")
        _add_response_preview_widgets(self._response_layout, plot_data)


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
) -> ResponsePlotData:
    """Compute response plot data for one selected ROI label."""
    label_value = _roi_label_value(roi_label)
    if label_value is None or recording.roi_labels_layer is None:
        msg = "Selected ROI is missing."
        raise ValueError(msg)
    label_image = np.asarray(cast(_LayerWithData, recording.roi_labels_layer).data)
    selected = np.where(label_image.astype(np.int64, copy=False) == label_value, 1, 0)
    return compute_response_plot_data_from_labels(
        recording.recording,
        _SelectedRoiLayer(data=selected.astype(np.int64, copy=False)),
        source_path=recording.display_path,
    )


def _add_response_preview_widgets(
    layout: QHBoxLayout,
    plot_data: ResponsePlotData,
) -> None:
    """Add one compact response plot per epoch to a card layout."""
    epoch_indices = tuple(range(len(plot_data.epochs)))
    if len(epoch_indices) == 0:
        return
    time_min, time_max = global_time_bounds(plot_data, epoch_indices)
    value_min, value_max = global_value_bounds(plot_data, (0,), epoch_indices)
    for epoch in plot_data.epochs:
        layout.addWidget(
            EpochPlotWidget(
                epoch,
                show_sem=True,
                roi_indices=(0,),
                roi_colors=(QColor("#d95f02"),),
                time_min=time_min,
                time_max=time_max,
                value_min=value_min,
                value_max=value_max,
                plot_size=_ROI_PREVIEW_PLOT_SIZE,
            ),
        )
    layout.addStretch(1)


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
            widget.setParent(None)
