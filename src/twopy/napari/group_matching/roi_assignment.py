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
from pathlib import Path
from typing import Protocol

import numpy as np
from qtpy.QtCore import QEvent, QObject, QSignalBlocker, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
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
    QSpinBox,
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
    SmoothingOptions,
)
from twopy.analysis.trials import is_baseline_epoch_name
from twopy.napari.group_matching import response_preview
from twopy.napari.group_matching.cards import (
    GROUP_MATCHING_CARD_SPACING,
    card_columns_for_width,
    clear_card_grid_layout,
)
from twopy.napari.group_matching.cards import (
    fov_group_display_id as _fov_group_display_id,
)
from twopy.napari.group_matching.images import (
    THUMBNAIL_SIZE as GROUP_MATCHING_THUMBNAIL_SIZE,
)
from twopy.napari.group_matching.responses import SelectedRoiResponseCache
from twopy.napari.group_matching.roi_cards import (
    ROI_CARD_HEIGHT,
    ROI_CARD_WIDTH,
    ROI_CONTROL_HEIGHT,
    RoiRecordingCard,
    roi_labels_from_layer_data,
)
from twopy.napari.group_matching.style import (
    group_matching_button,
    group_matching_section,
)
from twopy.napari.group_matching.tables import ToggleSelectionTable
from twopy.napari.plotting.data import EpochResponsePlotData, ResponsePlotData
from twopy.napari.plotting.form_controls import plot_form_layout, set_plot_control_width
from twopy.napari.plotting.normalization_options import NormalizationOptionsWidget
from twopy.napari.plotting.preview_strip import ResponsePreviewStrip
from twopy.napari.plotting.processing_options import SmoothingOptionsWidget
from twopy.napari.plotting.widgets import clear_layout
from twopy.napari.session import LoadedNapariRecording
from twopy.napari.text import configure_placeholder
from twopy.stimulus import stimulus_epoch_names_by_number

__all__ = [
    "RoiAssignmentView",
    "RoiRecordingCard",
    "roi_labels_from_layer_data",
]

MATCH_TABLE_FILENAME = "roi_matches.csv"
THUMBNAIL_SIZE = GROUP_MATCHING_THUMBNAIL_SIZE
_ROI_PREVIEW_PLOT_SIZE = 180
_ROI_TRACE_ALPHA = int(round(255 * 0.75))
_ROI_SIDEBAR_MIN_WIDTH = 340
_ROI_SIDEBAR_MAX_WIDTH = 370
_ROI_CARD_WIDTH = ROI_CARD_WIDTH
_ROI_CARD_HEIGHT = ROI_CARD_HEIGHT
_ROI_CONTROL_HEIGHT = ROI_CONTROL_HEIGHT
_FOV_FILTER_POPUP_MIN_WIDTH = 128
_ROI_TABLE_VISIBLE_ROWS = 5
_ROI_TABLE_ROW_HEIGHT = 24


class GroupMatchingState(Protocol):
    """Napari control-state fields needed by the ROI assignment view."""

    loaded_recordings: list[LoadedNapariRecording]


class RoiAssignmentView(QWidget):
    """Visual ROI assignment view filtered by finalized FOV groups."""

    def __init__(
        self,
        *,
        state: GroupMatchingState,
        fov_groups: dict[Path, str],
        current_rois: dict[Path, str],
        output_path: Path,
        on_output_path_changed: Callable[[Path], None],
        on_back: Callable[[], None],
    ) -> None:
        """Create the second-stage ROI assignment view.

        Args:
            state: Shared loaded-recording state.
            fov_groups: Recording-to-FOV assignments from the first stage.
            current_rois: Mutable unsaved ROI group keyed by recording path.
            output_path: Initial ROI match CSV path owned by the parent panel.
            on_output_path_changed: Callback for explicit ROI CSV path choices.
            on_back: Callback returning the popup to FOV assignment.
        """
        super().__init__()
        self._state = state
        self._fov_groups = fov_groups
        self._current_rois = current_rois
        self._on_output_path_changed = on_output_path_changed
        self._on_back = on_back
        self._cards: list[RoiRecordingCard] = []
        self._card_grid_rows = 0
        self._card_grid_columns = 0
        self._hidden_response_recordings: set[Path] = set()
        self._selected_responses: tuple[response_preview.SelectedRoiResponse, ...] = ()
        self._selected_group_dirty = False
        self._normalization_options = NormalizationOptions()
        self._smoothing_options = SmoothingOptions()
        self._response_cache = SelectedRoiResponseCache()
        self._plot_size = _ROI_PREVIEW_PLOT_SIZE
        self._epoch_visibility: dict[int, bool] = {}
        self._epoch_visibility_signature: tuple[tuple[int, str], ...] = ()
        self._status_label = QLabel("Choose ROIs in the cards, then save a group.")
        self._status_label.setObjectName("group_matching_caption")
        self._status_label.setMinimumWidth(0)
        self._status_label.setWordWrap(True)
        self._status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse,
        )
        self._response_status = QLabel("Choose ROIs to compare responses.")
        self._response_status.setWordWrap(True)
        self._plot_size_widget = self._create_plot_size_widget()
        self._epoch_widget = QGroupBox("Epoch")
        self._epoch_layout = QVBoxLayout()
        self._epoch_layout.setContentsMargins(6, 6, 6, 6)
        self._epoch_layout.setSpacing(4)
        self._epoch_widget.setLayout(self._epoch_layout)
        self._smoothing_widget = SmoothingOptionsWidget(
            self._smoothing_options,
            on_change=self._set_smoothing_options,
        )
        self._normalization_widget = self._create_normalization_widget()
        self._normalization_widget.setMinimumWidth(0)
        self._legend_widget = QWidget()
        self._legend_layout = QGridLayout()
        self._legend_layout.setContentsMargins(0, 0, 0, 0)
        self._legend_layout.setSpacing(6)
        self._legend_widget.setLayout(self._legend_layout)
        self._response_strip = ResponsePreviewStrip(
            "roi_response_preview",
            trace_alpha=_ROI_TRACE_ALPHA,
            title_pixel_size=response_preview.response_epoch_title_size,
        )
        self._response_widget = self._response_strip.widget
        self._mean_response_strip = ResponsePreviewStrip(
            "roi_mean_response_preview",
            trace_alpha=_ROI_TRACE_ALPHA,
            title_pixel_size=response_preview.response_epoch_title_size,
        )
        self._mean_response_widget = self._mean_response_strip.widget
        self._show_roi_responses_checkbox = QCheckBox("ROI responses")
        self._show_roi_responses_checkbox.setObjectName("show_roi_responses")
        self._show_roi_responses_checkbox.setChecked(True)
        self._show_roi_responses_checkbox.toggled.connect(
            self._set_response_panel_visibility,
        )
        self._show_combined_response_checkbox = QCheckBox("Combined response")
        self._show_combined_response_checkbox.setObjectName("show_combined_response")
        self._show_combined_response_checkbox.setChecked(True)
        self._show_combined_response_checkbox.toggled.connect(
            self._set_response_panel_visibility,
        )
        self._path_edit = QLineEdit(str(output_path.expanduser()))
        self._path_edit.setObjectName("roi_match_path")
        self._path_edit.setMinimumWidth(0)
        self._path_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        configure_placeholder(self._path_edit, "ROI CSV path")
        self._path_edit.editingFinished.connect(self._path_editing_finished)
        self._note_edit = QLineEdit("")
        self._note_edit.setObjectName("roi_match_note")
        configure_placeholder(self._note_edit, "Optional note")
        self._note_edit.textChanged.connect(
            lambda _text: self._mark_selected_group_dirty(),
        )
        self._fov_filter = QComboBox()
        self._fov_filter.setObjectName("fov_filter")
        self._fov_filter.setMinimumWidth(72)
        fov_filter_view = self._fov_filter.view()
        if fov_filter_view is not None:
            fov_filter_view.setMinimumWidth(_FOV_FILTER_POPUP_MIN_WIDTH)
        self._fov_filter.currentIndexChanged.connect(lambda _index: self.refresh())
        self._grid_widget = QWidget()
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(GROUP_MATCHING_CARD_SPACING)
        self._grid_widget.setLayout(self._grid)
        self._group_table = self._create_group_table()
        load_button = group_matching_button("Load ROI CSV")
        load_button.clicked.connect(self.load_match_row_path)
        browse_button = group_matching_button("Browse save path")
        browse_button.clicked.connect(self.browse_match_row_save_path)
        add_button = group_matching_button("Add new group", role="primary")
        add_button.clicked.connect(self.add_match_group)
        save_button = group_matching_button("Overwrite selected group")
        save_button.clicked.connect(self.save_selected_match_group)
        remove_button = group_matching_button("Remove selected group", role="danger")
        remove_button.clicked.connect(self.remove_selected_match_group)
        clear_button = group_matching_button("Clear ROI selection", role="quiet")
        clear_button.clicked.connect(self.clear_current_group)
        back_button = group_matching_button("Back to FOV assignment", role="quiet")
        back_button.clicked.connect(self.return_to_fov_assignment)
        close_button = group_matching_button("Save and close", role="primary")
        close_button.clicked.connect(self.close_group_matching_window)

        path_controls = QVBoxLayout()
        path_controls.addWidget(self._path_edit)
        path_button_row = QHBoxLayout()
        path_button_row.addWidget(load_button)
        path_button_row.addWidget(browse_button)
        path_controls.addLayout(path_button_row)

        fov_filter_label = QLabel("FOV ID")
        fov_filter_label.setObjectName("fov_id_label")
        controls = QHBoxLayout()
        controls.addStretch(1)
        controls.addWidget(fov_filter_label, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addWidget(self._fov_filter, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addStretch(1)

        note_row = QHBoxLayout()
        note_row.addWidget(QLabel("Note"))
        note_row.addWidget(self._note_edit, 1)

        match_action_panel = QVBoxLayout()
        match_action_panel.addLayout(note_row)
        match_action_panel.addWidget(add_button)
        match_action_panel.addWidget(save_button)
        match_action_panel.addWidget(remove_button)
        match_action_panel.addWidget(clear_button)
        match_action_panel.addStretch(1)

        plot_toggle_row = QHBoxLayout()
        plot_toggle_row.addStretch(1)
        plot_toggle_row.addWidget(self._show_roi_responses_checkbox)
        plot_toggle_row.addWidget(self._show_combined_response_checkbox)
        plot_toggle_row.addStretch(1)

        preview_settings_panel = QVBoxLayout()
        preview_settings_panel.addLayout(plot_toggle_row)
        preview_settings_panel.addWidget(self._plot_size_widget)
        preview_settings_panel.addWidget(self._epoch_widget)
        preview_settings_panel.addWidget(self._smoothing_widget)
        preview_settings_panel.addWidget(self._normalization_widget)
        preview_settings_panel.addStretch(1)

        finish_panel = QVBoxLayout()
        finish_panel.addWidget(back_button)
        finish_panel.addWidget(close_button)

        group_panel = QVBoxLayout()
        group_panel.setAlignment(Qt.AlignmentFlag.AlignTop)
        group_panel.addWidget(self._group_table, 1)

        card_section_layout = QVBoxLayout()
        card_section_layout.addWidget(self._grid_widget)

        selected_rois_panel = QVBoxLayout()
        selected_rois_panel.setContentsMargins(0, 0, 0, 0)
        selected_rois_panel.setSpacing(5)
        selected_rois_panel.addWidget(self._response_status)
        selected_rois_panel.addWidget(self._legend_widget)
        roi_response_panel = QVBoxLayout()
        roi_response_panel.setContentsMargins(0, 0, 0, 0)
        roi_response_panel.addWidget(self._response_widget)
        combined_response_panel = QVBoxLayout()
        combined_response_panel.setContentsMargins(0, 0, 0, 0)
        combined_response_panel.addWidget(self._mean_response_widget)

        title = QLabel("ROI assignment")
        title.setObjectName("group_matching_title")
        subtitle = QLabel(
            "Pick matching ROIs inside the selected FOV, then save the cell group.",
        )
        subtitle.setObjectName("group_matching_subtitle")
        subtitle.setWordWrap(True)

        left_column = QVBoxLayout()
        left_column.setSpacing(10)
        left_column.addWidget(title)
        left_column.addWidget(subtitle)
        left_column.addWidget(group_matching_section("ROI file", path_controls))
        left_column.addWidget(self._status_label)
        left_column.addWidget(group_matching_section("FOV filter", controls))
        left_column.addWidget(
            group_matching_section("Current selection", match_action_panel),
        )
        left_column.addWidget(
            group_matching_section("Saved groups", group_panel),
            1,
        )
        left_column.addWidget(
            group_matching_section("Plot settings", preview_settings_panel),
        )
        left_column.addWidget(group_matching_section("Finish", finish_panel))
        left_column.addStretch(1)

        left_content = QWidget()
        left_content.setMinimumWidth(0)
        left_content.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        left_content.setLayout(left_column)
        left_scroll = QScrollArea()
        left_scroll.setObjectName("roi_assignment_left_scroll")
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_content)
        left_scroll.setMinimumWidth(_ROI_SIDEBAR_MIN_WIDTH)
        left_scroll.setMaximumWidth(_ROI_SIDEBAR_MAX_WIDTH)

        right_column = QVBoxLayout()
        right_column.setSpacing(10)
        self._selected_rois_section = group_matching_section(
            "Selected ROIs",
            selected_rois_panel,
        )
        self._roi_response_section = group_matching_section(
            "ROI responses",
            roi_response_panel,
        )
        self._combined_response_section = group_matching_section(
            "Combined responses",
            combined_response_panel,
        )
        right_column.addWidget(self._selected_rois_section)
        right_column.addWidget(self._roi_response_section)
        right_column.addWidget(self._combined_response_section)
        right_column.addWidget(
            group_matching_section("ROI cards", card_section_layout),
            1,
        )
        right_content = QWidget()
        right_content.setLayout(right_column)
        self._workspace_scroll = QScrollArea()
        self._workspace_scroll.setObjectName("roi_assignment_workspace_scroll")
        self._workspace_scroll.setWidgetResizable(True)
        self._workspace_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self._workspace_scroll.setWidget(right_content)
        workspace_viewport = self._workspace_scroll.viewport()
        assert workspace_viewport is not None
        self._workspace_viewport = workspace_viewport
        self._workspace_viewport.installEventFilter(self)

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        workspace = QHBoxLayout()
        workspace.setSpacing(12)
        workspace.addWidget(left_scroll)
        workspace.addWidget(self._workspace_scroll, 1)
        outer_layout.addLayout(workspace)
        self.setLayout(outer_layout)
        self.refresh_fov_filter()

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:
        """Relayout fixed-size ROI cards when the workspace width changes."""
        if (
            a0 is getattr(self, "_workspace_viewport", None)
            and a1 is not None
            and a1.type() == QEvent.Type.Resize
        ):
            self._layout_cards()
            self._render_response_preview()
        return super().eventFilter(a0, a1)

    def _create_plot_size_widget(self) -> QGroupBox:
        """Return the ROI preview plot-size controls."""
        group = QGroupBox("Plot")
        layout = plot_form_layout()
        plot_size_spin = QSpinBox()
        plot_size_spin.setRange(120, 900)
        plot_size_spin.setSingleStep(20)
        plot_size_spin.setSuffix(" px")
        plot_size_spin.setValue(self._plot_size)
        plot_size_spin.valueChanged.connect(self._set_plot_size)
        set_plot_control_width(plot_size_spin)
        layout.addRow("Size", plot_size_spin)
        group.setLayout(layout)
        return group

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
        table = ToggleSelectionTable(0, 3)
        table.setObjectName("roi_match_group_table")
        table.setHorizontalHeaderLabels(("Group", "ROIs", "Note"))
        table.setFixedHeight(
            _ROI_TABLE_ROW_HEIGHT * _ROI_TABLE_VISIBLE_ROWS + _ROI_TABLE_ROW_HEIGHT + 4,
        )
        vertical_header = table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
            vertical_header.setDefaultSectionSize(_ROI_TABLE_ROW_HEIGHT)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        horizontal_header = table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setDefaultAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(0, 54)
        table.setColumnWidth(1, 190)
        table.setColumnWidth(2, 70)
        table.itemSelectionChanged.connect(self._restore_selected_group)
        return table

    def refresh_fov_filter(self) -> None:
        """Refresh the FOV filter choices from saved FOV assignments."""
        current = self._selected_fov_group_id()
        choices = tuple(sorted(set(self._fov_groups.values())))
        self._fov_filter.blockSignals(True)
        try:
            self._fov_filter.clear()
            for fov_group_id in choices:
                self._fov_filter.addItem(
                    _fov_group_display_id(fov_group_id),
                    fov_group_id,
                )
            current_index = self._fov_filter.findData(current)
            if current_index >= 0:
                self._fov_filter.setCurrentIndex(current_index)
            elif len(choices) > 0:
                self._fov_filter.setCurrentIndex(0)
        finally:
            self._fov_filter.blockSignals(False)
        self.refresh()

    def set_output_path(self, path: Path, *, load_rows: bool) -> None:
        """Set the current ROI match CSV path from parent-owned state."""
        self._path_edit.setText(str(path.expanduser()))
        if load_rows:
            self.load_match_rows_from_path()

    def refresh(self) -> None:
        """Refresh the visual ROI cards from current FOV filter state."""
        clear_card_grid_layout(self._grid, delete_widgets=True)
        self._cards.clear()
        recordings = self._filtered_recordings()
        self._response_cache.retain_recordings(
            tuple(
                recording.recording.path.expanduser()
                for recording in self._state.loaded_recordings
            ),
        )
        self._refresh_epoch_controls_from_names(_epoch_names_for_recordings(recordings))
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
        self._layout_cards()
        if len(recordings) == 0:
            self._status_label.setText("No recordings assigned to this FOV.")
        elif self._status_label.text() == "No recordings assigned to this FOV.":
            self._status_label.setText("Choose ROIs in the cards, then save a group.")
        self._refresh_group_table()
        self.refresh_response_preview()

    def _layout_cards(self) -> None:
        """Wrap fixed-size ROI cards to fit the current workspace width."""
        clear_card_grid_layout(self._grid, delete_widgets=False)
        for row_index in range(self._card_grid_rows + 1):
            self._grid.setRowStretch(row_index, 0)
        for column_index in range(self._card_grid_columns):
            self._grid.setColumnStretch(column_index, 0)
        columns = _roi_card_columns_for_width(
            self._workspace_viewport.width(),
            spacing=self._grid.spacing(),
        )
        for index, card in enumerate(self._cards):
            self._grid.addWidget(card, index // columns, index % columns)
        self._card_grid_columns = columns
        self._card_grid_rows = (len(self._cards) + columns - 1) // columns
        self._grid.setRowStretch(self._card_grid_rows, 1)

    def refresh_response_preview(self) -> None:
        """Refresh the shared response preview from selected card ROIs."""
        self._sync_current_rois_from_cards()
        self._selected_responses = self._selected_response_data()
        self._render_response_preview()

    def _render_response_preview(self) -> None:
        """Redraw cached ROI response previews without recomputing responses."""
        clear_layout(self._legend_layout)
        self._hidden_response_recordings.intersection_update(
            {response.recording_path for response in self._selected_responses},
        )
        self._update_selected_rois_section_title()
        if len(self._selected_responses) == 0:
            self._show_response_status("Choose ROIs to compare responses.")
            return
        response_preview.add_response_legend(
            self._legend_layout,
            self._selected_responses,
            hidden_recordings=self._hidden_response_recordings,
            set_visible=self._set_response_visible,
            max_width=self._workspace_viewport.width() - 48,
        )
        visible_responses = tuple(
            response
            for response in self._selected_responses
            if response.recording_path not in self._hidden_response_recordings
        )
        if len(visible_responses) == 0:
            self._show_response_status("All selected recording traces are hidden.")
            return
        combined = response_preview.combined_response_plot_data(visible_responses)
        if combined is None:
            self._show_response_status(
                "Responses unavailable for the selected ROI combination.",
            )
            return
        self._refresh_epoch_controls(combined)
        epoch_indices = self._visible_response_epoch_indices(combined)
        if len(epoch_indices) == 0:
            self._show_response_status("Choose at least one epoch to plot.")
            return
        self._set_response_status(
            response_preview.selected_response_visibility_status(
                total_count=len(self._selected_responses),
                visible_count=len(visible_responses),
            ),
        )
        self._response_strip.render(
            combined,
            epoch_indices=epoch_indices,
            roi_colors=tuple(response.color for response in visible_responses),
            plot_size=self._plot_size,
        )
        self._mean_response_strip.render(
            response_preview.mean_response_plot_data(combined),
            epoch_indices=epoch_indices,
            roi_colors=(QColor("#f2c14e"),),
            plot_size=self._plot_size,
        )
        self._set_response_panel_visibility()

    def _show_response_status(self, text: str) -> None:
        """Show a response status message and hide cached plot strips."""
        self._response_strip.clear()
        self._mean_response_strip.clear()
        self._set_response_status(text)

    def _set_response_status(self, text: str) -> None:
        """Show the Selected ROIs status text without changing cached plots."""
        self._response_status.setText(text)
        self._response_status.setVisible(True)

    def _set_response_visible(self, recording_path: Path, visible: bool) -> None:
        """Show or hide one selected recording in the shared response preview."""
        if visible:
            self._hidden_response_recordings.discard(recording_path)
        else:
            self._hidden_response_recordings.add(recording_path)
        self._render_response_preview()

    def _set_response_panel_visibility(self, _checked: bool | None = None) -> None:
        """Show or hide the two response-preview rows from plot settings."""
        show_roi_responses = self._show_roi_responses_checkbox.isChecked()
        show_combined_response = self._show_combined_response_checkbox.isChecked()
        self._roi_response_section.setVisible(show_roi_responses)
        self._combined_response_section.setVisible(show_combined_response)

    def _refresh_epoch_controls_from_names(self, epoch_names: dict[int, str]) -> None:
        """Refresh epoch checkboxes from metadata before responses are selected."""
        epochs = tuple(
            EpochResponsePlotData(
                epoch_name=epoch_name,
                epoch_number=epoch_number,
                roi_labels=(),
                time_seconds=np.array([], dtype=np.float64),
                mean_values=np.empty((0, 0), dtype=np.float64),
                sem_values=np.empty((0, 0), dtype=np.float64),
            )
            for epoch_number, epoch_name in sorted(epoch_names.items())
        )
        self._refresh_epoch_controls_for_epochs(epochs)

    def _refresh_epoch_controls(self, plot_data: ResponsePlotData) -> None:
        """Refresh epoch checkboxes from selected response plot data."""
        self._refresh_epoch_controls_for_epochs(plot_data.epochs)

    def _refresh_epoch_controls_for_epochs(
        self,
        epochs: tuple[EpochResponsePlotData, ...],
    ) -> None:
        """Render compact epoch visibility checkboxes."""
        signature = tuple((epoch.epoch_number, epoch.epoch_name) for epoch in epochs)
        if signature == self._epoch_visibility_signature:
            return
        self._epoch_visibility = {
            index: not is_baseline_epoch_name(epoch.epoch_name)
            for index, epoch in enumerate(epochs)
        }
        self._epoch_visibility_signature = signature
        _clear_nested_layout(self._epoch_layout)
        if len(epochs) == 0:
            self._epoch_layout.addWidget(QLabel("No epochs"))
            return
        button_row = QHBoxLayout()
        select_all_button = group_matching_button("Select all", role="quiet")
        select_none_button = group_matching_button("None", role="quiet")
        select_all_button.clicked.connect(lambda: self._set_all_epochs_visible(True))
        select_none_button.clicked.connect(lambda: self._set_all_epochs_visible(False))
        button_row.addWidget(select_all_button)
        button_row.addWidget(select_none_button)
        self._epoch_layout.addLayout(button_row)
        for index, epoch in enumerate(epochs):
            checkbox = QCheckBox(response_preview.response_epoch_title(epoch))
            checkbox.setObjectName("roi_epoch_visibility_checkbox")
            checkbox.setChecked(self._epoch_visibility.get(index, True))
            checkbox.stateChanged.connect(
                lambda _state, row=index, box=checkbox: self._set_epoch_visible(
                    row,
                    box.isChecked(),
                ),
            )
            self._epoch_layout.addWidget(checkbox)

    def _set_epoch_visible(self, index: int, visible: bool) -> None:
        """Set one epoch visibility flag and redraw cached response plots."""
        self._epoch_visibility[index] = visible
        self._render_response_preview()

    def _set_all_epochs_visible(self, visible: bool) -> None:
        """Set all epoch visibility flags and redraw cached response plots once."""
        checkboxes = self._epoch_widget.findChildren(
            QCheckBox,
            "roi_epoch_visibility_checkbox",
        )
        blockers = [QSignalBlocker(checkbox) for checkbox in checkboxes]
        try:
            for checkbox in checkboxes:
                checkbox.setChecked(visible)
        finally:
            del blockers
        for index in range(len(self._epoch_visibility)):
            self._epoch_visibility[index] = visible
        self._render_response_preview()

    def _visible_response_epoch_indices(
        self,
        plot_data: ResponsePlotData,
    ) -> tuple[int, ...]:
        """Return currently visible response epoch rows."""
        return tuple(
            index
            for index in range(len(plot_data.epochs))
            if self._epoch_visibility.get(index, True)
        )

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
            fov_group_id=self._selected_fov_group_id(),
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
            fov_group_id=self._selected_fov_group_id(),
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
            fov_group_id=self._selected_fov_group_id(),
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
        self._on_output_path_changed(self.output_path())
        self.load_match_rows_from_path()

    def browse_match_row_save_path(self) -> None:
        """Choose where future ROI match groups will be saved."""
        selected_path = self._choose_match_save_path()
        if selected_path is None:
            return
        self._path_edit.setText(str(selected_path))
        self._on_output_path_changed(self.output_path())
        if selected_path.exists():
            self.load_match_rows_from_path()
        else:
            self._status_label.setText(f"ROI CSV will be saved to {selected_path}")

    def load_match_rows_from_path(self) -> None:
        """Load existing ROI match rows into the saved-groups table."""
        path = self.output_path()
        if not path.exists():
            self._refresh_group_table()
            self._status_label.setText(f"ROI CSV will be saved to {path}")
            return
        try:
            rows = load_manual_roi_match_rows(path)
        except (OSError, ValueError) as error:
            self._status_label.setText(f"Could not load ROI CSV: {error}")
            return
        self._refresh_group_table()
        self._status_label.setText(f"Loaded {len(rows)} ROI match rows from {path}")

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

    def _path_editing_finished(self) -> None:
        """Load the typed ROI CSV path and mark it as user-owned."""
        self._on_output_path_changed(self.output_path())
        self.load_match_rows_from_path()

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

    def _set_smoothing_options(self, options: SmoothingOptions) -> None:
        """Update response smoothing options and redraw previews."""
        self._smoothing_options = options
        self.refresh_response_preview()

    def _set_plot_size(self, value: int) -> None:
        """Update response preview plot size and redraw previews."""
        self._plot_size = int(value)
        self._render_response_preview()

    def _selected_fov_group_id(self) -> str:
        """Return the stored FOV id for the compact visible filter value."""
        fov_group_id = self._fov_filter.currentData()
        return fov_group_id if isinstance(fov_group_id, str) else ""

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
        selected_fov = self._selected_fov_group_id()
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
        fov_group_id = self._selected_fov_group_id()
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
            # macOS Qt accessibility can log a table-cache warning if Hover
            # Text or another AX client reads rows during this rebuild. Qt
            # refreshes that cache itself; saved ROI rows are not corrupt.
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
        self._hidden_response_recordings.clear()
        if group_cell_id is None:
            for card in self._cards:
                card.set_selected_roi("")
                self._current_rois.pop(card.recording_path, None)
            self._set_note_text("")
            self._selected_group_dirty = False
            self._update_selected_rois_section_title()
            self.refresh_response_preview()
            return
        group = self._matched_group_by_id(group_cell_id)
        if group is None:
            self._update_selected_rois_section_title()
            return
        rois_by_path = {
            row.recording_path.expanduser(): row.roi_label for row in group.rows
        }
        for card in self._cards:
            card.set_selected_roi(rois_by_path.get(card.recording_path, ""))
        self._set_note_text(group.note)
        self._selected_group_dirty = False
        self._status_label.setText(f"Loaded saved group {group_cell_id}.")
        self._update_selected_rois_section_title()
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
        self._update_selected_rois_section_title()

    def _update_selected_rois_section_title(self) -> None:
        """Show the active saved group id in the selected-ROI subsection."""
        group_cell_id = self._selected_group_cell_id()
        title = (
            "Selected ROIs"
            if group_cell_id is None
            else f"Selected ROIs - Group {group_cell_id}"
        )
        self._selected_rois_section.setTitle(title)

    def _selected_response_data(
        self,
    ) -> tuple[response_preview.SelectedRoiResponse, ...]:
        """Return response data for card ROIs selected in the popup."""
        responses: list[response_preview.SelectedRoiResponse] = []
        for card in self._cards:
            roi_label = card.selected_roi_label()
            if roi_label is None:
                continue
            try:
                plot_data = response_preview.selected_roi_response_plot_data(
                    card.recording,
                    roi_label,
                    cache=self._response_cache,
                    normalization_options=self._normalization_options,
                    smoothing_options=self._smoothing_options,
                )
            except ValueError:
                continue
            responses.append(
                response_preview.SelectedRoiResponse(
                    recording_path=card.recording_path,
                    roi_label=roi_label,
                    plot_data=plot_data,
                    color=card.trace_color,
                ),
            )
        return tuple(responses)


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


def _roi_card_columns_for_width(width: int, *, spacing: int) -> int:
    """Return how many fixed-width ROI cards fit in one row."""
    return card_columns_for_width(
        width,
        card_width=_ROI_CARD_WIDTH,
        spacing=spacing,
    )


def _clear_nested_layout(layout: QLayout) -> None:
    """Remove widgets and child layouts from one preview layout."""
    while layout.count() > 0:
        item = layout.takeAt(0)
        if item is None:
            continue
        child_layout = item.layout()
        if child_layout is not None:
            _clear_nested_layout(child_layout)
            child_layout.setParent(None)
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
