"""Manual group-analysis controls for loaded napari recordings.

Inputs: recordings already loaded in the shared napari viewer, their mean-image
layers, and their ROI Labels layers.
Outputs: plain CSV FOV grouping and ROI match tables that downstream group
analysis can load without depending on napari.

The popup is intentionally staged. FOV assignment comes first because ROI
matching across fields of view should not be the default path.
"""

from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import QEvent
from qtpy.QtWidgets import (
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.group_matching import load_manual_fov_group_rows
from twopy.napari.group_matching.fov_assignment import (
    FOV_GROUP_TABLE_FILENAME,
    FovAssignmentView,
    FovRecordingCard,
    mean_image_thumbnail_pixmap,
)
from twopy.napari.group_matching.protocols import GroupMatchingState
from twopy.napari.group_matching.roi_assignment import (
    MATCH_TABLE_FILENAME,
    RoiAssignmentView,
    roi_labels_from_layer_data,
)
from twopy.napari.group_matching.style import (
    GROUP_MATCHING_OUTER_MARGIN,
    style_group_matching_panel,
)

__all__ = [
    "FovAssignmentView",
    "FovRecordingCard",
    "GroupMatchingPanel",
    "RoiAssignmentView",
    "mean_image_thumbnail_pixmap",
    "roi_labels_from_layer_data",
]


@dataclass
class GroupMatchingCsvPaths:
    """Session-level CSV paths for manual group matching.

    Args:
        fov_path: Current FOV assignment CSV path.
        roi_path: Current ROI match CSV path.
        roi_path_is_auto: Whether the ROI path still follows the FOV folder.

    Returns:
        Mutable path state owned by the group-matching panel.

    The ROI path should follow recording-list folders only while it is
    auto-managed. Once a user chooses a specific ROI CSV, that explicit choice
    is preserved across later FOV path changes.
    """

    fov_path: Path
    roi_path: Path
    roi_path_is_auto: bool = True

    @classmethod
    def default(cls) -> "GroupMatchingCsvPaths":
        """Return default CSV paths rooted at the current working directory."""
        return cls(
            fov_path=Path.cwd() / FOV_GROUP_TABLE_FILENAME,
            roi_path=Path.cwd() / MATCH_TABLE_FILENAME,
        )

    def retarget_folder(self, folder: Path) -> None:
        """Point auto-managed CSV paths at one recording-list folder."""
        self.fov_path = folder.expanduser() / FOV_GROUP_TABLE_FILENAME
        self._retarget_auto_roi_path()

    def set_fov_path(self, path: Path) -> None:
        """Set the FOV CSV path and move the auto ROI path beside it."""
        self.fov_path = path.expanduser()
        self._retarget_auto_roi_path()

    def set_manual_roi_path(self, path: Path) -> None:
        """Store an explicit ROI CSV path chosen by the user."""
        self.roi_path = path.expanduser()
        self.roi_path_is_auto = False

    def _retarget_auto_roi_path(self) -> None:
        """Move the ROI CSV path only while it is still auto-managed."""
        if self.roi_path_is_auto:
            self.roi_path = self.fov_path.parent / MATCH_TABLE_FILENAME


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
        self._csv_paths = GroupMatchingCsvPaths.default()
        self._theme_style_refreshing = False
        self._stack = QStackedWidget()
        self._fov_view = FovAssignmentView(
            state=state,
            fov_groups=self._fov_groups,
            fov_notes=self._fov_notes,
            output_path=self._csv_paths.fov_path,
            on_output_path_changed=self._set_fov_output_path,
            on_finalize=self.finalize_fov_assignments,
        )
        self._roi_view = RoiAssignmentView(
            state=state,
            fov_groups=self._fov_groups,
            current_rois=self._current_rois,
            output_path=self._csv_paths.roi_path,
            on_output_path_changed=self._set_manual_roi_output_path,
            on_back=self.show_fov_assignment,
        )
        self._stack.addWidget(self._fov_view)
        self._stack.addWidget(self._roi_view)

        style_group_matching_panel(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(
            GROUP_MATCHING_OUTER_MARGIN,
            GROUP_MATCHING_OUTER_MARGIN,
            GROUP_MATCHING_OUTER_MARGIN,
            GROUP_MATCHING_OUTER_MARGIN,
        )
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
            self._csv_paths.set_fov_path(self._fov_view.output_path())
            self._sync_roi_output_path(load_rows=True)
            self._roi_view.refresh_fov_filter()
            self._stack.setCurrentWidget(self._roi_view)

    def show_fov_assignment(self) -> None:
        """Return to the FOV assignment view."""
        self._stack.setCurrentWidget(self._fov_view)
        self._fov_view.refresh()

    def set_recording_csv_folder_defaults(self, folder: Path) -> None:
        """Point group-matching CSV defaults at a recording-list folder.

        Args:
            folder: Folder that owns a loaded-recordings CSV.

        Returns:
            None.

        Loading a recording-list CSV establishes the working folder for manual
        group matching. Existing FOV and ROI CSVs in that folder are loaded when
        present, but missing files still become the default save targets.
        """
        self._csv_paths.retarget_folder(folder)
        self._sync_csv_paths_to_views(load_roi_rows=True)
        self._fov_view.load_fov_groups_from_path()
        self._roi_view.refresh_fov_filter()

    def clear_loaded_recording_state(self) -> None:
        """Clear matching decisions that belong to unloaded recordings."""
        self._fov_groups.clear()
        self._fov_notes.clear()
        self._current_rois.clear()
        self._roi_view.refresh_fov_filter()

    def remove_loaded_recording_state(self, recording_path: Path) -> None:
        """Clear matching decisions that belong to one unloaded recording."""
        path = recording_path.expanduser()
        self._fov_groups.pop(path, None)
        self._fov_notes.pop(path, None)
        self._current_rois.pop(path, None)
        self._roi_view.refresh_fov_filter()

    def _set_fov_output_path(self, path: Path) -> None:
        """Store a user-visible FOV path change and retarget auto ROI output."""
        self._csv_paths.set_fov_path(path)
        self._sync_roi_output_path(load_rows=True)

    def _set_manual_roi_output_path(self, path: Path) -> None:
        """Store a user-chosen ROI path so future FOV changes preserve it."""
        self._csv_paths.set_manual_roi_path(path)

    def _sync_csv_paths_to_views(self, *, load_roi_rows: bool) -> None:
        """Mirror panel-owned CSV paths into both staged views."""
        self._fov_view.set_output_path(self._csv_paths.fov_path)
        self._sync_roi_output_path(load_rows=load_roi_rows)

    def _sync_roi_output_path(self, *, load_rows: bool) -> None:
        """Mirror the panel-owned ROI CSV path into the ROI view."""
        self._roi_view.set_output_path(self._csv_paths.roi_path, load_rows=load_rows)

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
