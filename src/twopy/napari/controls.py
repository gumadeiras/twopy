"""Magicgui controls for the twopy napari adapter.

Inputs: napari viewer, current ROI Labels layer, and paths chosen by the user.
Outputs: one tabbed right-sidebar dock for loading converted recordings,
tracking loaded recordings, and controlling response plots.

The controls own GUI callbacks only. Loading recordings still goes through the
same typed helpers used by scripts.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from magicgui.widgets import FileEdit
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QLabel,
    QListView,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from twopy.converted import RecordingData
from twopy.napari.errors import exception_message_for_user
from twopy.napari.group_matching import GroupMatchingPanel
from twopy.napari.load_workflow import (
    RecordingLoadFailure,
    RecordingLoadResult,
    SourceUnavailableWarning,
    load_recording_paths,
)
from twopy.napari.loaded_recordings_csv import (
    load_recording_paths_csv,
    write_loaded_recordings_csv,
)
from twopy.napari.paths import (
    DEFAULT_PATH_TEXT,
    PathInput,
    is_default_path,
)
from twopy.napari.protocols import NapariViewer
from twopy.napari.session import (
    LoadedNapariRecording,
    LoadedRecordingsPanel,
    render_loaded_recordings_panel,
    select_loaded_recording,
    unload_all_loaded_recordings,
    unload_loaded_recording,
)
from twopy.napari.sidebar import (
    TWOPY_SIDEBAR_MINIMUM_WIDTH,
    GroupMatchingWindowButton,
    create_twopy_sidebar_widget,
)
from twopy.napari.state import (
    read_last_recording_folder,
)

__all__ = ["NapariSidebarWidgets", "add_twopy_magicgui_controls"]


@dataclass
class NapariSidebarWidgets:
    """Widgets created for the twopy napari right sidebar.

    Inputs: Qt widgets and napari dock objects.
    Outputs: grouped references for scripts and tests.
    """

    load_widget: object
    loaded_recordings_widget: object
    group_matching_widget: object
    response_options_widget: object | None
    sidebar_widget: object
    sidebar_dock_widget: object


@dataclass
class NapariControlState:
    """Mutable state shared by the small magicgui controls.

    Inputs: napari viewer, current ROI Labels layer, and default save path.
    Outputs: state that button callbacks can update after loading recordings.

    The mutable state is intentionally narrow. It keeps GUI callback ownership
    in this module without moving recording loading or ROI conversion out of
    the typed core helpers.
    """

    viewer: NapariViewer
    roi_labels_layer: object | None
    roi_save_file: Path
    recording: RecordingData | None
    response_plot_widget: object | None
    trial_timeline_controller: object | None = None
    loaded_recordings: list[LoadedNapariRecording] = field(default_factory=list)
    selected_recording_index: int | None = None
    loaded_recordings_panel: LoadedRecordingsPanel | None = None
    group_matching_panel: object | None = None
    recording_required_widgets: list[object] = field(default_factory=list)
    recording_picker_path: Path | None = None
    recording_picker_display_text: str = DEFAULT_PATH_TEXT
    recording_picker_widget: FileEdit | None = None
    database_search_dialog: object | None = None
    is_loading: bool = False
    defer_timeline_updates: bool = False
    replace_selected_on_next_load: bool = False


def add_twopy_magicgui_controls(
    viewer: NapariViewer,
    *,
    roi_labels_layer: object | None,
    roi_save_file: Path,
    recording: RecordingData | None = None,
    response_plot_widget: object | None = None,
    response_options_widget: object | None = None,
    mean_image_layer: object | None = None,
    movie_layer: object | None = None,
    trial_timeline_controller: object | None = None,
    dock_name: str = "twopy",
    dock_area: str = "right",
) -> NapariSidebarWidgets:
    """Add the twopy right sidebar to a napari viewer.

    Args:
        viewer: Napari viewer that should receive the dock widget.
        roi_labels_layer: Editable napari Labels layer that contains ROI
            integer labels. ``None`` creates a disabled save path with a clear
            status message.
        roi_save_file: Default destination for saved ROI HDF5 files.
        recording: Optional loaded recording used to populate GUI defaults.
        response_plot_widget: Optional response plotting widget to refresh
            after a new recording is loaded.
        response_options_widget: Optional response-options tabs to place in the
            same right sidebar as the Load tab.
        mean_image_layer: Optional mean-image layer for an already loaded
            startup recording.
        movie_layer: Optional movie layer for an already loaded startup
            recording.
        trial_timeline_controller: Optional timeline controller that follows
            selected loaded recordings.
        dock_name: Dock widget title.
        dock_area: Napari dock area.

    Returns:
        Created load, loaded-recordings, and sidebar widgets plus the napari
        dock widget that contains them.

    The sidebar keeps twopy's right-side controls in one tabbed dock. The loader
    itself stays small and uses the same helpers as scripts, so a future plugin
    can wrap this without changing how recordings load.
    """
    state = NapariControlState(
        viewer=viewer,
        roi_labels_layer=roi_labels_layer,
        roi_save_file=roi_save_file,
        recording=recording,
        response_plot_widget=response_plot_widget,
        trial_timeline_controller=trial_timeline_controller,
    )
    if recording is not None and mean_image_layer is not None:
        state.loaded_recordings.append(
            LoadedNapariRecording(
                recording=recording,
                roi_save_file=roi_save_file,
                mean_image_layer=mean_image_layer,
                movie_layer=movie_layer,
                roi_labels_layer=roi_labels_layer,
            )
        )
        state.selected_recording_index = 0
    load_widget = _make_twopy_load_widget(state)
    group_matching_widget = GroupMatchingPanel(state)
    state.group_matching_panel = group_matching_widget
    response_load_button = _response_load_tab_button(response_plot_widget)
    group_matching_button = GroupMatchingWindowButton(group_matching_widget)
    save_recording_list_button = _save_loaded_recordings_button(state)
    state.recording_required_widgets = [
        widget
        for widget in (
            save_recording_list_button,
            response_load_button,
            group_matching_button,
        )
        if widget is not None
    ]
    loaded_recordings_widget = _make_loaded_recordings_widget(state)
    sidebar_widget = create_twopy_sidebar_widget(
        load_widget=load_widget,
        loaded_recordings_widget=loaded_recordings_widget,
        save_recording_list_button=save_recording_list_button,
        group_matching_button=group_matching_button,
        response_load_button=response_load_button,
        response_options_widget=response_options_widget,
    )
    sidebar_dock_widget = viewer.window.add_dock_widget(
        sidebar_widget,
        name=dock_name,
        area=dock_area,
    )
    if isinstance(sidebar_dock_widget, QWidget):
        sidebar_dock_widget.setMinimumWidth(TWOPY_SIDEBAR_MINIMUM_WIDTH)
    return NapariSidebarWidgets(
        load_widget=load_widget,
        loaded_recordings_widget=loaded_recordings_widget,
        group_matching_widget=group_matching_widget,
        response_options_widget=response_options_widget,
        sidebar_widget=sidebar_widget,
        sidebar_dock_widget=sidebar_dock_widget,
    )


def _make_twopy_load_widget(state: NapariControlState) -> object:
    """Create the right-side widget for loading recordings.

    Args:
        state: Mutable napari control state.

    Returns:
        Qt widget with compact recording-load controls.
    """
    from magicgui import magicgui

    @magicgui(
        call_button=False,
        layout="vertical",
    )
    def load_recording(
        recording_folder: Path = Path(DEFAULT_PATH_TEXT),
        roi_file_to_load: Path = Path(DEFAULT_PATH_TEXT),
    ) -> str:
        """Load a recording into the current napari viewer.

        Args:
            recording_folder: Source recording folder, converted folder, or
                direct ``recording_data.h5`` path. Source folders are converted
                first when twopy output does not exist yet.
            roi_file_to_load: Optional existing ROI HDF5 file. ``default`` uses
                ``rois.h5`` in the selected recording folder when it exists.

        Returns:
            Human-readable status for the dock widget.
        """
        replace_selected = state.replace_selected_on_next_load
        state.replace_selected_on_next_load = False
        selected_recording_path = _resolve_recording_folder_value(
            recording_folder,
            state,
        )
        try:
            result = load_recording_paths(
                state,
                (selected_recording_path,),
                roi_file_to_load=roi_file_to_load,
                replace_selected=replace_selected,
                remember_selected_folder=True,
            )
            if result.failures:
                return result.failures[0].message
            _show_source_unavailable_warnings(result.source_warnings)
            _sync_recording_picker_to_selected(state)
            return result.status_text
        finally:
            state.replace_selected_on_next_load = False

    load_recording.recording_folder.mode = "d"
    load_recording.roi_file_to_load.mode = "r"

    load_recording.recording_folder.label = "Recording"
    load_recording.roi_file_to_load.label = "ROI file"
    cast(FileEdit, load_recording.recording_folder).choose_btn.text = "Load manually"
    cast(FileEdit, load_recording.roi_file_to_load).choose_btn.text = "Load ROI"
    _configure_recording_folder_picker(
        cast(FileEdit, load_recording.recording_folder),
        state,
        lambda path: load_recording(recording_folder=path),
    )

    def load_after_selection(_value: object) -> None:
        """Load when the user changes the recording picker.

        Args:
            _value: Ignored widget signal value.

        Returns:
            None.
        """
        if state.is_loading or not _has_recording_to_load(
            load_recording.recording_folder.value,
            state,
        ):
            return
        load_recording()

    def reload_after_roi_change(_value: object) -> None:
        """Reload the current recording after the ROI file changes.

        Args:
            _value: Ignored widget signal value.

        Returns:
            None.

        The visible picker text may be shortened for display and is not always a
        real filesystem path, so reloads reuse the stored recording path.
        """
        if state.is_loading or not _has_recording_to_load(
            load_recording.recording_folder.value,
            state,
        ):
            return
        state.replace_selected_on_next_load = True
        load_recording(
            recording_folder=_current_recording_path_for_reload(
                load_recording.recording_folder.value,
                state,
            ),
        )

    load_recording.recording_folder.changed.connect(load_after_selection)
    load_recording.roi_file_to_load.changed.connect(reload_after_roi_change)

    return LoadRecordingPanel(
        load_recording=load_recording,
        on_search_database=lambda: _show_database_search_dialog(state),
        on_load_manually=lambda: _load_manual_recordings_from_dialog(state),
        on_load_csv_list=lambda: _load_recording_csvs_from_dialog(state),
    )


def _save_loaded_recordings_button(state: NapariControlState) -> QPushButton:
    """Return the Load-tab button that exports the loaded-recording list."""
    button = QPushButton("Save loaded list")
    button.clicked.connect(
        lambda _checked=False: _save_loaded_recordings_from_dialog(state)
    )
    return button


def _response_load_tab_button(response_plot_widget: object | None) -> object | None:
    """Return the response action button that belongs on the Load tab.

    Args:
        response_plot_widget: Optional response plot widget.

    Returns:
        Qt button exposed by the response plot widget, or ``None``.
    """
    if response_plot_widget is None:
        return None
    load_tab_button = getattr(response_plot_widget, "load_tab_button", None)
    if load_tab_button is None:
        return None
    button = load_tab_button()
    if isinstance(button, QWidget):
        return button
    msg = f"Response load action expected a QWidget, got {type(button).__name__}."
    raise TypeError(msg)


class LoadRecordingPanel(QWidget):
    """Compact Qt panel that owns the recording-load controls.

    Args:
        load_recording: Magicgui callable widget whose child controls and
            callbacks perform recording loading.

    Outputs:
        QWidget with a tight form layout for the Load tab.

    The magicgui callable still owns validation and callback wiring. This panel
    only arranges the existing controls with shorter labels so the path fields
    and frame slider have enough horizontal room.
    """

    load_recording: object

    def __init__(
        self,
        *,
        load_recording: object,
        on_search_database: Callable[[], None],
        on_load_manually: Callable[[], None],
        on_load_csv_list: Callable[[], None],
    ) -> None:
        """Create a compact Load Recording panel.

        Args:
            load_recording: Magicgui callable widget built by
                ``_make_twopy_load_widget``.
            on_search_database: Callback that opens the database-search window.
            on_load_manually: Callback that opens the manual multi-folder
                selection dialog.
            on_load_csv_list: Callback that opens the recording-list CSV picker.

        Returns:
            None.
        """
        super().__init__()
        self.load_recording = load_recording

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        title = QLabel("Load Recording")
        title.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        layout.addWidget(title)

        search_button = QPushButton("Search database")
        search_button.clicked.connect(on_search_database)
        layout.addWidget(search_button)
        load_manually_button = QPushButton("Load manually")
        load_manually_button.clicked.connect(lambda _checked=False: on_load_manually())
        layout.addWidget(load_manually_button)
        load_csv_button = QPushButton("Load CSV list")
        load_csv_button.clicked.connect(lambda _checked=False: on_load_csv_list())
        layout.addWidget(load_csv_button)
        self.setLayout(layout)


def _show_source_unavailable_warnings(
    warnings: tuple[SourceUnavailableWarning, ...],
) -> None:
    """Warn about cache-backed loads whose source paths were unavailable."""
    for warning in warnings:
        _show_source_unavailable_warning(
            source_path=warning.source_path,
            cache_path=warning.cache_path,
        )


def _show_source_unavailable_warning(*, source_path: Path, cache_path: Path) -> None:
    """Warn that napari opened cached data without source storage available.

    Args:
        source_path: Requested source recording path that is currently missing.
        cache_path: Cached converted HDF5 path being opened.

    Returns:
        None.
    """
    QMessageBox.warning(
        None,
        "Source path unavailable",
        (
            "The requested source path is unavailable, so twopy opened the local "
            "cached copy.\n\n"
            f"Source path:\n{source_path}\n\n"
            f"Cached data:\n{cache_path}\n\n"
            "Save and export outputs will not sync back to the source location "
            "until that path is available again."
        ),
    )


def _load_manual_recordings_from_dialog(state: NapariControlState) -> None:
    """Load all recording folders selected by the manual chooser.

    Args:
        state: Mutable napari control state.

    Returns:
        None.
    """
    selected_paths = _choose_recording_paths(state)
    _load_selected_recording_paths(state, selected_paths)


def _load_recording_csvs_from_dialog(state: NapariControlState) -> None:
    """Load recordings from selected recording-list CSV files.

    Args:
        state: Mutable napari control state.

    Returns:
        None.
    """
    selected_paths = _choose_recording_csv_paths(state)
    _load_selected_recording_paths(state, selected_paths)
    _load_adjacent_fov_groups_for_recording_csvs(state, selected_paths)


def _load_selected_recording_paths(
    state: NapariControlState,
    selected_paths: tuple[Path, ...],
) -> None:
    """Expand selected folders or CSVs, then load their recordings."""
    if len(selected_paths) == 0:
        return
    try:
        recording_paths = _recording_paths_from_manual_selections(selected_paths)
    except Exception as error:
        _show_selection_error(selected_paths, error)
        return
    result = load_recording_paths(
        state,
        recording_paths,
        remember_selected_folder=True,
    )
    _show_source_unavailable_warnings(result.source_warnings)
    _sync_recording_picker_to_selected(state)
    if result.failures:
        _show_load_errors(result)


def _choose_recording_paths(state: NapariControlState) -> tuple[Path, ...]:
    """Return manually selected recording folders.

    Args:
        state: Mutable napari control state used to choose the starting folder.

    Returns:
        Selected folder paths, or an empty tuple when the user cancels.

    Directory loading keeps a standard file dialog with normal navigation and
    places/sidebar affordances. The CSV path stays separate because Qt's normal
    dialog API does not support selecting directories and files in one clean mode.
    """
    dialog = QFileDialog()
    dialog.setWindowTitle("Choose recording folders")
    start_path = _recording_picker_start_path(state)
    if start_path is not None:
        dialog.setDirectory(start_path)
    dialog.setFileMode(QFileDialog.FileMode.Directory)
    dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    _allow_extended_dialog_selection(dialog)
    if dialog.exec() != QFileDialog.DialogCode.Accepted:
        return ()
    return tuple(Path(path).expanduser() for path in dialog.selectedFiles())


def _choose_recording_csv_paths(state: NapariControlState) -> tuple[Path, ...]:
    """Return selected loaded-recordings CSV files."""
    selected_paths, _selected_filter = QFileDialog.getOpenFileNames(
        None,
        "Choose loaded-recordings CSV",
        _recording_picker_start_path(state) or "",
        "CSV files (*.csv);;All files (*)",
    )
    return tuple(Path(path).expanduser() for path in selected_paths)


def _recording_paths_from_manual_selections(
    paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Expand selected CSV lists into concrete recording paths."""
    recording_paths: list[Path] = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            recording_paths.extend(load_recording_paths_csv(path))
        else:
            recording_paths.append(path)
    return tuple(recording_paths)


def _load_adjacent_fov_groups_for_recording_csvs(
    state: NapariControlState,
    csv_paths: tuple[Path, ...],
) -> None:
    """Load a FOV CSV beside the selected loaded-recordings CSV, if present."""
    panel = state.group_matching_panel
    if not isinstance(panel, GroupMatchingPanel):
        return
    for csv_path in csv_paths:
        if csv_path.suffix.lower() != ".csv":
            continue
        if panel.load_fov_groups_from_folder(csv_path.expanduser().parent):
            return


def _save_loaded_recordings_from_dialog(state: NapariControlState) -> None:
    """Save the current loaded-recordings list to a user-selected CSV path."""
    if len(state.loaded_recordings) == 0:
        return
    output_path = _choose_loaded_recordings_csv_save_path(state)
    if output_path is None:
        return
    try:
        write_loaded_recordings_csv(state.loaded_recordings, output_path)
    except Exception as error:
        _show_selection_error((output_path,), error)


def _choose_loaded_recordings_csv_save_path(state: NapariControlState) -> Path | None:
    """Return the CSV path selected for saving loaded recordings."""
    start_path = _recording_picker_start_path(state)
    suggested_path = (
        Path(start_path) / "loaded_recordings.csv"
        if start_path is not None
        else Path("loaded_recordings.csv")
    )
    selected_path, _selected_filter = QFileDialog.getSaveFileName(
        None,
        "Save loaded recordings CSV",
        str(suggested_path),
        "CSV files (*.csv);;All files (*)",
    )
    if selected_path == "":
        return None
    output_path = Path(selected_path).expanduser()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".csv")
    return output_path


def _show_selection_error(paths: tuple[Path, ...], error: Exception) -> None:
    """Show a manual-selection parse or save failure in the load error dialog."""
    path = paths[0] if len(paths) == 1 else None
    _show_load_errors(
        RecordingLoadResult(
            loaded_count=0,
            failures=(
                RecordingLoadFailure(
                    path=path,
                    message=exception_message_for_user(error),
                ),
            ),
        ),
    )


def _allow_extended_dialog_selection(dialog: QFileDialog) -> None:
    """Allow selecting more than one row in a non-native file dialog."""
    for view in (*dialog.findChildren(QListView), *dialog.findChildren(QTreeView)):
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)


def _show_load_errors(result: RecordingLoadResult) -> None:
    """Show recording-load failures in a scrollable error dialog."""
    from twopy.napari.database_search import RecordingLoadErrorDialog

    RecordingLoadErrorDialog(result).exec()


def _show_database_search_dialog(state: NapariControlState) -> None:
    """Open or raise the Load-tab database-search window.

    Args:
        state: Mutable napari control state.

    Returns:
        None.
    """
    from twopy.napari.database_search import ExperimentSearchDialog

    if not isinstance(state.database_search_dialog, ExperimentSearchDialog):
        state.database_search_dialog = ExperimentSearchDialog(
            on_load_recording_paths=lambda paths: _load_recordings_from_database_search(
                state,
                paths,
            ),
        )
    dialog = state.database_search_dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()


def _load_recordings_from_database_search(
    state: NapariControlState,
    paths: tuple[Path, ...],
) -> RecordingLoadResult:
    """Load source paths selected in database search and render warnings."""
    result = load_recording_paths(
        state,
        paths,
        remember_selected_folder=False,
    )
    _show_source_unavailable_warnings(result.source_warnings)
    _sync_recording_picker_to_selected(state)
    return result


def _make_loaded_recordings_widget(state: NapariControlState) -> object:
    """Create the loaded-recordings list panel.

    Args:
        state: Mutable napari control state.

    Returns:
        Qt panel that lists loaded recordings and can unload the selected one.
    """

    def select_recording(index: int | None) -> None:
        """Select a loaded recording and mirror it in the Load-tab picker."""
        select_loaded_recording(state, index)
        _sync_recording_picker_to_selected(state)

    def unload_recording(index: int) -> None:
        """Unload a recording and mirror the next active item in the picker."""
        unload_loaded_recording(state, index)
        _sync_recording_picker_to_selected(state)

    def unload_all_recordings() -> None:
        """Unload all recordings and reset the Load-tab picker."""
        unload_all_loaded_recordings(state)
        _sync_recording_picker_to_selected(state)

    panel = LoadedRecordingsPanel(
        on_select=select_recording,
        on_unload=unload_recording,
        on_unload_all=unload_all_recordings,
    )
    state.loaded_recordings_panel = panel
    render_loaded_recordings_panel(state)
    return panel


def _configure_recording_folder_picker(
    widget: FileEdit,
    state: NapariControlState,
    load_selected_path: Callable[[Path], object],
) -> None:
    """Keep the picker display compact while remembering dialog start folders.

    Args:
        widget: Recording-folder FileEdit widget.
        state: Current napari control state.
        load_selected_path: Callback that loads the selected recording path.

    Returns:
        None.

    Magicgui normally shows full paths. Twopy displays a shortened tail path so
    the useful recording date/folder stays visible in the compact dock.
    """
    state.recording_picker_widget = widget
    widget.choose_btn.changed.disconnect(widget._on_choose_clicked)

    def choose_recording_folder() -> None:
        """Open the folder dialog and load even when the same folder is picked."""
        result = widget._show_file_dialog(
            widget.mode,
            caption="Choose directory",
            start_path=_recording_picker_start_path(state),
            filter=widget.filter,
        )
        if result is None or isinstance(result, tuple):
            return
        with widget.changed.blocked():
            widget.line_edit.value = str(result)
        load_selected_path(result)

    widget.choose_btn.changed.connect(choose_recording_folder)
    if state.recording is not None:
        _set_recording_picker_display_path(
            widget,
            state,
            state.recording.source_session_dir,
        )
    else:
        _set_recording_picker_display_default(widget, state)


def _sync_recording_picker_to_selected(state: NapariControlState) -> None:
    """Mirror the active loaded recording in the Load-tab Recording field.

    Args:
        state: Current napari control state.

    Returns:
        None.

    The loaded-recordings list owns the active analysis context. The Recording
    field is display state only, so it must follow that list after manual
    loads, database loads, selection changes, and unloads.
    """
    widget = state.recording_picker_widget
    if widget is None:
        return
    if state.selected_recording_index is None or state.recording is None:
        _set_recording_picker_display_default(widget, state)
        return
    _set_recording_picker_display_path(
        widget,
        state,
        state.recording.source_session_dir,
    )


def _set_recording_picker_display_default(
    widget: FileEdit,
    state: NapariControlState,
) -> None:
    """Show ``default`` instead of a long recording path in the picker field.

    Args:
        widget: Recording-folder FileEdit widget.
        state: Current napari control state.

    Returns:
        None.
    """
    state.recording_picker_path = None
    state.recording_picker_display_text = DEFAULT_PATH_TEXT
    with widget.changed.blocked():
        widget.line_edit.value = DEFAULT_PATH_TEXT


def _set_recording_picker_display_path(
    widget: FileEdit,
    state: NapariControlState,
    path: Path,
) -> None:
    """Show a shortened loaded-recording path in the picker field.

    Args:
        widget: Recording-folder FileEdit widget.
        state: Current napari control state.
        path: Loaded recording folder to display and remember.

    Returns:
        None.
    """
    state.recording_picker_path = path.expanduser()
    state.recording_picker_display_text = _short_path_text(
        state.recording_picker_path,
        max_chars=72,
    )
    with widget.changed.blocked():
        widget.line_edit.value = state.recording_picker_display_text


def _short_path_text(path: Path, *, max_chars: int) -> str:
    """Return a compact path string that keeps the folder tail visible.

    Args:
        path: Path to display.
        max_chars: Maximum text length.

    Returns:
        Full path when it fits, otherwise ``...`` plus the path tail.
    """
    text = str(path)
    if path.is_dir() and not text.endswith("/"):
        text = f"{text}/"
    if len(text) <= max_chars:
        return text
    return f"...{text[-(max_chars - 3) :]}"


def _recording_picker_start_path(state: NapariControlState) -> str | None:
    """Return the folder shown first when choosing a recording.

    Args:
        state: Current napari control state.

    Returns:
        Absolute path string, or ``None`` for the platform default.
    """
    if state.recording_picker_path is not None:
        return str(state.recording_picker_path.resolve())
    if state.recording is not None:
        return str(state.recording.path.expanduser().parent.resolve())
    remembered_folder = read_last_recording_folder()
    if remembered_folder is None:
        return None
    return str(remembered_folder.resolve())


def _resolve_recording_folder_value(
    recording_folder: PathInput,
    state: NapariControlState,
) -> Path:
    """Resolve the picker value into the recording path to load.

    Args:
        recording_folder: Value from the visible picker field.
        state: Current napari control state.

    Returns:
        Selected path, or the current recording when the field shows
        ``default``.
    """
    if _is_displayed_recording_path(recording_folder, state):
        if state.recording_picker_path is not None:
            return state.recording_picker_path
        if state.recording is not None:
            return state.recording.path
    if not is_default_path(recording_folder):
        return Path(recording_folder).expanduser()
    if state.recording_picker_path is not None:
        return state.recording_picker_path
    if state.recording is not None:
        return state.recording.path
    return Path(recording_folder)


def _current_recording_path_for_reload(
    recording_folder: PathInput,
    state: NapariControlState,
) -> Path:
    """Return a concrete path for reloading the current recording.

    Args:
        recording_folder: Visible recording picker value.
        state: Current napari control state.

    Returns:
        Stored real recording path when available, otherwise the picker value.
    """
    if state.recording is not None:
        return state.recording.path
    if state.recording_picker_path is not None:
        return state.recording_picker_path
    return _resolve_recording_folder_value(recording_folder, state)


def _has_recording_to_load(
    recording_folder: PathInput,
    state: NapariControlState,
) -> bool:
    """Return whether a load-relevant widget change can load a recording.

    Args:
        recording_folder: Visible recording picker value.
        state: Current napari control state.

    Returns:
        ``True`` when a concrete path is selected or a recording is already
        loaded.
    """
    return (
        not is_default_path(recording_folder)
        or state.recording_picker_path is not None
        or state.recording is not None
    )


def _is_displayed_recording_path(
    recording_folder: PathInput,
    state: NapariControlState,
) -> bool:
    """Return whether the picker value is twopy's shortened display text.

    Args:
        recording_folder: Visible recording picker value.
        state: Current napari control state.

    Returns:
        ``True`` when the field text matches the remembered display value.
    """
    return state.recording_picker_path is not None and _normalized_display_path_text(
        str(recording_folder)
    ) == _normalized_display_path_text(state.recording_picker_display_text)


def _normalized_display_path_text(text: str) -> str:
    """Return path-display text normalized for widget comparison.

    Args:
        text: Text from magicgui or twopy's shortened display state.

    Returns:
        Text without a trailing path separator.
    """
    return text.rstrip("/")
