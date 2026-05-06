"""Magicgui controls for the twopy napari adapter.

Inputs: napari viewer, current ROI Labels layer, and paths chosen by the user.
Outputs: small dock widgets for loading converted recordings and saving ROI
masks.

The control panel owns GUI callbacks only. Loading recordings and saving ROIs
still go through the same typed helpers used by scripts.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from magicgui.widgets import FileEdit

from twopy.converted import RecordingData
from twopy.napari.constants import DEFAULT_PATH_TEXT
from twopy.napari.layout import (
    place_loaded_recordings_dock_after_load,
    place_save_rois_dock_before_layer_list,
)
from twopy.napari.loading import resolve_or_convert_recording
from twopy.napari.movie import resolve_movie_frame_range
from twopy.napari.paths import (
    PathInput,
    is_default_path,
    resolve_widget_output_path,
)
from twopy.napari.protocols import NapariViewer
from twopy.napari.roi import (
    roi_label_image_from_layer_for_recording,
    save_napari_label_rois,
)
from twopy.napari.session import (
    LoadedNapariRecording,
    LoadedRecordingsPanel,
    record_loaded_view,
    render_loaded_recordings_panel,
    select_loaded_recording,
    unload_loaded_recording,
    update_selected_roi_save_file,
)
from twopy.napari.state import (
    read_last_recording_folder,
    recording_folder_for_state,
    write_last_recording_folder,
)
from twopy.napari.text import counted_noun

__all__ = ["NapariControlDocks", "add_twopy_magicgui_controls"]


class _RoiSavePathReceiver(Protocol):
    """Small protocol for widgets that mirror the selected ROI save path."""

    def set_roi_save_file(self, roi_save_file: Path | None) -> None:
        """Store the ROI HDF5 path for the selected recording.

        Args:
            roi_save_file: ROI output path, or ``None`` when no recording is
                selected.

        Returns:
            None.
        """
        ...


@dataclass
class NapariControlDocks:
    """Dock widgets created for twopy napari controls.

    Inputs: Qt widgets and napari dock objects.
    Outputs: grouped references for scripts and tests.
    """

    load_widget: object
    load_dock_widget: object
    loaded_recordings_widget: object
    loaded_recordings_dock_widget: object
    save_rois_widget: object
    save_rois_dock_widget: object


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
    loaded_recordings: list[LoadedNapariRecording] = field(default_factory=list)
    selected_recording_index: int | None = None
    loaded_recordings_panel: LoadedRecordingsPanel | None = None
    recording_picker_path: Path | None = None
    recording_picker_display_text: str = DEFAULT_PATH_TEXT
    is_loading: bool = False
    replace_selected_on_next_load: bool = False


def add_twopy_magicgui_controls(
    viewer: NapariViewer,
    *,
    roi_labels_layer: object | None,
    roi_save_file: Path,
    recording: RecordingData | None = None,
    response_plot_widget: object | None = None,
    mean_image_layer: object | None = None,
    movie_layer: object | None = None,
    dock_name: str = "twopy",
    dock_area: str = "right",
) -> NapariControlDocks:
    """Add the first twopy magicgui control panel to a napari viewer.

    Args:
        viewer: Napari viewer that should receive the dock widget.
        roi_labels_layer: Editable napari Labels layer that contains ROI
            integer labels. ``None`` creates a disabled save path with a clear
            status message.
        roi_save_file: Default destination for saved ROI HDF5 files.
        recording: Optional loaded recording used to populate GUI defaults.
        response_plot_widget: Optional response plotting widget to refresh
            after a new recording is loaded.
        mean_image_layer: Optional mean-image layer for an already loaded
            startup recording.
        movie_layer: Optional movie layer for an already loaded startup
            recording.
        dock_name: Dock widget title.
        dock_area: Napari dock area.

    Returns:
        Created load and save-ROI widgets plus their napari dock widgets.

    The panel is intentionally small. It loads converted recordings and saves
    the current Labels layer through the same helpers used by scripts, so a
    future plugin can wrap this without changing ROI semantics.
    """
    state = NapariControlState(
        viewer=viewer,
        roi_labels_layer=roi_labels_layer,
        roi_save_file=roi_save_file,
        recording=recording,
        response_plot_widget=response_plot_widget,
    )
    if recording is not None and mean_image_layer is not None:
        state.loaded_recordings.append(
            LoadedNapariRecording(
                display_path=recording.path.parent,
                recording=recording,
                roi_save_file=roi_save_file,
                mean_image_layer=mean_image_layer,
                movie_layer=movie_layer,
                roi_labels_layer=roi_labels_layer,
            )
        )
        state.selected_recording_index = 0
    load_widget = _make_twopy_load_widget(state)
    load_dock_widget = viewer.window.add_dock_widget(
        load_widget,
        name=dock_name,
        area=dock_area,
    )
    loaded_recordings_widget = _make_loaded_recordings_widget(state)
    loaded_recordings_dock_widget = viewer.window.add_dock_widget(
        loaded_recordings_widget,
        name="twopy loaded recordings",
        area=dock_area,
    )
    place_loaded_recordings_dock_after_load(
        viewer,
        load_dock_widget,
        loaded_recordings_dock_widget,
    )
    save_rois_widget = _make_twopy_save_rois_widget(state)
    save_rois_dock_widget = viewer.window.add_dock_widget(
        save_rois_widget,
        name="twopy save ROIs",
        area="left",
    )
    place_save_rois_dock_before_layer_list(viewer, save_rois_dock_widget)
    return NapariControlDocks(
        load_widget=load_widget,
        load_dock_widget=load_dock_widget,
        loaded_recordings_widget=loaded_recordings_widget,
        loaded_recordings_dock_widget=loaded_recordings_dock_widget,
        save_rois_widget=save_rois_widget,
        save_rois_dock_widget=save_rois_dock_widget,
    )


def _make_twopy_load_widget(state: NapariControlState) -> object:
    """Create the right-side magicgui widget for loading recordings.

    Args:
        state: Mutable napari control state.

    Returns:
        magicgui container with recording-load controls.
    """
    from magicgui import magicgui
    from magicgui.widgets import Container, Label

    default_movie_end_frame = _default_movie_end_frame(state.recording)

    @magicgui(
        call_button=False,
        layout="vertical",
        movie_frame_range={
            "widget_type": "RangeSlider",
            "min": 0,
            "max": max(default_movie_end_frame, 0),
            "step": 1,
            "tracking": False,
        },
    )
    def load_recording(
        recording_folder: Path = Path(DEFAULT_PATH_TEXT),
        roi_file_to_load: Path = Path(DEFAULT_PATH_TEXT),
        movie_frame_range: tuple[int, int] = (0, default_movie_end_frame),
        load_movie: bool = True,
    ) -> str:
        """Load a recording into the current napari viewer.

        Args:
            recording_folder: Source recording folder, converted folder, or
                direct ``recording_data.h5`` path. Source folders are converted
                first when twopy output does not exist yet.
            roi_file_to_load: Optional existing ROI HDF5 file. ``default`` uses
                ``rois.h5`` in the selected recording folder when it exists.
            movie_frame_range: First and last movie frames to load. When the
                empty-launch widget still shows ``(0, 0)``, twopy resolves it
                to the recording's full frame range after the folder is
                selected.
            load_movie: Whether to load movie frames.

        Returns:
            Human-readable status for the dock widget.
        """
        if state.is_loading:
            return "Recording load already in progress."
        state.is_loading = True
        replace_selected = state.replace_selected_on_next_load
        state.replace_selected_on_next_load = False
        selected_recording_path = _resolve_recording_folder_value(
            recording_folder,
            state,
        )
        try:
            resolved_recording = resolve_or_convert_recording(selected_recording_path)
            paths = resolved_recording.paths
            roi_path = (
                paths.roi_file_to_load
                if is_default_path(roi_file_to_load)
                else _resolve_optional_roi_path(roi_file_to_load)
            )
            loaded_recording = _load_recording_for_defaults(
                recording_data_path=paths.recording_data_path,
                movie_path=paths.movie_path,
            )
            movie_range = None
            if load_movie:
                movie_range = resolve_movie_frame_range(
                    start_frame=int(movie_frame_range[0]),
                    end_frame=int(movie_frame_range[1]),
                    frame_count=loaded_recording.movie.shape[0],
                    zero_end_means_last=True,
                )

            # Import here to avoid a top-level cycle: viewer creates controls,
            # while controls can load recordings into that same viewer.
            from twopy.napari.viewer import open_recording_in_napari

            view = open_recording_in_napari(
                paths.recording_data_path,
                viewer=state.viewer,
                roi_set=roi_path,
                roi_save_file=paths.roi_save_file,
                movie_path=paths.movie_path,
                movie_frame_range=movie_range,
                add_controls=False,
            )
            record_loaded_view(
                state,
                view=view,
                roi_save_file=paths.roi_save_file,
                replace_selected=replace_selected,
            )
            write_last_recording_folder(
                recording_folder_for_state(
                    selected_recording_path,
                    paths.recording_data_path,
                ),
            )
            frame_end = _default_movie_end_frame(view.recording)
            load_recording.movie_frame_range.max = frame_end
            load_recording.movie_frame_range.value = movie_range or (0, frame_end)
            _set_recording_picker_display_path(
                cast(FileEdit, load_recording.recording_folder),
                state,
                paths.recording_data_path.parent,
            )
            status = (
                "Converted and loaded" if resolved_recording.was_converted else "Loaded"
            )
            return f"{status} {paths.recording_data_path}"
        finally:
            state.replace_selected_on_next_load = False
            state.is_loading = False

    load_recording.recording_folder.mode = "d"
    load_recording.roi_file_to_load.mode = "r"

    load_recording.recording_folder.label = "recording folder or file"
    load_recording.roi_file_to_load.label = "ROI file to load"
    load_recording.movie_frame_range.label = "movie frames"
    load_recording.load_movie.label = "load movie"
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

    def reload_after_option_change(_value: object) -> None:
        """Reload the current recording after a non-path option changes.

        Args:
            _value: Ignored widget signal value.

        Returns:
            None.

        Secondary controls such as the movie-frame slider should reuse the
        resolved recording path stored in state. The visible picker text may be
        shortened for display and is not always a real filesystem path.
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
    load_recording.roi_file_to_load.changed.connect(reload_after_option_change)
    load_recording.movie_frame_range.changed.connect(reload_after_option_change)
    load_recording.load_movie.changed.connect(reload_after_option_change)

    return Container(
        widgets=[
            Label(value="Load Recording"),
            load_recording,
        ],
        layout="vertical",
        labels=False,
    )


def _make_loaded_recordings_widget(state: NapariControlState) -> object:
    """Create the loaded-recordings list panel.

    Args:
        state: Mutable napari control state.

    Returns:
        Qt panel that lists loaded recordings and can unload the selected one.
    """
    panel = LoadedRecordingsPanel(
        on_select=lambda index: select_loaded_recording(state, index),
        on_unload=lambda index: unload_loaded_recording(state, index),
    )
    state.loaded_recordings_panel = panel
    render_loaded_recordings_panel(state)
    return panel


def _make_twopy_save_rois_widget(state: NapariControlState) -> object:
    """Create the left-side magicgui widget for saving ROI labels.

    Args:
        state: Mutable napari control state.

    Returns:
        magicgui widget with one Save ROIs button.
    """
    from magicgui import magicgui

    @magicgui(call_button="Save ROIs", layout="vertical")
    def save_rois(roi_save_file: Path = Path(DEFAULT_PATH_TEXT)) -> str:
        """Save current napari ROI labels to twopy ROI HDF5.

        Args:
            roi_save_file: Destination ROI HDF5 path. ``default`` uses the
                current recording's default ROI output path.

        Returns:
            Human-readable status for the dock widget.
        """
        if state.roi_labels_layer is None:
            return "No ROI Labels layer is available."
        if state.recording is None:
            return "No recording is selected."
        try:
            label_image = roi_label_image_from_layer_for_recording(
                state.roi_labels_layer,
                state.recording,
            )
        except ValueError as error:
            return str(error)
        if not np.any(label_image > 0):
            return "No ROI labels to save."

        resolved_output_path = resolve_widget_output_path(
            roi_save_file,
            default=state.roi_save_file,
        )
        roi_set = save_napari_label_rois(label_image, resolved_output_path)
        state.roi_save_file = resolved_output_path
        update_selected_roi_save_file(state, resolved_output_path)
        if hasattr(state.response_plot_widget, "set_roi_save_file"):
            cast(_RoiSavePathReceiver, state.response_plot_widget).set_roi_save_file(
                resolved_output_path
            )
        return (
            f"Saved {counted_noun(len(roi_set.labels), 'ROI', 'ROIs')} "
            f"to {resolved_output_path}"
        )

    save_rois.roi_save_file.mode = "w"
    save_rois.roi_save_file.label = "ROI save file"
    return save_rois


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
        _set_recording_picker_display_path(widget, state, state.recording.path.parent)
    else:
        _set_recording_picker_display_default(widget, state)


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
    if state.recording_picker_path is not None:
        return state.recording_picker_path
    if state.recording is not None:
        return state.recording.path
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


def _resolve_optional_roi_path(path: PathInput) -> Path | None:
    """Resolve an optional ROI file selected by the user.

    Args:
        path: Candidate ROI file path.

    Returns:
        Existing ROI file path, or ``None`` for empty/default values.
    """
    if is_default_path(path):
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        msg = f"ROI file does not exist: {resolved}"
        raise ValueError(msg)
    return resolved.resolve()


def _default_movie_end_frame(recording: RecordingData | None) -> int:
    """Return the default inclusive movie end frame for a loaded recording.

    Args:
        recording: Optional loaded recording.

    Returns:
        Last available frame index, or ``0`` before a recording is loaded.
    """
    if recording is None:
        return 0
    return max(recording.movie.shape[0] - 1, 0)


def _load_recording_for_defaults(
    *,
    recording_data_path: Path,
    movie_path: Path | None,
) -> RecordingData:
    """Load recording metadata before resolving frame defaults.

    Args:
        recording_data_path: Converted ``recording_data.h5`` path.
        movie_path: Optional explicit converted movie path.

    Returns:
        Loaded recording object.

    Loading the small manifest lets the empty-launch widget convert its initial
    ``0`` end frame into the actual final frame before adding the movie layer.
    """
    from twopy.converted import load_converted_recording

    return load_converted_recording(recording_data_path, movie_path=movie_path)
