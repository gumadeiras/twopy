"""Magicgui controls for the twopy napari adapter.

Inputs: napari viewer, current ROI Labels layer, and paths chosen by the user.
Outputs: a small dock widget for loading converted recordings and saving ROI
masks.

The control panel owns GUI callbacks only. Loading recordings and saving ROIs
still go through the same typed helpers used by scripts.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from magicgui.widgets import FileEdit

from twopy.converted import RecordingData
from twopy.napari.constants import DEFAULT_PATH_TEXT
from twopy.napari.movie import resolve_movie_frame_range
from twopy.napari.paths import (
    is_default_path,
    resolve_recording_paths,
    resolve_widget_output_path,
)
from twopy.napari.plotting import refresh_response_plot_widget
from twopy.napari.protocols import NapariViewer
from twopy.napari.roi import roi_label_image_from_layer, save_napari_label_rois
from twopy.napari.state import (
    read_last_recording_folder,
    recording_folder_for_state,
    write_last_recording_folder,
)

__all__ = ["add_twopy_magicgui_controls"]


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
    is_loading: bool = False


def add_twopy_magicgui_controls(
    viewer: NapariViewer,
    *,
    roi_labels_layer: object | None,
    roi_save_file: Path,
    recording: RecordingData | None = None,
    response_plot_widget: object | None = None,
    dock_name: str = "twopy",
    dock_area: str = "right",
) -> tuple[object, object]:
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
        dock_name: Dock widget title.
        dock_area: Napari dock area.

    Returns:
        ``(widget, dock_widget)`` created by magicgui and napari.

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
    widget = _make_twopy_control_widget(state)
    dock_widget = viewer.window.add_dock_widget(
        widget,
        name=dock_name,
        area=dock_area,
    )
    return widget, dock_widget


def _make_twopy_control_widget(state: NapariControlState) -> object:
    """Create a magicgui widget for loading recordings and saving ROIs.

    Args:
        state: Mutable napari control state.

    Returns:
        magicgui container with Load Recording and Save ROIs buttons.
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
        roi_save_file: Path = Path(DEFAULT_PATH_TEXT),
        movie_frame_range: tuple[int, int] = (0, default_movie_end_frame),
        load_movie: bool = True,
    ) -> str:
        """Load a converted recording into the current napari viewer.

        Args:
            recording_folder: Converted folder, source recording folder with
                ``twopy`` output, or direct ``recording_data.h5`` path.
            roi_file_to_load: Optional existing ROI HDF5 file. ``default`` uses
                ``rois.h5`` in the selected recording folder when it exists.
            roi_save_file: Output path used by Save ROIs after loading.
                ``default`` writes ``rois.h5`` in the selected recording folder.
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
        paths = resolve_recording_paths(
            _resolve_recording_folder_value(recording_folder, state),
        )
        try:
            roi_path = (
                paths.roi_file_to_load
                if is_default_path(roi_file_to_load)
                else _resolve_optional_roi_path(roi_file_to_load)
            )
            resolved_roi_save_file = resolve_widget_output_path(
                roi_save_file,
                default=paths.roi_save_file,
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
                roi_save_file=resolved_roi_save_file,
                movie_path=paths.movie_path,
                movie_frame_range=movie_range,
                add_controls=False,
            )
            state.roi_labels_layer = view.roi_labels_layer
            state.roi_save_file = resolved_roi_save_file
            state.recording = view.recording
            write_last_recording_folder(
                recording_folder_for_state(recording_folder, paths.recording_data_path),
            )
            frame_end = _default_movie_end_frame(view.recording)
            load_recording.movie_frame_range.max = frame_end
            load_recording.movie_frame_range.value = movie_range or (0, frame_end)
            _set_recording_picker_display_default(load_recording.recording_folder)
            refresh_response_plot_widget(
                state.response_plot_widget,
                recording=view.recording,
                roi_labels_layer=view.roi_labels_layer,
            )
            return f"Loaded {paths.recording_data_path}"
        finally:
            state.is_loading = False

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
        label_image = roi_label_image_from_layer(state.roi_labels_layer)
        if not np.any(label_image > 0):
            return "No ROI labels to save."

        resolved_output_path = resolve_widget_output_path(
            roi_save_file,
            default=state.roi_save_file,
        )
        roi_set = save_napari_label_rois(label_image, resolved_output_path)
        state.roi_save_file = resolved_output_path
        return f"Saved {len(roi_set.labels)} ROI(s) to {resolved_output_path}"

    load_recording.recording_folder.mode = "d"
    load_recording.roi_file_to_load.mode = "r"
    load_recording.roi_save_file.mode = "w"
    save_rois.roi_save_file.mode = "w"

    load_recording.recording_folder.label = "recording folder or file"
    load_recording.roi_file_to_load.label = "ROI file to load"
    load_recording.roi_save_file.label = "ROI save file"
    load_recording.movie_frame_range.label = "movie frames"
    load_recording.load_movie.label = "load movie"
    save_rois.roi_save_file.label = "ROI save file"
    _configure_recording_folder_picker(
        cast(FileEdit, load_recording.recording_folder),
        state,
        lambda path: load_recording(recording_folder=path),
    )

    def load_after_selection(_value: object) -> None:
        """Load when the user changes a load-relevant widget.

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

    load_recording.recording_folder.changed.connect(load_after_selection)
    load_recording.roi_file_to_load.changed.connect(load_after_selection)
    load_recording.movie_frame_range.changed.connect(load_after_selection)
    load_recording.load_movie.changed.connect(load_after_selection)

    return Container(
        widgets=[
            Label(value="Load Recording"),
            load_recording,
            Label(value="Save ROIs"),
            save_rois,
        ],
        layout="vertical",
        labels=False,
    )


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

    Magicgui normally uses the visible path field as the next dialog start
    directory. For twopy that clutters the dock with long recording paths, so
    the field stays at ``default`` and the remembered/current folder is supplied
    only when the user clicks the chooser.
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
        _set_recording_picker_display_default(widget)

    widget.choose_btn.changed.connect(choose_recording_folder)
    _set_recording_picker_display_default(widget)


def _set_recording_picker_display_default(widget: FileEdit) -> None:
    """Show ``default`` instead of a long recording path in the picker field.

    Args:
        widget: Recording-folder FileEdit widget.

    Returns:
        None.
    """
    with widget.changed.blocked():
        widget.line_edit.value = DEFAULT_PATH_TEXT


def _recording_picker_start_path(state: NapariControlState) -> str | None:
    """Return the folder shown first when choosing a recording.

    Args:
        state: Current napari control state.

    Returns:
        Absolute path string, or ``None`` for the platform default.
    """
    if state.recording is not None:
        return str(state.recording.path.expanduser().parent.resolve())
    remembered_folder = read_last_recording_folder()
    if remembered_folder is None:
        return None
    return str(remembered_folder.resolve())


def _resolve_recording_folder_value(
    recording_folder: Path,
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
    if not is_default_path(recording_folder):
        return recording_folder
    if state.recording is not None:
        return state.recording.path
    return recording_folder


def _has_recording_to_load(recording_folder: Path, state: NapariControlState) -> bool:
    """Return whether a load-relevant widget change can load a recording.

    Args:
        recording_folder: Visible recording picker value.
        state: Current napari control state.

    Returns:
        ``True`` when a concrete path is selected or a recording is already
        loaded.
    """
    return not is_default_path(recording_folder) or state.recording is not None


def _resolve_optional_roi_path(path: Path) -> Path | None:
    """Resolve an optional ROI file selected by the user.

    Args:
        path: Candidate ROI file path.

    Returns:
        Existing ROI file path, or ``None`` for empty/default values.
    """
    if is_default_path(path):
        return None
    resolved = path.expanduser()
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
