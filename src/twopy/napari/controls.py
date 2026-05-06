"""Magicgui controls for the twopy napari adapter.

Inputs: napari viewer, current ROI Labels layer, and paths chosen by the user.
Outputs: a small dock widget for loading converted recordings and saving ROI
masks.

The control panel owns GUI callbacks only. Loading recordings and saving ROIs
still go through the same typed helpers used by scripts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from twopy.converted import RecordingData
from twopy.napari.constants import DEFAULT_PATH_TEXT
from twopy.napari.metadata import format_recording_metadata
from twopy.napari.movie import resolve_movie_frame_range
from twopy.napari.paths import (
    is_default_path,
    resolve_recording_paths,
    resolve_widget_output_path,
)
from twopy.napari.protocols import NapariViewer
from twopy.napari.roi import roi_label_image_from_layer, save_napari_label_rois
from twopy.napari.state import (
    read_last_recording_folder,
    recording_folder_for_state,
    write_last_recording_folder,
)

__all__ = ["add_twopy_magicgui_controls"]


class _PathValueWidget(Protocol):
    """Small protocol for magicgui path widgets used by callbacks."""

    value: Path


class _LoadRecordingWidget(Protocol):
    """Small protocol for the load-recording magicgui function widget."""

    recording_folder: _PathValueWidget

    def __call__(self) -> object:
        """Run the load-recording function widget.

        Returns:
            Function widget result.
        """
        ...


class _TextButton(Protocol):
    """Small protocol for a text button widget."""

    text: str


class _MetadataTextWidget(Protocol):
    """Small protocol for the metadata text widget."""

    visible: bool


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


def add_twopy_magicgui_controls(
    viewer: NapariViewer,
    *,
    roi_labels_layer: object | None,
    roi_save_file: Path,
    recording: RecordingData | None = None,
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
        recording: Optional loaded recording used to populate defaults and the
            metadata panel.
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
    from magicgui.widgets import Container, Label, PushButton, TextEdit

    default_movie_end_frame = _default_movie_end_frame(state.recording)
    default_recording_folder = _default_recording_folder(state.recording)

    @magicgui(call_button=False, layout="vertical")
    def load_recording(
        recording_folder: Path = default_recording_folder,
        roi_file_to_load: Path = Path(DEFAULT_PATH_TEXT),
        roi_save_file: Path = Path(DEFAULT_PATH_TEXT),
        movie_start_frame: int = 0,
        movie_end_frame: int = default_movie_end_frame,
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
            movie_start_frame: First movie preview frame.
            movie_end_frame: Last movie frame to load. When the empty-launch
                widget still shows ``0``, twopy resolves it to the recording's
                final frame after the folder is selected.
            load_movie: Whether to load movie frames.

        Returns:
            Human-readable status for the dock widget.
        """
        paths = resolve_recording_paths(recording_folder)
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
                start_frame=movie_start_frame,
                end_frame=movie_end_frame,
                frame_count=loaded_recording.movie.shape[0],
                zero_end_means_last=True,
            )

        # Import here to avoid a top-level cycle: viewer creates controls, while
        # controls can load recordings into that same viewer.
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
        load_recording.movie_end_frame.value = _default_movie_end_frame(view.recording)
        metadata_text.value = format_recording_metadata(view.recording)
        metadata_button.text = "Recording Metadata v"
        metadata_text.visible = True
        return f"Loaded {paths.recording_data_path}"

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

    metadata_button = PushButton(text="Recording Metadata >")
    metadata_text = TextEdit(value=format_recording_metadata(state.recording))
    metadata_text.read_only = True
    metadata_text.min_height = 220
    metadata_text.visible = False
    metadata_button.clicked.connect(
        lambda _value: _toggle_metadata_panel(metadata_button, metadata_text),
    )

    load_recording.recording_folder.mode = "d"
    load_recording.roi_file_to_load.mode = "r"
    load_recording.roi_save_file.mode = "w"
    save_rois.roi_save_file.mode = "w"

    load_recording.recording_folder.label = "recording folder or file"
    load_recording.roi_file_to_load.label = "ROI file to load"
    load_recording.roi_save_file.label = "ROI save file"
    load_recording.movie_start_frame.label = "movie start frame"
    load_recording.movie_end_frame.label = "movie end frame"
    load_recording.load_movie.label = "load movie"
    save_rois.roi_save_file.label = "ROI save file"
    load_recording.recording_folder.changed.connect(
        lambda _value: _load_after_recording_selection(load_recording),
    )

    return Container(
        widgets=[
            Label(value="Load Recording"),
            load_recording,
            metadata_button,
            metadata_text,
            Label(value="Save ROIs"),
            save_rois,
        ],
        layout="vertical",
        labels=False,
    )


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


def _default_recording_folder(recording: RecordingData | None) -> Path:
    """Return the default recording folder shown in the load control.

    Args:
        recording: Optional loaded recording.

    Returns:
        Loaded recording folder, last selected folder, or ``default``.
    """
    if recording is not None:
        return recording.path.expanduser().parent
    remembered_folder = read_last_recording_folder()
    if remembered_folder is not None:
        return remembered_folder
    return Path(DEFAULT_PATH_TEXT)


def _load_after_recording_selection(
    load_recording_widget: _LoadRecordingWidget,
) -> None:
    """Load a recording after the folder picker changes.

    Args:
        load_recording_widget: magicgui function widget with load controls.

    Returns:
        None.
    """
    result = load_recording_widget.recording_folder.value
    if is_default_path(result):
        return
    load_recording_widget()


def _toggle_metadata_panel(
    button: _TextButton,
    metadata_text: _MetadataTextWidget,
) -> None:
    """Show or hide the metadata text panel.

    Args:
        button: Button widget used as the disclosure control.
        metadata_text: Text widget containing metadata.

    Returns:
        None.
    """
    metadata_text.visible = not bool(metadata_text.visible)
    button.text = (
        "Recording Metadata v" if metadata_text.visible else "Recording Metadata >"
    )


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
