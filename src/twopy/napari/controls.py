"""Magicgui controls for the twopy napari adapter.

Inputs: napari viewer, current ROI Labels layer, and paths chosen by the user.
Outputs: a small dock widget for loading converted recordings and saving ROI
masks.

The control panel owns GUI callbacks only. Loading recordings and saving ROIs
still go through the same typed helpers used by scripts.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twopy.napari.constants import DEFAULT_PATH_TEXT
from twopy.napari.paths import (
    is_default_path,
    resolve_recording_paths,
    resolve_widget_output_path,
)
from twopy.napari.protocols import NapariViewer
from twopy.napari.roi import roi_label_image_from_layer, save_napari_label_rois

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
    roi_output_path: Path


def add_twopy_magicgui_controls(
    viewer: NapariViewer,
    *,
    roi_labels_layer: object | None,
    roi_output_path: Path,
    dock_name: str = "twopy",
    dock_area: str = "right",
) -> tuple[object, object]:
    """Add the first twopy magicgui control panel to a napari viewer.

    Args:
        viewer: Napari viewer that should receive the dock widget.
        roi_labels_layer: Editable napari Labels layer that contains ROI
            integer labels. ``None`` creates a disabled save path with a clear
            status message.
        roi_output_path: Default destination for saved ROI HDF5 files.
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
        roi_output_path=roi_output_path,
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

    @magicgui(call_button="Load Recording", layout="vertical")
    def load_recording(
        recording_path: Path = Path(DEFAULT_PATH_TEXT),
        roi_set_path: Path = Path(DEFAULT_PATH_TEXT),
        roi_output_path: Path = Path(DEFAULT_PATH_TEXT),
        movie_start_frame: int = 0,
        movie_stop_frame: str = "last",
        load_movie_preview: bool = True,
    ) -> str:
        """Load a converted recording into the current napari viewer.

        Args:
            recording_path: Converted folder, source recording folder with
                ``twopy`` output, or direct ``recording_data.h5`` path.
            roi_set_path: Optional existing ``rois.h5`` file. ``default`` uses
                ``rois.h5`` in the selected recording folder when it exists.
            roi_output_path: Output path used by Save ROIs after loading.
                ``default`` writes ``rois.h5`` in the selected recording folder.
            movie_start_frame: First movie preview frame.
            movie_stop_frame: Exclusive stop frame, or ``last`` for the final
                movie frame.
            load_movie_preview: Whether to load preview movie frames.

        Returns:
            Human-readable status for the dock widget.
        """
        paths = resolve_recording_paths(recording_path)
        roi_path = (
            paths.roi_set_path
            if is_default_path(roi_set_path)
            else _resolve_optional_roi_path(roi_set_path)
        )
        resolved_roi_output_path = resolve_widget_output_path(
            roi_output_path,
            default=paths.roi_output_path,
        )
        movie_range = (
            resolve_movie_frame_range(
                start_frame=movie_start_frame,
                stop_frame=movie_stop_frame,
            )
            if load_movie_preview
            else None
        )

        # Import here to avoid a top-level cycle: viewer creates controls, while
        # controls can load recordings into that same viewer.
        from twopy.napari.viewer import open_recording_in_napari

        view = open_recording_in_napari(
            paths.recording_data_path,
            viewer=state.viewer,
            roi_set=roi_path,
            roi_output_path=resolved_roi_output_path,
            movie_path=paths.movie_path,
            movie_frame_range=movie_range,
            add_controls=False,
        )
        state.roi_labels_layer = view.roi_labels_layer
        state.roi_output_path = resolved_roi_output_path
        return f"Loaded {paths.recording_data_path}"

    @magicgui(call_button="Save ROIs", layout="vertical")
    def save_rois(output_path: Path = Path(DEFAULT_PATH_TEXT)) -> str:
        """Save current napari ROI labels to twopy ROI HDF5.

        Args:
            output_path: Destination ROI HDF5 path. ``default`` uses the
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
            output_path,
            default=state.roi_output_path,
        )
        roi_set = save_napari_label_rois(label_image, resolved_output_path)
        state.roi_output_path = resolved_output_path
        return f"Saved {len(roi_set.labels)} ROI(s) to {resolved_output_path}"

    load_recording.recording_path.mode = "d"
    load_recording.roi_set_path.mode = "r"
    load_recording.roi_output_path.mode = "w"
    save_rois.output_path.mode = "w"

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


def resolve_movie_frame_range(
    *,
    start_frame: int,
    stop_frame: str,
) -> tuple[int, int | None]:
    """Resolve movie preview frame inputs from the widget.

    Args:
        start_frame: First movie frame.
        stop_frame: Exclusive stop frame text, or ``last``.

    Returns:
        ``(start, stop)`` where ``stop=None`` means the final movie frame.
    """
    if start_frame < 0:
        msg = f"movie_start_frame must be at least 0; got {start_frame}"
        raise ValueError(msg)
    if stop_frame.strip().lower() in {"last", DEFAULT_PATH_TEXT, ""}:
        return (start_frame, None)

    stop = int(stop_frame)
    if stop < start_frame:
        msg = (
            "movie_stop_frame must be greater than or equal to "
            f"movie_start_frame; got {stop} < {start_frame}"
        )
        raise ValueError(msg)
    return (start_frame, stop)


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
