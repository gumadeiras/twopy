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

from twopy.napari.constants import DEFAULT_MOVIE_PREVIEW_FRAMES
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
    from magicgui.widgets import Container

    @magicgui(call_button="Load Recording")
    def load_recording(
        recording_data_path: Path = Path("recording_data.h5"),
        roi_set_path: Path = Path(""),
        roi_output_path: Path = Path(""),
        movie_start_frame: int = 0,
        movie_stop_frame: int = DEFAULT_MOVIE_PREVIEW_FRAMES,
        load_movie_preview: bool = True,
    ) -> str:
        """Load a converted recording into the current napari viewer.

        Args:
            recording_data_path: Path to converted ``recording_data.h5``.
            roi_set_path: Optional existing ``rois.h5`` file.
            roi_output_path: Output path used by Save ROIs after loading.
            movie_start_frame: First movie preview frame.
            movie_stop_frame: Exclusive stop frame for the movie preview.
            load_movie_preview: Whether to load preview movie frames.

        Returns:
            Human-readable status for the dock widget.
        """
        recording_path = resolve_load_widget_recording_path(recording_data_path)
        roi_path = roi_set_path.expanduser() if roi_set_path.is_file() else None
        resolved_roi_output_path = resolve_widget_output_path(
            roi_output_path,
            default=recording_path.parent / "rois.h5",
        )
        movie_range = (
            (int(movie_start_frame), int(movie_stop_frame))
            if load_movie_preview
            else None
        )

        # Import here to avoid a top-level cycle: viewer creates controls, while
        # controls can load recordings into that same viewer.
        from twopy.napari.viewer import open_recording_in_napari

        view = open_recording_in_napari(
            recording_path,
            viewer=state.viewer,
            roi_set=roi_path,
            roi_output_path=resolved_roi_output_path,
            movie_frame_range=movie_range,
            add_controls=False,
        )
        state.roi_labels_layer = view.roi_labels_layer
        state.roi_output_path = resolved_roi_output_path
        return f"Loaded {recording_path}"

    @magicgui(call_button="Save ROIs")
    def save_rois(output_path: Path = Path("")) -> str:
        """Save current napari ROI labels to twopy ROI HDF5.

        Args:
            output_path: Destination ROI HDF5 path.

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

    return Container(widgets=[load_recording, save_rois])


def resolve_load_widget_recording_path(recording_data_path: Path) -> Path:
    """Validate the Load Recording widget path.

    Args:
        recording_data_path: Candidate ``recording_data.h5`` path.

    Returns:
        Existing path.
    """
    resolved = recording_data_path.expanduser()
    if not resolved.is_file():
        msg = f"Recording file does not exist: {resolved}"
        raise ValueError(msg)
    return resolved.resolve()


def resolve_widget_output_path(path: Path, *, default: Path) -> Path:
    """Resolve an optional magicgui output path.

    Args:
        path: Path value from a magicgui widget. An untouched empty Path widget
            arrives as ``"."``.
        default: Default path to use when the widget was left empty.

    Returns:
        Expanded output path.
    """
    if str(path) in {"", "."}:
        return default.expanduser()
    return path.expanduser()
