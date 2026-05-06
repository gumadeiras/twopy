"""Thin napari adapter for converted twopy recordings.

Inputs: converted ``recording_data.h5`` files and optional ROI label data.
Outputs: napari image/label layers plus saved twopy ROI HDF5 files.

This module contains GUI wiring only. Recording loading, ROI semantics, and
analysis remain in core modules so scripts and tests can use the same code
without starting napari.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from twopy.converted import RecordingData, load_converted_recording
from twopy.roi import (
    RoiSet,
    load_roi_set,
    make_roi_set_from_label_image,
    roi_set_to_label_image,
    save_roi_set,
)

__all__ = [
    "NapariRecordingView",
    "add_twopy_magicgui_controls",
    "open_recording_in_napari",
    "roi_label_image_from_layer",
    "save_napari_label_rois",
]


class _NapariWindow(Protocol):
    """Small protocol for adding dock widgets to a napari viewer.

    Inputs: widget objects.
    Outputs: dock widgets owned by napari.
    """

    def add_dock_widget(
        self,
        widget: object,
        *,
        name: str,
        area: str,
    ) -> object:
        """Add a dock widget to the viewer window.

        Args:
            widget: Widget object to dock.
            name: Dock title shown by napari.
            area: Napari dock area name.

        Returns:
            The dock widget created by napari.
        """
        ...


class _NapariViewer(Protocol):
    """Small viewer protocol used to keep tests independent from napari.

    Inputs: image or label layer data.
    Outputs: napari layer objects.
    """

    @property
    def window(self) -> _NapariWindow:
        """Return the viewer window used to add dock widgets.

        Inputs: viewer object.
        Outputs: napari window object.
        """
        ...

    def add_image(self, data: object, *, name: str, **kwargs: object) -> object:
        """Add an image layer to the viewer.

        Args:
            data: Array-like image data.
            name: Layer name shown in napari.
            kwargs: napari image-layer options.

        Returns:
            The layer object created by napari.
        """
        ...

    def add_labels(self, data: object, *, name: str, **kwargs: object) -> object:
        """Add a labels layer to the viewer.

        Args:
            data: Two-dimensional integer label image.
            name: Layer name shown in napari.
            kwargs: napari label-layer options.

        Returns:
            The layer object created by napari.
        """
        ...


class _NapariLayerWithData(Protocol):
    """Small protocol for napari layers that expose array-like data.

    Inputs: a napari layer object.
    Outputs: access to its ``data`` attribute.
    """

    data: object


@dataclass(frozen=True)
class NapariRecordingView:
    """Objects created when a converted recording is opened in napari.

    Inputs: a converted recording path and optional ROI set.
    Outputs: viewer, loaded recording, and the layer objects added to the
    viewer.

    Keeping these references makes script-driven napari sessions auditable:
    callers can inspect which recording was loaded and where each layer came
    from.
    """

    viewer: object
    recording: RecordingData
    mean_image_layer: object
    movie_layer: object | None
    roi_labels_layer: object | None
    controls_widget: object | None
    controls_dock_widget: object | None


def open_recording_in_napari(
    recording_data_path: Path,
    *,
    roi_set: RoiSet | Path | None = None,
    viewer: _NapariViewer | None = None,
    movie_frame_range: tuple[int, int] | None = None,
    mean_image_layer_name: str = "mean image",
    movie_layer_name: str = "aligned movie",
    roi_layer_name: str = "rois",
    add_roi_labels_layer: bool = True,
    add_controls: bool = True,
    roi_output_path: Path | None = None,
) -> NapariRecordingView:
    """Open a converted recording in napari.

    Args:
        recording_data_path: Path to converted ``recording_data.h5``.
        roi_set: Optional ``RoiSet`` or saved ROI HDF5 path to display.
        viewer: Optional existing napari viewer. When omitted, a new viewer is
            created.
        movie_frame_range: Optional ``(start, stop)`` frame range to load as a
            movie preview. ``None`` avoids loading movie frames and shows only
            the mean image.
        mean_image_layer_name: Napari layer name for the mean image.
        movie_layer_name: Napari layer name for the optional movie preview.
        roi_layer_name: Napari layer name for ROI labels.
        add_roi_labels_layer: Whether to add a napari Labels layer for ROI
            drawing or editing. When ``roi_set`` is omitted, the layer starts as
            an empty integer image with the movie spatial shape.
        add_controls: Whether to add the small magicgui twopy control panel.
        roi_output_path: Default ROI HDF5 path used by the Save ROIs button.
            When omitted, twopy uses ``rois.h5`` beside ``recording_data.h5``.

    Returns:
        ``NapariRecordingView`` with the loaded recording and created layers.

    The large aligned movie is not loaded by default. Pass a bounded frame
    range when you want an interactive movie preview without pulling the whole
    recording into memory.
    """
    recording = load_converted_recording(recording_data_path)
    resolved_viewer = _create_viewer() if viewer is None else viewer
    mean_layer = resolved_viewer.add_image(
        recording.mean_image,
        name=mean_image_layer_name,
        colormap="gray",
    )

    movie_layer = None
    if movie_frame_range is not None:
        start_frame, stop_frame = movie_frame_range
        movie = recording.movie.read_frames(start_frame, stop_frame)
        movie_layer = resolved_viewer.add_image(
            movie,
            name=movie_layer_name,
            colormap="gray",
        )

    roi_layer = None
    if add_roi_labels_layer:
        roi_layer = resolved_viewer.add_labels(
            _roi_label_image_for_display(roi_set, recording),
            name=roi_layer_name,
        )
    controls_widget = None
    controls_dock = None
    if add_controls:
        controls_widget, controls_dock = add_twopy_magicgui_controls(
            resolved_viewer,
            roi_labels_layer=roi_layer,
            roi_output_path=_resolve_roi_output_path(
                recording_data_path=recording.path,
                roi_set=roi_set,
                explicit_roi_output_path=roi_output_path,
            ),
        )

    return NapariRecordingView(
        viewer=resolved_viewer,
        recording=recording,
        mean_image_layer=mean_layer,
        movie_layer=movie_layer,
        roi_labels_layer=roi_layer,
        controls_widget=controls_widget,
        controls_dock_widget=controls_dock,
    )


def add_twopy_magicgui_controls(
    viewer: _NapariViewer,
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

    The panel is intentionally small. It saves the current Labels layer through
    the same ROI helpers used by scripts, so a future plugin can wrap this
    without changing ROI semantics.
    """
    widget = _make_roi_save_widget(
        roi_labels_layer=roi_labels_layer,
        roi_output_path=roi_output_path,
    )
    dock_widget = viewer.window.add_dock_widget(
        widget,
        name=dock_name,
        area=dock_area,
    )
    return widget, dock_widget


def roi_label_image_from_layer(layer: object) -> npt.NDArray[np.int64]:
    """Read integer label data from a napari Labels layer.

    Args:
        layer: Napari Labels layer, or a test object with a ``data`` attribute.

    Returns:
        Two-dimensional integer label image.

    This small helper keeps scripts from reaching into napari layer internals
    in several different ways. The label image can be passed directly to
    ``save_napari_label_rois``.
    """
    data = cast(_NapariLayerWithData, layer).data
    return np.asarray(data, dtype=np.int64)


def save_napari_label_rois(
    label_image: npt.ArrayLike,
    path: Path,
    *,
    labels: tuple[str, ...] | None = None,
    label_prefix: str = "roi",
) -> RoiSet:
    """Save a napari labels-layer image as a twopy ROI set.

    Args:
        label_image: Two-dimensional labels-layer data where ``0`` is
            background and positive integers are ROIs.
        path: Destination ROI HDF5 file.
        labels: Optional text labels matching positive ROI values.
        label_prefix: Prefix for generated ROI labels when ``labels`` is
            omitted.

    Returns:
        Saved ``RoiSet``.

    This keeps napari editing output in the same HDF5 format that scripts and
    analysis functions already load.
    """
    roi_set = make_roi_set_from_label_image(
        np.asarray(label_image),
        labels=labels,
        label_prefix=label_prefix,
    )
    save_roi_set(roi_set, path)
    return roi_set


def _make_roi_save_widget(
    *,
    roi_labels_layer: object | None,
    roi_output_path: Path,
) -> object:
    """Create a magicgui widget that saves the current ROI Labels layer.

    Args:
        roi_labels_layer: Napari Labels layer to save.
        roi_output_path: Default output path.

    Returns:
        magicgui function widget.
    """
    from magicgui import magicgui

    @magicgui(call_button="Save ROIs")
    def save_rois(output_path: Path = roi_output_path) -> str:
        """Save current napari ROI labels to twopy ROI HDF5.

        Args:
            output_path: Destination ROI HDF5 path.

        Returns:
            Human-readable status for the dock widget.
        """
        if roi_labels_layer is None:
            return "No ROI Labels layer is available."
        label_image = roi_label_image_from_layer(roi_labels_layer)
        if not np.any(label_image > 0):
            return "No ROI labels to save."

        roi_set = save_napari_label_rois(label_image, Path(output_path))
        return f"Saved {len(roi_set.labels)} ROI(s) to {Path(output_path)}"

    return save_rois


def _roi_label_image_for_display(
    roi_set: RoiSet | Path | None,
    recording: RecordingData,
) -> npt.NDArray[np.int64]:
    """Create the label image shown in napari for ROI drawing/editing.

    Args:
        roi_set: Optional existing ROI set or saved ROI HDF5 path.
        recording: Loaded converted recording.

    Returns:
        Integer label image with the movie spatial shape.
    """
    if roi_set is None:
        return np.zeros(recording.movie.shape[1:], dtype=np.int64)
    loaded_roi_set = _resolve_roi_set(roi_set)
    _validate_roi_shape(loaded_roi_set, recording)
    return roi_set_to_label_image(loaded_roi_set)


def _resolve_roi_output_path(
    *,
    recording_data_path: Path,
    roi_set: RoiSet | Path | None,
    explicit_roi_output_path: Path | None,
) -> Path:
    """Resolve the default path used by the magicgui Save ROIs button.

    Args:
        recording_data_path: Converted recording manifest path.
        roi_set: Optional ROI object or path opened in napari.
        explicit_roi_output_path: Optional caller-provided output path.

    Returns:
        Path for saving ROI HDF5 files.
    """
    if explicit_roi_output_path is not None:
        return explicit_roi_output_path.expanduser()
    if isinstance(roi_set, Path):
        return roi_set.expanduser()
    return recording_data_path.expanduser().parent / "rois.h5"


def _create_viewer() -> _NapariViewer:
    """Create a napari viewer lazily.

    Args:
        None.

    Returns:
        A napari viewer object exposing the small viewer protocol used here.
    """
    import napari

    return cast(_NapariViewer, napari.Viewer())


def _resolve_roi_set(roi_set: RoiSet | Path) -> RoiSet:
    """Return a ROI set object from either an object or HDF5 path.

    Args:
        roi_set: Existing ``RoiSet`` or saved ROI HDF5 path.

    Returns:
        Loaded ``RoiSet``.
    """
    if isinstance(roi_set, Path):
        return load_roi_set(roi_set)
    return roi_set


def _validate_roi_shape(roi_set: RoiSet, recording: RecordingData) -> None:
    """Validate that ROI labels can be displayed over the recording.

    Args:
        roi_set: ROI masks to display.
        recording: Loaded converted recording.

    Returns:
        None.
    """
    if roi_set.masks.shape[1:] != recording.movie.shape[1:]:
        msg = (
            f"ROI mask shape {roi_set.masks.shape[1:]} does not match movie "
            f"shape {recording.movie.shape[1:]}"
        )
        raise ValueError(msg)
