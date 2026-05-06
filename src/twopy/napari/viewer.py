"""Open converted twopy recordings in napari.

Inputs: converted ``recording_data.h5`` paths, optional ROI masks, and optional
movie preview frame ranges.
Outputs: napari image/label layers plus the twopy control dock.

This module owns viewer/layer creation only. It does not query the database,
read source MATLAB/TIFF files, or perform analysis.
"""

from pathlib import Path
from typing import cast

from twopy.converted import load_converted_recording
from twopy.napari.controls import add_twopy_magicgui_controls
from twopy.napari.protocols import NapariViewer
from twopy.napari.roi import resolve_roi_output_path, roi_label_image_for_display
from twopy.napari.types import NapariRecordingView
from twopy.roi import RoiSet

__all__ = ["open_recording_in_napari"]


def open_recording_in_napari(
    recording_data_path: Path,
    *,
    roi_set: RoiSet | Path | None = None,
    viewer: NapariViewer | None = None,
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
    resolved_viewer = create_viewer() if viewer is None else viewer
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
            roi_label_image_for_display(roi_set, recording),
            name=roi_layer_name,
        )
    controls_widget = None
    controls_dock = None
    if add_controls:
        controls_widget, controls_dock = add_twopy_magicgui_controls(
            resolved_viewer,
            roi_labels_layer=roi_layer,
            roi_output_path=resolve_roi_output_path(
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


def create_viewer() -> NapariViewer:
    """Create a napari viewer lazily.

    Args:
        None.

    Returns:
        A napari viewer object exposing the small viewer protocol used here.
    """
    import napari

    return cast(NapariViewer, napari.Viewer())
