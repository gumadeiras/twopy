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
from twopy.napari.display import (
    display_image_from_movie_image,
    display_metadata_for_spatial_crop,
)
from twopy.napari.movie import exclusive_stop, resolve_movie_frame_range
from twopy.napari.plotting import (
    add_twopy_response_options_widget,
    add_twopy_response_plot_widget,
)
from twopy.napari.protocols import NapariViewer
from twopy.napari.roi import resolve_roi_save_file, roi_label_image_for_display
from twopy.napari.types import NapariRecordingView
from twopy.roi import RoiSet

__all__ = ["open_recording_in_napari"]


def open_recording_in_napari(
    recording_data_path: Path,
    *,
    roi_set: RoiSet | Path | None = None,
    movie_path: Path | None = None,
    viewer: NapariViewer | None = None,
    movie_frame_range: tuple[int, int | None] | None = None,
    mean_image_layer_name: str = "mean image",
    movie_layer_name: str = "aligned movie",
    roi_layer_name: str = "rois",
    add_roi_labels_layer: bool = True,
    add_controls: bool = True,
    roi_save_file: Path | None = None,
) -> NapariRecordingView:
    """Open a converted recording in napari.

    Args:
        recording_data_path: Path to converted ``recording_data.h5``.
        roi_set: Optional ``RoiSet`` or saved ROI HDF5 path to display.
        movie_path: Optional explicit ``aligned_movie.h5`` path. Folder-based
            loading passes this when the movie file is beside the recording
            manifest.
        viewer: Optional existing napari viewer. When omitted, a new viewer is
            created.
        movie_frame_range: Optional ``(start, end)`` frame range to load as a
            movie preview. ``end=None`` means the final movie frame. ``None``
            avoids loading movie frames and shows only the mean image.
        mean_image_layer_name: Napari layer name for the mean image.
        movie_layer_name: Napari layer name for the optional movie preview.
        roi_layer_name: Napari layer name for ROI labels.
        add_roi_labels_layer: Whether to add a napari Labels layer for ROI
            drawing or editing. When ``roi_set`` is omitted, the layer starts as
            an empty integer image with the movie spatial shape.
        add_controls: Whether to add the small magicgui twopy control panel.
        roi_save_file: Default ROI HDF5 path used by the Save ROIs button.
            When omitted, twopy uses ``rois.h5`` beside ``recording_data.h5``.

    Returns:
        ``NapariRecordingView`` with the loaded recording and created layers.

    Converted movies are usually small enough for direct interactive loading.
    Analysis code still uses chunked readers; this viewer path favors simple,
    auditable GUI behavior.
    """
    recording = load_converted_recording(recording_data_path, movie_path=movie_path)
    resolved_viewer = create_viewer() if viewer is None else viewer
    display_crop = recording.alignment_valid_crop
    layer_metadata = display_metadata_for_spatial_crop(display_crop)
    mean_layer = resolved_viewer.add_image(
        display_image_from_movie_image(display_crop.crop_image(recording.mean_image)),
        name=mean_image_layer_name,
        colormap="gray",
        metadata=layer_metadata,
    )

    movie_layer = None
    if movie_frame_range is not None:
        start_frame, end_frame = resolve_movie_frame_range(
            start_frame=movie_frame_range[0],
            end_frame=movie_frame_range[1],
            frame_count=recording.movie.shape[0],
        )
        movie = recording.movie.read_frames(
            start_frame,
            exclusive_stop(end_frame),
            spatial_crop=display_crop,
        )
        movie_layer = resolved_viewer.add_image(
            display_image_from_movie_image(movie),
            name=movie_layer_name,
            colormap="gray",
            metadata=layer_metadata,
        )

    roi_layer = None
    if add_roi_labels_layer:
        roi_layer = resolved_viewer.add_labels(
            roi_label_image_for_display(
                roi_set,
                recording,
                spatial_crop=display_crop,
            ),
            name=roi_layer_name,
            opacity=0.5,
            blending="additive",
            metadata=layer_metadata,
        )
    controls_widget = None
    controls_dock = None
    save_rois_widget = None
    save_rois_dock = None
    response_plot_widget = None
    response_plot_dock = None
    response_options_widget = None
    response_options_dock = None
    if add_controls:
        response_plot_widget, response_plot_dock = add_twopy_response_plot_widget(
            resolved_viewer,
            recording=recording,
            roi_labels_layer=roi_layer,
        )
        control_docks = add_twopy_magicgui_controls(
            resolved_viewer,
            roi_labels_layer=roi_layer,
            roi_save_file=resolve_roi_save_file(
                recording_data_path=recording.path,
                roi_set=roi_set,
                explicit_roi_save_file=roi_save_file,
            ),
            recording=recording,
            response_plot_widget=response_plot_widget,
        )
        controls_widget = control_docks.load_widget
        controls_dock = control_docks.load_dock_widget
        save_rois_widget = control_docks.save_rois_widget
        save_rois_dock = control_docks.save_rois_dock_widget
        response_options_widget, response_options_dock = (
            add_twopy_response_options_widget(
                resolved_viewer,
                response_plot_widget,
            )
        )
        _place_response_options_after_load_dock(
            resolved_viewer,
            controls_dock,
            response_options_dock,
        )

    return NapariRecordingView(
        viewer=resolved_viewer,
        recording=recording,
        mean_image_layer=mean_layer,
        movie_layer=movie_layer,
        roi_labels_layer=roi_layer,
        controls_widget=controls_widget,
        controls_dock_widget=controls_dock,
        save_rois_widget=save_rois_widget,
        save_rois_dock_widget=save_rois_dock,
        response_plot_widget=response_plot_widget,
        response_plot_dock_widget=response_plot_dock,
        response_options_widget=response_options_widget,
        response_options_dock_widget=response_options_dock,
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


def _place_response_options_after_load_dock(
    viewer: NapariViewer,
    controls_dock_widget: object | None,
    response_options_dock_widget: object | None,
) -> None:
    """Stack response options immediately under the Load Recording dock.

    Args:
        viewer: Napari viewer that owns the dock widgets.
        controls_dock_widget: Dock widget returned for the load controls.
        response_options_dock_widget: Dock widget returned for response options.

    Returns:
        None.

    This is only a visual napari layout tweak. The controls and options widgets
    remain separate so a future plugin wrapper can place them however it wants.
    """
    if controls_dock_widget is None or response_options_dock_widget is None:
        return
    try:
        from qtpy.QtCore import Qt
    except ImportError:
        return

    qt_window = getattr(viewer.window, "_qt_window", None)
    if qt_window is None:
        return
    if not hasattr(qt_window, "splitDockWidget") or not hasattr(
        qt_window,
        "resizeDocks",
    ):
        return
    qt_window.splitDockWidget(
        controls_dock_widget,
        response_options_dock_widget,
        Qt.Orientation.Vertical,
    )
    qt_window.resizeDocks(
        [controls_dock_widget, response_options_dock_widget],
        [260, 10000],
        Qt.Orientation.Vertical,
    )
