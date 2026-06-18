"""Open converted twopy recordings in napari.

Inputs: converted ``recording_data.h5`` paths, optional ROI masks, and optional
movie preview frame ranges.
Outputs: napari image/label layers plus the twopy control dock.

This module owns viewer/layer creation only. It does not query the database,
read source MATLAB/TIFF files, or perform analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from twopy._version import __version__
from twopy.converted import ConvertedMovie, RecordingData, load_converted_recording
from twopy.napari.controls import add_twopy_magicgui_controls
from twopy.napari.display import display_metadata_for_spatial_crop
from twopy.napari.empty_state import (
    hide_empty_viewer_message,
    show_empty_viewer_message,
)
from twopy.napari.movie import exclusive_stop, resolve_movie_frame_range
from twopy.napari.output_routing import NapariOutputRoute
from twopy.napari.plotting import (
    add_twopy_response_plot_widget,
    create_twopy_response_options_widget,
)
from twopy.napari.protocols import NapariViewer
from twopy.napari.roi import resolve_roi_save_file, roi_label_image_for_display
from twopy.napari.trial_timeline import (
    TrialTimelineController,
    movie_frame_metadata,
)
from twopy.napari.types import NapariRecordingView
from twopy.roi import RoiSet
from twopy.spatial import SpatialCrop

APPLICATION_TITLE = f"twopy {__version__}"

__all__ = [
    "APPLICATION_TITLE",
    "PreparedNapariRecordingViewData",
    "add_prepared_recording_to_viewer",
    "create_viewer",
    "open_recording_in_napari",
    "prepare_recording_view_data",
]


@dataclass(frozen=True)
class PreparedNapariRecordingViewData:
    """Data read before Qt creates napari layers for one recording.

    Args:
        recording: Loaded converted recording metadata and lazy movie reader.
        mean_image: Cropped mean image shown as the static image layer.
        movie: Optional cropped movie frames for the interactive movie layer.
        roi_labels: Optional cropped ROI labels shown in the Labels layer.
        layer_metadata: Display metadata shared by image and Labels layers.
        movie_contrast_limits: Optional display contrast limits for the movie.
        movie_frame_span: Optional inclusive start and exclusive stop frame
            used for movie-layer metadata.

    The object contains arrays and plain metadata only. Worker threads can build
    it safely, while Qt layer creation stays in ``add_prepared_recording_to_viewer``.
    """

    recording: RecordingData
    mean_image: npt.NDArray[np.float64]
    movie: npt.NDArray[np.float64] | None
    roi_labels: npt.NDArray[np.int64] | None
    layer_metadata: dict[str, object]
    movie_contrast_limits: tuple[float, float] | None
    movie_frame_span: tuple[int, int] | None


class _LabelsLayerWithBrushSize(Protocol):
    """Small layer shape needed to configure ROI painting defaults."""

    brush_size: int


class _ImageLayerWithContrastRange(Protocol):
    """Small layer shape needed to configure contrast slider bounds."""

    contrast_limits_range: tuple[float, float]


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
    output_route: NapariOutputRoute | None = None,
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
        add_controls: Whether to add the twopy response plot and right
            sidebar controls.
        roi_save_file: Default ROI HDF5 path used by the response-options
            persistence button. When omitted, twopy uses ``rois.h5`` beside
            ``recording_data.h5``.
        output_route: Optional local and published output folders resolved by
            the launcher or Load tab.

    Returns:
        ``NapariRecordingView`` with the loaded recording and created layers.

    Converted movies are usually small enough for direct interactive loading.
    Analysis code still uses chunked readers; this viewer path favors simple,
    auditable GUI behavior. When ``add_controls`` is true, the twopy control
    docks load ``config.yml`` for output routing, pixel calibration, custom
    workflows, and database-backed loading.
    """
    resolved_viewer = create_viewer() if viewer is None else viewer
    prepared = prepare_recording_view_data(
        recording_data_path,
        roi_set=roi_set,
        movie_path=movie_path,
        movie_frame_range=movie_frame_range,
        add_roi_labels_layer=add_roi_labels_layer,
    )
    view = add_prepared_recording_to_viewer(
        prepared,
        viewer=resolved_viewer,
        mean_image_layer_name=mean_image_layer_name,
        movie_layer_name=movie_layer_name,
        roi_layer_name=roi_layer_name,
    )
    recording = prepared.recording
    mean_layer = view.mean_image_layer
    movie_layer = view.movie_layer
    roi_layer = view.roi_labels_layer
    load_widget = None
    loaded_recordings_widget = None
    twopy_sidebar_widget = None
    twopy_sidebar_dock = None
    response_plot_widget = None
    response_plot_dock = None
    response_options_widget = None
    trial_timeline_controller = None
    if add_controls:
        trial_timeline_controller = TrialTimelineController(resolved_viewer)
        resolved_roi_save_file = resolve_roi_save_file(
            recording_data_path=recording.path,
            roi_set=roi_set,
            explicit_roi_save_file=roi_save_file,
        )
        response_plot_widget, response_plot_dock = add_twopy_response_plot_widget(
            resolved_viewer,
            recording=recording,
            roi_labels_layer=roi_layer,
            roi_save_file=resolved_roi_save_file,
            output_route=output_route,
        )
        response_options_widget = create_twopy_response_options_widget(
            response_plot_widget,
        )
        control_docks = add_twopy_magicgui_controls(
            resolved_viewer,
            roi_labels_layer=roi_layer,
            roi_save_file=resolved_roi_save_file,
            recording=recording,
            output_route=output_route,
            response_plot_widget=response_plot_widget,
            response_options_widget=response_options_widget,
            mean_image_layer=mean_layer,
            movie_layer=movie_layer,
            trial_timeline_controller=trial_timeline_controller,
        )
        load_widget = control_docks.load_widget
        loaded_recordings_widget = control_docks.loaded_recordings_widget
        twopy_sidebar_widget = control_docks.sidebar_widget
        twopy_sidebar_dock = control_docks.sidebar_dock_widget
        trial_timeline_controller.set_context(recording, movie_layer)

    return NapariRecordingView(
        viewer=resolved_viewer,
        recording=recording,
        mean_image_layer=mean_layer,
        movie_layer=movie_layer,
        roi_labels_layer=roi_layer,
        load_widget=load_widget,
        loaded_recordings_widget=loaded_recordings_widget,
        twopy_sidebar_widget=twopy_sidebar_widget,
        twopy_sidebar_dock_widget=twopy_sidebar_dock,
        response_plot_widget=response_plot_widget,
        response_plot_dock_widget=response_plot_dock,
        response_options_widget=response_options_widget,
        trial_timeline_controller=trial_timeline_controller,
    )


def prepare_recording_view_data(
    recording_data_path: Path,
    *,
    roi_set: RoiSet | Path | None = None,
    movie_path: Path | None = None,
    movie_frame_range: tuple[int, int | None] | None = None,
    add_roi_labels_layer: bool = True,
) -> PreparedNapariRecordingViewData:
    """Read recording arrays before creating napari layers.

    Args:
        recording_data_path: Path to converted ``recording_data.h5``.
        roi_set: Optional ROI set or saved ROI HDF5 path.
        movie_path: Optional explicit ``aligned_movie.h5`` path.
        movie_frame_range: Optional movie-frame range to read.
        add_roi_labels_layer: Whether ROI labels should be prepared.

    Returns:
        Prepared data ready for main-thread layer creation.

    HDF5 file reads can be slow for full movies. Keeping those reads here lets
    GUI callers run this function in a worker while preserving Qt's main-thread
    ownership of napari layer objects.
    """
    recording = load_converted_recording(recording_data_path, movie_path=movie_path)
    display_crop = recording.alignment_valid_crop
    layer_metadata = display_metadata_for_spatial_crop(display_crop)
    mean_image = display_crop.crop_image(recording.mean_image)
    movie = None
    movie_contrast_limits = None
    movie_frame_span = None
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
        movie_contrast_limits = sampled_movie_auto_contrast_limits(
            recording.movie,
            spatial_crop=display_crop,
        )
        movie_frame_span = (start_frame, end_frame + 1)
    roi_labels = (
        roi_label_image_for_display(
            roi_set,
            recording,
            spatial_crop=display_crop,
        )
        if add_roi_labels_layer
        else None
    )
    return PreparedNapariRecordingViewData(
        recording=recording,
        mean_image=mean_image,
        movie=movie,
        roi_labels=roi_labels,
        layer_metadata=layer_metadata,
        movie_contrast_limits=movie_contrast_limits,
        movie_frame_span=movie_frame_span,
    )


def add_prepared_recording_to_viewer(
    prepared: PreparedNapariRecordingViewData,
    *,
    viewer: NapariViewer,
    mean_image_layer_name: str = "mean image",
    movie_layer_name: str = "aligned movie",
    roi_layer_name: str = "rois",
) -> NapariRecordingView:
    """Create napari layers from prepared recording arrays.

    Args:
        prepared: Worker-safe data returned by ``prepare_recording_view_data``.
        viewer: Napari viewer receiving new layers.
        mean_image_layer_name: Name for the mean-image layer.
        movie_layer_name: Name for the optional movie layer.
        roi_layer_name: Name for the optional Labels layer.

    Returns:
        Napari view object with created layers and loaded recording metadata.
    """
    hide_empty_viewer_message(viewer)
    recording = prepared.recording
    mean_image = prepared.mean_image
    mean_layer = viewer.add_image(
        mean_image,
        name=mean_image_layer_name,
        colormap="gray",
        opacity=0.5,
        gamma=1.3,
        contrast_limits=contrast_limits_from_data_range(
            mean_image,
            minimum_fraction=0.10,
            maximum_fraction=1.0,
        ),
        metadata=prepared.layer_metadata,
    )
    set_image_contrast_limits_range(mean_layer, data=mean_image)

    movie_layer = None
    if prepared.movie is not None:
        if prepared.movie_contrast_limits is None or prepared.movie_frame_span is None:
            msg = "Prepared movie data must include contrast limits and frame span."
            raise ValueError(msg)
        movie_layer = viewer.add_image(
            prepared.movie,
            name=movie_layer_name,
            colormap="gray",
            blending="additive",
            contrast_limits=prepared.movie_contrast_limits,
            metadata=prepared.layer_metadata
            | movie_frame_metadata(
                start_frame=prepared.movie_frame_span[0],
                stop_frame=prepared.movie_frame_span[1],
            ),
        )
        set_image_contrast_limits_range(movie_layer, data=prepared.movie)

    roi_layer = None
    if prepared.roi_labels is not None:
        roi_layer = viewer.add_labels(
            prepared.roi_labels,
            name=roi_layer_name,
            opacity=0.5,
            blending="additive",
            metadata=prepared.layer_metadata,
        )
        set_labels_brush_size(roi_layer, brush_size=6)

    return NapariRecordingView(
        viewer=viewer,
        recording=recording,
        mean_image_layer=mean_layer,
        movie_layer=movie_layer,
        roi_labels_layer=roi_layer,
        load_widget=None,
        loaded_recordings_widget=None,
        twopy_sidebar_widget=None,
        twopy_sidebar_dock_widget=None,
        response_plot_widget=None,
        response_plot_dock_widget=None,
        response_options_widget=None,
        trial_timeline_controller=None,
    )


def create_viewer() -> NapariViewer:
    """Create a napari viewer lazily.

    Args:
        None.

    Returns:
        A napari viewer object exposing the small viewer protocol used here.
    """
    import napari

    viewer = cast(NapariViewer, napari.Viewer(title=APPLICATION_TITLE))
    show_empty_viewer_message(viewer)
    return viewer


def set_labels_brush_size(layer: object, *, brush_size: int) -> None:
    """Set the brush size on a napari Labels layer after creation.

    Args:
        layer: Labels layer returned by ``viewer.add_labels``.
        brush_size: Brush diameter used by the ROI painting tool.

    Returns:
        None.

    Napari exposes ``brush_size`` as a layer property, not a
    ``Viewer.add_labels`` argument in all supported versions. Setting it here
    keeps ROI drawing defaults explicit without making recording loading depend
    on version-specific layer-constructor keywords.
    """
    labels_layer = cast(_LabelsLayerWithBrushSize, layer)
    labels_layer.brush_size = brush_size


def set_image_contrast_limits_range(layer: object, *, data: npt.ArrayLike) -> None:
    """Allow napari contrast sliders to cover the full loaded data range.

    Args:
        layer: Image layer returned by ``viewer.add_image``.
        data: Image or movie values shown in that layer.

    Returns:
        None.

    ``contrast_limits`` sets the initial slider handles. Napari initializes the
    slider's allowed range from those handles, so we reset
    ``contrast_limits_range`` afterward to keep the full data range available.
    """
    image_layer = cast(_ImageLayerWithContrastRange, layer)
    image_layer.contrast_limits_range = contrast_limits_from_data_range(
        data,
        minimum_fraction=0.0,
        maximum_fraction=1.0,
    )


def sampled_movie_auto_contrast_limits(
    movie: ConvertedMovie,
    *,
    spatial_crop: SpatialCrop,
    sample_count: int = 10,
    edge_fraction: float = 0.10,
    random_seed: int = 0,
) -> tuple[float, float]:
    """Estimate movie display contrast from random interior frames.

    Args:
        movie: Lazy converted aligned-movie reader.
        spatial_crop: Spatial crop shown in napari.
        sample_count: Maximum number of frames to sample.
        edge_fraction: Fraction of recording edges excluded before sampling.
        random_seed: Seed used for deterministic frame sampling.

    Returns:
        ``(minimum, maximum)`` contrast limits for the movie layer.

    The napari movie layer can be a short preview range, but display contrast
    should reflect the recording rather than the first frames a user happened
    to load. We sample a few frames away from recording edges, compute simple
    per-frame auto contrast limits, then average those limits. The fixed seed
    keeps the same recording visually stable across launches.
    """
    frame_indices = interior_random_frame_indices(
        frame_count=movie.shape[0],
        sample_count=sample_count,
        edge_fraction=edge_fraction,
        random_seed=random_seed,
    )
    limits = [
        contrast_limits_from_data_range(
            movie.read_frames(
                int(frame_index),
                int(frame_index) + 1,
                spatial_crop=spatial_crop,
            )[0],
            minimum_fraction=0.0,
            maximum_fraction=1.0,
        )
        for frame_index in frame_indices
    ]
    if len(limits) == 0:
        return (0.0, 1.0)
    lower = float(np.mean([limit[0] for limit in limits]))
    upper = float(np.mean([limit[1] for limit in limits]))
    if upper <= lower:
        return contrast_limits_from_data_range(
            movie.read_frames(0, 1, spatial_crop=spatial_crop)[0],
            minimum_fraction=0.0,
            maximum_fraction=1.0,
        )
    return (lower, upper)


def interior_random_frame_indices(
    *,
    frame_count: int,
    sample_count: int,
    edge_fraction: float,
    random_seed: int,
) -> npt.NDArray[np.int64]:
    """Return deterministic random frame indices away from recording edges.

    Args:
        frame_count: Number of frames in the aligned movie.
        sample_count: Maximum number of frames to return.
        edge_fraction: Fraction excluded from both start and end.
        random_seed: Seed used by NumPy's random generator.

    Returns:
        Sorted frame indices.

    Very short recordings may not have enough frames to remove both edge
    regions. In that case this falls back to all frames, because a stable
    display estimate is better than failing the viewer load.
    """
    if frame_count <= 0 or sample_count <= 0:
        return np.asarray((), dtype=np.int64)
    edge_count = int(np.floor(frame_count * edge_fraction))
    start = edge_count
    stop = frame_count - edge_count
    if start >= stop:
        start = 0
        stop = frame_count
    candidates = np.arange(start, stop, dtype=np.int64)
    if candidates.size <= sample_count:
        return candidates
    rng = np.random.default_rng(random_seed)
    sampled = rng.choice(candidates, size=sample_count, replace=False)
    return np.sort(sampled)


def contrast_limits_from_data_range(
    data: npt.ArrayLike,
    *,
    minimum_fraction: float,
    maximum_fraction: float,
) -> tuple[float, float]:
    """Return contrast limits at fractions of one data range.

    Args:
        data: Image or movie values shown in napari.
        minimum_fraction: Fraction between the finite data minimum and maximum
            used for the lower contrast limit.
        maximum_fraction: Fraction between the finite data minimum and maximum
            used for the upper contrast limit.

    Returns:
        ``(minimum, maximum)`` contrast limits.

    This keeps the viewer display easier to inspect without changing the
    underlying converted data. Fractions are range positions, not percentiles,
    so the result is deterministic and cheap for already-loaded preview arrays.
    """
    values = np.asarray(data, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return (0.0, 1.0)
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    if maximum <= minimum:
        return (minimum, maximum)
    value_range = maximum - minimum
    return (
        minimum + value_range * minimum_fraction,
        minimum + value_range * maximum_fraction,
    )
