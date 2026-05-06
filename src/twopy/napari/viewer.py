"""Open converted twopy recordings in napari.

Inputs: converted ``recording_data.h5`` paths, optional ROI masks, and optional
movie preview frame ranges.
Outputs: napari image/label layers plus the twopy control dock.

This module owns viewer/layer creation only. It does not query the database,
read source MATLAB/TIFF files, or perform analysis.
"""

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt

from twopy.converted import load_converted_recording
from twopy.napari.constants import DEFAULT_MOVIE_CHUNK_FRAMES
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
    movie_path: Path | None = None,
    viewer: NapariViewer | None = None,
    movie_frame_range: tuple[int, int | None] | None = None,
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
        movie_path: Optional explicit ``aligned_movie.h5`` path. Folder-based
            loading passes this when the movie file is beside the recording
            manifest.
        viewer: Optional existing napari viewer. When omitted, a new viewer is
            created.
        movie_frame_range: Optional ``(start, stop)`` frame range to load as a
            movie preview. ``stop=None`` means the final movie frame. ``None``
            avoids loading movie frames and shows only the mean image.
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

    Movie layer data is dask-backed, so loading the full movie keeps memory
    bounded and reads HDF5 chunks only as napari needs them.
    """
    recording = load_converted_recording(recording_data_path, movie_path=movie_path)
    resolved_viewer = create_viewer() if viewer is None else viewer
    mean_layer = resolved_viewer.add_image(
        recording.mean_image,
        name=mean_image_layer_name,
        colormap="gray",
    )

    movie_layer = None
    if movie_frame_range is not None:
        start_frame, stop_frame = movie_frame_range
        movie = _lazy_movie_layer_data(
            movie_path=recording.movie.path,
            dataset_name=recording.movie.dataset_name,
            movie_shape=recording.movie.shape,
            movie_dtype=recording.movie.dtype,
            start_frame=start_frame,
            stop_frame=stop_frame,
        )
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


def _lazy_movie_layer_data(
    *,
    movie_path: Path,
    dataset_name: str,
    movie_shape: tuple[int, int, int],
    movie_dtype: str,
    start_frame: int,
    stop_frame: int | None,
) -> object:
    """Create dask-backed movie data for a napari image layer.

    Args:
        movie_path: Converted aligned movie HDF5 path.
        dataset_name: HDF5 dataset path for the movie.
        movie_shape: Full movie shape.
        movie_dtype: Movie dtype string stored during conversion.
        start_frame: First frame to expose.
        stop_frame: Exclusive stop frame, or ``None`` for the final frame.

    Returns:
        Dask array with shape ``(frames, axis0, axis1)``.

    Opening HDF5 inside each chunk avoids keeping a global file handle alive in
    the GUI. Napari requests chunks as the user scrolls through time.
    """
    import dask
    import dask.array as da

    start, stop = _normalize_movie_frame_range(
        frame_count=movie_shape[0],
        start_frame=start_frame,
        stop_frame=stop_frame,
    )
    chunk_arrays = []
    for chunk_start in range(start, stop, DEFAULT_MOVIE_CHUNK_FRAMES):
        chunk_stop = min(chunk_start + DEFAULT_MOVIE_CHUNK_FRAMES, stop)
        delayed_chunk = dask.delayed(_read_movie_chunk)(
            movie_path=movie_path,
            dataset_name=dataset_name,
            start_frame=chunk_start,
            stop_frame=chunk_stop,
        )
        chunk_arrays.append(
            da.from_delayed(
                delayed_chunk,
                shape=(chunk_stop - chunk_start, movie_shape[1], movie_shape[2]),
                dtype=np.dtype(movie_dtype),
            ),
        )

    return da.concatenate(chunk_arrays, axis=0)


def _normalize_movie_frame_range(
    *,
    frame_count: int,
    start_frame: int,
    stop_frame: int | None,
) -> tuple[int, int]:
    """Validate and clamp a requested movie frame range.

    Args:
        frame_count: Number of frames in the converted movie.
        start_frame: Requested first frame.
        stop_frame: Requested exclusive stop frame, or ``None`` for the final
            frame.

    Returns:
        Valid ``(start, stop)`` indices.
    """
    if start_frame < 0:
        msg = f"movie start frame must be at least 0; got {start_frame}"
        raise ValueError(msg)
    stop = frame_count if stop_frame is None else min(stop_frame, frame_count)
    if stop <= start_frame:
        msg = f"movie frame range is empty: start={start_frame}, stop={stop}"
        raise ValueError(msg)
    return start_frame, stop


def _read_movie_chunk(
    *,
    movie_path: Path,
    dataset_name: str,
    start_frame: int,
    stop_frame: int,
) -> npt.NDArray[np.float64]:
    """Read one movie chunk from HDF5.

    Args:
        movie_path: Converted aligned movie HDF5 path.
        dataset_name: HDF5 dataset path for the movie.
        start_frame: First frame to read.
        stop_frame: Exclusive stop frame.

    Returns:
        Movie chunk array.
    """
    with h5py.File(movie_path, "r") as h5_file:
        return cast(
            npt.NDArray[np.float64],
            h5_file[dataset_name][start_frame:stop_frame],
        )
