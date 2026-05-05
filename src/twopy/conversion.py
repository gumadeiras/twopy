"""Convert microscope source files into twopy-owned HDF5 recording files.

Inputs: one validated two-photon recording session directory.
Outputs: gzip-compressed HDF5 files containing twopy-owned datasets and
metadata for later analysis. The large aligned movie is stored separately from
the smaller recording metadata file.

Source MATLAB and TIFF files are read-only inputs. Analysis code should operate
on converted twopy HDF5 files, not directly on microscope source files.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import h5py
import numpy as np
import numpy.typing as npt
import scipy.io

from twopy.session import TwoPhotonSessionFiles, discover_session_files

__all__ = [
    "AcquisitionMetadata",
    "AlignedMovieSource",
    "ConvertedRecording",
    "PhotodiodeSignals",
    "SourceConversionInputs",
    "StimulusParameters",
    "StimulusTimeline",
    "convert_recording_to_twopy",
    "load_source_conversion_inputs",
]

CONVERTED_RECORDING_FILENAME = "recording_data.h5"
CONVERTED_ALIGNED_MOVIE_FILENAME = "aligned_movie.h5"
ALIGNED_MOVIE_DATASET = "movie/aligned"
SYNCHRONIZATION_MODEL = {
    "imaging_clock": "imaging computer",
    "stimulus_clock": "stimulus presentation computer",
    "clock_relationship": (
        "independent computers with different frame rates; align by photodiode, "
        "not by nominal frame counts alone"
    ),
    "sync_signal": "photodiode",
    "event_encoding": (
        "stimulus computer flashes the photodiode at start, trial transitions, "
        "and end; flash pattern or duration identifies the event type"
    ),
    "analysis_rule": (
        "decode photodiode events before assigning imaging frames to stimulus "
        "trials or computing responses"
    ),
}


class _MatlabStruct(Protocol):
    """Small protocol for SciPy MATLAB structs.

    Inputs: SciPy ``mat_struct`` objects.
    Outputs: field names used by conversion helpers.
    """

    _fieldnames: list[str] | tuple[str, ...]


@dataclass(frozen=True)
class AlignedMovieSource:
    """Lazy source for the aligned imaging movie before conversion.

    Inputs: HDF5-backed MATLAB file path and dataset metadata.
    Outputs: movie shape/dtype metadata plus methods for reading frames.

    This object is only for conversion. Downstream analysis should read the
    copied movie from the twopy aligned-movie HDF5 file.
    """

    path: Path
    dataset_name: str
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...] | None
    compression: str | None

    def read_frames(
        self,
        start: int | None = None,
        stop: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """Read a frame range from the aligned source movie.

        Args:
            start: First frame index to read. ``None`` starts at frame zero.
            stop: Exclusive stop frame index. ``None`` reads through the end.

        Returns:
            NumPy array with shape ``(frames, x, y)`` for the requested range.

        This method exists for conversion checks and small previews, not for
        long-lived analysis workflows.
        """
        frame_slice = slice(start, stop)
        with h5py.File(self.path, "r") as mat_file:
            dataset = mat_file[self.dataset_name]
            return cast(npt.NDArray[np.float64], dataset[frame_slice, :, :])

    def mean_image(
        self,
        *,
        start_frame: int | None = None,
        stop_frame: int | None = None,
        chunk_frames: int = 128,
    ) -> npt.NDArray[np.float64]:
        """Compute a mean image over a frame range without loading all frames.

        Args:
            start_frame: First frame included in the mean. ``None`` uses zero.
            stop_frame: Exclusive stop frame. ``None`` uses the movie end.
            chunk_frames: Number of frames read per chunk.

        Returns:
            Float64 mean image with shape ``(x, y)``.

        The computation streams the movie in chunks, keeping memory use bounded
        by ``chunk_frames`` rather than the full movie length.
        """
        start, stop = _normalize_frame_range(
            frame_count=self.shape[0],
            start_frame=start_frame,
            stop_frame=stop_frame,
        )
        if chunk_frames < 1:
            msg = f"chunk_frames must be at least 1; got {chunk_frames}"
            raise ValueError(msg)

        spatial_shape = self.shape[1:]
        accumulator = np.zeros(spatial_shape, dtype=np.float64)
        with h5py.File(self.path, "r") as mat_file:
            dataset = mat_file[self.dataset_name]
            for chunk_start in range(start, stop, chunk_frames):
                chunk_stop = min(chunk_start + chunk_frames, stop)
                block = np.asarray(dataset[chunk_start:chunk_stop, :, :])
                accumulator += block.sum(axis=0, dtype=np.float64)

        return accumulator / float(stop - start)


@dataclass(frozen=True)
class AcquisitionMetadata:
    """Recording acquisition metadata from ``imageDescription.mat``.

    Inputs: MATLAB ScanImage ``state`` struct.
    Outputs: selected scalar fields in a dictionary.

    The raw MATLAB struct is not exposed because it is source-format data. The
    converted HDF5 file stores this dictionary as HDF5 attributes.
    """

    path: Path
    fields: dict[str, object]


@dataclass(frozen=True)
class StimulusParameters:
    """Stimulus epoch definitions from ``stimParams.mat``.

    Inputs: MATLAB ``stimParams`` structs.
    Outputs: one plain dictionary per epoch.

    Field names and values vary with the stimulus design, so dictionaries keep
    the conversion auditable without inventing a premature fixed schema.
    """

    path: Path
    epochs: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class StimulusTimeline:
    """Per-stimulus-frame timeline from ``stimdata.mat``.

    Inputs: MATLAB ``stimData`` table.
    Outputs: the full numeric table plus common columns as direct arrays.

    The stimulus timeline comes from the stimulus presentation computer. Its
    clock is independent from the imaging computer clock, so response analysis
    must use photodiode synchronization before assigning imaging frames to
    stimulus trials.

    The first observed columns are time, stimulus frame number, and epoch. The
    converted HDF5 file preserves the full table for later analysis.
    """

    path: Path
    data: npt.NDArray[np.float64]
    time_seconds: npt.NDArray[np.float64]
    frame_numbers: npt.NDArray[np.int64]
    epoch_numbers: npt.NDArray[np.int64]


@dataclass(frozen=True)
class PhotodiodeSignals:
    """Photodiode signals used to align imaging frames and stimulus frames.

    Inputs: frame-resolution and high-resolution MATLAB photodiode files.
    Outputs: NumPy arrays ready to write into the twopy HDF5 file.

    Imaging and stimulus presentation run on different computers with different
    frame rates. The stimulus computer flashes the photodiode at key timepoints
    such as start, trial transitions, and end. These signals are the bridge
    between clocks and must be decoded before trial-level response analysis.
    """

    imaging_res_path: Path
    high_res_path: Path
    imaging_res_pd: npt.NDArray[np.float64]
    high_res_pd: npt.NDArray[np.float64]


@dataclass(frozen=True)
class SourceConversionInputs:
    """All source inputs needed to create a twopy converted recording.

    Inputs: one recording session directory.
    Outputs: validated file paths and Python objects used only for conversion.

    This object is the boundary between microscope source data and twopy-owned
    analysis files.
    """

    session_files: TwoPhotonSessionFiles
    aligned_movie: AlignedMovieSource
    acquisition: AcquisitionMetadata
    stimulus_parameters: StimulusParameters
    stimulus_timeline: StimulusTimeline
    photodiode: PhotodiodeSignals


@dataclass(frozen=True)
class ConvertedRecording:
    """Summary of converted twopy recording files.

    Inputs: conversion output path and source recording folder.
    Outputs: HDF5 paths plus mean-image frame range and movie shape.

    Later analysis modules should accept these converted HDF5 paths, not source
    MATLAB file paths. ``path`` points at the small recording metadata file;
    ``movie_path`` points at the large aligned movie file.
    """

    path: Path
    movie_path: Path
    source_session_dir: Path
    movie_shape: tuple[int, ...]
    mean_image_start_frame: int
    mean_image_stop_frame: int


def load_source_conversion_inputs(session_dir: Path) -> SourceConversionInputs:
    """Load the source inputs needed for conversion into twopy format.

    Args:
        session_dir: Timestamped two-photon recording directory.

    Returns:
        A typed source-input bundle for conversion only.

    Raises:
        FileNotFoundError: If the session or required files are missing.
        KeyError: If an expected MATLAB variable or HDF5 dataset is absent.
        ValueError: If a loaded table has an unexpected shape.
    """
    files = discover_session_files(session_dir)
    return SourceConversionInputs(
        session_files=files,
        aligned_movie=_load_aligned_movie_source(files.aligned_movie_mat),
        acquisition=_load_acquisition_metadata(files.image_description_mat),
        stimulus_parameters=_load_stimulus_parameters(
            files.stimulus_data_dir / "stimParams.mat",
        ),
        stimulus_timeline=_load_stimulus_timeline(
            files.stimulus_data_dir / "stimdata.mat",
        ),
        photodiode=_load_photodiode_signals(
            imaging_res_path=files.imaging_res_pd_mat,
            high_res_path=files.high_res_pd_mat,
        ),
    )


def convert_recording_to_twopy(
    session_dir: Path,
    output_dir: Path,
    *,
    mean_start_frame: int | None = None,
    mean_stop_frame: int | None = None,
) -> ConvertedRecording:
    """Convert one recording folder into twopy HDF5 files.

    Args:
        session_dir: Source recording folder.
        output_dir: Directory that will receive converted twopy HDF5 files.
        mean_start_frame: Optional first frame for the mean image.
        mean_stop_frame: Optional exclusive stop frame for the mean image.

    Returns:
        Summary of the converted HDF5 files.

    The conversion copies source data into twopy-owned datasets. The aligned
    movie gets its own file because it dominates size, while the recording file
    keeps metadata, stimulus, photodiode, and the quick-look mean image.
    """
    inputs = load_source_conversion_inputs(session_dir)
    start_frame, stop_frame = _normalize_frame_range(
        frame_count=inputs.aligned_movie.shape[0],
        start_frame=mean_start_frame,
        stop_frame=mean_stop_frame,
    )
    mean_image = inputs.aligned_movie.mean_image(
        start_frame=start_frame,
        stop_frame=stop_frame,
    )

    destination_dir = output_dir.expanduser()
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / CONVERTED_RECORDING_FILENAME
    movie_path = destination_dir / CONVERTED_ALIGNED_MOVIE_FILENAME

    _write_aligned_movie_file(inputs, movie_path)

    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = "converted-recording"
        h5_file.attrs["source_session_dir"] = str(inputs.session_files.session_dir)

        movie_group = h5_file.create_group("movie")
        movie_group.attrs["aligned_movie_file"] = CONVERTED_ALIGNED_MOVIE_FILENAME
        movie_group.attrs["aligned_movie_dataset"] = ALIGNED_MOVIE_DATASET
        movie_group.attrs["aligned_movie_shape"] = inputs.aligned_movie.shape
        movie_group.attrs["aligned_movie_dtype"] = inputs.aligned_movie.dtype
        mean_dataset = movie_group.create_dataset(
            "mean_image",
            data=mean_image,
            compression="gzip",
        )
        mean_dataset.attrs["start_frame"] = start_frame
        mean_dataset.attrs["stop_frame"] = stop_frame

        metadata_group = h5_file.create_group("metadata")
        _write_attrs(metadata_group, inputs.acquisition.fields)

        stimulus_group = h5_file.create_group("stimulus")
        stimulus_group.attrs["clock_source"] = SYNCHRONIZATION_MODEL["stimulus_clock"]
        stimulus_group.create_dataset(
            "timeline",
            data=inputs.stimulus_timeline.data,
            compression="gzip",
        )
        stimulus_group.create_dataset(
            "parameters_json",
            data=json.dumps(_json_ready(inputs.stimulus_parameters.epochs)),
        )

        photodiode_group = h5_file.create_group("photodiode")
        _write_synchronization_metadata(photodiode_group)
        imaging_res_pd = photodiode_group.create_dataset(
            "imaging_res_pd",
            data=inputs.photodiode.imaging_res_pd,
            compression="gzip",
        )
        imaging_res_pd.attrs["sample_domain"] = "one sample per aligned imaging frame"
        high_res_pd = photodiode_group.create_dataset(
            "high_res_pd",
            data=inputs.photodiode.high_res_pd,
            compression="gzip",
        )
        high_res_pd.attrs["sample_domain"] = (
            "high-rate photodiode signal for precise event detection"
        )

    return ConvertedRecording(
        path=output_path,
        movie_path=movie_path,
        source_session_dir=inputs.session_files.session_dir,
        movie_shape=inputs.aligned_movie.shape,
        mean_image_start_frame=start_frame,
        mean_image_stop_frame=stop_frame,
    )


def _load_aligned_movie_source(path: Path) -> AlignedMovieSource:
    """Read aligned movie dataset metadata without loading frames.

    Args:
        path: ``alignedMovie.mat`` path.

    Returns:
        Lazy aligned movie source.
    """
    dataset_name = "imgFrames_ch1"
    with h5py.File(path, "r") as mat_file:
        if dataset_name not in mat_file:
            msg = f"Missing dataset {dataset_name!r} in {path}"
            raise KeyError(msg)
        dataset = mat_file[dataset_name]
        return AlignedMovieSource(
            path=path,
            dataset_name=dataset_name,
            shape=tuple(int(length) for length in dataset.shape),
            dtype=str(dataset.dtype),
            chunks=(
                tuple(int(length) for length in dataset.chunks)
                if dataset.chunks is not None
                else None
            ),
            compression=dataset.compression,
        )


def _load_acquisition_metadata(path: Path) -> AcquisitionMetadata:
    """Load selected ScanImage acquisition metadata from ``imageDescription.mat``.

    Args:
        path: MATLAB file containing the ``state`` struct.

    Returns:
        Acquisition metadata with selected fields flattened.
    """
    state = _load_scipy_variable(path, "state")
    return AcquisitionMetadata(
        path=path,
        fields={
            key: _nested_matlab_field(state, key)
            for key in (
                "configName",
                "software.version",
                "acq.linesPerFrame",
                "acq.pixelsPerLine",
                "acq.numberOfFrames",
                "acq.numberOfChannelsSave",
                "acq.frameRate",
                "acq.zoomFactor",
                "acq.pixelTime",
                "acq.msPerLine",
                "acq.zStepSize",
                "acq.scanAngleMultiplierFast",
                "acq.scanAngleMultiplierSlow",
                "motor.absXPosition",
                "motor.absYPosition",
                "motor.absZPosition",
            )
        },
    )


def _load_stimulus_parameters(path: Path) -> StimulusParameters:
    """Load stimulus epoch parameter structs.

    Args:
        path: ``stimParams.mat`` path.

    Returns:
        Stimulus parameters as one dictionary per epoch.
    """
    raw_params = _load_scipy_variable(path, "stimParams")
    params_array = (
        raw_params
        if isinstance(raw_params, np.ndarray)
        else np.array([raw_params], dtype=object)
    )
    return StimulusParameters(
        path=path,
        epochs=tuple(
            _matlab_struct_to_dict(param)
            for param in params_array.ravel()
            if hasattr(param, "_fieldnames")
        ),
    )


def _load_stimulus_timeline(path: Path) -> StimulusTimeline:
    """Load the numeric stimulus timeline table.

    Args:
        path: ``stimdata.mat`` path.

    Returns:
        Stimulus timeline with direct views/copies for common columns.

    Raises:
        ValueError: If the table does not include time, frame, and epoch columns.
    """
    data = np.asarray(_load_scipy_variable(path, "stimData"), dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 3:
        msg = f"stimData must be a 2D table with at least 3 columns: {path}"
        raise ValueError(msg)

    return StimulusTimeline(
        path=path,
        data=data,
        time_seconds=data[:, 0],
        frame_numbers=data[:, 1].astype(np.int64),
        epoch_numbers=data[:, 2].astype(np.int64),
    )


def _load_photodiode_signals(
    *,
    imaging_res_path: Path,
    high_res_path: Path,
) -> PhotodiodeSignals:
    """Load frame-resolution and high-resolution photodiode vectors.

    Args:
        imaging_res_path: ``imagingResPd.mat`` path.
        high_res_path: ``highResPd.mat`` path.

    Returns:
        Photodiode arrays as float64 vectors.
    """
    return PhotodiodeSignals(
        imaging_res_path=imaging_res_path,
        high_res_path=high_res_path,
        imaging_res_pd=np.asarray(
            _load_scipy_variable(imaging_res_path, "imagingResPd"),
            dtype=np.float64,
        ),
        high_res_pd=np.asarray(
            _load_scipy_variable(high_res_path, "highResPd"),
            dtype=np.float64,
        ),
    )


def _write_synchronization_metadata(group: h5py.Group) -> None:
    """Write the timing model that makes photodiode data interpretable.

    Args:
        group: HDF5 group that receives synchronization attributes.

    Returns:
        None. The function writes human-readable HDF5 attributes.

    These attributes are deliberately plain language. Later response-analysis
    code should read them as the contract: stimulus rows and imaging frames are
    on independent clocks until photodiode events align them.
    """
    for key, value in SYNCHRONIZATION_MODEL.items():
        group.attrs[key] = value


def _write_aligned_movie_file(
    inputs: SourceConversionInputs,
    movie_path: Path,
) -> None:
    """Write the large aligned movie into its own twopy HDF5 file.

    Args:
        inputs: Loaded source conversion inputs.
        movie_path: Destination HDF5 path for aligned movie frames.

    Returns:
        None. The function writes the full aligned movie file.
    """
    with h5py.File(movie_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = "aligned-movie"
        h5_file.attrs["source_session_dir"] = str(inputs.session_files.session_dir)
        movie_group = h5_file.create_group("movie")
        _copy_aligned_movie(inputs.aligned_movie, movie_group)


def _copy_aligned_movie(source: AlignedMovieSource, movie_group: h5py.Group) -> None:
    """Copy aligned movie frames into an HDF5 group in chunks.

    Args:
        source: Lazy source movie metadata and path.
        movie_group: Destination HDF5 group in the aligned movie file.

    Returns:
        None. The function writes ``movie/aligned``.
    """
    chunk_frames = source.chunks[0] if source.chunks is not None else 64
    with h5py.File(source.path, "r") as source_file:
        source_dataset = source_file[source.dataset_name]
        aligned = movie_group.create_dataset(
            "aligned",
            shape=source.shape,
            dtype=source_dataset.dtype,
            chunks=source.chunks,
            compression="gzip",
        )
        aligned.attrs["source_path"] = str(source.path)
        aligned.attrs["source_dataset"] = source.dataset_name

        for start in range(0, source.shape[0], chunk_frames):
            stop = min(start + chunk_frames, source.shape[0])
            aligned[start:stop, :, :] = source_dataset[start:stop, :, :]


def _write_attrs(group: h5py.Group, values: dict[str, object]) -> None:
    """Write scalar metadata values as HDF5 attributes.

    Args:
        group: HDF5 group that receives attributes.
        values: Metadata dictionary.

    Returns:
        None.
    """
    for key, value in values.items():
        ready = _json_ready(value)
        if isinstance(ready, str | int | float | bool):
            group.attrs[key] = ready
        else:
            group.attrs[key] = json.dumps(ready)


def _normalize_frame_range(
    *,
    frame_count: int,
    start_frame: int | None,
    stop_frame: int | None,
) -> tuple[int, int]:
    """Normalize and validate an optional movie frame range.

    Args:
        frame_count: Total number of movie frames.
        start_frame: Optional first frame.
        stop_frame: Optional exclusive stop frame.

    Returns:
        Validated ``(start, stop)`` frame range.
    """
    start = 0 if start_frame is None else start_frame
    stop = frame_count if stop_frame is None else stop_frame
    if start < 0 or stop > frame_count or start >= stop:
        msg = (
            "Invalid frame range for mean image: "
            f"start={start}, stop={stop}, frame_count={frame_count}"
        )
        raise ValueError(msg)
    return start, stop


def _load_scipy_variable(path: Path, name: str) -> object:
    """Load one variable from an older MATLAB file through SciPy.

    Args:
        path: MATLAB file path.
        name: Variable name to load.

    Returns:
        Loaded MATLAB variable as a Python/NumPy object.

    Raises:
        KeyError: If the variable is absent.
    """
    loaded = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    if name not in loaded:
        msg = f"Missing variable {name!r} in {path}"
        raise KeyError(msg)
    return cast(object, loaded[name])


def _nested_matlab_field(root: object, dotted_name: str) -> object:
    """Read a dotted field path from a MATLAB struct.

    Args:
        root: MATLAB struct-like object.
        dotted_name: Field path such as ``acq.frameRate``.

    Returns:
        Python value for the nested field.
    """
    value = root
    for part in dotted_name.split("."):
        value = getattr(value, part)
    return _matlab_value_to_python(value)


def _matlab_struct_to_dict(value: object) -> dict[str, object]:
    """Convert a MATLAB struct-like object to a plain dictionary.

    Args:
        value: Object returned by SciPy for a MATLAB struct.

    Returns:
        Dictionary of field names to converted values.
    """
    struct = cast(_MatlabStruct, value)
    fields = tuple(struct._fieldnames)
    return {field: _matlab_value_to_python(getattr(struct, field)) for field in fields}


def _matlab_value_to_python(value: object) -> object:
    """Convert small MATLAB values into plain Python-friendly values.

    Args:
        value: MATLAB value loaded by SciPy.

    Returns:
        Scalars as Python scalars, struct arrays as tuples, and numeric arrays as
        NumPy arrays.
    """
    if hasattr(value, "_fieldnames"):
        return _matlab_struct_to_dict(value)

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ()
        if value.dtype == object:
            return tuple(_matlab_value_to_python(item) for item in value.ravel())
        if value.size == 1:
            return _matlab_value_to_python(value.item())
        return value

    if isinstance(value, np.generic):
        return value.item()

    return value


def _json_ready(value: object) -> object:
    """Convert Python/NumPy values into JSON-compatible objects.

    Args:
        value: Converted MATLAB value or Python scalar.

    Returns:
        JSON-compatible nested scalars, lists, and dictionaries.
    """
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
