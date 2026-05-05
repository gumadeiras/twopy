"""Write converted twopy data into HDF5 files.

Inputs: loaded source conversion objects and computed summary images.
Outputs: HDF5 files owned by twopy, using gzip for large compressible arrays.

This module owns the on-disk converted layout. Source MATLAB and TIFF files are
read-only inputs handled before these writers run.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from twopy.conversion.matlab_values import json_ready
from twopy.conversion.types import (
    AlignedMovieSource,
    PhotodiodeSignals,
    SourceConversionInputs,
    StimulusData,
    StimulusParameters,
)

__all__ = [
    "ALIGNED_MOVIE_DATASET",
    "CONVERTED_ALIGNED_MOVIE_FILENAME",
    "CONVERTED_RECORDING_FILENAME",
    "write_aligned_movie_file",
    "write_recording_data_file",
]

CONVERTED_RECORDING_FILENAME = "recording_data.h5"
CONVERTED_ALIGNED_MOVIE_FILENAME = "aligned_movie.h5"
ALIGNED_MOVIE_DATASET = "movie/aligned"
IMAGING_RES_PD_SAMPLE_DOMAIN = "one sample per aligned imaging frame"
HIGH_RES_PD_SAMPLE_DOMAIN = "high-rate photodiode signal for precise event detection"
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


def write_recording_data_file(
    *,
    inputs: SourceConversionInputs,
    output_path: Path,
    mean_image: npt.NDArray[np.float64],
    mean_start_frame: int,
    mean_stop_frame: int,
) -> None:
    """Write non-movie converted recording data into the main HDF5 file.

    Args:
        inputs: Loaded source conversion inputs.
        output_path: Destination ``recording_data.h5`` path.
        mean_image: Precomputed quick-look image.
        mean_start_frame: First frame included in the mean image.
        mean_stop_frame: Exclusive stop frame for the mean image.

    Returns:
        None. The function writes ``recording_data.h5``.
    """
    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = "converted-recording"
        h5_file.attrs["source_session_dir"] = str(inputs.session_files.session_dir)

        _write_movie_reference_group(
            h5_file=h5_file,
            movie=inputs.aligned_movie,
            mean_image=mean_image,
            mean_start_frame=mean_start_frame,
            mean_stop_frame=mean_stop_frame,
        )
        metadata_group = h5_file.create_group("metadata")
        _write_attrs(metadata_group, inputs.acquisition.fields)
        run_group = h5_file.create_group("run")
        _write_attrs(run_group, inputs.run.fields)
        _write_stimulus_group(
            h5_file,
            inputs.stimulus_data,
            inputs.stimulus_parameters,
        )
        _write_photodiode_group(h5_file, inputs.photodiode)
        _write_frame_count_audit_group(h5_file, inputs)


def write_aligned_movie_file(
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


def _write_movie_reference_group(
    *,
    h5_file: h5py.File,
    movie: AlignedMovieSource,
    mean_image: npt.NDArray[np.float64],
    mean_start_frame: int,
    mean_stop_frame: int,
) -> None:
    """Write mean image data and a pointer to the separate aligned movie file.

    Args:
        h5_file: Open ``recording_data.h5`` file.
        movie: Source aligned movie metadata.
        mean_image: Precomputed quick-look image.
        mean_start_frame: First frame included in the mean image.
        mean_stop_frame: Exclusive stop frame for the mean image.

    Returns:
        None. The function writes the ``movie`` group.
    """
    movie_group = h5_file.create_group("movie")
    movie_group.attrs["aligned_movie_file"] = CONVERTED_ALIGNED_MOVIE_FILENAME
    movie_group.attrs["aligned_movie_dataset"] = ALIGNED_MOVIE_DATASET
    movie_group.attrs["aligned_movie_shape"] = movie.shape
    movie_group.attrs["aligned_movie_dtype"] = movie.dtype

    mean_dataset = movie_group.create_dataset(
        "mean_image",
        data=mean_image,
    )
    mean_dataset.attrs["start_frame"] = mean_start_frame
    mean_dataset.attrs["stop_frame"] = mean_stop_frame


def _write_stimulus_group(
    h5_file: h5py.File,
    stimulus_data: StimulusData,
    parameters: StimulusParameters,
) -> None:
    """Write converted stimulus data and epoch parameters.

    Args:
        h5_file: Open ``recording_data.h5`` file.
        stimulus_data: Stimulus data loaded from ``stimdata.mat``.
        parameters: Stimulus epoch parameters loaded from ``stimParams.mat``.

    Returns:
        None. The function writes the ``stimulus`` group.
    """
    stimulus_group = h5_file.create_group("stimulus")
    stimulus_group.attrs["clock_source"] = SYNCHRONIZATION_MODEL["stimulus_clock"]
    stimulus_group.create_dataset(
        "data",
        data=stimulus_data.data,
        compression="gzip",
    )
    stimulus_group.create_dataset(
        "data_column_names",
        data=np.asarray(stimulus_data.column_names, dtype=h5py.string_dtype("utf-8")),
    )
    stimulus_group.create_dataset(
        "parameters_json",
        data=json.dumps(json_ready(parameters.epochs)),
    )


def _write_photodiode_group(
    h5_file: h5py.File,
    photodiode: PhotodiodeSignals,
) -> None:
    """Write photodiode vectors plus synchronization metadata.

    Args:
        h5_file: Open ``recording_data.h5`` file.
        photodiode: Photodiode signals loaded from source MATLAB files.

    Returns:
        None. The function writes the ``photodiode`` group.
    """
    photodiode_group = h5_file.create_group("photodiode")
    _write_synchronization_metadata(photodiode_group)

    imaging_res_pd = photodiode_group.create_dataset(
        "imaging_res_pd",
        data=photodiode.imaging_res_pd,
        compression="gzip",
    )
    imaging_res_pd.attrs["sample_domain"] = IMAGING_RES_PD_SAMPLE_DOMAIN

    high_res_pd = photodiode_group.create_dataset(
        "high_res_pd",
        data=photodiode.high_res_pd,
        compression="gzip",
    )
    high_res_pd.attrs["sample_domain"] = HIGH_RES_PD_SAMPLE_DOMAIN


def _write_frame_count_audit_group(
    h5_file: h5py.File,
    inputs: SourceConversionInputs,
) -> None:
    """Write source frame-count checks into the converted recording file.

    Args:
        h5_file: Open ``recording_data.h5`` file.
        inputs: Loaded source conversion inputs with frame-count audit values.

    Returns:
        None. The function writes the ``frame_counts`` group.

    These values keep the observed ScanImage one-frame metadata offset visible
    before response-analysis code starts assigning frames to trials.
    """
    frame_counts = h5_file.create_group("frame_counts")
    audit = inputs.frame_counts
    frame_counts.attrs["aligned_movie_frames"] = audit.aligned_movie_frames
    frame_counts.attrs["imaging_res_pd_samples"] = audit.imaging_res_pd_samples
    frame_counts.attrs["acquisition_number_of_frames"] = (
        audit.acquisition_number_of_frames
    )
    frame_counts.attrs["imaging_res_pd_minus_movie"] = audit.imaging_res_pd_minus_movie
    frame_counts.attrs["acquisition_minus_movie"] = audit.acquisition_minus_movie


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
        ready = json_ready(value)
        if isinstance(ready, str | int | float | bool):
            group.attrs[key] = ready
        else:
            group.attrs[key] = json.dumps(ready)
