"""Load microscope source files into typed conversion objects.

Inputs: one timestamped two-photon recording directory.
Outputs: Python objects used only to write converted twopy HDF5 files.

This module is the read-only boundary with MATLAB source data.
"""

from pathlib import Path

import h5py
import numpy as np

from twopy.conversion.matlab_values import (
    load_scipy_variable,
    matlab_struct_to_dict,
    nested_matlab_field,
)
from twopy.conversion.types import (
    AcquisitionMetadata,
    AlignedMovieSource,
    PhotodiodeSignals,
    SourceConversionInputs,
    StimulusParameters,
    StimulusTimeline,
)
from twopy.session import discover_session_files

__all__ = ["load_source_conversion_inputs"]

ACQUISITION_METADATA_FIELDS = (
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
    state = load_scipy_variable(path, "state")
    return AcquisitionMetadata(
        path=path,
        fields={
            key: nested_matlab_field(state, key) for key in ACQUISITION_METADATA_FIELDS
        },
    )


def _load_stimulus_parameters(path: Path) -> StimulusParameters:
    """Load stimulus epoch parameter structs.

    Args:
        path: ``stimParams.mat`` path.

    Returns:
        Stimulus parameters as one dictionary per epoch.
    """
    raw_params = load_scipy_variable(path, "stimParams")
    params_array = (
        raw_params
        if isinstance(raw_params, np.ndarray)
        else np.array([raw_params], dtype=object)
    )
    return StimulusParameters(
        path=path,
        epochs=tuple(
            matlab_struct_to_dict(param)
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
    data = np.asarray(load_scipy_variable(path, "stimData"), dtype=np.float64)
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
            load_scipy_variable(imaging_res_path, "imagingResPd"),
            dtype=np.float64,
        ),
        high_res_pd=np.asarray(
            load_scipy_variable(high_res_path, "highResPd"),
            dtype=np.float64,
        ),
    )
