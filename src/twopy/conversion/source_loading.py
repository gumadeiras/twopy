"""Load microscope source files into typed conversion objects.

Inputs: one timestamped two-photon recording directory.
Outputs: Python objects used only to write converted twopy HDF5 files.

This module is the read-only boundary with MATLAB source data.
"""

from pathlib import Path

import h5py
import numpy as np

from twopy.conversion.alignment_crop import load_alignment_valid_crop
from twopy.conversion.matlab_values import (
    load_scipy_variable,
    matlab_struct_to_dict,
    nested_matlab_field,
)
from twopy.conversion.stimulus_code import load_stimulus_code_metadata
from twopy.conversion.types import (
    AcquisitionMetadata,
    AlignedMovieSource,
    FrameCountAudit,
    PhotodiodeSignals,
    RunMetadata,
    SourceConversionInputs,
    StimulusData,
    StimulusParameters,
)
from twopy.session import discover_session_files

__all__ = ["load_source_conversion_inputs"]

STABLE_STIMULUS_DATA_COLUMNS = (
    "time_seconds",
    "stimulus_frame_number",
    "epoch_number",
    "closed_loop_01",
    "closed_loop_02",
    "closed_loop_03",
    "closed_loop_04",
    "closed_loop_05",
    "closed_loop_06",
    "closed_loop_07",
    "closed_loop_08",
    "closed_loop_09",
    "closed_loop_10",
    "stimulus_specific_01",
    "stimulus_specific_02",
    "stimulus_specific_03",
    "stimulus_specific_04",
    "stimulus_specific_05",
    "stimulus_specific_06",
    "stimulus_specific_07",
    "stimulus_specific_08",
    "stimulus_specific_09",
    "stimulus_specific_10",
    "stimulus_specific_11",
    "stimulus_specific_12",
    "stimulus_specific_13",
    "stimulus_specific_14",
    "stimulus_specific_15",
    "stimulus_specific_16",
    "stimulus_specific_17",
    "stimulus_specific_18",
    "stimulus_specific_19",
    "stimulus_specific_20",
    "photodiode_flash",
    "trailing_empty",
)
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
RUN_METADATA_FIELDS = (
    ("flyId", "fly_id"),
    ("genotype", "genotype"),
    ("rigName", "rig_name"),
    ("rigTemperature", "rig_temperature"),
    ("runNumber", "run_number"),
)
STIMULUS_PARAMETER_FILES = (
    ("stimParams.mat", "stimParams"),
    ("chosenparams.mat", "params"),
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
        ValueError: If loaded tables or frame counts have unexpected shapes.
    """
    files = discover_session_files(session_dir)
    aligned_movie = _load_aligned_movie_source(files.aligned_movie_mat)
    acquisition = _load_acquisition_metadata(files.image_description_mat)
    photodiode = _load_photodiode_signals(
        imaging_res_path=files.imaging_res_pd_mat,
        high_res_path=files.high_res_pd_mat,
    )
    frame_counts = _audit_frame_counts(
        aligned_movie=aligned_movie,
        acquisition=acquisition,
        photodiode=photodiode,
    )
    alignment_crop = load_alignment_valid_crop(
        alignment_path=files.alignment_text,
        aligned_movie=aligned_movie,
        acquisition=acquisition,
        photodiode=photodiode,
    )
    stimulus_parameters = _load_stimulus_parameters(files.stimulus_data_dir)
    return SourceConversionInputs(
        session_files=files,
        aligned_movie=aligned_movie,
        acquisition=acquisition,
        run=_load_run_metadata(files.stimulus_data_dir / "runDetails.mat"),
        stimulus_parameters=stimulus_parameters,
        stimulus_code=load_stimulus_code_metadata(
            files.stimulus_filebackup_zip,
            stimulus_parameters,
        ),
        stimulus_data=_load_stimulus_data(
            files.stimulus_data_dir / "stimdata.mat",
        ),
        photodiode=photodiode,
        frame_counts=frame_counts,
        alignment_crop=alignment_crop,
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


def _load_run_metadata(path: Path) -> RunMetadata:
    """Load stimulus-run metadata from ``runDetails.mat``.

    Args:
        path: MATLAB file containing run-level scalar metadata.

    Returns:
        Run metadata with selected fields flattened.

    The source file uses MATLAB-style names such as ``rigName``. twopy converts
    them to snake_case before writing HDF5 so converted files have one naming
    style.
    """
    return RunMetadata(
        path=path,
        fields={
            target_key: load_scipy_variable(path, source_key)
            for source_key, target_key in RUN_METADATA_FIELDS
        },
    )


def _load_stimulus_parameters(stimulus_dir: Path) -> StimulusParameters:
    """Load stimulus epoch parameter structs.

    Args:
        stimulus_dir: ``stimulusData`` folder that should contain structured
            stimulus epoch parameters.

    Returns:
        Stimulus parameters as one dictionary per epoch.

    Newer recordings use ``stimParams.mat`` with variable ``stimParams``. Older
    recordings can instead use ``chosenparams.mat`` with variable ``params``.
    """
    for filename, variable_name in STIMULUS_PARAMETER_FILES:
        path = stimulus_dir / filename
        if path.is_file():
            raw_params = load_scipy_variable(path, variable_name)
            break
    else:
        expected = ", ".join(
            filename for filename, _variable in STIMULUS_PARAMETER_FILES
        )
        msg = (
            f"Missing stimulus parameter MAT file in {stimulus_dir}: "
            f"expected one of {expected}"
        )
        raise FileNotFoundError(msg)

    params_array = (
        raw_params
        if isinstance(raw_params, np.ndarray)
        else np.array([raw_params], dtype=object)
    )
    return StimulusParameters(
        path=path,
        epochs=tuple(
            matlab_struct_to_dict(param)
            for param in np.ravel(params_array)
            if hasattr(param, "_fieldnames")
        ),
    )


def _load_stimulus_data(path: Path) -> StimulusData:
    """Load the numeric stimulus data table.

    Args:
        path: ``stimdata.mat`` path.

    Returns:
        Stimulus data with direct views/copies for common columns.

    Raises:
        ValueError: If the table does not include time, frame, and epoch columns.
    """
    data = np.asarray(load_scipy_variable(path, "stimData"), dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 3:
        msg = f"stimData must be a 2D table with at least 3 columns: {path}"
        raise ValueError(msg)

    return StimulusData(
        path=path,
        data=data,
        column_names=_load_stimulus_data_column_names(column_count=data.shape[1]),
        time_seconds=data[:, 0],
        frame_numbers=data[:, 1].astype(np.int64),
        epoch_numbers=data[:, 2].astype(np.int64),
    )


def _load_stimulus_data_column_names(
    *,
    column_count: int,
) -> tuple[str, ...]:
    """Return code-derived labels for ``stimData`` columns.

    Args:
        column_count: Number of columns in the MATLAB ``stimData`` table.

    Returns:
        One plain-language column label per MATLAB table column.

    ``filebackup/utils/ledUtils/WriteStimData.m`` is the source of truth for
    the stable order. It writes time, stimulus frame, epoch, ten closed-loop
    slots, twenty stimulus-specific slots, the photodiode flash value, then a
    trailing comma. The stimulus-specific slots are decoded from the backed-up
    MATLAB stimulus function selected by ``stimtype``.
    """
    if column_count <= len(STABLE_STIMULUS_DATA_COLUMNS):
        return STABLE_STIMULUS_DATA_COLUMNS[:column_count]
    return STABLE_STIMULUS_DATA_COLUMNS + tuple(
        f"extra_column_{index + 1:02d}"
        for index in range(len(STABLE_STIMULUS_DATA_COLUMNS), column_count)
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


def _audit_frame_counts(
    *,
    aligned_movie: AlignedMovieSource,
    acquisition: AcquisitionMetadata,
    photodiode: PhotodiodeSignals,
) -> FrameCountAudit:
    """Check frame-count relationships that response analysis will rely on.

    Args:
        aligned_movie: Aligned movie source metadata.
        acquisition: ScanImage acquisition metadata from ``imageDescription.mat``.
        photodiode: Loaded photodiode arrays.

    Returns:
        Frame-count audit values for validation and HDF5 metadata.

    Raises:
        ValueError: If the imaging-resolution photodiode does not match the
            aligned movie frame count, or if acquisition metadata differs by
            more than the observed one-frame ScanImage offset.

    Random sampling of real recordings showed ``imagingResPd`` and
    ``alignedMovie`` agree exactly, while ``acq.numberOfFrames`` is commonly one
    less than the aligned movie. Treat that as a known ScanImage metadata
    convention, not a dropped frame. Keep the offset visible for later response
    analysis instead of normalizing it away.
    """
    movie_frames = aligned_movie.shape[0]
    imaging_res_pd_samples = int(photodiode.imaging_res_pd.size)
    acquisition_frames = _required_int_field(acquisition, "acq.numberOfFrames")

    imaging_delta = imaging_res_pd_samples - movie_frames
    if imaging_delta != 0:
        msg = (
            "imagingResPd sample count must match aligned movie frames: "
            f"imaging_res_pd={imaging_res_pd_samples}, movie={movie_frames}"
        )
        raise ValueError(msg)

    acquisition_delta = acquisition_frames - movie_frames
    if acquisition_delta not in {-1, 0}:
        msg = (
            "Unexpected ScanImage acquisition frame count offset: "
            f"acq.numberOfFrames={acquisition_frames}, movie={movie_frames}"
        )
        raise ValueError(msg)

    return FrameCountAudit(
        aligned_movie_frames=movie_frames,
        imaging_res_pd_samples=imaging_res_pd_samples,
        acquisition_number_of_frames=acquisition_frames,
        imaging_res_pd_minus_movie=imaging_delta,
        acquisition_minus_movie=acquisition_delta,
    )


def _required_int_field(acquisition: AcquisitionMetadata, key: str) -> int:
    """Read one required integer acquisition metadata field.

    Args:
        acquisition: Loaded acquisition metadata.
        key: Field name to read.

    Returns:
        Field value as an integer.

    Raises:
        ValueError: If the field cannot be interpreted as an integer.
    """
    value = acquisition.fields[key]
    if isinstance(value, str | bytes | int | float | np.generic):
        return int(value)

    value_type = type(value).__name__
    msg = f"Acquisition field {key!r} must be integer-like, got {value_type}"
    raise ValueError(msg)
