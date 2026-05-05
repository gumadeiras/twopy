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
    FrameCountAudit,
    PhotodiodeSignals,
    RunMetadata,
    SourceConversionInputs,
    StimulusData,
    StimulusParameters,
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
RUN_METADATA_FIELDS = (
    ("flyId", "fly_id"),
    ("genotype", "genotype"),
    ("rigName", "rig_name"),
    ("rigTemperature", "rig_temperature"),
    ("runNumber", "run_number"),
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
    return SourceConversionInputs(
        session_files=files,
        aligned_movie=aligned_movie,
        acquisition=acquisition,
        run=_load_run_metadata(files.stimulus_data_dir / "runDetails.mat"),
        stimulus_parameters=_load_stimulus_parameters(
            files.stimulus_data_dir / "stimParams.mat",
        ),
        stimulus_data=_load_stimulus_data(
            files.stimulus_data_dir / "stimdata.mat",
        ),
        photodiode=photodiode,
        frame_counts=frame_counts,
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
        column_names=_load_stimulus_data_column_names(
            path.parent / "textStimData.csv",
            column_count=data.shape[1],
        ),
        time_seconds=data[:, 0],
        frame_numbers=data[:, 1].astype(np.int64),
        epoch_numbers=data[:, 2].astype(np.int64),
    )


def _load_stimulus_data_column_names(
    csv_path: Path,
    *,
    column_count: int,
) -> tuple[str, ...]:
    """Load or infer labels for ``stimData`` columns.

    Args:
        csv_path: Optional ``textStimData.csv`` path next to ``stimdata.mat``.
        column_count: Number of columns in the MATLAB ``stimData`` table.

    Returns:
        One plain-language column label per MATLAB table column.

    The CSV header is the only observed source that names column groups. It has
    blank cells for repeated group columns, so this function expands those into
    auditable names instead of pretending every protocol-specific column is
    fully understood.
    """
    if not csv_path.is_file():
        return _fallback_stimulus_column_names(column_count)

    first_line = csv_path.read_text(encoding="utf-8").splitlines()[0]
    raw_names = first_line.split(",")
    if raw_names and raw_names[-1] == "":
        raw_names = raw_names[:-1]

    names = _expand_stimulus_column_header(tuple(raw_names))
    if len(names) < column_count:
        names = names + tuple(
            f"unlabeled_column_{index + 1:02d}"
            for index in range(len(names), column_count)
        )
    return names[:column_count]


def _expand_stimulus_column_header(raw_names: tuple[str, ...]) -> tuple[str, ...]:
    """Expand sparse stimulus CSV header cells into one label per column.

    Args:
        raw_names: Header cells from ``textStimData.csv``.

    Returns:
        Column labels with group names repeated and numbered.
    """
    names: list[str] = []
    current_group = ""
    group_counts: dict[str, int] = {}
    explicit_names = {
        "Time": "time_seconds",
        "FrameNumber": "stimulus_frame_number",
        "Epoch": "epoch_number",
        "Flash": "flash",
    }
    grouped_names = {
        "ClosedLoopStimulusData": "closed_loop_stimulus_data",
        "StimulusData": "stimulus_data",
    }

    for raw_name in raw_names:
        if raw_name in explicit_names:
            current_group = ""
            names.append(explicit_names[raw_name])
            continue
        if raw_name in grouped_names:
            current_group = grouped_names[raw_name]

        if current_group == "":
            names.append(f"unlabeled_column_{len(names) + 1:02d}")
            continue

        group_counts[current_group] = group_counts.get(current_group, 0) + 1
        names.append(f"{current_group}_{group_counts[current_group]:02d}")

    return tuple(names)


def _fallback_stimulus_column_names(column_count: int) -> tuple[str, ...]:
    """Generate conservative labels when no CSV header is available.

    Args:
        column_count: Number of MATLAB ``stimData`` columns.

    Returns:
        One label per column.
    """
    base_names = ("time_seconds", "stimulus_frame_number", "epoch_number")
    if column_count <= len(base_names):
        return base_names[:column_count]
    return base_names + tuple(
        f"unknown_stimulus_column_{index + 1:02d}"
        for index in range(len(base_names), column_count)
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
