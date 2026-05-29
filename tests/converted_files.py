"""Shared writers for tiny twopy-converted HDF5 test files.

Inputs: temporary folders plus small movie, timing, and metadata arrays.
Outputs: inspectable ``recording_data.h5`` and ``aligned_movie.h5`` files.

Several tests need real converted files to exercise HDF5 loading and path
resolution. This helper keeps the common file layout in one place while tests
still choose the data values that matter for each scenario.
"""

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from twopy.conversion.types import FrameCountAudit
from twopy.spatial import SpatialCrop
from twopy.spatial_orientation import write_twopy_spatial_orientation


def write_aligned_movie_file(
    path: Path,
    movie_values: npt.NDArray[np.float64],
    *,
    compression: str | None = None,
) -> None:
    """Write one minimal converted aligned-movie HDF5 file.

    Args:
        path: Destination ``aligned_movie.h5`` path.
        movie_values: Movie data shaped ``(frames, axis0, axis1)``.
        compression: Optional HDF5 compression filter.

    Returns:
        None.
    """
    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = "aligned-movie"
        dataset = h5_file.create_dataset(
            "movie/aligned",
            data=movie_values,
            compression=compression,
        )
        write_twopy_spatial_orientation(dataset)


def write_converted_recording_files(
    root: Path,
    *,
    movie_values: npt.NDArray[np.float64] | None = None,
    alignment_valid_crop: SpatialCrop | None = None,
    source_session_dir: Path | None = None,
    acquisition_metadata: dict[str, object] | None = None,
    run_metadata: dict[str, object] | None = None,
    stimulus_data: npt.NDArray[np.float64] | None = None,
    stimulus_data_column_names: tuple[str, ...] = ("time_seconds", "epoch_number"),
    stimulus_parameters_json: str = "[]",
    function_lookup_json: str = "{}",
    stimulus_specific_columns_json: str = "{}",
    imaging_res_pd: npt.NDArray[np.float64] | None = None,
    high_res_pd: npt.NDArray[np.float64] | None = None,
    alignment_shift_pixels: npt.NDArray[np.float64] | None = None,
    motion_artifact_mask: npt.NDArray[np.bool_] | None = None,
    frame_counts: FrameCountAudit | None = None,
) -> Path:
    """Write one minimal converted recording and its aligned movie.

    Args:
        root: Folder that receives both HDF5 files.
        movie_values: Optional movie shaped ``(frames, axis0, axis1)``.
        alignment_valid_crop: Optional crop metadata for displayed frames.
        source_session_dir: Optional source recording path stored on the file.
        acquisition_metadata: Optional microscope metadata attributes.
        run_metadata: Optional run metadata attributes.
        stimulus_data: Optional converted stimulus table.
        stimulus_data_column_names: Column names stored with ``stimulus_data``.
        stimulus_parameters_json: JSON list of stimulus epoch dictionaries.
        function_lookup_json: JSON mapping from stimulus types to functions.
        stimulus_specific_columns_json: JSON metadata for stimulus-specific slots.
        imaging_res_pd: Optional imaging-resolution photodiode vector.
        high_res_pd: Optional high-resolution photodiode vector.
        alignment_shift_pixels: Optional per-frame alignment shifts.
        motion_artifact_mask: Optional per-frame motion mask.
        frame_counts: Optional frame-count audit metadata override.

    Returns:
        Path to the written ``recording_data.h5`` file.
    """
    root.mkdir(parents=True, exist_ok=True)
    resolved_movie = (
        np.arange(12, dtype=np.float64).reshape(3, 2, 2)
        if movie_values is None
        else movie_values
    )
    frame_count = resolved_movie.shape[0]
    resolved_imaging_res_pd = (
        np.zeros(frame_count, dtype=np.float64)
        if imaging_res_pd is None
        else imaging_res_pd
    )
    resolved_high_res_pd = (
        np.zeros(frame_count, dtype=np.float64) if high_res_pd is None else high_res_pd
    )
    resolved_frame_counts = frame_counts or FrameCountAudit(
        aligned_movie_frames=frame_count,
        imaging_res_pd_samples=len(resolved_imaging_res_pd),
        acquisition_number_of_frames=frame_count,
        imaging_res_pd_minus_movie=len(resolved_imaging_res_pd) - frame_count,
        acquisition_minus_movie=0,
    )
    resolved_crop = (
        SpatialCrop(
            axis0_start=0,
            axis0_stop=resolved_movie.shape[1],
            axis1_start=0,
            axis1_stop=resolved_movie.shape[2],
            original_shape=resolved_movie.shape[1:],
            source="full_frame",
        )
        if alignment_valid_crop is None
        else alignment_valid_crop
    )

    movie_path = root / "aligned_movie.h5"
    write_aligned_movie_file(movie_path, resolved_movie)

    recording_path = root / "recording_data.h5"
    with h5py.File(recording_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = "converted-recording"
        h5_file.attrs["source_session_dir"] = str(source_session_dir or root / "source")

        movie_group = h5_file.create_group("movie")
        movie_group.attrs["aligned_movie_file"] = "aligned_movie.h5"
        movie_group.attrs["aligned_movie_dataset"] = "movie/aligned"
        movie_group.attrs["aligned_movie_shape"] = resolved_movie.shape
        movie_group.attrs["aligned_movie_dtype"] = "float64"
        write_twopy_spatial_orientation(movie_group)
        mean_image = movie_group.create_dataset(
            "mean_image",
            data=resolved_movie.mean(axis=0),
        )
        write_twopy_spatial_orientation(mean_image)

        crop_group = movie_group.create_group("alignment_valid_crop")
        crop_group.attrs["source"] = resolved_crop.source
        crop_group.attrs["axis0_start"] = resolved_crop.axis0_start
        crop_group.attrs["axis0_stop"] = resolved_crop.axis0_stop
        crop_group.attrs["axis1_start"] = resolved_crop.axis1_start
        crop_group.attrs["axis1_stop"] = resolved_crop.axis1_stop
        crop_group.attrs["original_shape"] = resolved_crop.original_shape
        crop_group.create_dataset(
            "alignment_shift_pixels",
            data=(
                np.zeros(frame_count, dtype=np.float64)
                if alignment_shift_pixels is None
                else alignment_shift_pixels
            ),
        )
        crop_group.create_dataset(
            "motion_artifact_mask",
            data=(
                np.zeros(frame_count, dtype=np.bool_)
                if motion_artifact_mask is None
                else motion_artifact_mask
            ),
        )

        metadata_group = h5_file.create_group("metadata")
        for key, value in (acquisition_metadata or {"acq.frameRate": 10.0}).items():
            metadata_group.attrs[key] = value

        run_group = h5_file.create_group("run")
        for key, value in (run_metadata or {}).items():
            run_group.attrs[key] = value

        stimulus_group = h5_file.create_group("stimulus")
        stimulus_group.create_dataset(
            "data",
            data=(
                np.zeros((1, len(stimulus_data_column_names)), dtype=np.float64)
                if stimulus_data is None
                else stimulus_data
            ),
        )
        stimulus_group.create_dataset(
            "data_column_names",
            data=np.asarray(
                stimulus_data_column_names,
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        stimulus_group.create_dataset("parameters_json", data=stimulus_parameters_json)
        stimulus_group.create_dataset("function_lookup_json", data=function_lookup_json)
        stimulus_group.create_dataset(
            "stimulus_specific_columns_json",
            data=stimulus_specific_columns_json,
        )

        photodiode_group = h5_file.create_group("photodiode")
        photodiode_group.create_dataset(
            "imaging_res_pd",
            data=resolved_imaging_res_pd,
        )
        photodiode_group.create_dataset(
            "high_res_pd",
            data=resolved_high_res_pd,
        )

        frame_counts = h5_file.create_group("frame_counts")
        frame_counts.attrs["aligned_movie_frames"] = (
            resolved_frame_counts.aligned_movie_frames
        )
        frame_counts.attrs["imaging_res_pd_samples"] = (
            resolved_frame_counts.imaging_res_pd_samples
        )
        frame_counts.attrs["acquisition_number_of_frames"] = (
            resolved_frame_counts.acquisition_number_of_frames
        )
        frame_counts.attrs["imaging_res_pd_minus_movie"] = (
            resolved_frame_counts.imaging_res_pd_minus_movie
        )
        frame_counts.attrs["acquisition_minus_movie"] = (
            resolved_frame_counts.acquisition_minus_movie
        )

    return recording_path
