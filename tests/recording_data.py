"""Shared builders for small in-memory ``RecordingData`` test objects.

Inputs: focused arrays and metadata used by unit tests.
Outputs: complete ``RecordingData`` instances without writing converted HDF5.

The tests in this package often need a scientifically meaningful recording
container but only exercise one or two fields. This helper keeps that common
contract explicit while each test still supplies the data that matters for its
scenario.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData
from twopy.spatial import SpatialCrop, full_frame_crop

DEFAULT_RECORDING_ROOT = Path("/tmp/twopy-test-recording")


def minimal_recording_data(
    *,
    root: Path = DEFAULT_RECORDING_ROOT,
    frame_count: int = 3,
    movie_shape: tuple[int, int, int] | None = None,
    acquisition_metadata: dict[str, object] | None = None,
    run_metadata: dict[str, object] | None = None,
    synchronization_metadata: dict[str, str] | None = None,
    stimulus_data: npt.NDArray[np.float64] | None = None,
    stimulus_data_column_names: tuple[str, ...] = (),
    stimulus_parameters: tuple[dict[str, object], ...] = (),
    stimulus_function_lookup: dict[str, str] | None = None,
    stimulus_specific_columns: dict[str, dict[str, object]] | None = None,
    imaging_res_pd: npt.NDArray[np.float64] | None = None,
    high_res_pd: npt.NDArray[np.float64] | None = None,
    mean_image: npt.NDArray[np.float64] | None = None,
    alignment_valid_crop: SpatialCrop | None = None,
    alignment_offset_pixels: npt.NDArray[np.float64] | None = None,
    alignment_shift_pixels: npt.NDArray[np.float64] | None = None,
    motion_artifact_mask: npt.NDArray[np.bool_] | None = None,
    acquisition_frame_count: int | None = None,
) -> RecordingData:
    """Create a complete converted-recording object for focused unit tests.

    Args:
        root: Placeholder converted-recording folder used for object paths.
        frame_count: Number of imaging frames when arrays do not define it.
        movie_shape: Optional lazy movie shape. Defaults to ``frame_count`` by
            a two-by-two field of view.
        acquisition_metadata: Optional microscope acquisition metadata.
        run_metadata: Optional run metadata.
        synchronization_metadata: Optional synchronization metadata.
        stimulus_data: Optional converted stimulus table.
        stimulus_data_column_names: Stable names for ``stimulus_data`` columns.
        stimulus_parameters: Decoded stimulus epoch metadata.
        stimulus_function_lookup: Stimulus-type to function-name mapping.
        stimulus_specific_columns: Decoded MATLAB stimulus-specific slots.
        imaging_res_pd: Optional imaging-resolution photodiode vector.
        high_res_pd: Optional high-resolution photodiode vector.
        mean_image: Optional mean image for display or analysis helpers.
        alignment_valid_crop: Optional valid crop for aligned movie frames.
        alignment_offset_pixels: Optional per-frame x/y alignment offsets.
        alignment_shift_pixels: Optional per-frame alignment shifts.
        motion_artifact_mask: Optional per-frame motion artifact mask.
        acquisition_frame_count: Optional acquisition metadata frame count for
            audit tests.

    Returns:
        ``RecordingData`` with default placeholder paths and caller-provided
        scientific arrays.
    """
    movie_shape = (frame_count, 2, 2) if movie_shape is None else movie_shape
    frame_count = movie_shape[0]
    imaging_res_pd = (
        np.zeros(frame_count, dtype=np.float64)
        if imaging_res_pd is None
        else imaging_res_pd
    )
    high_res_pd = (
        np.zeros(frame_count, dtype=np.float64) if high_res_pd is None else high_res_pd
    )
    mean_image = (
        np.zeros(movie_shape[1:], dtype=np.float64)
        if mean_image is None
        else mean_image
    )
    alignment_shift_pixels = (
        np.zeros(frame_count, dtype=np.float64)
        if alignment_shift_pixels is None
        else alignment_shift_pixels
    )
    alignment_offset_pixels = (
        np.zeros((frame_count, 2), dtype=np.float64)
        if alignment_offset_pixels is None
        else alignment_offset_pixels
    )
    motion_artifact_mask = (
        np.zeros(frame_count, dtype=np.bool_)
        if motion_artifact_mask is None
        else motion_artifact_mask
    )
    stimulus_data = (
        np.zeros((0, 0), dtype=np.float64) if stimulus_data is None else stimulus_data
    )
    acquisition_frame_count = (
        frame_count if acquisition_frame_count is None else acquisition_frame_count
    )

    return RecordingData(
        path=root / "recording_data.h5",
        movie=ConvertedMovie(
            path=root / "aligned_movie.h5",
            dataset_name="movie/aligned",
            shape=movie_shape,
            dtype="float64",
        ),
        source_session_dir=root / "source",
        acquisition_metadata={}
        if acquisition_metadata is None
        else acquisition_metadata,
        run_metadata={} if run_metadata is None else run_metadata,
        synchronization_metadata=(
            {} if synchronization_metadata is None else synchronization_metadata
        ),
        stimulus_data=stimulus_data,
        stimulus_data_column_names=stimulus_data_column_names,
        stimulus_parameters=stimulus_parameters,
        stimulus_function_lookup=(
            {} if stimulus_function_lookup is None else stimulus_function_lookup
        ),
        stimulus_specific_columns=(
            {} if stimulus_specific_columns is None else stimulus_specific_columns
        ),
        imaging_res_pd=imaging_res_pd,
        high_res_pd=high_res_pd,
        mean_image=mean_image,
        alignment_valid_crop=(
            full_frame_crop(movie_shape[1:])
            if alignment_valid_crop is None
            else alignment_valid_crop
        ),
        alignment_offset_pixels=alignment_offset_pixels,
        alignment_shift_pixels=alignment_shift_pixels,
        motion_artifact_mask=motion_artifact_mask,
        frame_counts=FrameCountAudit(
            aligned_movie_frames=frame_count,
            imaging_res_pd_samples=len(imaging_res_pd),
            acquisition_number_of_frames=acquisition_frame_count,
            imaging_res_pd_minus_movie=len(imaging_res_pd) - frame_count,
            acquisition_minus_movie=acquisition_frame_count - frame_count,
        ),
    )
