"""Public conversion workflow from microscope source files to twopy files.

Inputs: one source recording folder plus one output folder.
Outputs: a ``ConvertedRecording`` pointing at twopy-owned HDF5 files.

The workflow reads MATLAB/TIFF-derived source data only during conversion.
Analysis code should operate on the converted HDF5 files.
"""

from pathlib import Path

from twopy.config import DEFAULT_CONFIG_PATH, load_config, resolve_analysis_output_dir
from twopy.conversion.hdf5_writing import (
    CONVERTED_ALIGNED_MOVIE_FILENAME,
    CONVERTED_RECORDING_FILENAME,
    write_aligned_movie_file,
    write_recording_data_file,
)
from twopy.conversion.source_loading import load_source_conversion_inputs
from twopy.conversion.types import ConvertedRecording
from twopy.frame_ranges import normalize_frame_range

__all__ = ["convert_recording_to_twopy"]


def convert_recording_to_twopy(
    session_dir: Path,
    output_dir: Path | None = None,
    *,
    mean_start_frame: int | None = None,
    mean_stop_frame: int | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> ConvertedRecording:
    """Convert one recording folder into twopy HDF5 files.

    Args:
        session_dir: Source recording folder.
        output_dir: Optional directory that receives converted twopy HDF5 files.
            When omitted, twopy uses ``analysis_output`` from config.
        mean_start_frame: Optional first frame for the mean image.
        mean_stop_frame: Optional exclusive stop frame for the mean image.
        config_path: YAML config file used when ``output_dir`` is omitted.

    Returns:
        Summary of the converted HDF5 files.

    The conversion copies source data into twopy-owned datasets. The aligned
    movie gets its own file because it dominates size, while the recording data
    file keeps metadata, stimulus, photodiode, and the quick-look mean image.
    """
    inputs = load_source_conversion_inputs(session_dir)
    start_frame, stop_frame = normalize_frame_range(
        frame_count=inputs.aligned_movie.shape[0],
        start_frame=mean_start_frame,
        stop_frame=mean_stop_frame,
        context="frame range for mean image",
    )
    mean_image = inputs.aligned_movie.mean_image(
        start_frame=start_frame,
        stop_frame=stop_frame,
    )

    destination_dir = _resolve_conversion_output_dir(
        session_dir=inputs.session_files.session_dir,
        output_dir=output_dir,
        config_path=config_path,
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    recording_data_path = destination_dir / CONVERTED_RECORDING_FILENAME
    movie_path = destination_dir / CONVERTED_ALIGNED_MOVIE_FILENAME

    write_aligned_movie_file(inputs, movie_path)
    write_recording_data_file(
        inputs=inputs,
        output_path=recording_data_path,
        mean_image=mean_image,
        mean_start_frame=start_frame,
        mean_stop_frame=stop_frame,
    )

    return ConvertedRecording(
        path=recording_data_path,
        movie_path=movie_path,
        source_session_dir=inputs.session_files.session_dir,
        movie_shape=inputs.aligned_movie.shape,
        mean_image_start_frame=start_frame,
        mean_image_stop_frame=stop_frame,
    )


def _resolve_conversion_output_dir(
    *,
    session_dir: Path,
    output_dir: Path | None,
    config_path: Path,
) -> Path:
    """Choose where conversion output should be written.

    Args:
        session_dir: Source recording folder.
        output_dir: Caller-provided output directory, if any.
        config_path: Config file used when ``output_dir`` is omitted.

    Returns:
        Expanded output directory path.

    Passing ``output_dir`` is an explicit override. Otherwise conversion follows
    the same ``analysis_output`` routing that the rest of twopy uses.
    """
    if output_dir is not None:
        return output_dir.expanduser()

    config = load_config(config_path)
    return resolve_analysis_output_dir(config, session_dir)
