"""Public conversion workflow from microscope source files to twopy files.

Inputs: one source recording folder plus one output folder.
Outputs: a ``ConvertedRecording`` pointing at twopy-owned HDF5 files.

The workflow reads MATLAB/TIFF-derived source data only during conversion.
Analysis code should operate on the converted HDF5 files.
"""

from dataclasses import replace
from pathlib import Path

from twopy.config import (
    DEFAULT_CONFIG_PATH,
    TwopyConfig,
    load_config,
    resolve_analysis_work_dir,
)
from twopy.conversion.hdf5_writing import (
    write_aligned_movie_file,
    write_recording_data_file,
)
from twopy.conversion.orientation import (
    orient_source_array_to_twopy,
    twopy_shape_from_source_movie_shape,
)
from twopy.conversion.source_loading import load_source_conversion_inputs
from twopy.conversion.types import (
    ConvertedRecording,
    RunMetadata,
    SourceConversionInputs,
)
from twopy.database import database_hemisphere_for_recording_path
from twopy.filenames import ALIGNED_MOVIE_FILENAME, RECORDING_DATA_FILENAME
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
            When omitted, twopy uses the configured analysis work directory.
        mean_start_frame: Optional first frame for the mean image.
        mean_stop_frame: Optional exclusive stop frame for the mean image.
        config_path: Optional YAML config file. The default uses twopy's usual
            config search when choosing output paths or reading database
            metadata.

    Returns:
        Summary of the converted HDF5 files.

    The conversion copies source data into twopy-owned datasets. The aligned
    movie gets its own file because it dominates size, while the recording data
    file keeps metadata, stimulus, photodiode, and the quick-look mean image.
    """
    inputs = load_source_conversion_inputs(session_dir)
    config = _load_config_for_conversion(
        config_path,
        require=output_dir is None,
    )
    if config is not None:
        inputs = _with_database_run_metadata(inputs, config)
    start_frame, stop_frame = normalize_frame_range(
        frame_count=inputs.aligned_movie.shape[0],
        start_frame=mean_start_frame,
        stop_frame=mean_stop_frame,
        context="frame range for mean image",
    )
    mean_image = orient_source_array_to_twopy(
        inputs.aligned_movie.mean_image(
            start_frame=start_frame,
            stop_frame=stop_frame,
        ),
    )

    destination_dir = _resolve_conversion_output_dir(
        session_dir=inputs.session_files.session_dir,
        output_dir=output_dir,
        config=config,
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    recording_data_path = destination_dir / RECORDING_DATA_FILENAME
    movie_path = destination_dir / ALIGNED_MOVIE_FILENAME

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
        movie_shape=twopy_shape_from_source_movie_shape(inputs.aligned_movie.shape),
        mean_image_start_frame=start_frame,
        mean_image_stop_frame=stop_frame,
    )


def _resolve_conversion_output_dir(
    *,
    session_dir: Path,
    output_dir: Path | None,
    config: TwopyConfig | None,
) -> Path:
    """Choose where conversion output should be written.

    Args:
        session_dir: Source recording folder.
        output_dir: Caller-provided output directory, if any.
        config: Already loaded config, when available.

    Returns:
        Expanded output directory path.

    Passing ``output_dir`` uses that folder for one call. Otherwise conversion
    uses the same work directory as the rest of twopy. With
    ``analysis_caching: true``, that work directory is local cache storage.
    """
    if output_dir is not None:
        return output_dir.expanduser()

    if config is None:
        msg = "Conversion config is required when output_dir is omitted."
        raise ValueError(msg)
    return resolve_analysis_work_dir(config, session_dir)


def _load_config_for_conversion(
    config_path: Path,
    *,
    require: bool,
) -> TwopyConfig | None:
    """Load config, unless explicit output conversion can run without it."""
    try:
        return load_config(config_path)
    except FileNotFoundError:
        if require:
            raise
        return None


def _with_database_run_metadata(
    inputs: SourceConversionInputs,
    config: TwopyConfig,
) -> SourceConversionInputs:
    """Attach database recording metadata that is not stored in runDetails.mat."""
    try:
        hemisphere = database_hemisphere_for_recording_path(
            config,
            inputs.session_files.session_dir,
        )
    except FileNotFoundError:
        return inputs
    if hemisphere is None:
        return inputs

    fields = dict(inputs.run.fields)
    fields.setdefault("hemisphere", hemisphere)
    fields.setdefault("eye", hemisphere)
    return replace(
        inputs,
        run=RunMetadata(path=inputs.run.path, fields=fields),
    )
