"""Public conversion workflow from microscope source files to twopy files.

Inputs: one source recording folder plus one output folder.
Outputs: a ``ConvertedRecording`` pointing at twopy-owned HDF5 files.

The workflow reads MATLAB/TIFF-derived source data only during conversion.
Analysis code should operate on the converted HDF5 files.
"""

import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from twopy.cache_inventory import (
    enforce_analysis_cache_limit,
    record_analysis_cache_write,
)
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
    protected_cache_roots: tuple[Path, ...] = (),
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
        protected_cache_roots: Local cache roots that must not be removed if
            conversion triggers cache cleanup.

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
    recording_data_path = destination_dir / RECORDING_DATA_FILENAME
    movie_path = destination_dir / ALIGNED_MOVIE_FILENAME

    write_dir = _conversion_write_dir(destination_dir)
    try:
        write_aligned_movie_file(inputs, write_dir / ALIGNED_MOVIE_FILENAME)
        write_recording_data_file(
            inputs=inputs,
            output_path=write_dir / RECORDING_DATA_FILENAME,
            mean_image=mean_image,
            mean_start_frame=start_frame,
            mean_stop_frame=stop_frame,
        )
        _commit_converted_files(write_dir=write_dir, destination_dir=destination_dir)
    except Exception:
        shutil.rmtree(write_dir, ignore_errors=True)
        raise
    if config is not None and _uses_analysis_cache(config, destination_dir):
        record_analysis_cache_write(config, destination_dir)
        enforce_analysis_cache_limit(
            config,
            protected_roots=(*protected_cache_roots, destination_dir),
        )

    return ConvertedRecording(
        path=recording_data_path,
        movie_path=movie_path,
        source_session_dir=inputs.session_files.session_dir,
        movie_shape=twopy_shape_from_source_movie_shape(inputs.aligned_movie.shape),
        mean_image_start_frame=start_frame,
        mean_image_stop_frame=stop_frame,
    )


def _conversion_write_dir(destination_dir: Path) -> Path:
    """Create a same-parent temporary folder for one conversion write.

    Args:
        destination_dir: Final converted-recording folder.

    Returns:
        Empty temporary folder beside ``destination_dir``.

    Conversion writes can fail after the large movie file is written. Writing into
    a temporary folder first keeps the final output from becoming a partial
    converted recording.
    """
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination_dir.name}.convert.",
            dir=destination_dir.parent,
        ),
    )


def _commit_converted_files(*, write_dir: Path, destination_dir: Path) -> None:
    """Move completed conversion files into the final output folder.

    Args:
        write_dir: Temporary folder containing completed HDF5 files.
        destination_dir: Final converted-recording folder.

    Returns:
        None.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_dir.name}.previous.",
            dir=destination_dir.parent,
        ),
    )
    touched: list[str] = []
    try:
        for filename in (ALIGNED_MOVIE_FILENAME, RECORDING_DATA_FILENAME):
            source = write_dir / filename
            target = destination_dir / filename
            backup = backup_dir / filename
            if target.exists():
                target.replace(backup)
                touched.append(filename)
            source.replace(target)
            if filename not in touched:
                touched.append(filename)
    except Exception:
        _restore_converted_files(
            destination_dir=destination_dir,
            backup_dir=backup_dir,
            filenames=tuple(touched),
        )
        raise
    finally:
        shutil.rmtree(write_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)


def _restore_converted_files(
    *,
    destination_dir: Path,
    backup_dir: Path,
    filenames: tuple[str, ...],
) -> None:
    """Restore final converted files after a failed multi-file commit."""
    for filename in reversed(filenames):
        target = destination_dir / filename
        backup = backup_dir / filename
        if backup.exists():
            target.unlink(missing_ok=True)
            backup.replace(target)
            continue
        target.unlink(missing_ok=True)


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


def _uses_analysis_cache(config: TwopyConfig, path: Path) -> bool:
    """Return whether a path sits under the configured analysis cache."""
    if not config.analysis_caching:
        return False
    resolved_path = path.expanduser().resolve(strict=False)
    cache_root = config.analysis_cache_dir.expanduser().resolve(strict=False)
    try:
        resolved_path.relative_to(cache_root)
    except ValueError:
        return False
    return True


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
