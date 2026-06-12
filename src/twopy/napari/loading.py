"""Recording loading helpers for the twopy napari adapter.

Inputs: user-selected recording folders, converted folders, or recording files.
Outputs: converted recording paths ready for napari.

This module may trigger conversion when a selected source recording does not
yet have twopy HDF5 output. It keeps that write behavior out of the read-only
path resolver.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py

from twopy.analysis_cache import (
    build_analysis_sync_plan_from_roots,
    copy_analysis_sync_plan,
    copy_file_atomically,
    refresh_cached_analysis_outputs,
)
from twopy.config import (
    DEFAULT_CONFIG_PATH,
    TwopyConfig,
    data_path_match,
    data_path_recording_candidates,
    load_config,
    resolve_analysis_output_dir,
    resolve_analysis_work_dir,
)
from twopy.conversion import ConvertedRecording, convert_recording_to_twopy
from twopy.filenames import ALIGNED_MOVIE_FILENAME, RECORDING_DATA_FILENAME
from twopy.napari.output_routing import (
    NapariOutputRoute,
    default_output_route,
    same_output_path,
)
from twopy.napari.paths import NapariRecordingPaths, PathInput, is_default_path
from twopy.napari.paths import resolve_recording_paths as resolve_converted_paths
from twopy.session import discover_session_files

__all__ = [
    "ResolvedNapariRecording",
    "reconvert_recording_to_output",
    "resolve_or_convert_launch_recording",
    "resolve_or_convert_recording",
]


@dataclass(frozen=True)
class ResolvedNapariRecording:
    """Resolved recording paths plus load-path status.

    Inputs: one selected path.
    Outputs: converted paths, output routing, whether conversion ran, and
    whether source data was unavailable.
    """

    paths: NapariRecordingPaths
    output_route: NapariOutputRoute
    was_converted: bool
    source_unavailable: bool = False


def resolve_or_convert_recording(path: PathInput) -> ResolvedNapariRecording:
    """Resolve a selected recording path, converting source data if needed.

    Args:
        path: User-selected source folder, converted folder, or
            ``recording_data.h5`` path.

    Returns:
        Converted recording paths and whether conversion was run.

    Raises:
        ValueError: If the path is neither converted twopy output nor a source
            recording folder.

    Existing converted files win. If ``recording_data.h5`` exists but
    ``aligned_movie.h5`` is missing and the source recording can be identified,
    conversion rewrites that same output folder to repair the pair.
    """
    selected = Path.cwd() if is_default_path(path) else Path(path).expanduser()
    source_dir = _source_recording_dir(selected)
    if source_dir is not None:
        cached = _resolve_or_convert_cached_source_recording(source_dir)
        if cached is not None:
            return cached

    cached = _resolve_cached_unavailable_source_recording(selected)
    if cached is not None:
        return cached

    try:
        paths = resolve_converted_paths(selected)
    except ValueError as error:
        if source_dir is None:
            raise _configured_data_path_error(selected, error) from error
        converted = convert_recording_to_twopy(source_dir)
        converted_paths, output_route = _converted_paths_with_output_route(converted)
        return ResolvedNapariRecording(
            paths=converted_paths,
            output_route=output_route,
            was_converted=True,
        )

    localized = _localize_converted_paths_for_cache(paths=paths, selected=selected)
    if localized is not None:
        return localized

    if paths.movie_path is not None:
        return ResolvedNapariRecording(
            paths=paths,
            output_route=_converted_selection_output_route(
                paths=paths,
                selected=selected,
            ),
            was_converted=False,
        )

    source_dir = _source_recording_dir_for_converted_selection(
        selected=selected,
        recording_data_path=paths.recording_data_path,
    )
    if source_dir is None:
        return ResolvedNapariRecording(
            paths=paths,
            output_route=_converted_selection_output_route(
                paths=paths,
                selected=selected,
            ),
            was_converted=False,
        )

    converted = convert_recording_to_twopy(
        source_dir,
        output_dir=paths.recording_data_path.parent,
    )
    converted_paths, output_route = _converted_paths_with_output_route(converted)
    return ResolvedNapariRecording(
        paths=converted_paths,
        output_route=output_route,
        was_converted=True,
    )


def reconvert_recording_to_output(
    source_dir: Path,
    output_dir: Path,
    output_route: NapariOutputRoute,
) -> ResolvedNapariRecording:
    """Rebuild converted files for one source recording in a chosen output folder.

    Args:
        source_dir: Source microscope recording folder.
        output_dir: Existing converted output folder to overwrite.
        output_route: Local and final output folders already resolved for the
            loaded recording.

    Returns:
        Resolved converted paths after the recording and movie files are
        rewritten.

    The Load-tab Reconvert selected action already has an active converted
    recording, so it should not route through the normal resolver that reuses
    existing files or rediscover output routing. This helper forces conversion
    while preserving the selected recording's loaded route.
    """
    converted = convert_recording_to_twopy(source_dir, output_dir=output_dir)
    converted_paths = resolve_converted_paths(converted.path)
    if converted_paths.movie_path is not None:
        sync_plan = build_analysis_sync_plan_from_roots(
            local_root=output_route.local_root,
            publish_root=output_route.publish_root,
            local_paths=(
                converted_paths.recording_data_path,
                converted_paths.movie_path,
            ),
        )
        if sync_plan is not None:
            copy_analysis_sync_plan(sync_plan)
    return ResolvedNapariRecording(
        paths=converted_paths,
        output_route=output_route,
        was_converted=True,
    )


def _resolve_cached_unavailable_source_recording(
    source_dir: Path,
) -> ResolvedNapariRecording | None:
    """Resolve an existing cache entry for a currently unavailable source path.

    Args:
        source_dir: Source recording folder requested by the user.

    Returns:
        Cached converted paths, or ``None`` when no usable cache entry exists.
    """
    if source_dir.exists():
        return None
    try:
        config = load_config(DEFAULT_CONFIG_PATH)
    except FileNotFoundError:
        return None
    if not config.analysis_caching:
        return None

    resolved_source_dir = source_dir.expanduser().resolve(strict=False)
    try:
        output_dir = resolve_analysis_work_dir(config, resolved_source_dir)
        paths = resolve_converted_paths(output_dir)
    except ValueError:
        return None
    if paths.movie_path is None:
        return None
    return _refresh_cached_analysis_outputs(
        paths=paths,
        source_dir=source_dir,
        was_converted=False,
        source_unavailable=True,
    )


def _resolve_or_convert_cached_source_recording(
    source_dir: Path,
) -> ResolvedNapariRecording | None:
    """Resolve or create local cached output for a source recording.

    Args:
        source_dir: Valid source microscope recording folder.

    Returns:
        Cached converted paths, or ``None`` when config is missing or caching is
        disabled.
    """
    try:
        config = load_config(DEFAULT_CONFIG_PATH)
    except FileNotFoundError:
        return None
    if not config.analysis_caching:
        return None

    try:
        output_dir = resolve_analysis_work_dir(config, source_dir)
    except ValueError:
        return None
    try:
        paths = resolve_converted_paths(output_dir)
    except ValueError:
        return _convert_cached_source_recording(
            source_dir=source_dir,
            output_dir=output_dir,
        )

    if paths.movie_path is None:
        return _convert_cached_source_recording(
            source_dir=source_dir,
            output_dir=output_dir,
        )

    return _refresh_cached_analysis_outputs(
        paths=paths,
        source_dir=source_dir,
        was_converted=False,
    )


def _convert_cached_source_recording(
    *,
    source_dir: Path,
    output_dir: Path,
) -> ResolvedNapariRecording:
    """Convert one source recording into cache and refresh saved outputs.

    Args:
        source_dir: Valid source microscope recording folder.
        output_dir: Cache directory that should receive converted files.

    Returns:
        Resolved cached paths for the converted recording.
    """
    converted = convert_recording_to_twopy(source_dir, output_dir=output_dir)
    converted_paths, _ = _converted_paths_with_output_route(converted)
    return _refresh_cached_analysis_outputs(
        paths=converted_paths,
        source_dir=source_dir,
        was_converted=True,
    )


def _converted_paths_with_output_route(
    converted: ConvertedRecording,
) -> tuple[NapariRecordingPaths, NapariOutputRoute]:
    """Resolve converted files, choose the route, and publish HDF5 outputs."""
    converted_paths = resolve_converted_paths(converted.path)
    output_route = _source_output_route(
        paths=converted_paths,
        source_dir=converted.source_session_dir,
    )
    if converted_paths.movie_path is not None:
        sync_plan = build_analysis_sync_plan_from_roots(
            local_root=output_route.local_root,
            publish_root=output_route.publish_root,
            local_paths=(
                converted_paths.recording_data_path,
                converted_paths.movie_path,
            ),
        )
        if sync_plan is not None:
            copy_analysis_sync_plan(sync_plan)
    return converted_paths, output_route


def _source_output_route(
    *,
    paths: NapariRecordingPaths,
    source_dir: Path,
) -> NapariOutputRoute:
    """Return output routing for a source recording load."""
    return default_output_route(
        local_root=paths.recording_data_path.parent,
        source_session_dir=source_dir,
        fallback_publish_root=source_dir / "twopy",
    )


def _converted_selection_output_route(
    *,
    paths: NapariRecordingPaths,
    selected: Path,
) -> NapariOutputRoute:
    """Return output routing for a manually selected converted output."""
    source_dir = _source_dir_from_recording_data(paths.recording_data_path)
    selected_root = _selected_converted_root(
        selected=selected,
        paths=paths,
    )
    if source_dir is None:
        return NapariOutputRoute(
            local_root=paths.recording_data_path.parent,
            publish_root=selected_root,
        )

    publish_root = _configured_output_for_selected_converted_folder(
        source_dir=source_dir,
        selected_root=selected_root,
    )
    return NapariOutputRoute(
        local_root=paths.recording_data_path.parent,
        publish_root=publish_root,
    )


def _refresh_cached_analysis_outputs(
    *,
    paths: NapariRecordingPaths,
    source_dir: Path,
    was_converted: bool,
    source_unavailable: bool = False,
) -> ResolvedNapariRecording:
    """Pull published saved outputs into cache and resolve fresh paths.

    Args:
        paths: Cached converted paths before saved-output refresh.
        source_dir: Source microscope recording folder.
        was_converted: Whether conversion ran during this load.
        source_unavailable: Whether this load used cache because source data
            was unavailable.

    Returns:
        Resolved cached paths after optional saved-output refresh.
    """
    refresh_cached_analysis_outputs(
        source_session_dir=source_dir,
        local_output_dir=paths.recording_data_path.parent,
    )
    refreshed_paths = resolve_converted_paths(paths.recording_data_path)
    return ResolvedNapariRecording(
        paths=refreshed_paths,
        output_route=_source_output_route(
            paths=refreshed_paths,
            source_dir=source_dir,
        ),
        was_converted=was_converted,
        source_unavailable=source_unavailable,
    )


def _localize_converted_paths_for_cache(
    *,
    paths: NapariRecordingPaths,
    selected: Path,
) -> ResolvedNapariRecording | None:
    """Copy explicitly selected converted files into the local cache when needed.

    Args:
        paths: Converted paths resolved from the user selection.
        selected: Original path selected by the user.

    Returns:
        Localized paths, or ``None`` when caching does not apply.
    """
    try:
        config = load_config(DEFAULT_CONFIG_PATH)
    except FileNotFoundError:
        return None
    if not config.analysis_caching:
        return None
    if paths.movie_path is None:
        return None

    source_dir = _source_dir_from_recording_data(paths.recording_data_path)
    if source_dir is None:
        return None
    if not _source_is_under_data_path(config, source_dir):
        return None
    selected_root = _selected_converted_root(selected=selected, paths=paths)
    try:
        publish_root = resolve_analysis_output_dir(config, source_dir)
    except ValueError:
        return None
    if not same_output_path(selected_root, publish_root):
        return None
    try:
        output_dir = resolve_analysis_work_dir(config, source_dir)
    except ValueError:
        return None
    if output_dir.resolve(strict=False) == paths.recording_data_path.parent.resolve(
        strict=False,
    ):
        return None

    local_recording_path = output_dir / RECORDING_DATA_FILENAME
    local_movie_path = output_dir / ALIGNED_MOVIE_FILENAME
    if _source_file_is_newer(paths.recording_data_path, local_recording_path):
        copy_file_atomically(paths.recording_data_path, local_recording_path)
    if _source_file_is_newer(paths.movie_path, local_movie_path):
        copy_file_atomically(paths.movie_path, local_movie_path)
    refresh_cached_analysis_outputs(
        source_session_dir=source_dir,
        local_output_dir=output_dir,
    )
    localized_paths = resolve_converted_paths(local_recording_path)
    return ResolvedNapariRecording(
        paths=localized_paths,
        output_route=default_output_route(
            local_root=localized_paths.recording_data_path.parent,
            source_session_dir=source_dir,
            fallback_publish_root=paths.recording_data_path.parent,
        ),
        was_converted=False,
    )


def _selected_converted_root(
    *,
    selected: Path,
    paths: NapariRecordingPaths,
) -> Path:
    """Return the converted folder the user selected."""
    if selected.is_file():
        return selected.expanduser().parent
    return paths.recording_data_path.parent


def _configured_output_for_selected_converted_folder(
    *,
    source_dir: Path,
    selected_root: Path,
) -> Path:
    """Return the final output folder for a manually selected converted folder.

    Manual converted folders stay where the user selected them unless that
    folder is the configured output location. This keeps arbitrary converted
    folders from being silently redirected to a different publish root.
    """
    try:
        config = load_config(DEFAULT_CONFIG_PATH)
        configured_publish = resolve_analysis_output_dir(config, source_dir)
    except (FileNotFoundError, ValueError):
        return selected_root
    if same_output_path(selected_root, configured_publish):
        return configured_publish
    return selected_root


def _source_is_under_data_path(config: TwopyConfig, source_dir: Path) -> bool:
    """Return whether a source folder belongs to the configured data root.

    Args:
        config: Loaded twopy configuration with a ``data_paths`` attribute.
        source_dir: Source recording folder.

    Returns:
        ``True`` when direct converted-file selections should be localized into
        the configured cache mirror.
    """
    return data_path_match(config, source_dir.resolve(strict=False)) is not None


def _configured_data_path_error(selected: Path, error: ValueError) -> ValueError:
    """Return a load error that lists every configured mirror candidate.

    Args:
        selected: Missing path that failed the normal converted-output lookup.
        error: Original converted-output resolver error.

    Returns:
        Original error, or a clearer error when ``selected`` sits under one of
        the configured ``data_paths`` roots.

    Database search passes one source path into the normal loader so existing
    cache-only loads still work. When that path and cache are both absent, the
    user needs to see every configured data mirror that was relevant, not just
    the first fallback path chosen for cache routing.
    """
    try:
        config = load_config(DEFAULT_CONFIG_PATH)
    except (FileNotFoundError, ValueError):
        return error

    candidates = data_path_recording_candidates(config, selected)
    if len(candidates) <= 1:
        return error

    candidate_list = "\n".join(f"- {candidate}" for candidate in candidates)
    msg = (
        "Could not find source recording or converted twopy output under "
        "configured data_paths.\n"
        f"Checked configured data_paths:\n{candidate_list}\n\n"
        f"Converted-output check:\n{error}"
    )
    return ValueError(msg)


def _source_file_is_newer(source: Path, target: Path) -> bool:
    """Return whether ``source`` should replace ``target`` in the cache.

    Args:
        source: Source file selected by the user.
        target: Local cache file.

    Returns:
        ``True`` when the cache file is absent or older than the source file.
    """
    if not target.is_file():
        return True
    return source.stat().st_mtime_ns > target.stat().st_mtime_ns


def resolve_or_convert_launch_recording(
    recording_data_path: PathInput | None,
) -> ResolvedNapariRecording | None:
    """Resolve startup recording paths for the ``twopy`` command.

    Args:
        recording_data_path: Optional command-line path.

    Returns:
        Converted recording paths, or ``None`` when no startup recording is
        available.

    An explicit command-line path may be source or converted data. With no path,
    twopy checks the current folder for converted output first, then treats the
    current folder as a source recording only if it has the expected source file
    shape.
    """
    if recording_data_path is not None:
        return resolve_or_convert_recording(recording_data_path)

    cwd = Path.cwd()
    for candidate in (cwd / RECORDING_DATA_FILENAME, cwd / "twopy", cwd):
        try:
            return resolve_or_convert_recording(candidate)
        except ValueError:
            continue
    return None


def _source_recording_dir_for_converted_selection(
    *,
    selected: Path,
    recording_data_path: Path,
) -> Path | None:
    """Infer the source recording folder for incomplete converted output.

    Args:
        selected: User-selected path.
        recording_data_path: Existing converted ``recording_data.h5`` path.

    Returns:
        Source recording folder, or ``None`` when it cannot be inferred.
    """
    candidates = [
        _source_dir_from_recording_data(recording_data_path),
        selected,
        selected.parent,
        recording_data_path.parent.parent,
    ]
    if recording_data_path.parent.name == "twopy":
        candidates.insert(0, recording_data_path.parent.parent)
    return _first_source_recording_dir(tuple(candidates))


def _source_recording_dir(path: Path) -> Path | None:
    """Return the source recording folder implied by one selected path.

    Args:
        path: User-selected path.

    Returns:
        Source recording folder, or ``None``.
    """
    candidates = [path]
    if path.is_file():
        candidates.append(path.parent)
    if path.name == "twopy":
        candidates.append(path.parent)
    if path.parent.name == "twopy":
        candidates.append(path.parent.parent)
    return _first_source_recording_dir(tuple(candidates))


def _source_dir_from_recording_data(recording_data_path: Path) -> Path | None:
    """Read the source recording folder recorded during conversion.

    Args:
        recording_data_path: Existing converted recording file.

    Returns:
        Source recording path, or ``None`` if the file cannot be read.
    """
    try:
        with h5py.File(recording_data_path, "r") as h5_file:
            value = h5_file.attrs.get("source_session_dir")
    except OSError:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or value == "":
        return None
    return Path(value).expanduser().resolve()


def _first_source_recording_dir(candidates: tuple[Path | None, ...]) -> Path | None:
    """Return the first candidate with the expected source recording files.

    Args:
        candidates: Possible source recording folders.

    Returns:
        Resolved source folder, or ``None``.
    """
    seen: set[Path] = set()
    validation_error: FileNotFoundError | ValueError | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        expanded = candidate.expanduser()
        if expanded in seen:
            continue
        seen.add(expanded)
        try:
            return discover_session_files(expanded).session_dir.resolve()
        except (FileNotFoundError, ValueError) as error:
            if validation_error is None and _looks_like_source_recording_dir(expanded):
                validation_error = error
            continue
    if validation_error is not None:
        raise validation_error
    return None


def _looks_like_source_recording_dir(path: Path) -> bool:
    """Return whether a folder appears to be microscope source output.

    Args:
        path: Candidate folder.

    Returns:
        ``True`` when the folder contains stable source-recording markers.

    Discovery errors from source-like folders should reach the user. Errors
    from arbitrary folders should stay quiet so twopy can continue checking
    other candidates such as a parent source folder.
    """
    if not path.is_dir():
        return False
    fixed_markers = (
        "stimulusData",
        "alignedMovie.mat",
        "defaultAlignChannel.txt",
        "highResPd.mat",
        "imageDescription.mat",
        "imagingResPd.mat",
    )
    if any((path / marker).exists() for marker in fixed_markers):
        return True
    return any(_source_candidate_files(path, "*.tif")) or any(
        _source_candidate_files(path, "*_alignment.txt")
    )


def _source_candidate_files(path: Path, pattern: str) -> tuple[Path, ...]:
    """Return source files matching a flexible microscope filename pattern.

    Args:
        path: Candidate source folder.
        pattern: Glob pattern.

    Returns:
        Matching files excluding macOS AppleDouble sidecars.
    """
    return tuple(
        candidate
        for candidate in path.glob(pattern)
        if candidate.is_file() and not candidate.name.startswith("._")
    )
