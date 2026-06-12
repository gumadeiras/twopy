"""Local analysis-cache helpers for converted recordings.

Inputs: configured recording roots, local cache folders, and saved analysis files.
Outputs: cache refreshes and copy-back plans for twopy-owned outputs.

The cache keeps interactive reads and writes on local storage. Sync copies
converted files and user-visible analysis outputs back to the configured output
folder. Exported figure files are removed from the local cache after they copy
successfully because the configured output folder is their durable location.
"""

import os
import shutil
import tempfile
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from pathlib import Path

from twopy.config import (
    DEFAULT_CONFIG_PATH,
    TwopyConfig,
    load_config,
    resolve_analysis_output_dir,
)
from twopy.converted import RecordingData
from twopy.filenames import (
    ALIGNED_MOVIE_FILENAME,
    ANALYSIS_OUTPUT_FILENAME,
    RECORDING_DATA_FILENAME,
    RESPONSE_HEATMAPS_FILENAME,
    ROI_FILENAME,
)

__all__ = [
    "AnalysisSyncPlan",
    "AnalysisSyncResult",
    "build_analysis_sync_plan",
    "build_analysis_sync_plan_from_roots",
    "copy_file_atomically",
    "copy_analysis_sync_plan",
    "copy_converted_files_to_publish",
    "refresh_cached_analysis_outputs",
    "start_analysis_sync",
]

_CACHE_REFRESH_FILENAMES = (
    RECORDING_DATA_FILENAME,
    ALIGNED_MOVIE_FILENAME,
    ROI_FILENAME,
    ANALYSIS_OUTPUT_FILENAME,
    RESPONSE_HEATMAPS_FILENAME,
)
_REMOVE_AFTER_SYNC_SUFFIXES = frozenset((".pdf", ".png"))


@dataclass(frozen=True)
class AnalysisSyncPlan:
    """Files that should be copied from local cache to the output folder.

    Inputs: local and output roots plus local paths that were just written.
    Outputs: immutable copy plan consumed by a worker thread.
    """

    local_root: Path
    publish_root: Path
    local_paths: tuple[Path, ...]


@dataclass(frozen=True)
class AnalysisSyncResult:
    """Result of copying saved analysis outputs.

    Inputs: one completed sync plan.
    Outputs: output paths that were copied and local cache files removed.
    """

    publish_root: Path
    copied_paths: tuple[Path, ...]
    removed_local_paths: tuple[Path, ...] = ()


def refresh_cached_analysis_outputs(
    *,
    source_session_dir: Path,
    local_output_dir: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[Path, ...]:
    """Copy newer output files into the local cache.

    Args:
        source_session_dir: Source microscope recording folder.
        local_output_dir: Local cache/work directory for that recording.
        config_path: Optional YAML config file with cache and output settings.
            The default uses twopy's usual config search.

    Returns:
        Local paths refreshed from the configured output folder.

    When called outside the napari launcher, missing config or disabled caching
    means there is no cache refresh to do. HDF5 files needed by the GUI are
    pulled down; CSV and image exports are regenerated from saved analysis when
    needed.
    """
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        return ()
    if not config.analysis_caching:
        return ()

    try:
        publish_dir = resolve_analysis_output_dir(config, source_session_dir)
    except ValueError:
        return ()
    local_dir = local_output_dir.expanduser()
    if _same_path(publish_dir, local_dir):
        return ()

    refreshed: list[Path] = []
    for filename in _CACHE_REFRESH_FILENAMES:
        source = publish_dir / filename
        target = local_dir / filename
        if not source.is_file():
            continue
        if target.is_file() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            continue
        copy_file_atomically(source, target)
        refreshed.append(target)
    return tuple(refreshed)


def build_analysis_sync_plan(
    *,
    recording: RecordingData,
    local_paths: tuple[Path | None, ...],
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> AnalysisSyncPlan | None:
    """Plan which saved local analysis files should be copied back.

    Args:
        recording: Loaded local recording whose source path defines the output
            folder.
        local_paths: Files written by the save action.
        config_path: Optional YAML config file with cache and output settings.
            The default uses twopy's usual config search.

    Returns:
        A sync plan, or ``None`` when config is unavailable to this direct
        helper call, the output root is the local root, or no syncable files were
        written.
    """
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        return None
    try:
        plan = _build_sync_plan_from_config(
            source_session_dir=recording.source_session_dir,
            local_root=recording.path.expanduser().parent,
            local_paths=(*_converted_recording_paths(recording), *local_paths),
            config=config,
        )
    except ValueError:
        return None
    if len(plan.local_paths) == 0:
        return None
    return plan


def build_analysis_sync_plan_from_roots(
    *,
    local_root: Path,
    publish_root: Path,
    local_paths: tuple[Path | None, ...],
) -> AnalysisSyncPlan | None:
    """Plan publish syncing from already-resolved local and publish roots.

    Args:
        local_root: Folder where napari wrote the files.
        publish_root: Folder where those files should be visible.
        local_paths: Files written or refreshed by the save action.

    Returns:
        A sync plan, or ``None`` when the publish root is already local or no
        supplied paths sit under the local root.

    The caller has already resolved the publish destination. This helper keeps
    the sync decision independent from config inference without mixing GUI
    status text into the cache layer.
    """
    plan = _build_sync_plan_from_roots(
        local_root=local_root,
        publish_root=publish_root,
        local_paths=local_paths,
    )
    if len(plan.local_paths) == 0:
        return None
    return plan


def copy_converted_files_to_publish(
    *,
    recording_data_path: Path,
    movie_path: Path,
    source_session_dir: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> AnalysisSyncResult | None:
    """Copy converted recording files to the configured output folder.

    Args:
        recording_data_path: Local ``recording_data.h5`` file.
        movie_path: Local ``aligned_movie.h5`` file.
        source_session_dir: Source microscope recording folder that defines the
            configured output location.
        config_path: Optional YAML config file with cache and output settings.
            The default uses twopy's usual config search.

    Returns:
        Sync result, or ``None`` when config is unavailable to this direct
        helper call, the output root is the local root, or no copy-back is
        needed.
    """
    try:
        config = load_config(config_path)
        plan = _build_sync_plan_from_config(
            source_session_dir=source_session_dir,
            local_root=recording_data_path.expanduser().parent,
            local_paths=(recording_data_path, movie_path),
            config=config,
        )
    except (FileNotFoundError, ValueError):
        return None
    if len(plan.local_paths) == 0:
        return None
    return copy_analysis_sync_plan(plan)


def start_analysis_sync(
    plan: AnalysisSyncPlan,
    *,
    executor: Executor,
) -> Future[AnalysisSyncResult]:
    """Start copying analysis outputs in a worker executor.

    Args:
        plan: Files to copy from local cache to the output folder.
        executor: Executor owned by the caller.

    Returns:
        Future for the completed sync result.
    """
    return executor.submit(copy_analysis_sync_plan, plan)


def copy_analysis_sync_plan(plan: AnalysisSyncPlan) -> AnalysisSyncResult:
    """Copy all files in one sync plan.

    Args:
        plan: Local files and output root.

    Returns:
        Output paths copied by this sync.

    Copies go through a temporary file in the target directory before replace,
    so readers never see a partially written HDF5, CSV, or figure file. Figure
    files are cache artifacts and are deleted locally after the destination copy
    succeeds.
    """
    copied: list[Path] = []
    removed: list[Path] = []
    for local_path in plan.local_paths:
        if not local_path.is_file():
            continue
        relative = local_path.relative_to(plan.local_root)
        publish_path = plan.publish_root / relative
        if _same_path(local_path, publish_path):
            continue
        if _copy_is_current(local_path, publish_path):
            continue
        copy_file_atomically(local_path, publish_path)
        copied.append(publish_path)
        if _should_remove_after_sync(local_path):
            local_path.unlink()
            removed.append(local_path)
    return AnalysisSyncResult(
        publish_root=plan.publish_root,
        copied_paths=tuple(copied),
        removed_local_paths=tuple(removed),
    )


def _build_sync_plan_from_config(
    *,
    source_session_dir: Path,
    local_root: Path,
    local_paths: tuple[Path | None, ...],
    config: TwopyConfig,
) -> AnalysisSyncPlan:
    """Return a sync plan from an already-loaded config."""
    publish_root = resolve_analysis_output_dir(config, source_session_dir)
    return _build_sync_plan_from_roots(
        local_root=local_root,
        publish_root=publish_root,
        local_paths=local_paths,
    )


def _build_sync_plan_from_roots(
    *,
    local_root: Path,
    publish_root: Path,
    local_paths: tuple[Path | None, ...],
) -> AnalysisSyncPlan:
    """Return a sync plan from explicit local and publish folders."""
    local = local_root.expanduser().resolve(strict=False)
    publish = publish_root.expanduser().resolve(strict=False)
    if _same_path(local, publish):
        return AnalysisSyncPlan(
            local_root=local,
            publish_root=publish,
            local_paths=(),
        )

    syncable: list[Path] = []
    seen: set[Path] = set()
    for path in local_paths:
        if path is None:
            continue
        local_path = path.expanduser().resolve(strict=False)
        try:
            local_path.relative_to(local)
        except ValueError:
            continue
        if local_path in seen:
            continue
        seen.add(local_path)
        syncable.append(local_path)

    return AnalysisSyncPlan(
        local_root=local,
        publish_root=publish,
        local_paths=tuple(syncable),
    )


def copy_file_atomically(source: Path, target: Path) -> None:
    """Copy one file through a temporary target path before replacing.

    Args:
        source: Existing file to copy.
        target: Destination path.

    Returns:
        None.

    The temporary file lives beside the target so the final replace is atomic on
    one filesystem.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temp_path = Path(temp_name)
    try:
        with (
            os.fdopen(temp_descriptor, "wb") as temp_file,
            source.open(
                "rb",
            ) as source_file,
        ):
            shutil.copyfileobj(source_file, temp_file)
        shutil.copystat(source, temp_path)
        temp_path.replace(target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _same_path(left: Path, right: Path) -> bool:
    """Return whether two paths resolve to the same filesystem location."""
    return left.expanduser().resolve(strict=False) == right.expanduser().resolve(
        strict=False,
    )


def _copy_is_current(source: Path, target: Path) -> bool:
    """Return whether the target file already matches this source version."""
    if not target.is_file():
        return False
    source_stat = source.stat()
    target_stat = target.stat()
    return (
        target_stat.st_size == source_stat.st_size
        and target_stat.st_mtime_ns >= source_stat.st_mtime_ns
    )


def _converted_recording_paths(recording: RecordingData) -> tuple[Path, Path]:
    """Return the converted HDF5 files copied with every sync."""
    return (recording.path, recording.movie.path)


def _should_remove_after_sync(path: Path) -> bool:
    """Return whether a local cache file can be removed after sync."""
    return path.suffix.lower() in _REMOVE_AFTER_SYNC_SUFFIXES
