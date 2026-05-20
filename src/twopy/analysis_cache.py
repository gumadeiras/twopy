"""Local analysis-cache helpers for converted recordings.

Inputs: configured recording roots, local cache folders, and saved analysis files.
Outputs: cache refreshes and publish-sync plans for twopy-owned outputs.

The cache keeps interactive reads and writes on local storage. The publish sync
copies only user-visible analysis outputs back to the configured output folder;
converted movies remain derived local working files by default. Exported figure
files are removed from the local cache after they copy successfully because the
publish destination is their durable location.
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
    ANALYSIS_OUTPUT_FILENAME,
    RESPONSE_HEATMAPS_FILENAME,
    ROI_FILENAME,
)

__all__ = [
    "AnalysisSyncPlan",
    "AnalysisSyncResult",
    "build_analysis_sync_plan",
    "copy_file_atomically",
    "copy_analysis_sync_plan",
    "refresh_cached_analysis_outputs",
    "start_analysis_sync",
]

_CACHE_REFRESH_FILENAMES = (
    ROI_FILENAME,
    ANALYSIS_OUTPUT_FILENAME,
    RESPONSE_HEATMAPS_FILENAME,
)
_REMOVE_AFTER_SYNC_SUFFIXES = frozenset((".pdf", ".png"))


@dataclass(frozen=True)
class AnalysisSyncPlan:
    """Files that should be copied from local cache to publish storage.

    Inputs: local and publish roots plus local paths that were just written.
    Outputs: immutable copy plan consumed by a worker thread.
    """

    local_root: Path
    publish_root: Path
    local_paths: tuple[Path, ...]


@dataclass(frozen=True)
class AnalysisSyncResult:
    """Result of publishing saved analysis outputs.

    Inputs: one completed sync plan.
    Outputs: publish paths that were copied and local cache files removed.
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
    """Copy newer published analysis outputs into the local cache.

    Args:
        source_session_dir: Source microscope recording folder.
        local_output_dir: Local cache/work directory for that recording.
        config_path: YAML config file with cache and publish settings.

    Returns:
        Local paths refreshed from published outputs.

    Missing config or disabled caching means there is no cache refresh to do.
    Only the HDF5 files needed by the GUI reload path are pulled down; CSV and
    image exports are regenerated from saved analysis when needed.
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
    """Build a publish-sync plan for saved local analysis outputs.

    Args:
        recording: Loaded local recording whose source path defines publish
            routing.
        local_paths: Files written by the save action.
        config_path: YAML config file with cache and publish settings.

    Returns:
        A sync plan, or ``None`` when caching is disabled, config is missing,
        the publish root is the local root, or no syncable files were written.
    """
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        return None
    try:
        return _build_analysis_sync_plan_from_config(
            recording=recording,
            local_paths=local_paths,
            config=config,
        )
    except ValueError:
        return None


def start_analysis_sync(
    plan: AnalysisSyncPlan,
    *,
    executor: Executor,
) -> Future[AnalysisSyncResult]:
    """Start an analysis-output publish sync in a worker executor.

    Args:
        plan: Files to copy from local cache to publish storage.
        executor: Executor owned by the caller.

    Returns:
        Future for the completed sync result.
    """
    return executor.submit(copy_analysis_sync_plan, plan)


def copy_analysis_sync_plan(plan: AnalysisSyncPlan) -> AnalysisSyncResult:
    """Copy all files in one publish-sync plan.

    Args:
        plan: Local files and publish root.

    Returns:
        Publish paths copied by this sync.

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


def _build_analysis_sync_plan_from_config(
    *,
    recording: RecordingData,
    local_paths: tuple[Path | None, ...],
    config: TwopyConfig,
) -> AnalysisSyncPlan | None:
    """Return a sync plan from an already-loaded config."""
    if not config.analysis_caching:
        return None

    local_root = recording.path.expanduser().parent
    publish_root = resolve_analysis_output_dir(config, recording.source_session_dir)
    if _same_path(local_root, publish_root):
        return None

    syncable: list[Path] = []
    seen: set[Path] = set()
    for path in local_paths:
        if path is None:
            continue
        local_path = path.expanduser()
        try:
            local_path.relative_to(local_root)
        except ValueError:
            continue
        if local_path in seen:
            continue
        seen.add(local_path)
        syncable.append(local_path)

    if len(syncable) == 0:
        return None
    return AnalysisSyncPlan(
        local_root=local_root,
        publish_root=publish_root,
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


def _should_remove_after_sync(path: Path) -> bool:
    """Return whether a local cache file is disposable after publish sync."""
    return path.suffix.lower() in _REMOVE_AFTER_SYNC_SUFFIXES
