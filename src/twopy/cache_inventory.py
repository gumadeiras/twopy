"""Bound the local analysis cache with a small rebuildable inventory.

Inputs: the configured analysis cache folder plus recording-level cache writes.
Outputs: inventory updates and whole-recording cache cleanup results.

The inventory stores cache policy facts such as size, last use, and publish
state. The files on disk remain the source of truth: cleanup validates each
candidate against its publish folder before deleting a cache entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import h5py

from twopy.config import TwopyConfig, resolve_analysis_output_dir
from twopy.filenames import RECORDING_DATA_FILENAME
from twopy.typing_guards import string_key_mapping_or_none

__all__ = [
    "CACHE_INVENTORY_FILENAME",
    "CacheCleanupResult",
    "enforce_analysis_cache_limit",
    "record_analysis_cache_refresh",
    "record_analysis_cache_sync",
    "record_analysis_cache_use",
    "record_analysis_cache_write",
]

CACHE_INVENTORY_FILENAME = "twopy_cache_inventory.json"
_INVENTORY_VERSION = 1
_PUBLISHED_STATE = "published"
_LOCAL_ONLY_STATE = "local_only"
_UNKNOWN_STATE = "unknown"
CacheEntryState = Literal["published", "local_only", "unknown"]


@dataclass(frozen=True)
class CacheCleanupResult:
    """Result of enforcing the local analysis-cache size limit.

    Args:
        cache_root: Analysis cache folder that was inspected.
        max_bytes: Configured cache size limit.
        bytes_before: Known cache-entry bytes before cleanup.
        bytes_after: Known cache-entry bytes after cleanup.
        removed_roots: Cache entry folders removed by cleanup.
        skipped_roots: Candidate folders skipped because they were unsafe.

    Returns:
        Immutable cleanup summary for GUI status and tests.
    """

    cache_root: Path
    max_bytes: int
    bytes_before: int
    bytes_after: int
    removed_roots: tuple[Path, ...] = ()
    skipped_roots: tuple[Path, ...] = ()


@dataclass(frozen=True)
class _CacheEntry:
    """One recording-level analysis-cache entry."""

    root: Path
    source_session_dir: Path | None
    publish_root: Path | None
    size_bytes: int
    last_used_ns: int
    last_local_change_ns: int
    last_publish_sync_ns: int
    state: CacheEntryState


@dataclass(frozen=True)
class _CacheInventory:
    """Parsed cache inventory data."""

    cache_root: Path
    entries: tuple[_CacheEntry, ...]


def record_analysis_cache_use(config: TwopyConfig, recording_root: Path) -> None:
    """Record that one cache entry was used.

    Args:
        config: Loaded twopy configuration.
        recording_root: Folder containing the cached recording files.

    Returns:
        None.

    Last-use time is explicit policy state so cleanup never depends on platform
    filesystem access-time behavior.
    """
    _update_entry(config, recording_root, action="use")


def record_analysis_cache_write(config: TwopyConfig, recording_root: Path) -> None:
    """Record that one cache entry received local writes.

    Args:
        config: Loaded twopy configuration.
        recording_root: Folder containing the cached recording files.

    Returns:
        None.

    A local write makes the entry ineligible for eviction until a successful
    publish sync marks it current again.
    """
    _update_entry(config, recording_root, action="write")


def record_analysis_cache_refresh(
    config: TwopyConfig,
    *,
    local_root: Path,
    publish_root: Path,
) -> None:
    """Record that published files were copied into one cache entry.

    Args:
        config: Loaded twopy configuration.
        local_root: Local cache entry folder that received copied files.
        publish_root: Durable folder those files came from.

    Returns:
        None.

    Refreshing from published storage changes local size and recency, but it
    does not prove unrelated local files are published. The existing safety
    state is preserved.
    """
    _update_entry(
        config,
        local_root,
        action="refresh",
        publish_root=publish_root,
    )


def record_analysis_cache_sync(
    config: TwopyConfig,
    *,
    local_root: Path,
    publish_root: Path,
) -> None:
    """Record that one cache entry synced to its publish folder.

    Args:
        config: Loaded twopy configuration.
        local_root: Local cache entry folder.
        publish_root: Durable folder that received the published files.

    Returns:
        None.
    """
    _update_entry(
        config,
        local_root,
        action="sync",
        publish_root=publish_root,
    )


def enforce_analysis_cache_limit(
    config: TwopyConfig,
    *,
    protected_roots: tuple[Path, ...] = (),
) -> CacheCleanupResult:
    """Remove old safe cache entries until the configured size limit is met.

    Args:
        config: Loaded twopy configuration.
        protected_roots: Cache entries, or paths inside cache entries, that must
            not be removed during this cleanup.

    Returns:
        Cleanup result with removed and skipped roots.

    Cleanup is whole-recording only. It never removes individual HDF5 files from
    an entry because partial converted output is harder to inspect and recover.
    """
    cache_root = config.analysis_cache_dir.expanduser().resolve(strict=False)
    max_bytes = config.analysis_cache_max_bytes
    if not config.analysis_caching:
        return CacheCleanupResult(
            cache_root=cache_root,
            max_bytes=max_bytes,
            bytes_before=0,
            bytes_after=0,
        )

    inventory = _load_or_rebuild_inventory(config)
    entries = tuple(entry for entry in inventory.entries if entry.root.is_dir())
    total_bytes = sum(entry.size_bytes for entry in entries)
    if total_bytes <= max_bytes:
        if len(entries) != len(inventory.entries):
            _write_inventory(_CacheInventory(cache_root=cache_root, entries=entries))
        return CacheCleanupResult(
            cache_root=cache_root,
            max_bytes=max_bytes,
            bytes_before=total_bytes,
            bytes_after=total_bytes,
        )

    protected = tuple(
        path.expanduser().resolve(strict=False) for path in protected_roots
    )
    removed: list[Path] = []
    skipped: list[Path] = []
    remaining_entries = list(entries)
    for entry in sorted(entries, key=_eviction_sort_key):
        if total_bytes <= max_bytes:
            break
        if _entry_is_protected(entry, protected):
            skipped.append(entry.root)
            continue
        if not _entry_can_be_removed(entry, cache_root):
            skipped.append(entry.root)
            continue
        current_size = _directory_size(entry.root)
        try:
            shutil.rmtree(entry.root)
        except OSError:
            skipped.append(entry.root)
            continue
        _remove_empty_parents(entry.root.parent, cache_root)
        removed.append(entry.root)
        total_bytes -= current_size
        remaining_entries = [
            candidate for candidate in remaining_entries if candidate.root != entry.root
        ]

    refreshed_entries = tuple(
        replace(entry, size_bytes=_directory_size(entry.root))
        for entry in remaining_entries
        if entry.root.is_dir()
    )
    bytes_after = sum(entry.size_bytes for entry in refreshed_entries)
    _write_inventory(
        _CacheInventory(cache_root=cache_root, entries=refreshed_entries),
    )
    return CacheCleanupResult(
        cache_root=cache_root,
        max_bytes=max_bytes,
        bytes_before=sum(entry.size_bytes for entry in entries),
        bytes_after=bytes_after,
        removed_roots=tuple(removed),
        skipped_roots=tuple(skipped),
    )


def _update_entry(
    config: TwopyConfig,
    recording_root: Path,
    *,
    action: Literal["use", "write", "sync", "refresh"],
    publish_root: Path | None = None,
) -> None:
    """Update one inventory entry if the root is inside the analysis cache."""
    if not config.analysis_caching:
        return
    cache_root = config.analysis_cache_dir.expanduser().resolve(strict=False)
    root = recording_root.expanduser().resolve(strict=False)
    if not _is_relative_to(root, cache_root):
        return
    if not root.is_dir():
        return

    inventory = _load_or_rebuild_inventory(config)
    entries = {entry.root: entry for entry in inventory.entries if entry.root.is_dir()}
    existing = entries.get(root)
    now = time.time_ns()
    source_session_dir = (
        existing.source_session_dir
        if existing is not None
        else _source_session_dir_from_recording(root / RECORDING_DATA_FILENAME)
    )
    resolved_publish_root = publish_root
    if resolved_publish_root is None and existing is not None:
        resolved_publish_root = existing.publish_root
    if resolved_publish_root is None and source_session_dir is not None:
        resolved_publish_root = _publish_root_for_source(config, source_session_dir)

    size_bytes = (
        existing.size_bytes
        if action == "use" and existing is not None
        else _directory_size(root)
    )
    last_used_ns = existing.last_used_ns if existing is not None else now
    last_local_change_ns = existing.last_local_change_ns if existing is not None else 0
    last_publish_sync_ns = existing.last_publish_sync_ns if existing is not None else 0
    state: CacheEntryState = existing.state if existing is not None else _UNKNOWN_STATE

    if action == "use":
        last_used_ns = now
    elif action == "write":
        last_used_ns = now
        last_local_change_ns = now
        state = _LOCAL_ONLY_STATE
    elif action == "sync":
        last_used_ns = now
        last_publish_sync_ns = now
        state = _PUBLISHED_STATE
    else:
        last_used_ns = now

    entries[root] = _CacheEntry(
        root=root,
        source_session_dir=source_session_dir,
        publish_root=(
            resolved_publish_root.expanduser().resolve(strict=False)
            if resolved_publish_root is not None
            else None
        ),
        size_bytes=size_bytes,
        last_used_ns=last_used_ns,
        last_local_change_ns=last_local_change_ns,
        last_publish_sync_ns=last_publish_sync_ns,
        state=state,
    )
    _write_inventory(
        _CacheInventory(
            cache_root=cache_root,
            entries=tuple(sorted(entries.values(), key=lambda entry: str(entry.root))),
        ),
    )


def _load_or_rebuild_inventory(config: TwopyConfig) -> _CacheInventory:
    """Load inventory JSON, rebuilding it when absent or invalid."""
    cache_root = config.analysis_cache_dir.expanduser().resolve(strict=False)
    loaded = _read_inventory(cache_root)
    if loaded is not None:
        return loaded
    inventory = _rebuild_inventory(config)
    _write_inventory(inventory)
    return inventory


def _read_inventory(cache_root: Path) -> _CacheInventory | None:
    """Read inventory JSON, returning ``None`` when it cannot be trusted."""
    path = _inventory_path(cache_root)
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    data = string_key_mapping_or_none(loaded)
    if data is None or data.get("version") != _INVENTORY_VERSION:
        return None
    cache_root_text = data.get("cache_root")
    if not isinstance(cache_root_text, str):
        return None
    loaded_root = Path(cache_root_text).expanduser().resolve(strict=False)
    if loaded_root != cache_root:
        return None
    raw_entries = data.get("entries")
    raw_entry_map = string_key_mapping_or_none(raw_entries)
    if raw_entry_map is None:
        return None

    entries: list[_CacheEntry] = []
    for value in raw_entry_map.values():
        entry_data = string_key_mapping_or_none(value)
        if entry_data is None:
            return None
        entry = _entry_from_storage(entry_data)
        if entry is None or not _is_relative_to(entry.root, cache_root):
            return None
        entries.append(entry)
    return _CacheInventory(cache_root=cache_root, entries=tuple(entries))


def _rebuild_inventory(config: TwopyConfig) -> _CacheInventory:
    """Discover cache entries from disk and create unknown inventory records."""
    cache_root = config.analysis_cache_dir.expanduser().resolve(strict=False)
    if not cache_root.is_dir():
        return _CacheInventory(cache_root=cache_root, entries=())
    now = time.time_ns()
    entries: list[_CacheEntry] = []
    for recording_data_path in cache_root.rglob(RECORDING_DATA_FILENAME):
        root = recording_data_path.parent.resolve(strict=False)
        if not root.is_dir() or root.is_symlink():
            continue
        source_session_dir = _source_session_dir_from_recording(recording_data_path)
        publish_root = (
            _publish_root_for_source(config, source_session_dir)
            if source_session_dir is not None
            else None
        )
        entries.append(
            _CacheEntry(
                root=root,
                source_session_dir=source_session_dir,
                publish_root=publish_root,
                size_bytes=_directory_size(root),
                last_used_ns=_latest_mtime_ns(root),
                last_local_change_ns=0,
                last_publish_sync_ns=0,
                state=_UNKNOWN_STATE,
            ),
        )
    if len(entries) == 0:
        return _CacheInventory(cache_root=cache_root, entries=())
    return _CacheInventory(
        cache_root=cache_root,
        entries=tuple(
            sorted(
                (
                    entry
                    if entry.last_used_ns > 0
                    else replace(entry, last_used_ns=now)
                    for entry in entries
                ),
                key=lambda entry: str(entry.root),
            ),
        ),
    )


def _entry_from_storage(data: dict[str, object]) -> _CacheEntry | None:
    """Parse one stored entry."""
    root = _optional_path_from_storage(data.get("root"))
    if root is None:
        return None
    raw_state = data.get("state")
    if raw_state == _PUBLISHED_STATE:
        state: CacheEntryState = _PUBLISHED_STATE
    elif raw_state == _LOCAL_ONLY_STATE:
        state = _LOCAL_ONLY_STATE
    elif raw_state == _UNKNOWN_STATE:
        state = _UNKNOWN_STATE
    else:
        return None
    size_bytes = _non_negative_int(data.get("size_bytes"))
    last_used_ns = _non_negative_int(data.get("last_used_ns"))
    last_local_change_ns = _non_negative_int(data.get("last_local_change_ns"))
    last_publish_sync_ns = _non_negative_int(data.get("last_publish_sync_ns"))
    if (
        size_bytes is None
        or last_used_ns is None
        or last_local_change_ns is None
        or last_publish_sync_ns is None
    ):
        return None
    return _CacheEntry(
        root=root,
        source_session_dir=_optional_path_from_storage(data.get("source_session_dir")),
        publish_root=_optional_path_from_storage(data.get("publish_root")),
        size_bytes=size_bytes,
        last_used_ns=last_used_ns,
        last_local_change_ns=last_local_change_ns,
        last_publish_sync_ns=last_publish_sync_ns,
        state=state,
    )


def _entry_to_storage(entry: _CacheEntry) -> dict[str, object]:
    """Return stable JSON data for one inventory entry."""
    return {
        "root": str(entry.root),
        "source_session_dir": (
            str(entry.source_session_dir)
            if entry.source_session_dir is not None
            else None
        ),
        "publish_root": (
            str(entry.publish_root) if entry.publish_root is not None else None
        ),
        "size_bytes": entry.size_bytes,
        "last_used_ns": entry.last_used_ns,
        "last_local_change_ns": entry.last_local_change_ns,
        "last_publish_sync_ns": entry.last_publish_sync_ns,
        "state": entry.state,
    }


def _write_inventory(inventory: _CacheInventory) -> None:
    """Write inventory JSON through a same-folder temporary file."""
    inventory.cache_root.mkdir(parents=True, exist_ok=True)
    data = {
        "version": _INVENTORY_VERSION,
        "cache_root": str(inventory.cache_root),
        "entries": {
            _entry_key(entry.root): _entry_to_storage(entry)
            for entry in sorted(inventory.entries, key=lambda item: str(item.root))
        },
    }
    _write_json_atomically(_inventory_path(inventory.cache_root), data)


def _entry_can_be_removed(entry: _CacheEntry, cache_root: Path) -> bool:
    """Return whether one cache entry is safe to delete."""
    if entry.root.is_symlink() or not _is_relative_to(entry.root, cache_root):
        return False
    if entry.state != _PUBLISHED_STATE:
        return False
    if entry.source_session_dir is not None and not entry.source_session_dir.exists():
        return False
    if entry.publish_root is None:
        return False
    if _same_path(entry.root, entry.publish_root):
        return False
    return _published_tree_is_current(entry.root, entry.publish_root)


def _published_tree_is_current(local_root: Path, publish_root: Path) -> bool:
    """Return whether every local file has a current published copy."""
    for local_file in _iter_files(local_root):
        relative = local_file.relative_to(local_root)
        publish_file = publish_root / relative
        if not _file_copy_is_current(local_file, publish_file):
            return False
    return True


def _file_copy_is_current(source: Path, target: Path) -> bool:
    """Return whether target already contains the source file version."""
    try:
        if not target.is_file():
            return False
        source_stat = source.stat()
        target_stat = target.stat()
    except OSError:
        return False
    return (
        target_stat.st_size == source_stat.st_size
        and target_stat.st_mtime_ns >= source_stat.st_mtime_ns
    )


def _iter_files(root: Path) -> Iterator[Path]:
    """Yield regular files below a cache entry, ignoring symlinks."""
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            yield path


def _directory_size(root: Path) -> int:
    """Return byte size for regular files under one directory."""
    if not root.is_dir():
        return 0
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                            continue
                        if entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _latest_mtime_ns(root: Path) -> int:
    """Return latest modification time for files in one entry."""
    latest = 0
    for file_path in _iter_files(root):
        try:
            latest = max(latest, file_path.stat().st_mtime_ns)
        except OSError:
            continue
    return latest


def _source_session_dir_from_recording(recording_data_path: Path) -> Path | None:
    """Read source session path from a converted recording file."""
    if not recording_data_path.is_file():
        return None
    try:
        with h5py.File(recording_data_path, "r") as h5_file:
            value = h5_file.attrs.get("source_session_dir")
    except OSError:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or value == "":
        return None
    return Path(value).expanduser().resolve(strict=False)


def _publish_root_for_source(
    config: TwopyConfig,
    source_session_dir: Path,
) -> Path | None:
    """Return publish root for a source path, or ``None`` when it cannot map."""
    try:
        return (
            resolve_analysis_output_dir(config, source_session_dir)
            .expanduser()
            .resolve(strict=False)
        )
    except ValueError:
        return None


def _entry_is_protected(entry: _CacheEntry, protected_roots: tuple[Path, ...]) -> bool:
    """Return whether an entry overlaps a protected path."""
    return any(
        _is_relative_to(protected, entry.root) or _is_relative_to(entry.root, protected)
        for protected in protected_roots
    )


def _eviction_sort_key(entry: _CacheEntry) -> tuple[int, str]:
    """Return oldest-first eviction key."""
    return (entry.last_used_ns, str(entry.root))


def _remove_empty_parents(start: Path, stop: Path) -> None:
    """Remove empty parent folders up to but not including ``stop``."""
    current = start
    while current != stop and _is_relative_to(current, stop):
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _write_json_atomically(path: Path, data: Mapping[str, object]) -> None:
    """Write JSON through a temporary path before replacing the target."""
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temp_file:
            json.dump(data, temp_file, indent=2, sort_keys=True)
            temp_file.write("\n")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _optional_path_from_storage(value: object) -> Path | None:
    """Parse an optional path value from JSON."""
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        return None
    return Path(value).expanduser().resolve(strict=False)


def _non_negative_int(value: object) -> int | None:
    """Return non-negative integer JSON values."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def _entry_key(root: Path) -> str:
    """Return stable JSON key for a cache entry root."""
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()


def _inventory_path(cache_root: Path) -> Path:
    """Return inventory path for one cache root."""
    return cache_root / CACHE_INVENTORY_FILENAME


def _same_path(left: Path, right: Path) -> bool:
    """Return whether two paths resolve to the same filesystem location."""
    return left.expanduser().resolve(strict=False) == right.expanduser().resolve(
        strict=False,
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether path is inside parent."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
