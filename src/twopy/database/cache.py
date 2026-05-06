"""Local cache support for Clark lab SQLite database files.

Inputs: mounted SQLite DB paths and a local cache folder.
Outputs: local DB copies plus JSON manifests that record source metadata.

Copying exists for performance. Database queries over network-mounted volumes
can be slow, but transferring the database file locally is usually fast.
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import cast

from twopy.database.types import DatabaseAccess

__all__ = [
    "DEFAULT_DATABASE_CACHE_DIR",
    "ensure_local_database_copy",
    "validate_database_access",
]

DEFAULT_DATABASE_CACHE_DIR = Path.home() / ".cache" / "twopy" / "database"


def validate_database_access(database_access: DatabaseAccess) -> None:
    """Validate the database access mode.

    Args:
        database_access: Requested DB access mode.

    Returns:
        None when the mode is supported.

    Raises:
        ValueError: If the mode is not ``direct`` or ``copy``.
    """
    if database_access not in {"direct", "copy"}:
        msg = f"database_access must be 'direct' or 'copy'; got {database_access!r}"
        raise ValueError(msg)


def ensure_local_database_copy(source_path: Path, cache_dir: Path) -> Path:
    """Return a local cached copy of a SQLite database.

    Args:
        source_path: Source SQLite database path.
        cache_dir: Directory where local copies and manifests are stored.

    Returns:
        Local cached database path.

    The manifest stores source path, size, mtime, and SHA-256. If size and mtime
    match, twopy reuses the local copy. If metadata changed, twopy hashes source
    and cached files and copies only when content changed. This keeps repeated
    queries fast without repeatedly transferring unchanged DB files.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / source_path.name
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    source_stat = source_path.stat()
    source_metadata: dict[str, object] = {
        "source_path": str(source_path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
    }

    manifest = _read_copy_manifest(manifest_path)
    if cache_path.is_file() and _manifest_metadata_matches(manifest, source_metadata):
        return cache_path

    source_hash = _file_sha256(source_path)
    cached_hash = _file_sha256(cache_path) if cache_path.is_file() else None
    if source_hash != cached_hash:
        temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        shutil.copy2(source_path, temp_path)
        temp_path.replace(cache_path)

    _write_copy_manifest(
        manifest_path,
        {
            **source_metadata,
            "source_sha256": source_hash,
        },
    )
    return cache_path


def _read_copy_manifest(manifest_path: Path) -> dict[str, object] | None:
    """Read a DB copy manifest.

    Args:
        manifest_path: Manifest JSON path.

    Returns:
        Parsed manifest dictionary or ``None`` when absent or invalid.

    Invalid manifests are treated as cache misses because the source DB remains
    the source of truth.
    """
    if not manifest_path.is_file():
        return None

    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(loaded, dict):
        return None

    return cast(dict[str, object], loaded)


def _manifest_metadata_matches(
    manifest: dict[str, object] | None,
    source_metadata: dict[str, object],
) -> bool:
    """Check whether source file metadata matches a manifest.

    Args:
        manifest: Parsed manifest or ``None``.
        source_metadata: Current source path, size, and mtime metadata.

    Returns:
        ``True`` when the manifest points at the same unchanged source file.
    """
    if manifest is None:
        return False

    return all(manifest.get(key) == value for key, value in source_metadata.items())


def _write_copy_manifest(manifest_path: Path, manifest: dict[str, object]) -> None:
    """Write a DB copy manifest as stable JSON.

    Args:
        manifest_path: Manifest JSON path.
        manifest: Manifest fields to write.

    Returns:
        None.
    """
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 for one file.

    Args:
        path: File path to hash.

    Returns:
        Hex SHA-256 digest.

    The function streams chunks so database files do not need to fit in memory.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
