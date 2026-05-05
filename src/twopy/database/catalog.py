"""Find and prepare Clark lab SQLite database files.

Inputs: a lab database folder and an access mode.
Outputs: a catalog of SQLite file paths, optionally copied to a local cache.
"""

from pathlib import Path

from twopy.database.cache import (
    DEFAULT_DATABASE_CACHE_DIR,
    ensure_local_database_copy,
    validate_database_access,
)
from twopy.database.types import (
    DEFAULT_DATABASE_FILES,
    DatabaseAccess,
    DatabaseCatalog,
)

__all__ = ["read_database_catalog"]


def read_database_catalog(
    database_dir: Path,
    *,
    database_access: DatabaseAccess = "direct",
    cache_dir: Path | None = None,
) -> DatabaseCatalog:
    """Find known SQLite database files in a database folder.

    Args:
        database_dir: Folder that should contain lab SQLite DB files.
        database_access: ``direct`` reads DB files in place. ``copy`` uses local
            cached copies because network DB queries can be slow while file
            transfer is fast.
        cache_dir: Optional local cache directory for copied DB files.

    Returns:
        A catalog listing known DB files that exist.

    Raises:
        FileNotFoundError: If the database folder is missing or no known DB file
            exists.

    The function checks only the known Clark lab DB filenames so macOS resource
    fork files like ``._experimentLog.db`` are ignored.
    """
    root = database_dir.expanduser()
    validate_database_access(database_access)

    catalog = _read_direct_database_catalog(root)
    if database_access == "direct":
        return catalog

    local_cache_dir = (cache_dir or DEFAULT_DATABASE_CACHE_DIR).expanduser()
    return DatabaseCatalog(
        database_dir=root,
        sqlite_files=tuple(
            ensure_local_database_copy(database_path, local_cache_dir)
            for database_path in catalog.sqlite_files
        ),
    )


def _read_direct_database_catalog(database_dir: Path) -> DatabaseCatalog:
    """Find known SQLite files without copying them.

    Args:
        database_dir: Database folder to inspect.

    Returns:
        A catalog with direct database file paths.

    Raises:
        FileNotFoundError: If the folder or known DB files are absent.
    """
    root = database_dir.expanduser()
    if not root.is_dir():
        msg = f"Missing database directory: {root}"
        raise FileNotFoundError(msg)

    sqlite_files = tuple(
        path for name in DEFAULT_DATABASE_FILES if (path := root / name).is_file()
    )
    if not sqlite_files:
        expected = ", ".join(DEFAULT_DATABASE_FILES)
        msg = f"No known SQLite database files found in {root}; expected {expected}"
        raise FileNotFoundError(msg)

    return DatabaseCatalog(database_dir=root, sqlite_files=sqlite_files)
