"""Open SQLite databases with twopy's safety defaults.

Inputs: SQLite database file paths.
Outputs: read-only SQLite connections that return row dictionaries.
"""

import sqlite3
from pathlib import Path

__all__ = ["connect_read_only"]


def connect_read_only(database_path: Path) -> sqlite3.Connection:
    """Open one SQLite database in read-only mode.

    Args:
        database_path: SQLite file to open.

    Returns:
        SQLite connection configured for row dictionaries.

    Read-only URI mode prevents accidental writes to the lab database.
    """
    uri = database_path.expanduser().resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection
