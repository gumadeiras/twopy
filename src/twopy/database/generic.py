"""Schema-aware generic SQLite table search.

Inputs: lab database folder, optional table names, and explicit filters.
Outputs: ``DatabaseRecord`` rows with source DB and table provenance.

This module supports exploratory queries before twopy has a modeled schema for a
table.
"""

import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path

from twopy.database.catalog import read_database_catalog
from twopy.database.connection import connect_read_only
from twopy.database.sql import quote_identifier, validate_filter_columns
from twopy.database.types import (
    DatabaseAccess,
    DatabaseColumn,
    DatabaseRecord,
    DatabaseTable,
    DatabaseValue,
)

__all__ = ["find_experiments"]


def find_experiments(
    database_dir: Path,
    *,
    table_names: Sequence[str] | None = None,
    exact_filters: Mapping[str, DatabaseValue] | None = None,
    contains_filters: Mapping[str, str] | None = None,
    search_text: str | None = None,
    limit: int = 100,
    database_access: DatabaseAccess = "direct",
    cache_dir: Path | None = None,
) -> tuple[DatabaseRecord, ...]:
    """Find experiment rows across the known lab SQLite databases.

    Args:
        database_dir: Folder containing ``experimentLog.db`` and/or
            ``experimentInitLog.db``.
        table_names: Optional table names to search. ``None`` searches all user
            tables in each DB.
        exact_filters: Optional column-to-value equality filters.
        contains_filters: Optional column-to-text substring filters.
        search_text: Optional substring searched across all columns in each
            selected table.
        limit: Maximum number of records to return across all DB files.
        database_access: ``direct`` queries DB files in place. ``copy`` queries
            local cached copies because network DB queries can be slow while
            file transfer is fast.
        cache_dir: Optional cache directory used when ``database_access`` is
            ``copy``.

    Returns:
        Matching records with source DB path, table name, and row values.

    Raises:
        FileNotFoundError: If no database files can be found.
        ValueError: If filters reference columns absent from a selected table or
            if ``limit`` is less than one.

    Filters are combined with ``AND``. This conservative behavior keeps search
    results auditable: every returned row satisfies every requested constraint.
    """
    if limit < 1:
        msg = "Database query limit must be at least 1"
        raise ValueError(msg)

    catalog = read_database_catalog(
        database_dir,
        database_access=database_access,
        cache_dir=cache_dir,
    )
    remaining = limit
    records: list[DatabaseRecord] = []

    for database_path in catalog.sqlite_files:
        with closing(connect_read_only(database_path)) as connection:
            tables = _matching_tables(connection, database_path, table_names)
            for table in tables:
                table_records = _query_table(
                    connection=connection,
                    table=table,
                    exact_filters=exact_filters or {},
                    contains_filters=contains_filters or {},
                    search_text=search_text,
                    limit=remaining,
                )
                records.extend(table_records)
                remaining = limit - len(records)
                if remaining == 0:
                    return tuple(records)

    return tuple(records)


def _matching_tables(
    connection: sqlite3.Connection,
    database_path: Path,
    table_names: Sequence[str] | None,
) -> tuple[DatabaseTable, ...]:
    """Return selected user tables from one SQLite database.

    Args:
        connection: Open SQLite connection.
        database_path: Source DB path for returned metadata.
        table_names: Optional table names requested by the caller.

    Returns:
        Table metadata for the selected tables.

    Raises:
        ValueError: If a requested table is absent.
    """
    available = _read_tables(connection, database_path)
    if table_names is None:
        return available

    by_name = {table.name: table for table in available}
    missing = tuple(name for name in table_names if name not in by_name)
    if missing:
        msg = f"Missing table(s) in {database_path}: {', '.join(missing)}"
        raise ValueError(msg)

    return tuple(by_name[name] for name in table_names)


def _read_tables(
    connection: sqlite3.Connection,
    database_path: Path,
) -> tuple[DatabaseTable, ...]:
    """Read table and column metadata from one SQLite database.

    Args:
        connection: Open SQLite connection.
        database_path: Source DB path for returned metadata.

    Returns:
        User tables sorted by name.

    Internal SQLite tables are ignored because they are storage metadata, not
    experiment data.
    """
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """,
    ).fetchall()

    return tuple(
        DatabaseTable(
            database_path=database_path,
            name=str(row["name"]),
            columns=_read_columns(connection, str(row["name"])),
        )
        for row in rows
    )


def _read_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> tuple[DatabaseColumn, ...]:
    """Read column metadata for one SQLite table.

    Args:
        connection: Open SQLite connection.
        table_name: Table to inspect.

    Returns:
        Column metadata in SQLite table order.
    """
    rows = connection.execute(
        f"PRAGMA table_info({quote_identifier(table_name)})",
    ).fetchall()
    return tuple(
        DatabaseColumn(
            name=str(row["name"]),
            sqlite_type=str(row["type"]),
            is_primary_key=bool(row["pk"]),
        )
        for row in rows
    )


def _query_table(
    *,
    connection: sqlite3.Connection,
    table: DatabaseTable,
    exact_filters: Mapping[str, DatabaseValue],
    contains_filters: Mapping[str, str],
    search_text: str | None,
    limit: int,
) -> tuple[DatabaseRecord, ...]:
    """Query one table with validated filters.

    Args:
        connection: Open SQLite connection.
        table: Table metadata.
        exact_filters: Equality filters.
        contains_filters: Substring filters for named columns.
        search_text: Optional substring searched across all columns.
        limit: Maximum rows returned from this table.

    Returns:
        Matching records from this table.

    SQL identifiers are quoted after validation against discovered columns. SQL
    values are always passed as parameters.
    """
    column_names = tuple(column.name for column in table.columns)
    validate_filter_columns(table, tuple(exact_filters))
    validate_filter_columns(table, tuple(contains_filters))

    where_clauses: list[str] = []
    parameters: list[DatabaseValue] = []

    for column, value in exact_filters.items():
        where_clauses.append(f"{quote_identifier(column)} = ?")
        parameters.append(value)

    for column, value in contains_filters.items():
        where_clauses.append(f"CAST({quote_identifier(column)} AS TEXT) LIKE ?")
        parameters.append(f"%{value}%")

    if search_text:
        search_clause = " OR ".join(
            f"CAST({quote_identifier(column)} AS TEXT) LIKE ?"
            for column in column_names
        )
        where_clauses.append(f"({search_clause})")
        parameters.extend(f"%{search_text}%" for _ in column_names)

    where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"SELECT * FROM {quote_identifier(table.name)}{where_sql} LIMIT ?"
    parameters.append(limit)

    rows = connection.execute(sql, parameters).fetchall()
    return tuple(
        DatabaseRecord(
            database_path=table.database_path,
            table_name=table.name,
            values={column: row[column] for column in tuple(row.keys())},
        )
        for row in rows
    )
