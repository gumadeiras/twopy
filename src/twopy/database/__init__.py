"""Query Clark lab experiment SQLite databases.

Inputs: a DB folder. Outputs: typed table metadata and experiment records.
This layer can discover tables, inspect columns, and filter rows
without hard-coding experiment fields that we have not validated yet.
"""

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

from twopy.database.cache import (
    DEFAULT_DATABASE_CACHE_DIR,
)
from twopy.database.cache import (
    ensure_local_database_copy as _ensure_local_database_copy,
)
from twopy.database.cache import (
    validate_database_access as _validate_database_access,
)
from twopy.database.sql import quote_identifier as _quote_identifier
from twopy.database.sql import validate_filter_columns as _validate_filter_columns
from twopy.database.types import (
    DEFAULT_DATABASE_FILES,
    DatabaseAccess,
    DatabaseCatalog,
    DatabaseColumn,
    DatabaseExperiment,
    DatabaseRecord,
    DatabaseTable,
    DatabaseValue,
)

__all__ = [
    "DatabaseAccess",
    "DatabaseCatalog",
    "DatabaseColumn",
    "DatabaseExperiment",
    "DatabaseRecord",
    "DatabaseTable",
    "find_experiments",
    "find_stimulus_presentations",
    "find_recordings",
    "read_database_catalog",
]


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
            cached copies.
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
    _validate_database_access(database_access)

    catalog = _read_direct_database_catalog(root)
    if database_access == "direct":
        return catalog

    local_cache_dir = (cache_dir or DEFAULT_DATABASE_CACHE_DIR).expanduser()
    return DatabaseCatalog(
        database_dir=root,
        sqlite_files=tuple(
            _ensure_local_database_copy(database_path, local_cache_dir)
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
            local cached copies.
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
        with _connect_read_only(database_path) as connection:
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


def find_stimulus_presentations(
    database_dir: Path,
    *,
    date_year: int | None = None,
    date_month: int | None = None,
    date_day: int | None = None,
    path_contains: str | None = None,
    stimulus_contains: str | None = None,
    genotype_contains: str | None = None,
    sensor_contains: str | None = None,
    cell_type_contains: str | None = None,
    hemisphere: str | None = None,
    person_contains: str | None = None,
    limit: int = 100,
    database_access: DatabaseAccess = "direct",
    cache_dir: Path | None = None,
) -> tuple[DatabaseExperiment, ...]:
    """Find recording experiments with joined fly metadata.

    Args:
        database_dir: Folder containing lab SQLite DB files.
        date_year: Optional four-digit year matched against experiment date.
        date_month: Optional month number matched against experiment date.
        date_day: Optional day-of-month matched against experiment date.
        path_contains: Optional substring matched against ``relativeDataPath``.
        stimulus_contains: Optional substring matched against ``stimulusFunction``.
        genotype_contains: Optional substring matched against fly ``genotype``.
        sensor_contains: Optional substring matched against fly
            ``fluorescentProtein``.
        cell_type_contains: Optional substring matched against fly ``cellType``.
        hemisphere: Optional exact match against fly ``eye``.
        person_contains: Optional substring matched against fly ``surgeon``.
        limit: Maximum number of experiments to return.
        database_access: ``direct`` queries DB files in place. ``copy`` queries
            local cached copies.
        cache_dir: Optional cache directory used when ``database_access`` is
            ``copy``.

    Returns:
        Typed experiment rows sorted by database file order and DB row order.

    Raises:
        FileNotFoundError: If no database files can be found.
        ValueError: If ``limit`` is less than one.

    This query encodes the observed Clark lab schema while keeping the generic
    ``find_experiments`` function available for unmodeled tables.
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
    experiments: list[DatabaseExperiment] = []

    for database_path in catalog.sqlite_files:
        with _connect_read_only(database_path) as connection:
            if not _has_tables(connection, ("stimulusPresentation", "fly")):
                continue

            rows = _query_stimulus_presentations(
                connection=connection,
                date_year=date_year,
                date_month=date_month,
                date_day=date_day,
                path_contains=path_contains,
                stimulus_contains=stimulus_contains,
                genotype_contains=genotype_contains,
                sensor_contains=sensor_contains,
                cell_type_contains=cell_type_contains,
                hemisphere=hemisphere,
                person_contains=person_contains,
                limit=remaining,
            )
            experiments.extend(
                _experiment_from_row(database_path=database_path, row=row)
                for row in rows
            )
            remaining = limit - len(experiments)
            if remaining == 0:
                return tuple(experiments)

    return tuple(experiments)


def find_recordings(
    database_dir: Path,
    *,
    year: int | None = None,
    month: int | None = None,
    day: int | None = None,
    genotype: str | None = None,
    stimulus: str | None = None,
    sensor: str | None = None,
    cell_type: str | None = None,
    hemisphere: str | None = None,
    person: str | None = None,
    limit: int = 100,
    database_access: DatabaseAccess = "direct",
    cache_dir: Path | None = None,
) -> tuple[DatabaseExperiment, ...]:
    """Script-friendly entrance function for finding recording experiments.

    Args:
        database_dir: Folder containing lab SQLite DB files.
        year: Optional experiment year.
        month: Optional experiment month.
        day: Optional experiment day-of-month.
        genotype: Optional genotype substring.
        stimulus: Optional stimulus-function substring.
        sensor: Optional sensor substring from fly ``fluorescentProtein``.
        cell_type: Optional cell-type substring from fly ``cellType``.
        hemisphere: Optional exact fly-eye value, such as ``left`` or ``right``.
        person: Optional experimenter substring from fly ``surgeon``.
        limit: Maximum number of recordings to return.
        database_access: ``direct`` queries DB files in place. ``copy`` queries
            local cached copies.
        cache_dir: Optional cache directory used when ``database_access`` is
            ``copy``.

    Returns:
        Typed recording experiment rows.

    This wrapper uses researcher-facing names and delegates to the database
    schema-specific query below.
    """
    return find_stimulus_presentations(
        database_dir,
        date_year=year,
        date_month=month,
        date_day=day,
        stimulus_contains=stimulus,
        genotype_contains=genotype,
        sensor_contains=sensor,
        cell_type_contains=cell_type,
        hemisphere=hemisphere,
        person_contains=person,
        limit=limit,
        database_access=database_access,
        cache_dir=cache_dir,
    )


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


def _has_tables(connection: sqlite3.Connection, table_names: tuple[str, ...]) -> bool:
    """Check whether all named tables exist.

    Args:
        connection: Open SQLite connection.
        table_names: Required table names.

    Returns:
        ``True`` when every table exists, otherwise ``False``.

    This lets modeled queries skip unrelated SQLite files instead of failing.
    """
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """,
    ).fetchall()
    available = {str(row["name"]) for row in rows}
    return all(name in available for name in table_names)


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
        f"PRAGMA table_info({_quote_identifier(table_name)})",
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
    _validate_filter_columns(table, tuple(exact_filters))
    _validate_filter_columns(table, tuple(contains_filters))

    where_clauses: list[str] = []
    parameters: list[DatabaseValue] = []

    for column, value in exact_filters.items():
        where_clauses.append(f"{_quote_identifier(column)} = ?")
        parameters.append(value)

    for column, value in contains_filters.items():
        where_clauses.append(f"CAST({_quote_identifier(column)} AS TEXT) LIKE ?")
        parameters.append(f"%{value}%")

    if search_text:
        search_clause = " OR ".join(
            f"CAST({_quote_identifier(column)} AS TEXT) LIKE ?"
            for column in column_names
        )
        where_clauses.append(f"({search_clause})")
        parameters.extend(f"%{search_text}%" for _ in column_names)

    where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"SELECT * FROM {_quote_identifier(table.name)}{where_sql} LIMIT ?"
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


def _query_stimulus_presentations(
    *,
    connection: sqlite3.Connection,
    date_year: int | None,
    date_month: int | None,
    date_day: int | None,
    path_contains: str | None,
    stimulus_contains: str | None,
    genotype_contains: str | None,
    sensor_contains: str | None,
    cell_type_contains: str | None,
    hemisphere: str | None,
    person_contains: str | None,
    limit: int,
) -> Sequence[sqlite3.Row]:
    """Run the modeled stimulus-presentation query.

    Args:
        connection: Open SQLite connection.
        date_year: Optional experiment year.
        date_month: Optional experiment month.
        date_day: Optional experiment day-of-month.
        path_contains: Optional relative path substring.
        stimulus_contains: Optional stimulus-function substring.
        genotype_contains: Optional genotype substring.
        sensor_contains: Optional fluorescent-protein substring.
        cell_type_contains: Optional cell-type substring.
        hemisphere: Optional exact fly-eye value.
        person_contains: Optional surgeon substring.
        limit: Maximum rows to return.

    Returns:
        SQLite rows with stimulus and fly metadata.

    The query uses left join because a stimulus row is still useful even if fly
    metadata is incomplete.
    """
    where_clauses: list[str] = []
    parameters: list[DatabaseValue] = []

    _append_date_filters(
        where_clauses=where_clauses,
        parameters=parameters,
        year=date_year,
        month=date_month,
        day=date_day,
    )

    if path_contains:
        where_clauses.append('CAST(sp."relativeDataPath" AS TEXT) LIKE ?')
        parameters.append(f"%{path_contains}%")

    if stimulus_contains:
        where_clauses.append('CAST(sp."stimulusFunction" AS TEXT) LIKE ?')
        parameters.append(f"%{stimulus_contains}%")

    if genotype_contains:
        where_clauses.append('CAST(f."genotype" AS TEXT) LIKE ?')
        parameters.append(f"%{genotype_contains}%")

    if sensor_contains:
        where_clauses.append('CAST(f."fluorescentProtein" AS TEXT) LIKE ?')
        parameters.append(f"%{sensor_contains}%")

    if cell_type_contains:
        where_clauses.append('CAST(f."cellType" AS TEXT) LIKE ?')
        parameters.append(f"%{cell_type_contains}%")

    if hemisphere:
        where_clauses.append('f."eye" = ?')
        parameters.append(hemisphere)

    if person_contains:
        where_clauses.append('CAST(f."surgeon" AS TEXT) LIKE ?')
        parameters.append(f"%{person_contains}%")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    parameters.append(limit)

    return connection.execute(
        f"""
        SELECT
            sp."stimulusPresentationId",
            sp."fly",
            sp."relativeDataPath",
            sp."stimulusFunction",
            sp."date",
            sp."dataQuality",
            f."genotype",
            f."cellType",
            f."fluorescentProtein",
            f."eye",
            f."surgeon"
        FROM "stimulusPresentation" AS sp
        LEFT JOIN "fly" AS f ON f."flyId" = sp."fly"
        {where_sql}
        ORDER BY sp."stimulusPresentationId"
        LIMIT ?
        """,
        parameters,
    ).fetchall()


def _experiment_from_row(database_path: Path, row: sqlite3.Row) -> DatabaseExperiment:
    """Convert one joined SQLite row into a typed experiment.

    Args:
        database_path: Source SQLite file.
        row: SQLite row from the modeled query.

    Returns:
        A typed experiment search result.
    """
    return DatabaseExperiment(
        database_path=database_path,
        stimulus_presentation_id=int(row["stimulusPresentationId"]),
        fly_id=int(row["fly"]),
        relative_data_path=str(row["relativeDataPath"]),
        stimulus_function=_optional_string(row["stimulusFunction"]),
        date=_optional_string(row["date"]),
        genotype=_optional_string(row["genotype"]),
        cell_type=_optional_string(row["cellType"]),
        fluorescent_protein=_optional_string(row["fluorescentProtein"]),
        hemisphere=_optional_string(row["eye"]),
        person=_optional_string(row["surgeon"]),
        data_quality=_optional_int(row["dataQuality"]),
    )


def _append_date_filters(
    *,
    where_clauses: list[str],
    parameters: list[DatabaseValue],
    year: int | None,
    month: int | None,
    day: int | None,
) -> None:
    """Add flexible year, month, and day filters for experiment dates.

    Args:
        where_clauses: SQL where clauses being built.
        parameters: SQL parameter values being built.
        year: Optional experiment year.
        month: Optional experiment month.
        day: Optional experiment day-of-month.

    Returns:
        None. The function appends clauses and parameters in place.

    The lab DB stores dates as ``YYYY-MM-DD HH:MM:SS`` text. ``substr`` keeps the
    query simple and allows year-only, month-only, day-only, or combined filters.
    """
    if year is not None:
        _validate_range("year", year, 1, 9999)
        where_clauses.append('substr(CAST(sp."date" AS TEXT), 1, 4) = ?')
        parameters.append(f"{year:04d}")

    if month is not None:
        _validate_range("month", month, 1, 12)
        where_clauses.append('substr(CAST(sp."date" AS TEXT), 6, 2) = ?')
        parameters.append(f"{month:02d}")

    if day is not None:
        _validate_range("day", day, 1, 31)
        where_clauses.append('substr(CAST(sp."date" AS TEXT), 9, 2) = ?')
        parameters.append(f"{day:02d}")


def _validate_range(name: str, value: int, minimum: int, maximum: int) -> None:
    """Validate one integer filter range.

    Args:
        name: Filter name for error messages.
        value: Integer value to check.
        minimum: Smallest accepted value.
        maximum: Largest accepted value.

    Returns:
        None when the value is valid.

    Raises:
        ValueError: If the value falls outside the accepted range.
    """
    if not minimum <= value <= maximum:
        msg = f"{name} must be between {minimum} and {maximum}; got {value}"
        raise ValueError(msg)


def _optional_string(value: object) -> str | None:
    """Convert nullable SQLite values into optional strings.

    Args:
        value: SQLite value.

    Returns:
        ``None`` for database nulls, otherwise ``str(value)``.
    """
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    """Convert nullable SQLite values into optional integers.

    Args:
        value: SQLite value.

    Returns:
        ``None`` for database nulls, otherwise ``int(value)``.
    """
    if value is None:
        return None

    if isinstance(value, str | bytes | int | float):
        return int(value)

    value_type = type(value).__name__
    msg = f"Expected nullable integer-compatible SQLite value, got {value_type}"
    raise TypeError(msg)


def _connect_read_only(database_path: Path) -> sqlite3.Connection:
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
