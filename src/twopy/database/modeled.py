"""Modeled Clark lab recording queries.

Inputs: lab database folder and researcher-facing filters.
Outputs: typed ``DatabaseExperiment`` rows for recording discovery.

This module owns SQL that depends on the observed ``stimulusPresentation`` and
``fly`` tables.
"""

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from twopy.config import DatabaseAccess
from twopy.database.catalog import read_database_catalog
from twopy.database.connection import connect_read_only
from twopy.database.types import DatabaseExperiment, DatabaseValue

__all__ = ["find_recordings", "find_stimulus_presentations"]


def find_stimulus_presentations(
    database_dir: Path,
    *,
    date_year: int | None = None,
    date_month: int | None = None,
    date_day: int | None = None,
    date_contains: str | None = None,
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
        date_contains: Optional substring matched against full experiment date
            text.
        path_contains: Optional substring matched against ``relativeDataPath``.
        stimulus_contains: Optional substring matched against ``stimulusFunction``.
        genotype_contains: Optional substring matched against fly ``genotype``.
        sensor_contains: Optional substring matched against fly
            ``fluorescentProtein``.
        cell_type_contains: Optional substring matched against fly ``cellType``.
        hemisphere: Optional exact recording hemisphere, matched against the
            source fly ``eye`` column.
        person_contains: Optional substring matched against fly ``surgeon``.
        limit: Maximum number of experiments to return.
        database_access: ``direct`` queries DB files in place. ``copy`` queries
            local cached copies because network DB queries can be slow while
            file transfer is fast.
        cache_dir: Optional cache directory used when ``database_access`` is
            ``copy``.

    Returns:
        Typed experiment rows sorted by database file order and DB row order.

    Raises:
        FileNotFoundError: If no database files can be found.
        ValueError: If ``limit`` is less than one.
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
        with closing(connect_read_only(database_path)) as connection:
            if not _has_tables(connection, ("stimulusPresentation", "fly")):
                continue

            rows = _query_stimulus_presentations(
                connection=connection,
                date_year=date_year,
                date_month=date_month,
                date_day=date_day,
                date_contains=date_contains,
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
        hemisphere: Optional exact recording hemisphere, such as ``left`` or
            ``right``.
        person: Optional experimenter substring from fly ``surgeon``.
        limit: Maximum number of recordings to return.
        database_access: ``direct`` queries DB files in place. ``copy`` queries
            local cached copies because network DB queries can be slow while
            file transfer is fast.
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


def _query_stimulus_presentations(
    *,
    connection: sqlite3.Connection,
    date_year: int | None,
    date_month: int | None,
    date_day: int | None,
    date_contains: str | None,
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
        date_contains: Optional substring matched against experiment date.
        path_contains: Optional relative path substring.
        stimulus_contains: Optional stimulus-function substring.
        genotype_contains: Optional genotype substring.
        sensor_contains: Optional fluorescent-protein substring.
        cell_type_contains: Optional cell-type substring.
        hemisphere: Optional exact recording hemisphere.
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
    _append_optional_text_filters(
        where_clauses=where_clauses,
        parameters=parameters,
        filters={
            'CAST(sp."relativeDataPath" AS TEXT) LIKE ?': path_contains,
            'CAST(sp."date" AS TEXT) LIKE ?': date_contains,
            'CAST(sp."stimulusFunction" AS TEXT) LIKE ?': stimulus_contains,
            'CAST(f."genotype" AS TEXT) LIKE ?': genotype_contains,
            'CAST(f."fluorescentProtein" AS TEXT) LIKE ?': sensor_contains,
            'CAST(f."cellType" AS TEXT) LIKE ?': cell_type_contains,
            'CAST(f."surgeon" AS TEXT) LIKE ?': person_contains,
        },
    )

    if hemisphere:
        where_clauses.append('f."eye" = ?')
        parameters.append(hemisphere)

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


def _append_optional_text_filters(
    *,
    where_clauses: list[str],
    parameters: list[DatabaseValue],
    filters: dict[str, str | None],
) -> None:
    """Append optional substring filters to a modeled query.

    Args:
        where_clauses: SQL where clauses being built.
        parameters: SQL parameter values being built.
        filters: Mapping of SQL clauses to optional text values.

    Returns:
        None. The function appends one clause and parameter for each value.
    """
    for clause, value in filters.items():
        if value:
            where_clauses.append(clause)
            parameters.append(f"%{value}%")


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
