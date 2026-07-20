"""Typed values and records used by the database query layer.

Inputs: SQLite database metadata and rows.
Outputs: small immutable Python objects that describe database files, tables,
columns, raw records, and modeled recording experiments.

These definitions are split out so query code can stay short and auditable.
"""

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_DATABASE_FILES",
    "DatabaseCatalog",
    "DatabaseColumn",
    "DatabaseExperiment",
    "DatabaseRecord",
    "DatabaseTable",
    "DatabaseValue",
]

DatabaseValue = str | int | float | bytes | None

DEFAULT_DATABASE_FILES = ("experimentLog.db", "experimentInitLog.db")


@dataclass(frozen=True)
class DatabaseCatalog:
    """Known SQLite database files in the lab database folder.

    Inputs: database root folder and discovered SQLite file paths.
    Outputs: immutable paths used by query functions.

    This object exists so GUI code can show which DB files are available before
    users try to search experiments.
    """

    database_dir: Path
    sqlite_files: tuple[Path, ...]


@dataclass(frozen=True)
class DatabaseColumn:
    """Description of one SQLite table column.

    Inputs: SQLite ``PRAGMA table_info`` output.
    Outputs: column name, SQLite type text, and whether SQLite marks it as a
    primary-key column.

    Column metadata lets filtering code validate user-provided filter names
    before building SQL.
    """

    name: str
    sqlite_type: str
    is_primary_key: bool


@dataclass(frozen=True)
class DatabaseTable:
    """Description of one SQLite table.

    Inputs: SQLite database path, table name, and discovered columns.
    Outputs: immutable table metadata for query planning and GUI display.

    This is a small schema snapshot. It does not include table rows.
    """

    database_path: Path
    name: str
    columns: tuple[DatabaseColumn, ...]


@dataclass(frozen=True)
class DatabaseRecord:
    """One experiment-like row returned from a database query.

    Inputs: SQLite source path, table name, and row values.
    Outputs: a typed record that preserves source context with row data.

    The row values remain a dictionary because the real DB schema may evolve and
    has not yet been normalized into twopy-specific objects.
    """

    database_path: Path
    table_name: str
    values: dict[str, DatabaseValue]


@dataclass(frozen=True)
class DatabaseExperiment:
    """Joined stimulus-presentation and fly metadata for one experiment.

    Inputs: rows from ``stimulusPresentation`` and ``fly`` tables.
    Outputs: the fields most useful for finding a recording before loading data.

    This object is the first twopy-specific database view. It keeps the raw DB
    flexible while giving the GUI a stable, typed search result.
    """

    database_path: Path
    stimulus_presentation_id: int
    fly_id: int
    relative_data_path: str
    stimulus_function: str | None
    date: str | None
    genotype: str | None
    cell_type: str | None
    fluorescent_protein: str | None
    hemisphere: str | None
    person: str | None
    data_quality: int | None
