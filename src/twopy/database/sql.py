"""Small SQL helpers shared by database query functions.

Inputs: discovered SQLite table metadata and identifier names.
Outputs: validated filters and quoted identifiers for SQL strings.

Values still use SQLite parameters. These helpers only handle names, because
table and column identifiers cannot be passed as SQL parameters.
"""

from collections.abc import Sequence

from twopy.database.types import DatabaseTable

__all__ = ["quote_identifier", "validate_filter_columns"]


def quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier.

    Args:
        identifier: Table or column name from SQLite metadata.

    Returns:
        Identifier wrapped in double quotes with embedded quotes escaped.
    """
    return '"' + identifier.replace('"', '""') + '"'


def validate_filter_columns(table: DatabaseTable, names: Sequence[str]) -> None:
    """Ensure filter names are columns in the selected table.

    Args:
        table: Table being queried.
        names: Filter column names requested by the caller.

    Returns:
        None when every name exists.

    Raises:
        ValueError: If any filter name is not a table column.
    """
    available = {column.name for column in table.columns}
    missing = tuple(name for name in names if name not in available)
    if missing:
        msg = f"Missing column(s) in {table.name}: {', '.join(missing)}"
        raise ValueError(msg)
