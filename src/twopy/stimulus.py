"""Helpers for interpreting converted stimulus data columns.

Inputs: a loaded twopy ``RecordingData`` object.
Outputs: small typed mappings from stable ``stimulus/data`` column names to the
experiment-specific MATLAB code that wrote those columns.

The stable HDF5 column name stays generic because different ``stimtype`` values
can reuse the same slot for different meanings. These helpers make that
per-``stimtype`` mapping explicit for scripts and analysis code.
"""

from dataclasses import dataclass
from typing import cast

from twopy.converted import RecordingData

__all__ = ["StimulusSpecificColumnMapping", "map_stimulus_specific_column"]


@dataclass(frozen=True)
class StimulusSpecificColumnMapping:
    """Meaning of one stable stimulus-specific column for one ``stimtype``.

    Inputs: decoded metadata from ``stimulus/stimulus_specific_columns_json``.
    Outputs: a script-friendly record containing the stable column name, the
    ``stimtype``, the backed-up MATLAB function, and the source assignment.

    ``specific_name`` is the most readable available label: an inline MATLAB
    comment when present, otherwise the right-hand source expression.
    """

    stable_column_name: str
    stimtype: str
    specific_name: str
    source_expression: str
    source_comment: str | None
    source_line: int
    source_path: str
    stimulus_function: str


def map_stimulus_specific_column(
    recording: RecordingData,
    stable_column_name: str,
    *,
    stimtype: int | str | None = None,
) -> tuple[StimulusSpecificColumnMapping, ...]:
    """Map a stable stimulus column to experiment-specific meanings.

    Args:
        recording: Loaded converted recording.
        stable_column_name: Stable name such as ``stimulus_specific_04``.
        stimtype: Optional stimulus type. When omitted, mappings for all
            ``stimtype`` values in the recording are returned.

    Returns:
        One mapping per matching ``stimtype``. An unused stimulus-specific slot
        returns an empty tuple.

    Raises:
        ValueError: If ``stable_column_name`` is not a stimulus-specific column.
        KeyError: If the requested ``stimtype`` is not present in the recording.

    This function exists so analysis code can use the simple stable column
    names for data access while still showing the exact experiment-specific
    meaning from the backed-up MATLAB code.
    """
    if not stable_column_name.startswith("stimulus_specific_"):
        msg = f"Expected stimulus_specific_* column; got {stable_column_name!r}"
        raise ValueError(msg)

    selected_stimtypes = _selected_stimtypes(recording, stimtype)
    mappings: list[StimulusSpecificColumnMapping] = []
    for stimtype_key in selected_stimtypes:
        metadata = recording.stimulus_specific_columns[stimtype_key]
        mappings.extend(
            _mappings_from_metadata(
                stable_column_name=stable_column_name,
                stimtype=stimtype_key,
                metadata=metadata,
            ),
        )
    return tuple(mappings)


def _selected_stimtypes(
    recording: RecordingData,
    stimtype: int | str | None,
) -> tuple[str, ...]:
    """Choose which ``stimtype`` entries to inspect.

    Args:
        recording: Loaded converted recording.
        stimtype: Optional requested stimulus type.

    Returns:
        Tuple of stimulus type keys present in ``recording``.
    """
    if stimtype is None:
        return tuple(recording.stimulus_specific_columns)

    stimtype_key = str(int(stimtype)) if isinstance(stimtype, int) else str(stimtype)
    if stimtype_key not in recording.stimulus_specific_columns:
        msg = f"stimtype {stimtype_key!r} is not present in this recording"
        raise KeyError(msg)
    return (stimtype_key,)


def _mappings_from_metadata(
    *,
    stable_column_name: str,
    stimtype: str,
    metadata: dict[str, object],
) -> tuple[StimulusSpecificColumnMapping, ...]:
    """Build mappings from one decoded stimulus-function metadata block.

    Args:
        stable_column_name: Stable HDF5 column name to find.
        stimtype: Stimulus type key for this metadata block.
        metadata: Decoded JSON dictionary for one stimulus function.

    Returns:
        Matching column mappings.
    """
    columns = metadata.get("columns")
    if not isinstance(columns, list):
        msg = f"stimulus metadata for stimtype {stimtype!r} has no column list"
        raise ValueError(msg)

    mappings: list[StimulusSpecificColumnMapping] = []
    for column in columns:
        if not isinstance(column, dict):
            msg = f"stimulus metadata for stimtype {stimtype!r} is malformed"
            raise ValueError(msg)
        column_metadata = cast(dict[str, object], column)
        if column_metadata.get("column_name") != stable_column_name:
            continue

        source_comment = _optional_text(column_metadata.get("source_comment"))
        source_expression = str(column_metadata["source_expression"])
        source_line = _required_int(column_metadata["source_line"])
        mappings.append(
            StimulusSpecificColumnMapping(
                stable_column_name=stable_column_name,
                stimtype=stimtype,
                specific_name=source_comment or source_expression,
                source_expression=source_expression,
                source_comment=source_comment,
                source_line=source_line,
                source_path=str(metadata["source_path"]),
                stimulus_function=str(metadata["function"]),
            ),
        )
    return tuple(mappings)


def _optional_text(value: object) -> str | None:
    """Return optional metadata text as ``str | None``.

    Args:
        value: Raw decoded JSON value.

    Returns:
        Text when present, otherwise ``None``.
    """
    if value is None:
        return None
    return str(value)


def _required_int(value: object) -> int:
    """Return a required JSON metadata integer.

    Args:
        value: Raw decoded JSON value.

    Returns:
        Integer value.

    Raises:
        ValueError: If the value cannot honestly be read as an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int | str):
        msg = f"Expected integer metadata value; got {value!r}"
        raise ValueError(msg)
    return int(value)
