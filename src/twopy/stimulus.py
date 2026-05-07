"""Helpers for interpreting converted stimulus data columns.

Inputs: a loaded twopy ``RecordingData`` object.
Outputs: small typed mappings from stable ``stimulus/data`` column names to the
experiment-specific MATLAB code that wrote those columns.

The stable HDF5 column name stays generic because different ``stimtype`` values
can reuse the same slot for different meanings. These helpers make that
per-``stimtype`` mapping explicit for scripts and analysis code.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from twopy.converted import RecordingData
from twopy.typing_guards import require_string_key_mapping

__all__ = [
    "StimulusSpecificColumnMapping",
    "default_interleave_epoch_number",
    "is_interleave_epoch_name",
    "map_stimulus_specific_column",
    "psycho5_default_interleave_epoch_numbers",
    "stimulus_column_index",
    "stimulus_epoch_names_by_number",
]


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


def stimulus_column_index(recording: RecordingData, column_name: str) -> int:
    """Return the index for one converted stimulus data column.

    Args:
        recording: Loaded converted recording.
        column_name: Required column name from ``stimulus/data_column_names``.

    Returns:
        Integer column index.

    Raises:
        ValueError: If the column is not present.

    Keeping this lookup in one helper avoids slightly different error handling
    in timing, classification, and script-facing analysis code.
    """
    try:
        return recording.stimulus_data_column_names.index(column_name)
    except ValueError as error:
        msg = f"stimulus/data is missing required column {column_name!r}"
        raise ValueError(msg) from error


def stimulus_epoch_names_by_number(recording: RecordingData) -> dict[int, str]:
    """Map one-based stimulus epoch numbers to readable epoch names.

    Args:
        recording: Loaded converted recording with stimulus parameter metadata.

    Returns:
        Dictionary mapping epoch number to name.

    Epoch numbers in ``stimulus/data`` are one-based because they come from the
    stimulus computer. Missing names fall back to ``epoch_N`` at the call site.
    """
    names: dict[int, str] = {}
    for index, parameters in enumerate(recording.stimulus_parameters, start=1):
        epoch_name = parameters.get("epochName", f"epoch_{index}")
        names[index] = str(epoch_name)
    return names


def is_interleave_epoch_name(epoch_name: str) -> bool:
    """Return whether an epoch name looks like an interleave baseline.

    Args:
        epoch_name: Stimulus epoch name.

    Returns:
        ``True`` for ``gray`` or ``grey`` spelling, or for names containing
        ``interleave``.
    """
    lowered = epoch_name.lower()
    return "gray" in lowered or "grey" in lowered or "interleave" in lowered


def default_interleave_epoch_number(epoch_names: Mapping[int, str]) -> int:
    """Return the default interleave epoch from readable epoch names.

    Args:
        epoch_names: Mapping from stimulus epoch number to epoch name.

    Returns:
        First epoch whose name looks like gray/grey/interleave, otherwise
        epoch 1.
    """
    for epoch_number, epoch_name in sorted(epoch_names.items()):
        if is_interleave_epoch_name(epoch_name):
            return epoch_number
    return 1


def psycho5_default_interleave_epoch_numbers(
    recording: RecordingData,
) -> tuple[int, ...]:
    """Return the source-style default interleave epoch selection.

    Args:
        recording: Loaded converted recording with decoded stimulus parameters.

    Returns:
        One or more one-based epoch numbers to use as gray-interleave baselines.

    psycho5 first uses ``paramsHere(end).nextEpoch`` when present. Otherwise it
    selects epochs named exactly ``Gray Interleave`` and reverses the list when
    epoch one is the first of multiple gray interleaves. If neither source
    convention is present, twopy falls back to its readable-name helper so older
    scripts still get an explicit baseline instead of an empty selection.
    """
    if recording.stimulus_parameters:
        next_epoch = recording.stimulus_parameters[-1].get("nextEpoch")
        if next_epoch is not None:
            return _epoch_numbers_from_source_value(next_epoch, name="nextEpoch")

    exact_gray_epochs = tuple(
        index
        for index, parameters in enumerate(recording.stimulus_parameters, start=1)
        if parameters.get("epochName") == "Gray Interleave"
    )
    if len(exact_gray_epochs) > 1 and exact_gray_epochs[0] == 1:
        return exact_gray_epochs[::-1]
    if exact_gray_epochs:
        return exact_gray_epochs

    return (default_interleave_epoch_number(stimulus_epoch_names_by_number(recording)),)


def _epoch_numbers_from_source_value(value: object, *, name: str) -> tuple[int, ...]:
    """Decode one MATLAB-like epoch-number value from converted metadata.

    Args:
        value: Scalar or sequence value decoded from stimulus parameters.
        name: Field name used in validation messages.

    Returns:
        Tuple of one-based epoch numbers.
    """
    if isinstance(value, bool):
        msg = f"{name} must be an epoch number, not a boolean"
        raise ValueError(msg)
    if isinstance(value, int | float):
        return (_source_epoch_number(value, name=name),)
    if isinstance(value, list | tuple):
        epoch_numbers = tuple(_source_epoch_number(item, name=name) for item in value)
        if not epoch_numbers:
            msg = f"{name} must contain at least one epoch number"
            raise ValueError(msg)
        return epoch_numbers

    msg = f"{name} must be an epoch number or list of epoch numbers; got {value!r}"
    raise ValueError(msg)


def _source_epoch_number(value: object, *, name: str) -> int:
    """Validate one one-based epoch number from source metadata.

    Args:
        value: Candidate numeric epoch identifier.
        name: Field name used in validation messages.

    Returns:
        Integer epoch number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{name} entries must be numeric epoch numbers; got {value!r}"
        raise ValueError(msg)
    epoch_number = int(value)
    if epoch_number < 1 or float(value) != float(epoch_number):
        msg = f"{name} entries must be positive integer epoch numbers; got {value!r}"
        raise ValueError(msg)
    return epoch_number


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
        try:
            column_metadata = require_string_key_mapping(
                column,
                name=f"stimulus metadata for stimtype {stimtype!r} column",
            )
        except ValueError as error:
            msg = f"stimulus metadata for stimtype {stimtype!r} is malformed"
            raise ValueError(msg) from error
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
