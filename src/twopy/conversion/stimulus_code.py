"""Decode experiment-specific stimulus column metadata from backup code.

Inputs: ``stimulusData/filebackup.zip`` and loaded stimulus epoch parameters.
Outputs: small JSON-ready dictionaries that explain stimulus-specific
``stimData`` slots for the exact recording.

This module deliberately uses simple text parsing, not a MATLAB parser. The
goal is auditable provenance: record which backed-up function wrote which
``stimData.mat(N)`` slots.
"""

import re
from pathlib import Path
from zipfile import ZipFile

from twopy.conversion.types import StimulusCodeMetadata, StimulusParameters

__all__ = ["load_stimulus_code_metadata"]

LOOKUP_PATH = "paramfiles/stimulus_lookup.txt"
STIMFUNCTION_DIR = "stimfunctions"
STIMULUS_SLOT_COUNT = 20
STIMULUS_ASSIGNMENT_PATTERN = re.compile(
    r"stimData\.mat\((?P<slot>\d+)\)\s*=\s*(?P<expression>[^;]+);(?P<comment>.*)",
)


def load_stimulus_code_metadata(
    filebackup_zip: Path,
    parameters: StimulusParameters,
) -> StimulusCodeMetadata:
    """Load per-``stimtype`` stimulus-slot metadata from backup code.

    Args:
        filebackup_zip: ``stimulusData/filebackup.zip`` for one recording.
        parameters: Stimulus epoch parameters loaded from ``stimParams.mat``.

    Returns:
        Function lookup and slot assignments for the ``stimtype`` values used
        by this recording.

    Raises:
        FileNotFoundError: If the code backup zip is absent.
        KeyError: If lookup or function files needed by this recording are
            missing from the backup.

    Reading this zip is cheap relative to movie conversion and gives later
    analysis code the exact experiment-specific meaning of columns such as
    ``stimulus_specific_04``.
    """
    if not filebackup_zip.is_file():
        msg = f"Missing stimulus code backup zip: {filebackup_zip}"
        raise FileNotFoundError(msg)

    with ZipFile(filebackup_zip) as archive:
        function_lookup = _load_function_lookup(archive)
        used_stimtypes = _stimtypes_from_parameters(parameters)
        _require_lookup_entries(function_lookup, used_stimtypes)
        stimulus_specific_columns = {
            stimtype: _load_function_columns(
                archive=archive,
                stimtype=stimtype,
                function_name=function_lookup[stimtype],
            )
            for stimtype in used_stimtypes
        }

    return StimulusCodeMetadata(
        path=filebackup_zip,
        function_lookup={
            stimtype: function_lookup[stimtype] for stimtype in used_stimtypes
        },
        stimulus_specific_columns=stimulus_specific_columns,
    )


def _require_lookup_entries(
    function_lookup: dict[str, str],
    used_stimtypes: tuple[str, ...],
) -> None:
    """Fail clearly when a recording uses a ``stimtype`` missing from lookup.

    Args:
        function_lookup: Mapping loaded from ``stimulus_lookup.txt``.
        used_stimtypes: Stimulus types present in epoch parameters.

    Returns:
        None.

    Raises:
        KeyError: If any used ``stimtype`` is absent from the lookup file.
    """
    missing = tuple(
        stimtype for stimtype in used_stimtypes if stimtype not in function_lookup
    )
    if missing:
        msg = f"Missing stimtype entries in {LOOKUP_PATH}: {missing}"
        raise KeyError(msg)


def _load_function_lookup(archive: ZipFile) -> dict[str, str]:
    """Read ``stimtype`` to stimulus-function mappings from the backup zip.

    Args:
        archive: Open ``filebackup.zip``.

    Returns:
        Mapping from ``stimtype`` number as text to MATLAB function name.
    """
    lookup_text = _read_zip_text(archive, LOOKUP_PATH)
    lookup: dict[str, str] = {}
    for line in lookup_text.splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped.split(",", maxsplit=1)]
        if len(parts) != 2:
            continue
        lookup[parts[0]] = parts[1]
    return lookup


def _stimtypes_from_parameters(parameters: StimulusParameters) -> tuple[str, ...]:
    """Return unique ``stimtype`` values used by this recording.

    Args:
        parameters: Stimulus epoch parameter dictionaries.

    Returns:
        Ordered tuple of distinct ``stimtype`` values as text.
    """
    stimtypes: list[str] = []
    for epoch in parameters.epochs:
        if "stimtype" not in epoch:
            continue
        stimtype = _stimtype_to_text(epoch["stimtype"])
        if stimtype not in stimtypes:
            stimtypes.append(stimtype)
    return tuple(stimtypes)


def _stimtype_to_text(value: object) -> str:
    """Convert one MATLAB-loaded ``stimtype`` value into stable text.

    Args:
        value: ``stimtype`` value from one epoch parameter dictionary.

    Returns:
        Integer-like stimulus type as text.
    """
    if isinstance(value, int | float | str):
        return str(int(value))
    msg = f"stimtype must be an int-like scalar; got {value!r}"
    raise TypeError(msg)


def _load_function_columns(
    *,
    archive: ZipFile,
    stimtype: str,
    function_name: str,
) -> dict[str, object]:
    """Read one stimulus function and decode its ``stimData.mat`` writes.

    Args:
        archive: Open ``filebackup.zip``.
        stimtype: Stimulus type number from epoch parameters.
        function_name: MATLAB stimulus function name from the lookup file.

    Returns:
        JSON-ready metadata for that function and its assigned slots.
    """
    source_path = f"{STIMFUNCTION_DIR}/{function_name}.m"
    source_text = _read_zip_text(archive, source_path)
    return {
        "stimtype": stimtype,
        "function": function_name,
        "source_path": source_path,
        "columns": _parse_stimulus_assignments(source_text),
    }


def _parse_stimulus_assignments(source_text: str) -> list[dict[str, object]]:
    """Find simple ``stimData.mat(N) = ...;`` assignments in MATLAB text.

    Args:
        source_text: MATLAB function source text.

    Returns:
        One JSON-ready dictionary per assigned stimulus-specific slot.

    The parser intentionally keeps the right-hand expression as text. That is
    more honest than inventing semantic labels, and it is enough for a human to
    audit the exact source line later.
    """
    columns: list[dict[str, object]] = []
    for line_number, line in enumerate(source_text.splitlines(), start=1):
        match = STIMULUS_ASSIGNMENT_PATTERN.search(line)
        if match is None:
            continue

        slot = int(match.group("slot"))
        if not 1 <= slot <= STIMULUS_SLOT_COUNT:
            continue

        comment = _clean_matlab_comment(match.group("comment"))
        column = {
            "mat_slot": slot,
            "column_name": f"stimulus_specific_{slot:02d}",
            "source_expression": match.group("expression").strip(),
            "source_line": line_number,
        }
        if comment is not None:
            column["source_comment"] = comment
        columns.append(column)
    return columns


def _clean_matlab_comment(raw_comment: str) -> str | None:
    """Normalize an optional MATLAB inline comment.

    Args:
        raw_comment: Text after the semicolon on one MATLAB assignment line.

    Returns:
        Comment text without the leading ``%``, or ``None`` when absent.
    """
    comment = raw_comment.strip()
    if not comment.startswith("%"):
        return None
    return comment.lstrip("%").strip() or None


def _read_zip_text(archive: ZipFile, member_path: str) -> str:
    """Read one UTF-8-ish text file from a stimulus backup zip.

    Args:
        archive: Open ``filebackup.zip``.
        member_path: Path inside the zip.

    Returns:
        Decoded text.

    Raises:
        KeyError: If the zip member is absent.
    """
    with archive.open(member_path) as member:
        return member.read().decode("utf-8", errors="replace")
