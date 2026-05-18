"""Persist manual ROI matches across recordings.

Inputs: user-authored ROI match decisions from a group-analysis review.
Outputs: plain CSV rows that identify which ROI labels belong to the same
cell, or which ROI labels were reviewed and left unmatched.

This module is GUI-independent on purpose. Napari can help users make visual
decisions, but the scientific audit trail is the explicit match table written
here.
"""

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = [
    "ManualFovGroupRow",
    "ManualRoiMatchGroup",
    "ManualRoiMatchRow",
    "ManualRoiMatchStatus",
    "add_manual_roi_match_group",
    "append_manual_roi_match_rows",
    "load_manual_fov_group_rows",
    "load_manual_roi_match_rows",
    "make_manual_fov_group_rows",
    "make_manual_roi_match_rows",
    "matched_manual_roi_groups",
    "next_group_cell_id",
    "remove_manual_roi_match_group",
    "replace_manual_roi_match_group",
    "save_manual_fov_group_rows",
    "save_manual_roi_match_rows",
]

ManualRoiMatchStatus = Literal["matched", "unmatched", "ambiguous", "ignored"]

_FIELDNAMES = (
    "fov_group_id",
    "group_cell_id",
    "recording_path",
    "roi_label",
    "status",
    "note",
)
_FOV_FIELDNAMES = ("fov_group_id", "recording_path", "note")
_STATUSES: tuple[ManualRoiMatchStatus, ...] = (
    "matched",
    "unmatched",
    "ambiguous",
    "ignored",
)


@dataclass(frozen=True)
class ManualFovGroupRow:
    """One row assigning a recording to a manually reviewed FOV group.

    Inputs: one converted recording path and the FOV group label chosen by the
        user.
    Outputs: a typed row that can be saved beside downstream group-analysis
        artifacts.

    FOV grouping is deliberately separate from ROI matching. A recording can be
    visually assigned to a FOV before any cell-level match decisions are made.
    """

    fov_group_id: str
    recording_path: Path
    note: str = ""


@dataclass(frozen=True)
class ManualRoiMatchRow:
    """One row in a manual cross-recording ROI match table.

    Inputs: one recording path, one ROI label, and the group-cell decision that
        the user made for it.
    Outputs: a typed row that can be written to CSV or loaded from CSV.

    ``group_cell_id`` is scoped to the saved match file. Rows that share the
    same ``group_cell_id`` and ``status="matched"`` are treated as the same
    visually assigned cell across recordings.
    """

    fov_group_id: str
    group_cell_id: int
    recording_path: Path
    roi_label: str
    status: ManualRoiMatchStatus
    note: str = ""


@dataclass(frozen=True)
class ManualRoiMatchGroup:
    """Matched rows for one manually assigned cell group.

    Inputs: existing ``ManualRoiMatchRow`` objects from a match table.
    Outputs: one grouped decision for display or downstream group analysis.
    """

    group_cell_id: int
    rows: tuple[ManualRoiMatchRow, ...]
    note: str = ""


def make_manual_fov_group_rows(
    recording_groups: Mapping[Path, str],
    *,
    note: str = "",
    notes: Mapping[Path, str] | None = None,
) -> tuple[ManualFovGroupRow, ...]:
    """Create validated FOV-group rows from recording assignments.

    Args:
        recording_groups: Mapping from recording path to non-empty FOV group
            id.
        note: Optional fallback free-text note stored on each row.
        notes: Optional per-recording free-text notes stored on matching rows.

    Returns:
        Tuple of FOV group rows sorted by recording path.

    Raises:
        ValueError: If no recordings are assigned, a path is empty, or a FOV id
            is empty.
    """
    if len(recording_groups) == 0:
        msg = "Manual FOV group rows require at least one recording assignment."
        raise ValueError(msg)
    return tuple(
        ManualFovGroupRow(
            fov_group_id=_validate_fov_group_id(fov_group_id),
            recording_path=_validate_recording_path(recording_path),
            note=notes.get(recording_path, note) if notes is not None else note,
        )
        for recording_path, fov_group_id in sorted(
            recording_groups.items(),
            key=lambda item: str(item[0]),
        )
    )


def make_manual_roi_match_rows(
    recording_rois: Mapping[Path, str],
    *,
    group_cell_id: int,
    fov_group_id: str = "",
    status: ManualRoiMatchStatus = "matched",
    note: str = "",
) -> tuple[ManualRoiMatchRow, ...]:
    """Create validated match-table rows for one user decision.

    Args:
        recording_rois: Mapping from recording path to ROI label.
        group_cell_id: Positive cell-group identifier for this decision.
        fov_group_id: Optional user-facing FOV group identifier.
        status: Decision status for every emitted row.
        note: Optional free-text note stored on each row.

    Returns:
        Tuple of manual ROI match rows sorted by recording path.

    Raises:
        ValueError: If the group id, status, paths, or ROI labels are invalid.
    """
    _validate_group_cell_id(group_cell_id)
    _validate_status(status)
    if len(recording_rois) == 0:
        msg = "Manual ROI match rows require at least one recording ROI."
        raise ValueError(msg)

    rows: list[ManualRoiMatchRow] = []
    for recording_path, roi_label in sorted(
        recording_rois.items(),
        key=lambda item: str(item[0]),
    ):
        rows.append(
            ManualRoiMatchRow(
                fov_group_id=fov_group_id,
                group_cell_id=group_cell_id,
                recording_path=_validate_recording_path(recording_path),
                roi_label=_validate_roi_label(roi_label),
                status=status,
                note=note,
            ),
        )
    return tuple(rows)


def matched_manual_roi_groups(
    rows: Sequence[ManualRoiMatchRow],
    *,
    fov_group_id: str,
) -> tuple[ManualRoiMatchGroup, ...]:
    """Return matched ROI groups for one FOV id.

    Args:
        rows: Manual ROI match rows from memory or CSV.
        fov_group_id: FOV id whose matched groups should be returned.

    Returns:
        Groups sorted by ``group_cell_id``. Non-matched statuses and other FOVs
        are ignored.
    """
    _validate_fov_group_id(fov_group_id)
    grouped: dict[int, list[ManualRoiMatchRow]] = {}
    for row in rows:
        if row.status != "matched" or row.fov_group_id != fov_group_id:
            continue
        grouped.setdefault(row.group_cell_id, []).append(row)
    groups: list[ManualRoiMatchGroup] = []
    for group_cell_id, group_rows in sorted(grouped.items()):
        row_tuple = tuple(group_rows)
        groups.append(
            ManualRoiMatchGroup(
                group_cell_id=group_cell_id,
                rows=row_tuple,
                note=_shared_row_note(row_tuple),
            ),
        )
    return tuple(groups)


def add_manual_roi_match_group(
    rows: Sequence[ManualRoiMatchRow],
    recording_rois: Mapping[Path, str],
    *,
    fov_group_id: str,
    note: str = "",
) -> tuple[tuple[ManualRoiMatchRow, ...], int]:
    """Append one new matched ROI group to existing rows.

    Args:
        rows: Existing manual ROI match rows.
        recording_rois: Mapping from recording path to selected ROI label.
        fov_group_id: FOV id for the new matched group.
        note: Optional note copied to every new row.

    Returns:
        Updated row tuple and the assigned ``group_cell_id``.
    """
    _validate_fov_group_id(fov_group_id)
    group_cell_id = next_group_cell_id(rows)
    group_rows = make_manual_roi_match_rows(
        recording_rois,
        group_cell_id=group_cell_id,
        fov_group_id=fov_group_id,
        status="matched",
        note=note,
    )
    return (*tuple(rows), *group_rows), group_cell_id


def replace_manual_roi_match_group(
    rows: Sequence[ManualRoiMatchRow],
    recording_rois: Mapping[Path, str],
    *,
    fov_group_id: str,
    group_cell_id: int,
    note: str = "",
) -> tuple[ManualRoiMatchRow, ...]:
    """Replace one matched group inside one FOV while preserving other rows.

    Args:
        rows: Existing manual ROI match rows.
        recording_rois: Replacement recording-to-ROI selections.
        fov_group_id: FOV id containing the group.
        group_cell_id: Existing group id to replace.
        note: Optional note copied to every replacement row.

    Returns:
        Updated row tuple.
    """
    _validate_fov_group_id(fov_group_id)
    replacement_rows = make_manual_roi_match_rows(
        recording_rois,
        group_cell_id=group_cell_id,
        fov_group_id=fov_group_id,
        status="matched",
        note=note,
    )
    retained_rows = _rows_without_matched_group(rows, fov_group_id, group_cell_id)
    return (*retained_rows, *replacement_rows)


def remove_manual_roi_match_group(
    rows: Sequence[ManualRoiMatchRow],
    *,
    fov_group_id: str,
    group_cell_id: int,
) -> tuple[ManualRoiMatchRow, ...]:
    """Remove one matched group inside one FOV while preserving other rows."""
    return _rows_without_matched_group(rows, fov_group_id, group_cell_id)


def save_manual_fov_group_rows(
    rows: Iterable[ManualFovGroupRow],
    path: Path,
) -> None:
    """Write manual FOV grouping rows to a CSV file.

    Args:
        rows: Manual FOV group rows to persist.
        path: Destination CSV path.

    Returns:
        None.
    """
    _write_csv_rows(
        path,
        fieldnames=_FOV_FIELDNAMES,
        rows=(_fov_row_to_csv(row) for row in rows),
    )


def load_manual_fov_group_rows(path: Path) -> tuple[ManualFovGroupRow, ...]:
    """Load manual FOV grouping rows from a CSV file.

    Args:
        path: CSV file written by ``save_manual_fov_group_rows``.

    Returns:
        Tuple of validated FOV grouping rows.

    Raises:
        ValueError: If the file contains missing columns or invalid values.
    """
    csv_rows = _read_required_csv_rows(
        path,
        fieldnames=_FOV_FIELDNAMES,
        table_name="Manual FOV group file",
    )
    return tuple(_fov_row_from_csv(row) for row in csv_rows)


def append_manual_roi_match_rows(
    path: Path,
    rows: Sequence[ManualRoiMatchRow],
) -> tuple[ManualRoiMatchRow, ...]:
    """Append manual ROI match rows to an existing CSV file.

    Args:
        path: CSV match table path. Missing files are created.
        rows: Rows to append after the existing rows.

    Returns:
        All rows written to the file.
    """
    existing_rows = load_manual_roi_match_rows(path) if path.exists() else ()
    all_rows = (*existing_rows, *tuple(rows))
    save_manual_roi_match_rows(all_rows, path)
    return all_rows


def save_manual_roi_match_rows(
    rows: Iterable[ManualRoiMatchRow],
    path: Path,
) -> None:
    """Write manual ROI match rows to a CSV file.

    Args:
        rows: Manual match rows to persist.
        path: Destination CSV path.

    Returns:
        None.

    The CSV file is intentionally plain text so group-analysis decisions can be
    inspected, diffed, and edited without twopy.
    """
    _write_csv_rows(
        path,
        fieldnames=_FIELDNAMES,
        rows=(_row_to_csv(row) for row in rows),
    )


def load_manual_roi_match_rows(path: Path) -> tuple[ManualRoiMatchRow, ...]:
    """Load manual ROI match rows from a CSV file.

    Args:
        path: CSV file written by ``save_manual_roi_match_rows``.

    Returns:
        Tuple of validated manual ROI match rows.

    Raises:
        ValueError: If the file contains missing columns or invalid values.
    """
    input_path = path.expanduser()
    csv_rows = _read_required_csv_rows(
        input_path,
        fieldnames=_FIELDNAMES,
        table_name="Manual ROI match file",
    )
    return tuple(_row_from_csv(row, input_path=input_path) for row in csv_rows)


def next_group_cell_id(rows: Sequence[ManualRoiMatchRow]) -> int:
    """Return the next positive group-cell id for a match table.

    Args:
        rows: Existing manual match rows.

    Returns:
        One plus the highest existing group-cell id, or ``1`` for an empty
        match table.
    """
    if len(rows) == 0:
        return 1
    return max(row.group_cell_id for row in rows) + 1


def _rows_without_matched_group(
    rows: Sequence[ManualRoiMatchRow],
    fov_group_id: str,
    group_cell_id: int,
) -> tuple[ManualRoiMatchRow, ...]:
    """Return rows after removing one matched group in one FOV."""
    _validate_fov_group_id(fov_group_id)
    _validate_group_cell_id(group_cell_id)
    return tuple(
        row
        for row in rows
        if not (
            row.status == "matched"
            and row.fov_group_id == fov_group_id
            and row.group_cell_id == group_cell_id
        )
    )


def _shared_row_note(rows: tuple[ManualRoiMatchRow, ...]) -> str:
    """Return the common note for one matched group."""
    notes = {row.note for row in rows}
    return notes.pop() if len(notes) == 1 else ""


def _write_csv_rows(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, str]],
) -> None:
    """Write validated dictionaries to a required-header CSV file."""
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_required_csv_rows(
    path: Path,
    *,
    fieldnames: Sequence[str],
    table_name: str,
) -> tuple[dict[str, str], ...]:
    """Read rows from a CSV table that must contain known columns."""
    input_path = path.expanduser()
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            msg = f"{table_name} is empty: {input_path}"
            raise ValueError(msg)
        missing = tuple(field for field in fieldnames if field not in reader.fieldnames)
        if len(missing) > 0:
            msg = f"{table_name} missing columns {missing}: {input_path}"
            raise ValueError(msg)
        return tuple(
            {
                field: _required_csv_cell(row, field=field, input_path=input_path)
                for field in fieldnames
            }
            for row in reader
        )


def _required_csv_cell(
    row: Mapping[str, str | None],
    *,
    field: str,
    input_path: Path,
) -> str:
    """Return one required CSV cell value."""
    value = row[field]
    if value is None:
        msg = f"Manual group CSV row missing {field!r} value: {input_path}"
        raise ValueError(msg)
    return value


def _fov_row_to_csv(row: ManualFovGroupRow) -> dict[str, str]:
    """Convert one FOV group row into CSV strings."""
    return {
        "fov_group_id": _validate_fov_group_id(row.fov_group_id),
        "recording_path": str(_validate_recording_path(row.recording_path)),
        "note": row.note,
    }


def _fov_row_from_csv(values: Mapping[str, str]) -> ManualFovGroupRow:
    """Parse one FOV grouping CSV row."""
    return ManualFovGroupRow(
        fov_group_id=_validate_fov_group_id(values["fov_group_id"]),
        recording_path=_recording_path_from_csv(values["recording_path"]),
        note=values["note"],
    )


def _row_to_csv(row: ManualRoiMatchRow) -> dict[str, str]:
    """Convert one validated row into CSV strings."""
    _validate_group_cell_id(row.group_cell_id)
    _validate_status(row.status)
    return {
        "fov_group_id": row.fov_group_id,
        "group_cell_id": str(row.group_cell_id),
        "recording_path": str(_validate_recording_path(row.recording_path)),
        "roi_label": _validate_roi_label(row.roi_label),
        "status": row.status,
        "note": row.note,
    }


def _row_from_csv(
    values: Mapping[str, str],
    *,
    input_path: Path,
) -> ManualRoiMatchRow:
    """Parse and validate one CSV row."""
    try:
        group_cell_id = int(values["group_cell_id"])
    except ValueError as error:
        msg = (
            "Manual ROI match row has non-integer group_cell_id "
            f"{values['group_cell_id']!r}: {input_path}"
        )
        raise ValueError(msg) from error
    status = _parse_status(values["status"], input_path=input_path)
    return ManualRoiMatchRow(
        fov_group_id=values["fov_group_id"],
        group_cell_id=_validate_group_cell_id(group_cell_id),
        recording_path=_recording_path_from_csv(values["recording_path"]),
        roi_label=_validate_roi_label(values["roi_label"]),
        status=status,
        note=values["note"],
    )


def _parse_status(value: str, *, input_path: Path) -> ManualRoiMatchStatus:
    """Return a typed status from CSV text."""
    if value == "matched":
        return "matched"
    if value == "unmatched":
        return "unmatched"
    if value == "ambiguous":
        return "ambiguous"
    if value == "ignored":
        return "ignored"
    msg = f"Manual ROI match status {value!r} is invalid: {input_path}"
    raise ValueError(msg)


def _validate_status(status: ManualRoiMatchStatus) -> ManualRoiMatchStatus:
    """Return a status after validating it is one of the saved literals."""
    if status not in _STATUSES:
        msg = f"Manual ROI match status {status!r} is invalid."
        raise ValueError(msg)
    return status


def _validate_group_cell_id(group_cell_id: int) -> int:
    """Return a positive group-cell id."""
    if group_cell_id < 1:
        msg = f"Manual ROI group_cell_id must be positive; got {group_cell_id}."
        raise ValueError(msg)
    return group_cell_id


def _validate_recording_path(recording_path: Path) -> Path:
    """Return a non-empty recording path."""
    if str(recording_path) == "":
        msg = "Manual group recording_path cannot be empty."
        raise ValueError(msg)
    return recording_path.expanduser()


def _recording_path_from_csv(value: str) -> Path:
    """Return a validated recording path from raw CSV text."""
    if value.strip() == "":
        msg = "Manual group recording_path cannot be empty."
        raise ValueError(msg)
    return _validate_recording_path(Path(value))


def _validate_fov_group_id(fov_group_id: str) -> str:
    """Return a non-empty FOV group id."""
    if fov_group_id == "":
        msg = "Manual FOV group id cannot be empty."
        raise ValueError(msg)
    return fov_group_id


def _validate_roi_label(roi_label: str) -> str:
    """Return a non-empty ROI label."""
    if roi_label == "":
        msg = "Manual ROI match roi_label cannot be empty."
        raise ValueError(msg)
    return roi_label
