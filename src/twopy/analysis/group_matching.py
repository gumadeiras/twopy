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
    "ManualRoiMatchRow",
    "ManualRoiMatchStatus",
    "append_manual_roi_match_rows",
    "load_manual_fov_group_rows",
    "load_manual_roi_match_rows",
    "make_manual_fov_group_rows",
    "make_manual_roi_match_rows",
    "next_group_cell_id",
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


def make_manual_fov_group_rows(
    recording_groups: Mapping[Path, str],
    *,
    note: str = "",
) -> tuple[ManualFovGroupRow, ...]:
    """Create validated FOV-group rows from recording assignments.

    Args:
        recording_groups: Mapping from recording path to non-empty FOV group
            id.
        note: Optional free-text note stored on each row.

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
            note=note,
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
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_FOV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(_fov_row_to_csv(row))


def load_manual_fov_group_rows(path: Path) -> tuple[ManualFovGroupRow, ...]:
    """Load manual FOV grouping rows from a CSV file.

    Args:
        path: CSV file written by ``save_manual_fov_group_rows``.

    Returns:
        Tuple of validated FOV grouping rows.

    Raises:
        ValueError: If the file contains missing columns or invalid values.
    """
    input_path = path.expanduser()
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            msg = f"Manual FOV group file is empty: {input_path}"
            raise ValueError(msg)
        missing = tuple(
            field for field in _FOV_FIELDNAMES if field not in reader.fieldnames
        )
        if len(missing) > 0:
            msg = f"Manual FOV group file missing columns {missing}: {input_path}"
            raise ValueError(msg)
        rows = tuple(_fov_row_from_csv(row) for row in reader)
    return rows


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
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_csv(row))


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
    with input_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            msg = f"Manual ROI match file is empty: {input_path}"
            raise ValueError(msg)
        missing = tuple(
            field for field in _FIELDNAMES if field not in reader.fieldnames
        )
        if len(missing) > 0:
            msg = f"Manual ROI match file missing columns {missing}: {input_path}"
            raise ValueError(msg)
        rows = tuple(_row_from_csv(row, input_path=input_path) for row in reader)
    return rows


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
