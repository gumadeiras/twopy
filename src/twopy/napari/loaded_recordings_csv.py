"""CSV helpers for reloading a napari session's recording list.

Inputs: loaded napari recording entries or a plain CSV list of recording paths.
Outputs: auditable CSV rows and typed paths that the Load tab can pass through
the normal recording loader.
"""

import csv
from collections.abc import Sequence
from pathlib import Path

from twopy.filenames import RECORDING_DATA_FILENAME
from twopy.napari.session import LoadedNapariRecording

__all__ = [
    "LOADED_RECORDINGS_CSV_COLUMNS",
    "load_recording_paths_csv",
    "write_loaded_recordings_csv",
]

LOADED_RECORDINGS_CSV_COLUMNS = ("recording_path", "recording_data_path")
_SOURCE_PATH_COLUMNS = ("recording_path", "path", "recording", "folder")


def write_loaded_recordings_csv(
    recordings: Sequence[LoadedNapariRecording],
    path: Path,
) -> None:
    """Write the current loaded-recordings list to a reusable CSV file.

    Args:
        recordings: Loaded napari recordings in the visible list order.
        path: Destination CSV path.

    Returns:
        None.

    The primary ``recording_path`` column stores the path shown in the loaded
    list because that is the scientist-facing session list. The
    ``recording_data_path`` column stores the route resolved when the recording
    was loaded, so CSV writing cannot rediscover a different destination later.
    """
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LOADED_RECORDINGS_CSV_COLUMNS)
        writer.writeheader()
        for recording in recordings:
            writer.writerow(
                {
                    "recording_path": str(
                        recording.recording.source_session_dir.expanduser(),
                    ),
                    "recording_data_path": str(
                        recording.output_route.publish_root / RECORDING_DATA_FILENAME,
                    ),
                },
            )


def load_recording_paths_csv(path: Path) -> tuple[Path, ...]:
    """Read recording paths from a CSV selected in the Load tab.

    Args:
        path: CSV file with recording path columns or one path per row.

    Returns:
        Tuple of recording paths in file order.

    The loader accepts both twopy's exported headered CSV and simple hand-made
    one-column path lists so batch loading remains easy to inspect and edit.
    Twopy-exported CSVs reopen valid ``recording_data_path`` files directly and
    fall back to source-style path columns when that file is missing or invalid.
    """
    csv_path = path.expanduser()
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.reader(csv_file))
    if len(rows) == 0:
        return ()

    normalized_header = tuple(column.strip().lower() for column in rows[0])
    column_indexes = {
        column: normalized_header.index(column)
        for column in (*_SOURCE_PATH_COLUMNS, "recording_data_path")
        if column in normalized_header
    }
    if len(column_indexes) == 0:
        data_rows = rows
        first_data_line = 1
    else:
        data_rows = rows[1:]
        first_data_line = 2

    fallback_index = 0 if len(column_indexes) == 0 else -1
    paths: list[Path] = []
    for line_number, row in enumerate(data_rows, start=first_data_line):
        if len(row) == 0 or all(cell.strip() == "" for cell in row):
            continue
        recording_data_index = column_indexes.get("recording_data_path")
        if recording_data_index is not None and recording_data_index < len(row):
            recording_data_path = Path(row[recording_data_index].strip()).expanduser()
            if (
                recording_data_path.name == RECORDING_DATA_FILENAME
                and recording_data_path.is_file()
            ):
                paths.append(recording_data_path)
                continue

        for column in _SOURCE_PATH_COLUMNS:
            column_index = column_indexes.get(column, fallback_index)
            if (
                column_index >= 0
                and column_index < len(row)
                and row[column_index].strip()
            ):
                paths.append(Path(row[column_index].strip()).expanduser())
                break
        else:
            msg = (
                f"Recording CSV {csv_path} has no recording path on line {line_number}."
            )
            raise ValueError(msg)
    return tuple(paths)
