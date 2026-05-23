"""CSV helpers for reloading a napari session's recording list.

Inputs: loaded napari recording entries or a plain CSV list of recording paths.
Outputs: auditable CSV rows and typed paths that the Load tab can pass through
the normal recording loader.
"""

import csv
from collections.abc import Sequence
from pathlib import Path

from twopy.config import (
    DEFAULT_CONFIG_PATH,
    TwopyConfig,
    load_config,
    resolve_analysis_output_dir,
)
from twopy.converted import RecordingData
from twopy.filenames import RECORDING_DATA_FILENAME
from twopy.napari.session import LoadedNapariRecording

__all__ = [
    "LOADED_RECORDINGS_CSV_COLUMNS",
    "load_recording_paths_csv",
    "write_loaded_recordings_csv",
]

LOADED_RECORDINGS_CSV_COLUMNS = ("recording_path", "recording_data_path")
_SUPPORTED_PATH_COLUMNS = (
    "recording_path",
    "path",
    "recording",
    "folder",
    "recording_data_path",
)


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
    ``recording_data_path`` column stores the durable published HDF5 path when
    config routing can resolve one, so local cache paths stay an implementation
    detail.
    """
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        config = load_config(DEFAULT_CONFIG_PATH)
    except FileNotFoundError:
        config = None
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
                        _recording_data_csv_path(recording.recording, config),
                    ),
                },
            )


def load_recording_paths_csv(path: Path) -> tuple[Path, ...]:
    """Read recording paths from a CSV selected in the Load tab.

    Args:
        path: CSV file with a ``recording_path`` column or one path per row.

    Returns:
        Tuple of recording paths in file order.

    The loader accepts both twopy's exported headered CSV and simple hand-made
    one-column path lists so batch loading remains easy to inspect and edit.
    """
    csv_path = path.expanduser()
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.reader(csv_file))
    if len(rows) == 0:
        return ()

    path_column_index = _path_column_index(rows[0])
    if path_column_index is None:
        path_column_index = 0
        data_rows = rows
        first_data_line = 1
    else:
        data_rows = rows[1:]
        first_data_line = 2

    paths: list[Path] = []
    for line_number, row in enumerate(data_rows, start=first_data_line):
        if len(row) == 0 or all(cell.strip() == "" for cell in row):
            continue
        if path_column_index >= len(row) or row[path_column_index].strip() == "":
            msg = (
                f"Recording CSV {csv_path} has no recording path on line {line_number}."
            )
            raise ValueError(msg)
        paths.append(Path(row[path_column_index].strip()).expanduser())
    return tuple(paths)


def _path_column_index(header: Sequence[str]) -> int | None:
    """Return the first supported recording-path column index."""
    normalized_header = tuple(column.strip().lower() for column in header)
    for column_name in _SUPPORTED_PATH_COLUMNS:
        if column_name in normalized_header:
            return normalized_header.index(column_name)
    return None


def _recording_data_csv_path(
    recording: RecordingData,
    config: TwopyConfig | None,
) -> Path:
    """Return the durable ``recording_data.h5`` path for one CSV row."""
    if config is None:
        return recording.path.expanduser()
    try:
        return (
            resolve_analysis_output_dir(config, recording.source_session_dir)
            / RECORDING_DATA_FILENAME
        )
    except ValueError:
        return recording.path.expanduser()
