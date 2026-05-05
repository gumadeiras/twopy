"""Inspect all recognized files in a two-photon recording session.

Inputs: a session folder that follows the twopy data layout.
Outputs: typed summaries for MATLAB, TIFF, text, CSV, ZIP, and other files.

The goal is fast orientation: users and tests can see what the recording
contains without loading the full movie or committing to an analysis method.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import tifffile

from twopy.matlab import MatlabFileSummary, inspect_mat_file

__all__ = [
    "FileInspection",
    "SessionInspection",
    "TiffSummary",
    "inspect_recording_files",
]

FileKind = Literal["matlab", "tiff", "text", "csv", "zip", "other"]


@dataclass(frozen=True)
class TiffSummary:
    """Small metadata summary for one TIFF movie.

    Inputs: TIFF file path.
    Outputs: series count plus shape and dtype for each TIFF series.

    This avoids loading image pixels while still proving the movie can be opened
    and inspected by Python.
    """

    series_count: int
    series_shapes: tuple[tuple[int, ...], ...]
    series_dtypes: tuple[str, ...]


@dataclass(frozen=True)
class FileInspection:
    """Inspection result for one file inside a recording.

    Inputs: one file path under the session directory.
    Outputs: file kind, size, and type-specific summary fields.

    Optional fields stay ``None`` unless they apply to that file kind. This keeps
    the object simple and avoids a hierarchy before the data model is stable.
    """

    relative_path: Path
    kind: FileKind
    size_bytes: int
    matlab: MatlabFileSummary | None
    tiff: TiffSummary | None
    text_preview: tuple[str, ...] | None
    zip_entries: tuple[str, ...] | None


@dataclass(frozen=True)
class SessionInspection:
    """Inspection result for all recognized files in a recording.

    Inputs: session directory.
    Outputs: sorted file inspection entries.

    This is the Python object that the GUI can use to show what twopy found.
    """

    session_dir: Path
    files: tuple[FileInspection, ...]


def inspect_recording_files(session_dir: Path) -> SessionInspection:
    """Inspect all files below one recording session directory.

    Args:
        session_dir: Recording directory to inspect recursively.

    Returns:
        A sorted session inspection with one entry per file.

    Raises:
        FileNotFoundError: If ``session_dir`` is missing.

    The function inspects metadata only: MATLAB variable summaries, TIFF series
    metadata, short text previews, and ZIP entry names.
    """
    root = session_dir.expanduser()
    if not root.is_dir():
        msg = f"Missing recording directory: {root}"
        raise FileNotFoundError(msg)

    inspections = tuple(
        _inspect_file(root, path)
        for path in sorted(
            candidate for candidate in root.rglob("*") if candidate.is_file()
        )
    )

    return SessionInspection(session_dir=root, files=inspections)


def _inspect_file(root: Path, path: Path) -> FileInspection:
    """Inspect one file using the simplest reader for its suffix.

    Args:
        root: Session root used to compute relative paths.
        path: File to inspect.

    Returns:
        A file inspection with metadata for recognized file types.

    The suffix switch is intentionally boring and explicit. It is easier to
    audit than a registry while the project has only a handful of file types.
    """
    suffix = path.suffix.lower()
    size_bytes = path.stat().st_size

    if suffix == ".mat":
        return FileInspection(
            relative_path=path.relative_to(root),
            kind="matlab",
            size_bytes=size_bytes,
            matlab=inspect_mat_file(path),
            tiff=None,
            text_preview=None,
            zip_entries=None,
        )

    if suffix in {".tif", ".tiff"}:
        return FileInspection(
            relative_path=path.relative_to(root),
            kind="tiff",
            size_bytes=size_bytes,
            matlab=None,
            tiff=_inspect_tiff(path),
            text_preview=None,
            zip_entries=None,
        )

    if suffix == ".zip":
        return FileInspection(
            relative_path=path.relative_to(root),
            kind="zip",
            size_bytes=size_bytes,
            matlab=None,
            tiff=None,
            text_preview=None,
            zip_entries=_inspect_zip(path),
        )

    if suffix in {".txt", ".csv", ".batch"}:
        return FileInspection(
            relative_path=path.relative_to(root),
            kind="csv" if suffix == ".csv" else "text",
            size_bytes=size_bytes,
            matlab=None,
            tiff=None,
            text_preview=_text_preview(path),
            zip_entries=None,
        )

    return FileInspection(
        relative_path=path.relative_to(root),
        kind="other",
        size_bytes=size_bytes,
        matlab=None,
        tiff=None,
        text_preview=None,
        zip_entries=None,
    )


def _inspect_tiff(path: Path) -> TiffSummary:
    """Inspect TIFF series metadata without reading the full movie.

    Args:
        path: TIFF movie path.

    Returns:
        Series count, shape, and dtype metadata.

    tifffile exposes series metadata lazily, which is enough for an initial GUI
    summary and avoids loading hundreds of megabytes into memory.
    """
    with tifffile.TiffFile(path) as tif:
        shapes = tuple(
            tuple(int(length) for length in series.shape) for series in tif.series
        )
        dtypes = tuple(str(series.dtype) for series in tif.series)
        return TiffSummary(
            series_count=len(tif.series),
            series_shapes=shapes,
            series_dtypes=dtypes,
        )


def _text_preview(path: Path, line_count: int = 5) -> tuple[str, ...]:
    """Read the first few lines of a text-like file.

    Args:
        path: Text file path.
        line_count: Maximum number of lines to return.

    Returns:
        Preview lines without trailing newline characters.

    A short preview is enough to recognize metadata files without treating them
    as a stable parsed format too early.
    """
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as text_file:
        for _ in range(line_count):
            line = text_file.readline()
            if line == "":
                break
            lines.append(line.rstrip("\n"))
    return tuple(lines)


def _inspect_zip(path: Path) -> tuple[str, ...]:
    """List files contained inside a ZIP archive.

    Args:
        path: ZIP archive path.

    Returns:
        Sorted entry names stored in the archive.

    The archive is not extracted. Listing entries is enough for inspection and
    avoids writing data back into the source recording folder.
    """
    with ZipFile(path) as archive:
        return tuple(sorted(archive.namelist()))
