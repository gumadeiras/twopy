"""Inspect all recognized files in a two-photon recording session.

Inputs: a session folder that follows the twopy data layout.
Outputs: typed summaries for MATLAB, TIFF, text, CSV, ZIP, and other files.

The goal is fast orientation: users and tests can see what the recording
contains without loading the full movie or committing to an analysis method.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from zipfile import ZipFile

import tifffile

from twopy.matlab import MatlabFileSummary, inspect_mat_file

__all__ = [
    "FileInspection",
    "SessionInspection",
    "TIFF_METADATA_CSV_COLUMNS",
    "TiffSummary",
    "inspect_recording_files",
    "write_tiff_metadata_csv",
]

FileKind = Literal["matlab", "tiff", "text", "csv", "zip", "other"]

TIFF_METADATA_CSV_COLUMNS = (
    "relative_path",
    "size_bytes",
    "series_count",
    "series_shapes",
    "series_dtypes",
    "page_count",
    "first_page_shape",
    "first_page_dtype",
    "image_width",
    "image_length",
    "bits_per_sample",
    "samples_per_pixel",
    "x_resolution",
    "y_resolution",
    "resolution_unit",
    "image_description_chars",
    "scanimage_config_name",
    "scanimage_software_version",
    "scanimage_acq_lines_per_frame",
    "scanimage_acq_pixels_per_line",
    "scanimage_acq_number_of_frames",
    "scanimage_acq_number_of_channels_save",
    "scanimage_acq_frame_rate",
    "scanimage_acq_zoom_factor",
    "scanimage_acq_pixel_time",
    "scanimage_acq_ms_per_line",
    "scanimage_acq_z_step_size",
    "scanimage_acq_scan_angle_multiplier_fast",
    "scanimage_acq_scan_angle_multiplier_slow",
    "scanimage_acq_scan_rotation",
    "scanimage_acq_scan_shift_fast",
    "scanimage_acq_scan_shift_slow",
    "scanimage_acq_x_step",
    "scanimage_acq_y_step",
    "scanimage_motor_abs_x_position",
    "scanimage_motor_abs_y_position",
    "scanimage_motor_abs_z_position",
)

_SCANIMAGE_CSV_KEY_BY_COLUMN = {
    "scanimage_config_name": "configName",
    "scanimage_software_version": "software.version",
    "scanimage_acq_lines_per_frame": "acq.linesPerFrame",
    "scanimage_acq_pixels_per_line": "acq.pixelsPerLine",
    "scanimage_acq_number_of_frames": "acq.numberOfFrames",
    "scanimage_acq_number_of_channels_save": "acq.numberOfChannelsSave",
    "scanimage_acq_frame_rate": "acq.frameRate",
    "scanimage_acq_zoom_factor": "acq.zoomFactor",
    "scanimage_acq_pixel_time": "acq.pixelTime",
    "scanimage_acq_ms_per_line": "acq.msPerLine",
    "scanimage_acq_z_step_size": "acq.zStepSize",
    "scanimage_acq_scan_angle_multiplier_fast": "acq.scanAngleMultiplierFast",
    "scanimage_acq_scan_angle_multiplier_slow": "acq.scanAngleMultiplierSlow",
    "scanimage_acq_scan_rotation": "acq.scanRotation",
    "scanimage_acq_scan_shift_fast": "acq.scanShiftFast",
    "scanimage_acq_scan_shift_slow": "acq.scanShiftSlow",
    "scanimage_acq_x_step": "acq.xstep",
    "scanimage_acq_y_step": "acq.ystep",
    "scanimage_motor_abs_x_position": "motor.absXPosition",
    "scanimage_motor_abs_y_position": "motor.absYPosition",
    "scanimage_motor_abs_z_position": "motor.absZPosition",
}


@dataclass(frozen=True)
class TiffSummary:
    """Small metadata summary for one TIFF movie.

    Inputs: TIFF file path.
    Outputs: TIFF layout, tags, and parsed ScanImage metadata.

    This avoids loading image pixels while still proving the movie can be opened
    and inspected by Python. ``scanimage_state`` is a flat dictionary keyed by
    paths such as ``acq.zoomFactor`` for easy script access.
    """

    series_count: int
    series_shapes: tuple[tuple[int, ...], ...]
    series_dtypes: tuple[str, ...]
    page_count: int
    first_page_shape: tuple[int, ...]
    first_page_dtype: str
    tags: dict[str, str]
    scanimage_state: dict[str, str]


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


def inspect_recording_files(
    session_dir: Path,
    *,
    tiff_metadata_csv: Path | None = None,
) -> SessionInspection:
    """Inspect all files below one recording session directory.

    Args:
        session_dir: Recording directory to inspect recursively.
        tiff_metadata_csv: Optional CSV path that receives one row per TIFF
            movie with selected TIFF tags and ScanImage metadata.

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

    inspection = SessionInspection(session_dir=root, files=inspections)
    if tiff_metadata_csv is not None:
        write_tiff_metadata_csv(inspection, tiff_metadata_csv)
    return inspection


def write_tiff_metadata_csv(inspection: SessionInspection, path: Path) -> None:
    """Write selected TIFF and ScanImage metadata to a CSV file.

    Args:
        inspection: Session inspection returned by ``inspect_recording_files``.
        path: CSV file path to write.

    Returns:
        None. The function writes a human-readable metadata table to disk.

    The CSV keeps the high-value fields stable while the full parsed ScanImage
    dictionary remains available on each ``TiffSummary`` Python object.
    """
    output_path = path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TIFF_METADATA_CSV_COLUMNS)
        writer.writeheader()
        for file in inspection.files:
            if file.tiff is not None:
                writer.writerow(_tiff_csv_row(file))


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
        first_page = cast(tifffile.TiffPage, tif.pages[0])
        tags = _selected_tiff_tags(first_page)
        return TiffSummary(
            series_count=len(tif.series),
            series_shapes=shapes,
            series_dtypes=dtypes,
            page_count=len(tif.pages),
            first_page_shape=tuple(int(length) for length in first_page.shape),
            first_page_dtype=str(first_page.dtype),
            tags=tags,
            scanimage_state=_parse_scanimage_description(
                tags.get("ImageDescription", ""),
            ),
        )


def _selected_tiff_tags(page: tifffile.TiffPage) -> dict[str, str]:
    """Read the TIFF tags that are useful for imaging metadata.

    Args:
        page: First TIFF page from the movie.

    Returns:
        Mapping from TIFF tag name to string value.

    ScanImage writes the same metadata onto each page in the observed files, so
    the first page is enough and avoids iterating thousands of frames.
    """
    tag_names = (
        "ImageWidth",
        "ImageLength",
        "BitsPerSample",
        "SamplesPerPixel",
        "XResolution",
        "YResolution",
        "ResolutionUnit",
        "ImageDescription",
        "Software",
        "DateTime",
    )
    tags: dict[str, str] = {}
    for name in tag_names:
        tag = page.tags.get(name)
        if tag is not None:
            tags[name] = _metadata_value_to_text(cast(object, tag.value))
    return tags


def _parse_scanimage_description(description: str) -> dict[str, str]:
    """Parse ScanImage ``state.*`` lines from a TIFF ImageDescription.

    Args:
        description: Raw TIFF ``ImageDescription`` tag value.

    Returns:
        Flat metadata dictionary without the leading ``state.`` prefix.

    ScanImage stores one assignment per line. Keeping values as strings avoids
    guessing units or numeric intent too early while still making fields easy to
    search and write to CSV.
    """
    state: dict[str, str] = {}
    for raw_line in description.splitlines():
        line = raw_line.strip()
        if not line.startswith("state.") or "=" not in line:
            continue
        key, value = line.removeprefix("state.").split("=", 1)
        state[key] = _clean_scanimage_value(value)
    return state


def _clean_scanimage_value(value: str) -> str:
    """Normalize one ScanImage assignment value for Python and CSV use.

    Args:
        value: Raw value text from one ``state.*`` line.

    Returns:
        Value with surrounding whitespace and simple MATLAB string quotes removed.
    """
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == "'" and stripped[-1] == "'":
        return stripped[1:-1]
    return stripped


def _metadata_value_to_text(value: object) -> str:
    """Convert TIFF tag values into stable text.

    Args:
        value: TIFF tag value returned by tifffile.

    Returns:
        Compact string representation for Python summaries and CSV output.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        return " ".join(str(part) for part in value)
    return str(value)


def _tiff_csv_row(file: FileInspection) -> dict[str, str]:
    """Convert one TIFF file inspection into a CSV row.

    Args:
        file: File inspection with a non-``None`` TIFF summary.

    Returns:
        Dictionary keyed by ``TIFF_METADATA_CSV_COLUMNS``.
    """
    if file.tiff is None:
        msg = "Expected TIFF summary when writing TIFF metadata CSV"
        raise ValueError(msg)

    tiff = file.tiff
    row = {
        "relative_path": str(file.relative_path),
        "size_bytes": str(file.size_bytes),
        "series_count": str(tiff.series_count),
        "series_shapes": ";".join(
            _shape_to_text(shape) for shape in tiff.series_shapes
        ),
        "series_dtypes": ";".join(tiff.series_dtypes),
        "page_count": str(tiff.page_count),
        "first_page_shape": _shape_to_text(tiff.first_page_shape),
        "first_page_dtype": tiff.first_page_dtype,
        "image_width": tiff.tags.get("ImageWidth", ""),
        "image_length": tiff.tags.get("ImageLength", ""),
        "bits_per_sample": tiff.tags.get("BitsPerSample", ""),
        "samples_per_pixel": tiff.tags.get("SamplesPerPixel", ""),
        "x_resolution": tiff.tags.get("XResolution", ""),
        "y_resolution": tiff.tags.get("YResolution", ""),
        "resolution_unit": tiff.tags.get("ResolutionUnit", ""),
        "image_description_chars": str(len(tiff.tags.get("ImageDescription", ""))),
    }
    for column, key in _SCANIMAGE_CSV_KEY_BY_COLUMN.items():
        row[column] = tiff.scanimage_state.get(key, "")
    return row


def _shape_to_text(shape: tuple[int, ...]) -> str:
    """Format an array shape for one CSV cell.

    Args:
        shape: Tuple of array dimensions.

    Returns:
        Shape dimensions joined by ``x``.
    """
    return "x".join(str(length) for length in shape)


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
