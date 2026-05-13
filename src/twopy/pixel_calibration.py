"""Resolve microscope pixel-size calibrations from auditable CSV rows.

Inputs: a CSV registry with rig, scanner mode, zoom, and measured microns per
pixel.
Outputs: direct, interpolated, or explicitly extrapolated pixel-size estimates.

The microscope metadata identifies acquisition settings, but physical pixel
size comes from lab calibration. This module keeps that calibration as plain
data and exposes pure helpers that analysis or ROI code can call without
reading UI state, converted recordings, or workflow objects.
"""

import csv
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

__all__ = [
    "DEFAULT_PIXEL_CALIBRATION_PATH",
    "PixelCalibrationResolution",
    "PixelCalibrationResolutionMethod",
    "PixelCalibrationRow",
    "load_pixel_calibrations",
    "resolve_pixel_size_um",
]

PixelCalibrationResolutionMethod = Literal["direct", "interpolated", "extrapolated"]
DEFAULT_PIXEL_CALIBRATION_PATH = Path(__file__).with_name("calibrations") / (
    "pixel_size.csv"
)
_REQUIRED_COLUMNS = ("rig", "mode", "scanner", "zoom", "pixel_size_um", "measured_on")
_ZOOM_ABS_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PixelCalibrationRow:
    """One measured microscope pixel-size calibration.

    Args:
        rig: Rig name, such as ``day`` or ``night``.
        mode: Microscope scanner mode number.
        scanner: Scanner family, such as ``galvo`` or ``res``.
        zoom: Microscope zoom setting.
        pixel_size_um: Measured microns per image pixel.
        measured_on: Date when this row was measured.

    Rows are intentionally small and match the CSV registry directly. The
    values are the source of truth for calibrated grid sizing.
    """

    rig: str
    mode: int
    scanner: str
    zoom: float
    pixel_size_um: float
    measured_on: date


@dataclass(frozen=True)
class PixelCalibrationResolution:
    """Resolved pixel-size estimate for one acquisition setting.

    Args:
        pixel_size_um: Microns per image pixel used by downstream code.
        method: Whether the value was directly measured, interpolated, or
            explicitly extrapolated.
        rig: Matched rig.
        mode: Matched microscope scanner mode.
        scanner: Matched scanner family.
        zoom: Requested zoom setting.
        reference_zooms: Measured zooms used to produce this value.
        reference_dates: Measurement dates used to produce this value.

    ``reference_zooms`` and ``reference_dates`` make interpolated and
    extrapolated values auditable in output logs without pulling workbook
    details into runtime configuration.
    """

    pixel_size_um: float
    method: PixelCalibrationResolutionMethod
    rig: str
    mode: int
    scanner: str
    zoom: float
    reference_zooms: tuple[float, ...]
    reference_dates: tuple[date, ...]


def load_pixel_calibrations(path: Path) -> tuple[PixelCalibrationRow, ...]:
    """Load measured pixel-size rows from a CSV registry.

    Args:
        path: CSV file with ``rig``, ``mode``, ``scanner``, ``zoom``,
            ``pixel_size_um``, and ``measured_on`` columns.

    Returns:
        Immutable tuple of validated calibration rows.

    Raises:
        FileNotFoundError: If the CSV is absent.
        ValueError: If required columns or row values are invalid.

    CSV keeps this data diffable and easy to audit while preserving a typed
    boundary before analysis code consumes it.
    """
    calibration_path = path.expanduser()
    with calibration_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_columns(reader.fieldnames, calibration_path)
        rows = tuple(
            _parse_calibration_row(row, calibration_path, line_number)
            for line_number, row in enumerate(reader, start=2)
        )

    if len(rows) == 0:
        msg = f"Pixel calibration CSV has no data rows: {calibration_path}"
        raise ValueError(msg)
    _validate_unique_calibration_rows(rows, calibration_path)
    return rows


def resolve_pixel_size_um(
    calibrations: Iterable[PixelCalibrationRow],
    *,
    rig: str,
    mode: int,
    scanner: str,
    zoom: float,
    allow_extrapolation: bool = False,
) -> PixelCalibrationResolution:
    """Resolve microns per pixel for one acquisition setting.

    Args:
        calibrations: Measured calibration rows.
        rig: Requested rig name.
        mode: Requested microscope scanner mode.
        scanner: Requested scanner family.
        zoom: Requested zoom setting.
        allow_extrapolation: Whether zooms outside the measured range may use
            the nearest two rows from the same rig/mode/scanner group.

    Returns:
        Pixel-size resolution with method and reference zooms.

    Raises:
        ValueError: If no matching group exists, direct rows conflict,
            interpolation is impossible, or extrapolation is required but not
            allowed.

    The fit is linear in pixels per micron, not microns per pixel, because the
    calibration data are close to linear in magnification. Rows are never shared
    across rigs, scanner modes, or scanner families.
    """
    requested_zoom = _positive_float("zoom", zoom)
    matching_rows = sorted(
        (
            row
            for row in calibrations
            if _matches_calibration_group(
                row,
                rig=rig,
                mode=mode,
                scanner=scanner,
            )
        ),
        key=lambda row: row.zoom,
    )
    if len(matching_rows) == 0:
        msg = (
            "No pixel calibration rows match "
            f"rig={rig!r}, mode={mode!r}, scanner={scanner!r}"
        )
        raise ValueError(msg)

    direct_rows = [
        row
        for row in matching_rows
        if math.isclose(row.zoom, requested_zoom, abs_tol=_ZOOM_ABS_TOLERANCE)
    ]
    if len(direct_rows) > 0:
        pixel_size_um = _resolved_direct_pixel_size(direct_rows)
        first_row = direct_rows[0]
        return PixelCalibrationResolution(
            pixel_size_um=pixel_size_um,
            method="direct",
            rig=first_row.rig,
            mode=first_row.mode,
            scanner=first_row.scanner,
            zoom=requested_zoom,
            reference_zooms=(first_row.zoom,),
            reference_dates=(first_row.measured_on,),
        )

    if len(matching_rows) < 2:
        msg = (
            "Pixel calibration interpolation needs at least two rows for "
            f"rig={rig!r}, mode={mode!r}, scanner={scanner!r}"
        )
        raise ValueError(msg)

    for lower_row, upper_row in zip(matching_rows, matching_rows[1:], strict=False):
        if lower_row.zoom < requested_zoom < upper_row.zoom:
            return _resolve_from_pair(
                lower_row,
                upper_row,
                requested_zoom,
                method="interpolated",
            )

    if not allow_extrapolation:
        lower_bound = matching_rows[0].zoom
        upper_bound = matching_rows[-1].zoom
        msg = (
            f"Pixel calibration zoom {requested_zoom} is outside measured "
            f"range [{lower_bound}, {upper_bound}] for rig={rig!r}, "
            f"mode={mode!r}, scanner={scanner!r}; set allow_extrapolation=True "
            "to use the nearest two measured zooms"
        )
        raise ValueError(msg)

    if requested_zoom < matching_rows[0].zoom:
        return _resolve_from_pair(
            matching_rows[0],
            matching_rows[1],
            requested_zoom,
            method="extrapolated",
        )
    return _resolve_from_pair(
        matching_rows[-2],
        matching_rows[-1],
        requested_zoom,
        method="extrapolated",
    )


def _resolve_from_pair(
    lower_row: PixelCalibrationRow,
    upper_row: PixelCalibrationRow,
    zoom: float,
    *,
    method: PixelCalibrationResolutionMethod,
) -> PixelCalibrationResolution:
    """Resolve one estimate from two measured rows in the same group.

    Args:
        lower_row: First measured calibration row.
        upper_row: Second measured calibration row.
        zoom: Requested zoom.
        method: Interpolation or extrapolation marker.

    Returns:
        Resolution with pixel size computed from linear pixels-per-micron fit.
    """
    pixel_size_um = _linear_pixel_size_um(lower_row, upper_row, zoom)
    return PixelCalibrationResolution(
        pixel_size_um=pixel_size_um,
        method=method,
        rig=lower_row.rig,
        mode=lower_row.mode,
        scanner=lower_row.scanner,
        zoom=zoom,
        reference_zooms=(lower_row.zoom, upper_row.zoom),
        reference_dates=(lower_row.measured_on, upper_row.measured_on),
    )


def _linear_pixel_size_um(
    first_row: PixelCalibrationRow,
    second_row: PixelCalibrationRow,
    zoom: float,
) -> float:
    """Interpolate or extrapolate microns per pixel from two measured zooms.

    Args:
        first_row: First measured calibration row.
        second_row: Second measured calibration row.
        zoom: Requested zoom.

    Returns:
        Microns per pixel from a linear pixels-per-micron fit.
    """
    if math.isclose(first_row.zoom, second_row.zoom, abs_tol=_ZOOM_ABS_TOLERANCE):
        msg = f"Duplicate calibration zoom cannot define a line: {first_row.zoom}"
        raise ValueError(msg)
    first_pixels_per_um = 1.0 / first_row.pixel_size_um
    second_pixels_per_um = 1.0 / second_row.pixel_size_um
    slope = (second_pixels_per_um - first_pixels_per_um) / (
        second_row.zoom - first_row.zoom
    )
    resolved_pixels_per_um = first_pixels_per_um + slope * (zoom - first_row.zoom)
    if resolved_pixels_per_um <= 0:
        msg = f"Resolved pixel calibration is non-positive at zoom {zoom}"
        raise ValueError(msg)
    return 1.0 / resolved_pixels_per_um


def _matches_calibration_group(
    row: PixelCalibrationRow,
    *,
    rig: str,
    mode: int,
    scanner: str,
) -> bool:
    """Return whether one row matches a requested calibration group.

    Args:
        row: Candidate calibration row.
        rig: Requested rig.
        mode: Requested scanner mode.
        scanner: Requested scanner family.

    Returns:
        ``True`` when all group keys match exactly after text normalization.
    """
    return (
        _normalized_text(row.rig) == _normalized_text(rig)
        and row.mode == mode
        and _normalized_text(row.scanner) == _normalized_text(scanner)
    )


def _resolved_direct_pixel_size(rows: list[PixelCalibrationRow]) -> float:
    """Return one direct measured pixel size from duplicate zoom rows.

    Args:
        rows: Direct calibration rows at the requested zoom.

    Returns:
        Microns per pixel.

    Raises:
        ValueError: If duplicate direct rows disagree.
    """
    first_pixel_size = rows[0].pixel_size_um
    for row in rows[1:]:
        if not math.isclose(
            row.pixel_size_um,
            first_pixel_size,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            msg = f"Conflicting direct pixel calibration rows at zoom {row.zoom}"
            raise ValueError(msg)
    return first_pixel_size


def _validate_columns(fieldnames: Sequence[str] | None, path: Path) -> None:
    """Confirm a calibration CSV has all required columns.

    Args:
        fieldnames: Header columns reported by ``csv.DictReader``.
        path: CSV path, used for clear errors.
    """
    if fieldnames is None:
        msg = f"Pixel calibration CSV is missing a header row: {path}"
        raise ValueError(msg)
    missing = [column for column in _REQUIRED_COLUMNS if column not in fieldnames]
    if len(missing) > 0:
        msg = f"Pixel calibration CSV {path} is missing columns: {', '.join(missing)}"
        raise ValueError(msg)


def _validate_unique_calibration_rows(
    rows: tuple[PixelCalibrationRow, ...],
    path: Path,
) -> None:
    """Reject conflicting duplicate measurements in one calibration registry.

    Args:
        rows: Parsed calibration rows.
        path: CSV path, used for clear errors.

    Raises:
        ValueError: If the same rig/mode/scanner/zoom appears with a different
            pixel size or measurement date.
    """
    value_by_key: dict[tuple[str, int, str, float], tuple[float, date]] = {}
    for row in rows:
        key = (
            _normalized_text(row.rig),
            row.mode,
            _normalized_text(row.scanner),
            row.zoom,
        )
        existing_value = value_by_key.get(key)
        if existing_value is None:
            value_by_key[key] = (row.pixel_size_um, row.measured_on)
            continue
        existing_pixel_size, existing_measured_on = existing_value
        if (
            not math.isclose(
                existing_pixel_size,
                row.pixel_size_um,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            or row.measured_on != existing_measured_on
        ):
            msg = (
                f"Pixel calibration CSV {path} has conflicting duplicate rows "
                f"for rig={row.rig!r}, mode={row.mode!r}, "
                f"scanner={row.scanner!r}, zoom={row.zoom!r}"
            )
            raise ValueError(msg)


def _parse_calibration_row(
    row: dict[str, str | None],
    path: Path,
    line_number: int,
) -> PixelCalibrationRow:
    """Parse one CSV row into a validated calibration row.

    Args:
        row: Raw CSV row.
        path: CSV path, used for clear errors.
        line_number: One-based CSV line number.

    Returns:
        Typed calibration row.
    """
    rig = _required_text(row, "rig", path, line_number)
    scanner = _required_text(row, "scanner", path, line_number)
    mode = _required_int(row, "mode", path, line_number)
    zoom = _required_positive_float(row, "zoom", path, line_number)
    pixel_size_um = _required_positive_float(
        row,
        "pixel_size_um",
        path,
        line_number,
    )
    measured_on = _required_date(row, "measured_on", path, line_number)
    return PixelCalibrationRow(
        rig=rig,
        mode=mode,
        scanner=scanner,
        zoom=zoom,
        pixel_size_um=pixel_size_um,
        measured_on=measured_on,
    )


def _required_text(
    row: dict[str, str | None],
    key: str,
    path: Path,
    line_number: int,
) -> str:
    """Read one required non-empty text cell.

    Args:
        row: Raw CSV row.
        key: Column to read.
        path: CSV path, used for clear errors.
        line_number: One-based CSV line number.

    Returns:
        Stripped text value.
    """
    value = row.get(key)
    if value is None or value.strip() == "":
        msg = f"Pixel calibration {path}:{line_number} has empty {key!r}"
        raise ValueError(msg)
    return value.strip()


def _required_int(
    row: dict[str, str | None],
    key: str,
    path: Path,
    line_number: int,
) -> int:
    """Read one required integer cell.

    Args:
        row: Raw CSV row.
        key: Column to read.
        path: CSV path, used for clear errors.
        line_number: One-based CSV line number.

    Returns:
        Integer value.
    """
    value = _required_text(row, key, path, line_number)
    try:
        return int(value)
    except ValueError as error:
        msg = f"Pixel calibration {path}:{line_number} has invalid {key!r}: {value!r}"
        raise ValueError(msg) from error


def _required_positive_float(
    row: dict[str, str | None],
    key: str,
    path: Path,
    line_number: int,
) -> float:
    """Read one required positive float cell.

    Args:
        row: Raw CSV row.
        key: Column to read.
        path: CSV path, used for clear errors.
        line_number: One-based CSV line number.

    Returns:
        Positive float value.
    """
    value = _required_text(row, key, path, line_number)
    try:
        return _positive_float(key, float(value))
    except ValueError as error:
        msg = f"Pixel calibration {path}:{line_number} has invalid {key!r}: {value!r}"
        raise ValueError(msg) from error


def _required_date(
    row: dict[str, str | None],
    key: str,
    path: Path,
    line_number: int,
) -> date:
    """Read one required ISO date cell.

    Args:
        row: Raw CSV row.
        key: Column to read.
        path: CSV path, used for clear errors.
        line_number: One-based CSV line number.

    Returns:
        Parsed date value.
    """
    value = _required_text(row, key, path, line_number)
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        msg = (
            f"Pixel calibration {path}:{line_number} has invalid {key!r}: "
            f"{value!r}; expected YYYY-MM-DD"
        )
        raise ValueError(msg) from error


def _positive_float(key: str, value: float) -> float:
    """Validate one finite positive float.

    Args:
        key: Field name, used for clear errors.
        value: Candidate numeric value.

    Returns:
        The validated value.
    """
    if not math.isfinite(value) or value <= 0:
        msg = f"{key} must be a finite positive number; got {value!r}"
        raise ValueError(msg)
    return value


def _normalized_text(value: str) -> str:
    """Return a case-insensitive match key for calibration text.

    Args:
        value: Candidate text value.

    Returns:
        Stripped and case-folded text.
    """
    return value.strip().casefold()
