"""Resolve calibration profile fields from recording metadata and mappings.

Inputs: converted acquisition/run metadata plus an auditable CSV of historical
ScanImage config-to-mode mappings.
Outputs: optional rig, mode, and scanner fields that can prefill pixel-size
calibration choices without coupling analysis code to napari widgets.

Physical pixel size still comes from ``twopy.pixel_calibration``. This module
only answers which calibration group a recording appears to belong to, using
direct metadata first and explicit lab mappings only for fields ScanImage does
not store directly.
"""

import csv
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from twopy.pixel_calibration import PixelCalibrationRow

__all__ = [
    "DEFAULT_PIXEL_CALIBRATION_PROFILE_PATH",
    "PixelCalibrationGroup",
    "PixelCalibrationProfile",
    "PixelCalibrationProfileMapping",
    "PixelCalibrationProfileSource",
    "load_pixel_calibration_profile_mappings",
    "resolve_pixel_calibration_profile",
    "select_pixel_calibration_group",
]

PixelCalibrationProfileSource = Literal[
    "metadata",
    "metadata+mapping",
    "unresolved",
]
DEFAULT_PIXEL_CALIBRATION_PROFILE_PATH = (
    Path(__file__).with_name(
        "calibrations",
    )
    / "pixel_calibration_profiles.csv"
)
_REQUIRED_MAPPING_COLUMNS = (
    "config_name",
    "mode",
    "scanner",
    "pixels_per_line",
    "lines_per_frame",
    "pixel_time",
    "ms_per_line",
    "scan_angle_multiplier_fast",
    "scan_angle_multiplier_slow",
)
_ACQUISITION_FIELD_KEYS = {
    "pixels_per_line": ("acq.pixelsPerLine", "SI.hRoiManager.pixelsPerLine"),
    "lines_per_frame": ("acq.linesPerFrame", "SI.hRoiManager.linesPerFrame"),
    "pixel_time": ("acq.pixelTime", "SI.hScan2D.scanPixelTimeMean"),
    "ms_per_line": (
        "acq.msPerLine",
        "SI.hRoiManager.linePeriodMs",
        "SI.hRoiManager.linePeriod",
    ),
    "scan_angle_multiplier_fast": (
        "acq.scanAngleMultiplierFast",
        "SI.hRoiManager.scanAngleMultiplierFast",
    ),
    "scan_angle_multiplier_slow": (
        "acq.scanAngleMultiplierSlow",
        "SI.hRoiManager.scanAngleMultiplierSlow",
    ),
}
_ACQUISITION_FIELD_SCALE = {
    "SI.hRoiManager.linePeriod": 1000.0,
}
_CONFIG_NAME_KEYS = (
    "configName",
    "scanimage.configName",
    "SI.hConfigurationSaver.cfgFilename",
)
_SCANNER_KEYS = (
    "scanner",
    "scanimage.scanner",
    "scanimage.imagingSystem",
    "scanimage.scannerType",
    "SI.imagingSystem",
    "SI.hScan2D.scannerType",
)
_MODE_KEYS = ("calibration.mode", "scanimage.mode", "scanner.mode", "acq.mode")


@dataclass(frozen=True)
class PixelCalibrationProfileMapping:
    """One explicit ScanImage config-to-calibration-profile mapping.

    Args:
        config_name: ScanImage config filename or historical config name.
        mode: Lab calibration mode for recordings using this config.
        scanner: Calibration scanner family, such as ``galvo`` or ``res``.
        acquisition_fields: Optional acquisition fingerprint values used to
            reject stale config-name matches when metadata is available.

    The mapping exists because old ScanImage metadata records the config name
    and scanner timing fields, but not the lab's calibration mode number.
    """

    config_name: str
    mode: int
    scanner: str
    acquisition_fields: Mapping[str, float]


@dataclass(frozen=True)
class PixelCalibrationProfile:
    """Calibration-profile fields inferred for one recording.

    Args:
        rig: Calibration rig when directly inferable, otherwise ``None``.
        mode: Calibration mode when directly recorded or mapped from a known
            ScanImage config, otherwise ``None``.
        scanner: Scanner family when directly inferable or mapped.
        config_name: Normalized ScanImage config basename when present.
        source: Whether fields came from metadata, mappings, both, or neither.
        evidence: Short field names explaining what was used.

    The profile is intentionally partial. Callers should only auto-select a
    calibration group when the profile matches one unique measured group.
    """

    rig: str | None
    mode: int | None
    scanner: str | None
    config_name: str | None
    source: PixelCalibrationProfileSource
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class PixelCalibrationGroup:
    """One measured calibration group independent of zoom.

    Args:
        rig: Calibration rig.
        mode: Calibration scanner mode.
        scanner: Calibration scanner family.

    The pixel-size registry stores one row per zoom. UI code needs the unique
    group first, then uses the recording zoom to resolve microns per pixel.
    """

    rig: str
    mode: int
    scanner: str


def load_pixel_calibration_profile_mappings(
    path: Path,
) -> tuple[PixelCalibrationProfileMapping, ...]:
    """Load explicit ScanImage config-to-profile mappings from CSV.

    Args:
        path: CSV path with config name, calibration mode, scanner, and optional
            acquisition fingerprint columns.

    Returns:
        Immutable tuple of validated profile mappings.

    Raises:
        FileNotFoundError: If the CSV is absent.
        ValueError: If required columns or row values are invalid.

    The CSV is kept separate from measured pixel-size rows because it describes
    metadata interpretation, not physical calibration measurements.
    """
    mapping_path = path.expanduser()
    with mapping_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_mapping_columns(reader.fieldnames, mapping_path)
        return tuple(
            _parse_profile_mapping(row, mapping_path, line_number)
            for line_number, row in enumerate(reader, start=2)
        )


def resolve_pixel_calibration_profile(
    acquisition_metadata: Mapping[str, object],
    run_metadata: Mapping[str, object],
    *,
    mappings: Iterable[PixelCalibrationProfileMapping] = (),
) -> PixelCalibrationProfile:
    """Infer calibration-profile fields from converted recording metadata.

    Args:
        acquisition_metadata: Converted acquisition metadata attributes.
        run_metadata: Converted stimulus-run metadata attributes.
        mappings: Optional explicit config-to-profile mappings.

    Returns:
        Partial or complete calibration profile.

    Direct metadata wins because it is recording-specific. Mappings only fill
    gaps that ScanImage configs encode indirectly, primarily the lab calibration
    mode number for historical configs.
    """
    config_name, config_evidence = _metadata_config_name(acquisition_metadata)
    rig, rig_evidence = _metadata_rig(run_metadata)
    mode, mode_evidence = _metadata_mode(acquisition_metadata)
    scanner, scanner_evidence = _metadata_scanner(acquisition_metadata)
    mapping, mapping_evidence = _matching_profile_mapping(
        config_name,
        acquisition_metadata,
        mappings,
        scanner=scanner,
    )
    if mapping is not None:
        if mode is None:
            mode = mapping.mode
        if scanner is None:
            scanner = mapping.scanner

    evidence = tuple(
        item
        for item in (
            rig_evidence,
            mode_evidence,
            scanner_evidence,
            config_evidence,
            mapping_evidence,
        )
        if item is not None
    )
    return PixelCalibrationProfile(
        rig=rig,
        mode=mode,
        scanner=scanner,
        config_name=config_name,
        source=_profile_source(evidence),
        evidence=evidence,
    )


def select_pixel_calibration_group(
    profile: PixelCalibrationProfile,
    calibrations: Iterable[PixelCalibrationRow],
) -> PixelCalibrationGroup | None:
    """Select a unique measured calibration group for a profile.

    Args:
        profile: Partial or complete profile inferred from metadata.
        calibrations: Measured pixel-size calibration rows.

    Returns:
        Unique matching group, or ``None`` when the profile is incomplete,
        ambiguous, or points at a group without measured calibration rows.

    This keeps UI automation conservative: dropdowns are only preselected when
    the metadata and mapping evidence identify exactly one calibrated group.
    """
    groups = {
        PixelCalibrationGroup(row.rig, row.mode, row.scanner)
        for row in calibrations
        if _profile_matches_row(profile, row)
    }
    if not _has_complete_calibration_group(profile):
        return None
    if len(groups) == 1:
        return next(iter(groups))
    return None


def _profile_matches_row(
    profile: PixelCalibrationProfile,
    row: PixelCalibrationRow,
) -> bool:
    """Return whether one measured row matches all known profile fields."""
    if profile.rig is not None and _normalize_text(row.rig) != profile.rig:
        return False
    if profile.mode is not None and row.mode != profile.mode:
        return False
    return not (
        profile.scanner is not None
        and _normalize_scanner(row.scanner) != profile.scanner
    )


def _has_complete_calibration_group(profile: PixelCalibrationProfile) -> bool:
    """Return whether a profile contains every calibration group field."""
    return (
        profile.rig is not None
        and profile.mode is not None
        and profile.scanner is not None
    )


def _metadata_config_name(
    acquisition_metadata: Mapping[str, object],
) -> tuple[str | None, str | None]:
    """Return normalized config basename from known metadata keys."""
    for key in _CONFIG_NAME_KEYS:
        value = acquisition_metadata.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_config_name(value), key
    return None, None


def _metadata_rig(run_metadata: Mapping[str, object]) -> tuple[str | None, str | None]:
    """Return a calibration rig only when run metadata states it clearly."""
    value = run_metadata.get("calibration_rig", run_metadata.get("rig_name"))
    if not isinstance(value, str):
        return None, None
    text = _normalize_text(value)
    if text in {"day", "dayrig", "day_rig"} or "day" in text.split("_"):
        return "day", "run.rig_name"
    if text in {
        "night",
        "nightrig",
        "night_rig",
        "odorrig",
        "odor_rig",
    } or "night" in text.split("_"):
        return "night", "run.rig_name"
    return None, None


def _metadata_mode(
    acquisition_metadata: Mapping[str, object],
) -> tuple[int | None, str | None]:
    """Return directly recorded calibration mode when present."""
    for key in _MODE_KEYS:
        value = acquisition_metadata.get(key)
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed, key
    return None, None


def _metadata_scanner(
    acquisition_metadata: Mapping[str, object],
) -> tuple[str | None, str | None]:
    """Return directly recorded scanner family when present."""
    for key in _SCANNER_KEYS:
        value = acquisition_metadata.get(key)
        scanner = _normalize_scanner(value)
        if scanner is not None:
            return scanner, key
    return None, None


def _matching_profile_mapping(
    config_name: str | None,
    acquisition_metadata: Mapping[str, object],
    mappings: Iterable[PixelCalibrationProfileMapping],
    *,
    scanner: str | None,
) -> tuple[PixelCalibrationProfileMapping | None, str | None]:
    """Return the one profile mapping matching the recording fingerprint."""
    if config_name is None:
        return None, None
    matches = [
        mapping
        for mapping in mappings
        if _normalize_config_name(mapping.config_name) == config_name
        and (scanner is None or mapping.scanner == scanner)
        and _mapping_fingerprint_matches(mapping, acquisition_metadata)
    ]
    if len(matches) == 1:
        return matches[0], "profile_mapping.config_name"
    return None, None


def _mapping_fingerprint_matches(
    mapping: PixelCalibrationProfileMapping,
    acquisition_metadata: Mapping[str, object],
) -> bool:
    """Return whether available acquisition fields agree with a mapping."""
    for field, expected in mapping.acquisition_fields.items():
        observed = _metadata_float(acquisition_metadata, _ACQUISITION_FIELD_KEYS[field])
        if observed is not None and not math.isclose(
            observed,
            expected,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            return False
    return True


def _metadata_float(
    metadata: Mapping[str, object],
    keys: tuple[str, ...],
) -> float | None:
    """Read one optional float from the first available metadata key."""
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, int | float):
            return float(value) * _ACQUISITION_FIELD_SCALE.get(key, 1.0)
        if isinstance(value, str):
            try:
                return float(value) * _ACQUISITION_FIELD_SCALE.get(key, 1.0)
            except ValueError:
                continue
    return None


def _parse_profile_mapping(
    row: Mapping[str, str],
    path: Path,
    line_number: int,
) -> PixelCalibrationProfileMapping:
    """Parse one CSV row into a validated profile mapping."""
    config_name = _required_text(row, "config_name", path, line_number)
    mode = _required_int(row, "mode", path, line_number)
    scanner = _required_scanner(row, "scanner", path, line_number)
    acquisition_fields = {
        key: value
        for key, value in (
            (
                field,
                _optional_float(row.get(field), field, path, line_number),
            )
            for field in _ACQUISITION_FIELD_KEYS
        )
        if value is not None
    }
    return PixelCalibrationProfileMapping(
        config_name=config_name,
        mode=mode,
        scanner=scanner,
        acquisition_fields=acquisition_fields,
    )


def _validate_mapping_columns(fieldnames: Sequence[str] | None, path: Path) -> None:
    """Reject profile-mapping CSV files without the expected schema."""
    if fieldnames is None:
        msg = f"Pixel calibration profile CSV is missing a header row: {path}"
        raise ValueError(msg)
    missing = [
        column for column in _REQUIRED_MAPPING_COLUMNS if column not in fieldnames
    ]
    if missing:
        msg = (
            f"Pixel calibration profile CSV {path} is missing columns: "
            f"{', '.join(missing)}"
        )
        raise ValueError(msg)


def _required_text(
    row: Mapping[str, str],
    key: str,
    path: Path,
    line_number: int,
) -> str:
    """Return one required non-empty CSV text value."""
    value = row.get(key, "").strip()
    if value == "":
        msg = f"Pixel calibration profile {path}:{line_number} has empty {key!r}"
        raise ValueError(msg)
    return value


def _required_int(
    row: Mapping[str, str],
    key: str,
    path: Path,
    line_number: int,
) -> int:
    """Return one required positive integer CSV value."""
    value = row.get(key, "").strip()
    parsed = _optional_int(value)
    if parsed is None or parsed <= 0:
        msg = f"Pixel calibration profile {path}:{line_number} has invalid {key!r}"
        raise ValueError(msg)
    return parsed


def _required_scanner(
    row: Mapping[str, str],
    key: str,
    path: Path,
    line_number: int,
) -> str:
    """Return one required normalized scanner value."""
    scanner = _normalize_scanner(row.get(key, ""))
    if scanner is None:
        msg = f"Pixel calibration profile {path}:{line_number} has invalid {key!r}"
        raise ValueError(msg)
    return scanner


def _optional_float(
    value: str | None,
    key: str,
    path: Path,
    line_number: int,
) -> float | None:
    """Return one optional positive CSV float value."""
    if value is None or value.strip() == "":
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        msg = f"Pixel calibration profile {path}:{line_number} has invalid {key!r}"
        raise ValueError(msg) from error
    if not math.isfinite(parsed) or parsed <= 0:
        msg = f"Pixel calibration profile {path}:{line_number} has invalid {key!r}"
        raise ValueError(msg)
    return parsed


def _optional_int(value: object) -> int | None:
    """Return a positive integer from metadata-like values when possible."""
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        if parsed.is_integer():
            return int(parsed)
    return None


def _normalize_config_name(value: str) -> str:
    """Return a case-insensitive config basename without a ``.cfg`` suffix."""
    basename = re.split(r"[\\/]", value.strip())[-1]
    if basename.lower().endswith(".cfg"):
        basename = basename[:-4]
    return _normalize_text(basename)


def _normalize_scanner(value: object) -> str | None:
    """Normalize microscope scanner labels to calibration registry values."""
    if not isinstance(value, str):
        return None
    text = _normalize_text(value)
    if text in {"galvo", "galvos", "linear"}:
        return "galvo"
    if text in {"res", "resonant", "resonance"}:
        return "res"
    return None


def _normalize_text(value: str) -> str:
    """Return a stable comparison key for human-entered metadata text."""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _profile_source(evidence: tuple[str, ...]) -> PixelCalibrationProfileSource:
    """Classify which kind of evidence produced a profile."""
    has_metadata = any(item != "profile_mapping.config_name" for item in evidence)
    has_mapping = "profile_mapping.config_name" in evidence
    if has_metadata and has_mapping:
        return "metadata+mapping"
    if has_metadata:
        return "metadata"
    return "unresolved"
