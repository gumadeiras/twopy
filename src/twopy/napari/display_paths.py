"""Human-readable recording and output path labels for napari.

Inputs: loaded converted recordings and output paths.
Outputs: short strings for GUI labels and status messages.

Full recording paths are too long for the napari sidebars. These helpers keep
path presentation in one place while file-reading and file-writing code keeps
using full ``Path`` objects.
"""

import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

from twopy.config import DEFAULT_CONFIG_PATH, data_path_match, load_config
from twopy.converted import RecordingData

__all__ = [
    "RecordingDisplaySummary",
    "RecordingPathDisplaySummary",
    "format_recording_read_folder",
    "format_recording_queue_path",
    "format_output_folder",
    "format_recording_minute_label",
    "format_twopy_h5_output",
    "microscope_display_lines",
    "recording_path_display_summary",
    "recording_display_summary",
]

_YEAR_PATTERN = re.compile(r"^\d{4}$")
_DATE_PATTERN = re.compile(r"^\d{2}_\d{2}$")
_TIME_PATTERN = re.compile(r"^\d{2}_\d{2}_\d{2}$")


@dataclass(frozen=True)
class RecordingDisplaySummary:
    """Short recording identity shown in the response Metadata tab.

    Inputs: one converted recording path.
    Outputs: root data path, genotype folder, stimulus folder, and recording
    time.
    """

    root: Path
    genotype: str
    stimulus: str
    recording: str

    def lines(self) -> tuple[str, str, str, str]:
        """Return display lines for a compact Qt label.

        Args:
            None.

        Returns:
            Four user-facing lines.
        """
        return (
            f"Root: {self.root}",
            f"Genotype: {self.genotype}",
            f"Stimulus: {self.stimulus}",
            f"Recording: {self.recording}",
        )


@dataclass(frozen=True)
class RecordingPathDisplaySummary:
    """Short recording identity parsed directly from a selected path.

    Inputs: source recording folders, converted folders, or converted HDF5 paths.
    Outputs: root data path, genotype folder, stimulus folder, and date/time
    folder label when the lab path pattern is present.
    """

    root: Path
    genotype: str
    stimulus: str
    date: str


def recording_display_summary(recording: RecordingData) -> RecordingDisplaySummary:
    """Parse one recording into stable, readable identity fields.

    Args:
        recording: Loaded converted recording.

    Returns:
        Root/genotype/recording-time summary.

    ``config.yml`` gives the cleanest root and genotype split when present.
    The fallback still reads the final ``YYYY/MM_DD/HH_MM_SS`` path contract so
    napari remains usable from copied example folders.
    """
    recording_dir = recording.source_session_dir.expanduser()
    date_index = _recording_date_index(recording_dir)
    data_root, genotype, stimulus = _recording_identity_parts(
        recording_dir,
        date_index,
    )

    return RecordingDisplaySummary(
        root=data_root,
        genotype=genotype,
        stimulus=stimulus,
        recording=_recording_label(recording_dir, date_index),
    )


def recording_path_display_summary(path: Path) -> RecordingPathDisplaySummary:
    """Parse one selected recording path into display fields.

    Args:
        path: Source recording folder, converted folder, or converted HDF5 file.

    Returns:
        Root, genotype, stimulus, and ``YYYY/MM_DD/HH_MM_SS`` date/time text
        when the selected path includes that lab folder pattern. Unknown fields
        are explicit when the path does not match the microscope layout.

    The load-progress dialog uses this before a converted ``RecordingData``
    object exists, so it cannot call ``recording_display_summary(...)``.
    """
    recording_path = path.expanduser()
    date_index = _recording_date_index(recording_path)
    if date_index is None or date_index < 2:
        return RecordingPathDisplaySummary(
            root=recording_path.parent,
            genotype="unknown",
            stimulus="unknown",
            date=recording_path.name,
        )

    data_root, genotype, stimulus = _recording_identity_parts(
        recording_path,
        date_index,
    )
    parts = recording_path.parts
    return RecordingPathDisplaySummary(
        root=data_root,
        genotype=genotype,
        stimulus=stimulus,
        date=f"{parts[date_index]}/{parts[date_index + 1]}/{parts[date_index + 2]}",
    )


def microscope_display_lines(recording: RecordingData) -> tuple[str, ...]:
    """Return compact microscope metadata lines for the Metadata tab.

    Args:
        recording: Loaded converted recording.

    Returns:
        User-facing microscope, scan, and timing fields.
    """
    acquisition = recording.acquisition_metadata
    run = recording.run_metadata
    return (
        f"Rig: {_metadata_text(run, 'rig_name')}",
        f"Hemisphere: {_hemisphere_text(run)}",
        f"Config: {_metadata_text(acquisition, 'configName')}",
        f"Zoom: {_metadata_text(acquisition, 'acq.zoomFactor')}",
        f"Frame rate: {_metadata_number(acquisition, 'acq.frameRate', ' Hz')}",
        "Frame shape: "
        f"{_metadata_text(acquisition, 'acq.pixelsPerLine')} x "
        f"{_metadata_text(acquisition, 'acq.linesPerFrame')}",
        f"Frames: {_metadata_text(acquisition, 'acq.numberOfFrames')}",
    )


def format_twopy_h5_output(path: Path) -> str:
    """Return the compact display form for twopy HDF5 outputs.

    Args:
        path: Full output path.

    Returns:
        ``./twopy/<filename>.h5`` style string.
    """
    return f"./twopy/{path.name}"


def format_recording_minute_label(recording_dir: Path) -> str:
    """Return the compact recording date and minute for UI labels.

    Args:
        recording_dir: Source recording folder.

    Returns:
        ``YYYY.MM.DD HH:MM`` when the folder follows the lab
        ``YYYY/MM_DD/HH_MM_SS`` layout, otherwise the folder name.

    Group-matching cards need a short label that distinguishes recordings from
    different dates while hiding seconds that do not help manual matching.
    """
    root = recording_dir.expanduser()
    date_index = _recording_date_index(root)
    if date_index is None:
        return root.name

    parts = root.parts
    year = parts[date_index]
    month_day = parts[date_index + 1].replace("_", ".")
    hour, minute, _second = parts[date_index + 2].split("_")
    return f"{year}.{month_day} {hour}:{minute}"


def format_recording_queue_path(path: Path) -> str:
    """Return the compact queue label for one selected recording path.

    Args:
        path: Source recording folder, converted folder, or converted HDF5 file.

    Returns:
        ``.../YYYY/MM_DD/HH_MM_SS`` when the lab folder pattern is present,
        otherwise the final three path parts.
    """
    recording_path = path.expanduser()
    date_index = _recording_date_index(recording_path)
    parts = recording_path.parts
    if date_index is not None:
        return (
            f".../{parts[date_index]}/{parts[date_index + 1]}/{parts[date_index + 2]}"
        )
    if len(parts) < 3:
        return str(recording_path)
    return f".../{parts[-3]}/{parts[-2]}/{parts[-1]}"


def format_recording_read_folder(path: Path) -> str:
    """Return a compact label for the converted folder napari is reading.

    Args:
        path: Full folder containing ``recording_data.h5``.

    Returns:
        ``<root>/.../YYYY/MM_DD/HH_MM_SS`` plus any suffix after the recording
        timestamp, such as ``/twopy`` for source-local converted output.

    The source identity panel already shows genotype and stimulus. This label
    keeps the concrete read root visible while preserving enough suffix detail
    to distinguish source-local output from cache or analysis output.
    """
    read_folder = path.expanduser()
    date_index = _recording_date_index(read_folder)
    if date_index is None:
        return str(read_folder)

    summary = recording_path_display_summary(read_folder)
    suffix = "/".join(read_folder.parts[date_index:])
    return f"{summary.root}/.../{suffix}"


def _recording_identity_parts(
    recording_dir: Path,
    date_index: int | None,
) -> tuple[Path, str, str]:
    """Return root, genotype, and stimulus fields for one source-like path."""
    try:
        match = data_path_match(load_config(DEFAULT_CONFIG_PATH), recording_dir)
        if match is None:
            raise ValueError
        data_root, relative = match
    except (FileNotFoundError, ValueError):
        return _fallback_path_parts(recording_dir, date_index)

    genotype = relative.parts[0] if len(relative.parts) > 0 else "unknown"
    stimulus = relative.parts[1] if len(relative.parts) > 1 else "unknown"
    return data_root, genotype, stimulus


def _metadata_text(metadata: dict[str, object], key: str) -> str:
    """Return a display string for one metadata field."""
    value = metadata.get(key)
    if value is None or value == "":
        return "unknown"
    return str(value)


def _metadata_number(metadata: dict[str, object], key: str, suffix: str) -> str:
    """Return a compact numeric metadata string."""
    value = metadata.get(key)
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{value:g}{suffix}"
    return _metadata_text(metadata, key)


def _hemisphere_text(metadata: dict[str, object]) -> str:
    """Return the stored hemisphere display text from run metadata."""
    for key in ("hemisphere", "eye"):
        value = metadata.get(key)
        if value is not None and value != "":
            return str(value)
    return "unknown"


def format_output_folder(path: Path, recording: RecordingData | None) -> str:
    """Return a compact output-folder label for status messages.

    Args:
        path: File or folder path that was written.
        recording: Optional selected recording used to compute a twopy-relative
            folder.

    Returns:
        Short folder string such as ``./twopy`` or
        ``./twopy/exports/recording_view``.
    """
    folder = path if path.suffix == "" else path.parent
    if recording is None:
        return str(folder)

    twopy_dir = recording.path.parent
    try:
        relative = folder.relative_to(twopy_dir)
    except ValueError:
        return "./twopy"

    if str(relative) == ".":
        return "./twopy"
    return f"./twopy/{relative}"


def _recording_date_index(recording_dir: Path) -> int | None:
    """Return the index of the ``YYYY/MM_DD/HH_MM_SS`` path segment.

    Args:
        recording_dir: Source recording folder.

    Returns:
        Index of the year segment, or ``None`` when the path is not recognized.
    """
    parts = recording_dir.parts
    for index in range(len(parts) - 2):
        if (
            _YEAR_PATTERN.match(parts[index]) is not None
            and _DATE_PATTERN.match(parts[index + 1]) is not None
            and _TIME_PATTERN.match(parts[index + 2]) is not None
        ):
            return index
    return None


def _recording_label(recording_dir: Path, date_index: int | None) -> str:
    """Return the recording date label from the source recording path.

    Args:
        recording_dir: Source recording folder.
        date_index: Index of the year segment, if found.

    Returns:
        ``YYYY-MM-DD HH:MM:SS`` or the folder name as a fallback.
    """
    if date_index is None:
        return recording_dir.name

    parts = recording_dir.parts
    year = parts[date_index]
    month_day = parts[date_index + 1].replace("_", "-")
    time = parts[date_index + 2].replace("_", ":")
    return f"{year}-{month_day} {time}"


def _fallback_path_parts(
    recording_dir: Path,
    date_index: int | None,
) -> tuple[Path, str, str]:
    """Infer root, genotype, and stimulus when config is unavailable.

    Args:
        recording_dir: Source recording folder.
        date_index: Index of the year segment, if found.

    Returns:
        ``(root, genotype, stimulus)`` best effort.

    Stable lab recording paths end as
    ``<genotype>/<stimulus>/<YYYY>/<MM_DD>/<HH_MM_SS>``. The fallback keeps that
    contract explicit instead of guessing from folder names.
    """
    if date_index is None or date_index < 2:
        return recording_dir.parent, recording_dir.name, "unknown"

    parts = recording_dir.parts
    genotype_index = date_index - 2
    stimulus_index = date_index - 1
    return Path(*parts[:genotype_index]), parts[genotype_index], parts[stimulus_index]
