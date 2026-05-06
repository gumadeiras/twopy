"""Human-readable recording and output path labels for napari.

Inputs: loaded converted recordings and output paths.
Outputs: short strings for GUI labels and status messages.

Full recording paths are too long for the napari sidebars. These helpers keep
path presentation in one place while file-reading and file-writing code keeps
using full ``Path`` objects.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from twopy.config import DEFAULT_CONFIG_PATH, load_config
from twopy.converted import RecordingData

__all__ = [
    "RecordingDisplaySummary",
    "format_output_folder",
    "format_twopy_h5_output",
    "recording_display_summary",
]

_YEAR_PATTERN = re.compile(r"^\d{4}$")
_DATE_PATTERN = re.compile(r"^\d{2}_\d{2}$")
_TIME_PATTERN = re.compile(r"^\d{2}_\d{2}_\d{2}$")


@dataclass(frozen=True)
class RecordingDisplaySummary:
    """Short recording identity shown in the response Update tab.

    Inputs: one converted recording path.
    Outputs: root data path, genotype folder, and formatted recording time.
    """

    root: Path
    genotype: str
    recording: str

    def lines(self) -> tuple[str, str, str]:
        """Return display lines for a compact Qt label.

        Args:
            None.

        Returns:
            Three user-facing lines.
        """
        return (
            f"Root: {self.root}",
            f"Genotype: {self.genotype}",
            f"Recording: {self.recording}",
        )


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
    recording_label = _recording_label(recording_dir, date_index)

    try:
        data_root = load_config(DEFAULT_CONFIG_PATH).data_path.expanduser()
        relative = recording_dir.relative_to(data_root)
    except (FileNotFoundError, ValueError):
        data_root, genotype = _fallback_root_and_genotype(recording_dir, date_index)
    else:
        genotype = relative.parts[0] if len(relative.parts) > 0 else "unknown"

    return RecordingDisplaySummary(
        root=data_root,
        genotype=genotype,
        recording=recording_label,
    )


def format_twopy_h5_output(path: Path) -> str:
    """Return the compact display form for twopy HDF5 outputs.

    Args:
        path: Full output path.

    Returns:
        ``./twopy/<filename>.h5`` style string.
    """
    return f"./twopy/{path.name}"


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


def _fallback_root_and_genotype(
    recording_dir: Path,
    date_index: int | None,
) -> tuple[Path, str]:
    """Infer root and genotype when config is unavailable.

    Args:
        recording_dir: Source recording folder.
        date_index: Index of the year segment, if found.

    Returns:
        ``(root, genotype)`` best effort.

    If there is a stimulus folder between genotype and date, it usually has
    explicit stimulus naming. Otherwise the folder directly before the date is
    treated as the genotype folder.
    """
    if date_index is None or date_index == 0:
        return recording_dir.parent, recording_dir.name

    parts = recording_dir.parts
    genotype_index = date_index - 1
    if date_index >= 2 and _looks_like_stimulus_folder(parts[date_index - 1]):
        genotype_index = date_index - 2

    return Path(*parts[:genotype_index]), parts[genotype_index]


def _looks_like_stimulus_folder(folder_name: str) -> bool:
    """Return whether a path segment looks like an explicit stimulus folder.

    Args:
        folder_name: Candidate folder name.

    Returns:
        ``True`` when the folder name is likely not genotype.
    """
    lowered = folder_name.lower()
    return "=" in folder_name or "stim" in lowered
