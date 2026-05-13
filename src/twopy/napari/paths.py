"""Path resolution helpers for the twopy napari adapter.

Inputs: user-selected files or folders.
Outputs: concrete converted-recording, movie, and ROI paths.

These helpers keep GUI callbacks small. A user can point napari at a converted
output folder, a source recording folder with ``twopy/`` output, or a direct
``recording_data.h5`` file.
"""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from twopy.filenames import (
    ALIGNED_MOVIE_FILENAME,
    RECORDING_DATA_FILENAME,
    ROI_FILENAME,
)

__all__ = [
    "DEFAULT_PATH_TEXT",
    "NapariRecordingPaths",
    "PathInput",
    "is_default_path",
    "resolve_launch_recording_path",
    "resolve_recording_paths",
    "resolve_widget_output_path",
]

PathInput = str | PathLike[str]
DEFAULT_PATH_TEXT = "default"


@dataclass(frozen=True)
class NapariRecordingPaths:
    """Resolved files associated with one converted recording.

    Inputs: a user-selected file or folder.
    Outputs: paths that the viewer can load or save.
    """

    recording_data_path: Path
    movie_path: Path | None
    roi_file_to_load: Path | None
    roi_save_file: Path


def resolve_recording_paths(path: PathInput) -> NapariRecordingPaths:
    """Resolve a widget-selected file or folder into twopy recording paths.

    Args:
        path: User-selected folder, ``recording_data.h5`` file, or ``default``.
            Magicgui may provide this as either a ``Path`` or plain string.

    Returns:
        Existing recording data path plus optional movie and ROI paths.

    Raises:
        ValueError: If the input does not identify a converted recording.

    Folder selection is the intended GUI workflow. Direct file selection still
    works for scripts and audits.
    """
    selected = Path.cwd() if is_default_path(path) else Path(path).expanduser()
    recording_path = _resolve_recording_data_path(selected)
    recording_dir = recording_path.parent
    movie_path = _optional_file(recording_dir / ALIGNED_MOVIE_FILENAME)
    roi_path = _optional_file(recording_dir / ROI_FILENAME)
    return NapariRecordingPaths(
        recording_data_path=recording_path,
        movie_path=movie_path,
        roi_file_to_load=roi_path,
        roi_save_file=roi_path or recording_dir / ROI_FILENAME,
    )


def resolve_launch_recording_path(
    recording_data_path: PathInput | None,
) -> Path | None:
    """Resolve an optional command-line recording path.

    Args:
        recording_data_path: Optional caller path.

    Returns:
        Existing ``recording_data.h5`` path, or ``None`` when no path was
        supplied and no default recording exists.

    Raises:
        ValueError: If an explicit path was supplied and does not exist.
    """
    if recording_data_path is not None:
        return resolve_recording_paths(recording_data_path).recording_data_path

    cwd = Path.cwd()
    for candidate in (cwd / RECORDING_DATA_FILENAME, cwd / "twopy"):
        try:
            return resolve_recording_paths(candidate).recording_data_path
        except ValueError:
            continue
    return None


def resolve_widget_output_path(path: PathInput, *, default: Path) -> Path:
    """Resolve an optional widget output path.

    Args:
        path: Path value from a magicgui widget. Magicgui may provide this as
            either a ``Path`` or plain string.
        default: Default path to use when the widget says ``default``.

    Returns:
        Expanded output path.
    """
    if is_default_path(path):
        return default.expanduser()
    return Path(path).expanduser()


def is_default_path(path: PathInput) -> bool:
    """Return whether a widget path means "use twopy's default".

    Args:
        path: Path value from magicgui or a script.

    Returns:
        ``True`` when the path is empty, ``.``, or ``default``.
    """
    return str(path).strip().lower() in {"", ".", DEFAULT_PATH_TEXT}


def _resolve_recording_data_path(path: Path) -> Path:
    """Find ``recording_data.h5`` from a selected file or folder.

    Args:
        path: Expanded file or folder path.

    Returns:
        Existing recording data file.
    """
    candidates = _recording_data_candidates(path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    candidate_list = "\n".join(f"- {candidate}" for candidate in candidates)
    msg = (
        "Could not find recording_data.h5. Select a converted output folder, "
        "a source recording folder with twopy output, or recording_data.h5.\n"
        f"Checked:\n{candidate_list}"
    )
    raise ValueError(msg)


def _recording_data_candidates(path: Path) -> tuple[Path, ...]:
    """Return the recording-data paths implied by one selected path.

    Args:
        path: Expanded file or folder path.

    Returns:
        Candidate paths in preferred order.
    """
    if path.is_dir():
        return (
            path / RECORDING_DATA_FILENAME,
            path / "twopy" / RECORDING_DATA_FILENAME,
        )
    return (path,)


def _optional_file(path: Path) -> Path | None:
    """Return an optional file path when it exists.

    Args:
        path: Candidate file path.

    Returns:
        Resolved file path, or ``None``.
    """
    return path.resolve() if path.is_file() else None
