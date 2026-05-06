"""Local napari UI state for twopy.

Inputs: small local JSON file in the user's home directory.
Outputs: remembered GUI defaults such as the last recording folder.

This module stores convenience state only. It is never used for analysis
decisions, and missing or corrupt state simply falls back to defaults.
"""

import json
import os
from pathlib import Path

__all__ = [
    "read_last_recording_folder",
    "recording_folder_for_state",
    "write_last_recording_folder",
]

STATE_FILE_ENV_VAR = "TWOPY_NAPARI_STATE_FILE"


def read_last_recording_folder(*, state_file: Path | None = None) -> Path | None:
    """Read the last recording folder selected in napari.

    Args:
        state_file: Optional state file override for tests.

    Returns:
        Existing folder path, or ``None`` when no usable state exists.
    """
    path = _state_file_path(state_file)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    folder_text = data.get("last_recording_folder")
    if not isinstance(folder_text, str):
        return None
    folder = Path(folder_text).expanduser()
    return folder if folder.is_dir() else None


def write_last_recording_folder(
    folder: Path,
    *,
    state_file: Path | None = None,
) -> None:
    """Remember the last recording folder selected in napari.

    Args:
        folder: Folder to store.
        state_file: Optional state file override for tests.

    Returns:
        None.
    """
    resolved_folder = folder.expanduser().resolve()
    path = _state_file_path(state_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"last_recording_folder": str(resolved_folder)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    except OSError:
        return


def recording_folder_for_state(selected_path: Path, recording_data_path: Path) -> Path:
    """Choose which folder should be remembered after a successful load.

    Args:
        selected_path: Path selected by the user.
        recording_data_path: Resolved ``recording_data.h5`` path.

    Returns:
        Folder path to store.
    """
    expanded = selected_path.expanduser()
    if expanded.is_dir():
        return expanded
    return recording_data_path.expanduser().parent


def _state_file_path(state_file: Path | None) -> Path:
    """Return the JSON state file path.

    Args:
        state_file: Optional explicit path.

    Returns:
        Path to the napari state JSON file.
    """
    if state_file is not None:
        return state_file.expanduser()
    env_path = os.environ.get(STATE_FILE_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".config" / "twopy" / "napari_state.json"
