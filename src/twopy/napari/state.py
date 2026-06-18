"""Local napari UI state for twopy.

Inputs: small local JSON file in the user's home directory.
Outputs: remembered GUI defaults such as the last recording and CSV folders.

This module stores convenience state only. It is never used for analysis
decisions, and missing or corrupt state simply falls back to defaults.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "read_last_recording_csv_folder",
    "read_last_recording_folder",
    "read_version_check_cache",
    "recording_folder_for_state",
    "VersionCheckCache",
    "write_last_recording_csv_folder",
    "write_last_recording_folder",
    "write_version_check_cache",
]

STATE_FILE_ENV_VAR = "TWOPY_NAPARI_STATE_FILE"
LAST_RECORDING_FOLDER_KEY = "last_recording_folder"
LAST_RECORDING_CSV_FOLDER_KEY = "last_recording_csv_folder"
VERSION_CHECKED_AT_KEY = "version_check_checked_at"
VERSION_LATEST_VERSION_KEY = "version_check_latest_version"


@dataclass(frozen=True)
class VersionCheckCache:
    """Cached package-version lookup stored in local napari state.

    Args:
        checked_at: UTC ISO timestamp for the latest completed lookup.
        latest_version: Latest package version reported by the package index.

    The cache keeps launch-time update notices fast and quiet. It is UI state,
    not analysis data, so malformed or missing values are ignored.
    """

    checked_at: str
    latest_version: str


def read_last_recording_folder(*, state_file: Path | None = None) -> Path | None:
    """Read the last recording folder selected in napari.

    Args:
        state_file: Optional state file override for tests.

    Returns:
        Existing folder path, or ``None`` when no usable state exists.
    """
    return _read_existing_folder(LAST_RECORDING_FOLDER_KEY, state_file)


def read_last_recording_csv_folder(*, state_file: Path | None = None) -> Path | None:
    """Read the last folder used for loaded-recordings CSV files.

    Args:
        state_file: Optional state file override for tests.

    Returns:
        Existing folder path, or ``None`` when no usable state exists.
    """
    return _read_existing_folder(LAST_RECORDING_CSV_FOLDER_KEY, state_file)


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
    _write_folder(LAST_RECORDING_FOLDER_KEY, folder, state_file)


def write_last_recording_csv_folder(
    folder: Path,
    *,
    state_file: Path | None = None,
) -> None:
    """Remember the last folder used for loaded-recordings CSV files.

    Args:
        folder: Folder to store.
        state_file: Optional state file override for tests.

    Returns:
        None.
    """
    _write_folder(LAST_RECORDING_CSV_FOLDER_KEY, folder, state_file)


def read_version_check_cache(
    *,
    state_file: Path | None = None,
) -> VersionCheckCache | None:
    """Read the last completed package-version lookup.

    Args:
        state_file: Optional state file override for tests.

    Returns:
        Cached lookup fields, or ``None`` when no complete cache exists.
    """
    data = _read_state_data(state_file)
    checked_at = data.get(VERSION_CHECKED_AT_KEY)
    latest_version = data.get(VERSION_LATEST_VERSION_KEY)
    if not (isinstance(checked_at, str) and isinstance(latest_version, str)):
        return None
    return VersionCheckCache(
        checked_at=checked_at,
        latest_version=latest_version,
    )


def write_version_check_cache(
    cache: VersionCheckCache,
    *,
    state_file: Path | None = None,
) -> None:
    """Store the latest completed package-version lookup.

    Args:
        cache: Lookup fields to store.
        state_file: Optional state file override for tests.

    Returns:
        None.
    """
    data = _read_state_data(state_file)
    data[VERSION_CHECKED_AT_KEY] = cache.checked_at
    data[VERSION_LATEST_VERSION_KEY] = cache.latest_version
    _write_state_data(data, state_file)


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


def _read_existing_folder(key: str, state_file: Path | None) -> Path | None:
    """Read one existing folder value from the local state file.

    Args:
        key: JSON key to read.
        state_file: Optional explicit state file path.

    Returns:
        Existing folder path, or ``None`` when the key is absent or stale.
    """
    folder_text = _read_state_data(state_file).get(key)
    if not isinstance(folder_text, str):
        return None
    folder = Path(folder_text).expanduser()
    return folder if folder.is_dir() else None


def _write_folder(key: str, folder: Path, state_file: Path | None) -> None:
    """Write one folder value while preserving other local state keys.

    Args:
        key: JSON key to write.
        folder: Folder to store.
        state_file: Optional explicit state file path.

    Returns:
        None.
    """
    data = _read_state_data(state_file)
    data[key] = str(folder.expanduser().resolve())
    _write_state_data(data, state_file)


def _read_state_data(state_file: Path | None) -> dict[str, object]:
    """Return usable persisted napari state values.

    Args:
        state_file: Optional explicit state file path.

    Returns:
        JSON state mapping, or an empty mapping when state is absent or
        corrupt.
    """
    path = _state_file_path(state_file)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    state_data: dict[str, object] = {}
    for key, value in data.items():
        if isinstance(key, str):
            state_data[key] = value
    return state_data


def _write_state_data(data: dict[str, object], state_file: Path | None) -> None:
    """Persist napari state while treating write failures as non-fatal.

    Args:
        data: State values to write.
        state_file: Optional explicit state file path.

    Returns:
        None.
    """
    path = _state_file_path(state_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    except OSError:
        return
