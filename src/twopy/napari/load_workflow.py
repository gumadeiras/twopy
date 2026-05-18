"""Recording-load workflow for the twopy napari Load tab.

Inputs: selected recording paths and the current napari session state.
Outputs: loaded recordings, structured failures, and source-cache warnings.

This module owns the non-Qt decisions in the Load workflow: source or converted
path resolution, duplicate selection, selected-row replacement, loaded-count
accounting, and per-path failure collection. The controls module chooses paths
and renders dialogs; this module changes the recording session.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from twopy.napari.errors import exception_message_for_user
from twopy.napari.loading import ResolvedNapariRecording, resolve_or_convert_recording
from twopy.napari.paths import (
    DEFAULT_PATH_TEXT,
    PathInput,
    is_default_path,
)
from twopy.napari.session import (
    NapariSessionState,
    record_loaded_view,
    select_loaded_recording,
)
from twopy.napari.state import recording_folder_for_state, write_last_recording_folder
from twopy.napari.types import NapariRecordingView

__all__ = [
    "RecordingLoadFailure",
    "RecordingLoadResult",
    "SourceUnavailableWarning",
    "load_recording_paths",
]


class RecordingLoadState(NapariSessionState, Protocol):
    """Session state fields required by the recording-load workflow."""

    is_loading: bool


@dataclass(frozen=True)
class RecordingLoadFailure:
    """One recording path that failed during a load request.

    Inputs: source or converted path and the user-facing error message.
    Outputs: immutable failure item for dialogs and tests.
    """

    path: Path | None
    message: str


@dataclass(frozen=True)
class SourceUnavailableWarning:
    """One cache-backed load whose source path was unavailable.

    Inputs: requested source path and resolved cached recording path.
    Outputs: immutable warning item for the Qt layer to render.
    """

    source_path: Path
    cache_path: Path


@dataclass(frozen=True)
class RecordingLoadResult:
    """Result from loading one or more recording paths.

    Inputs: newly loaded count plus any warnings or failures collected per path.
    Outputs: structured result shared by manual, CSV, and database loads.
    """

    loaded_count: int
    failures: tuple[RecordingLoadFailure, ...] = ()
    source_warnings: tuple[SourceUnavailableWarning, ...] = ()
    status_text: str = ""


@dataclass(frozen=True)
class _SingleRecordingLoad:
    """Internal result for one path before batch failure aggregation."""

    status_text: str
    source_warnings: tuple[SourceUnavailableWarning, ...] = ()


def load_recording_paths(
    state: RecordingLoadState,
    paths: tuple[Path, ...],
    *,
    remember_selected_folder: bool,
    roi_file_to_load: PathInput = Path(DEFAULT_PATH_TEXT),
    replace_selected: bool = False,
    resolve_recording: Callable[[PathInput], ResolvedNapariRecording] = (
        resolve_or_convert_recording
    ),
    open_recording: Callable[..., NapariRecordingView] | None = None,
) -> RecordingLoadResult:
    """Load selected source or converted recording paths into napari.

    Args:
        state: Mutable napari session state.
        paths: Recording folders or converted recording paths to load.
        remember_selected_folder: Whether successful loads should update the
            next manual picker starting folder.
        roi_file_to_load: Optional ROI path for single-recording loads.
        replace_selected: Whether a single load should replace the selected row.
        resolve_recording: Recording path resolver. Tests can provide a local
            fake; production uses ``resolve_or_convert_recording``.
        open_recording: Converted-recording opener. Tests can provide a local
            fake; production uses ``open_recording_in_napari``.

    Returns:
        Structured load result with new-recording count, failures, source-cache
        warnings, and the last successful status string.

    This is the shared execution path for manual, CSV, and database loads. The
    caller remains responsible for choosing paths and showing dialogs.
    """
    if state.is_loading:
        message = "Recording load already in progress."
        return RecordingLoadResult(
            loaded_count=0,
            failures=(RecordingLoadFailure(path=None, message=message),),
            status_text=message,
        )
    if len(paths) == 0:
        message = "No recordings selected."
        return RecordingLoadResult(
            loaded_count=0,
            failures=(RecordingLoadFailure(path=None, message=message),),
            status_text=message,
        )

    state.is_loading = True
    state.defer_timeline_updates = True
    loaded_count = 0
    failures: list[RecordingLoadFailure] = []
    source_warnings: list[SourceUnavailableWarning] = []
    status_text = ""
    try:
        for path in paths:
            try:
                previous_loaded_count = len(state.loaded_recordings)
                single = _load_one_recording_path(
                    state,
                    path,
                    roi_file_to_load=(
                        roi_file_to_load if len(paths) == 1 else Path(DEFAULT_PATH_TEXT)
                    ),
                    replace_selected=replace_selected if len(paths) == 1 else False,
                    remember_selected_folder=remember_selected_folder,
                    resolve_recording=resolve_recording,
                    open_recording=open_recording,
                )
            except Exception as error:
                failures.append(
                    RecordingLoadFailure(
                        path=path,
                        message=exception_message_for_user(error),
                    ),
                )
                continue

            status_text = single.status_text
            source_warnings.extend(single.source_warnings)
            if len(state.loaded_recordings) > previous_loaded_count:
                loaded_count += 1
    finally:
        state.is_loading = False
        state.defer_timeline_updates = False
        if state.selected_recording_index is not None:
            select_loaded_recording(state, state.selected_recording_index)

    if status_text == "" and failures:
        status_text = failures[0].message
    return RecordingLoadResult(
        loaded_count=loaded_count,
        failures=tuple(failures),
        source_warnings=tuple(source_warnings),
        status_text=status_text,
    )


def _load_one_recording_path(
    state: RecordingLoadState,
    selected_recording_path: Path,
    *,
    roi_file_to_load: PathInput,
    replace_selected: bool,
    remember_selected_folder: bool,
    resolve_recording: Callable[[PathInput], ResolvedNapariRecording],
    open_recording: Callable[..., NapariRecordingView] | None,
) -> _SingleRecordingLoad:
    """Load one path and return its status plus cache warnings."""
    resolved_recording = resolve_recording(selected_recording_path)
    paths = resolved_recording.paths
    source_warnings: tuple[SourceUnavailableWarning, ...] = ()
    if resolved_recording.source_unavailable:
        source_warnings = (
            SourceUnavailableWarning(
                source_path=selected_recording_path,
                cache_path=paths.recording_data_path,
            ),
        )

    duplicate_index = _loaded_recording_index_for_path(
        state,
        paths.recording_data_path,
    )
    is_selected_replacement = (
        replace_selected and duplicate_index == state.selected_recording_index
    )
    if duplicate_index is not None and not is_selected_replacement:
        select_loaded_recording(state, duplicate_index)
        if remember_selected_folder:
            _remember_recording_folder(
                selected_recording_path,
                paths.recording_data_path,
            )
        return _SingleRecordingLoad(
            status_text=f"Already loaded {paths.recording_data_path}",
            source_warnings=source_warnings,
        )

    roi_path = (
        paths.roi_file_to_load
        if is_default_path(roi_file_to_load)
        else _resolve_optional_roi_path(roi_file_to_load)
    )

    resolved_open_recording = open_recording
    if resolved_open_recording is None:
        # Import here to avoid a top-level cycle: viewer creates controls,
        # while controls can load recordings into that same viewer.
        from twopy.napari.viewer import open_recording_in_napari

        resolved_open_recording = open_recording_in_napari

    view = resolved_open_recording(
        paths.recording_data_path,
        viewer=state.viewer,
        roi_set=roi_path,
        roi_save_file=paths.roi_save_file,
        movie_path=paths.movie_path,
        movie_frame_range=(0, None),
        add_controls=False,
    )
    record_loaded_view(
        state,
        view=view,
        roi_save_file=paths.roi_save_file,
        replace_selected=replace_selected,
    )
    if remember_selected_folder:
        _remember_recording_folder(
            selected_recording_path,
            paths.recording_data_path,
        )
    status = "Converted and loaded" if resolved_recording.was_converted else "Loaded"
    return _SingleRecordingLoad(
        status_text=f"{status} {paths.recording_data_path}",
        source_warnings=source_warnings,
    )


def _loaded_recording_index_for_path(
    state: RecordingLoadState,
    recording_data_path: Path,
) -> int | None:
    """Return the loaded-recording row for an already open recording."""
    requested = _recording_path_key(recording_data_path)
    for index, loaded in enumerate(state.loaded_recordings):
        if _recording_path_key(loaded.recording.path) == requested:
            return index
    return None


def _remember_recording_folder(
    selected_recording_path: Path,
    recording_data_path: Path,
) -> None:
    """Persist the next starting folder for manual recording selection."""
    write_last_recording_folder(
        recording_folder_for_state(
            selected_recording_path,
            recording_data_path,
        ),
    )


def _recording_path_key(path: Path) -> Path:
    """Return a stable comparison key for a converted recording path."""
    return path.expanduser().resolve(strict=False)


def _resolve_optional_roi_path(path: PathInput) -> Path | None:
    """Resolve an optional ROI file selected by the user."""
    if is_default_path(path):
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        msg = f"ROI file does not exist: {resolved}"
        raise ValueError(msg)
    return resolved.resolve()
