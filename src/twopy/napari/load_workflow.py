"""Recording-load workflow for the twopy napari Load tab.

Inputs: selected recording paths and the current napari session state.
Outputs: loaded recordings, structured failures, and source-cache warnings.

This module owns the non-Qt decisions in the Load workflow: source or converted
path resolution, duplicate selection, selected-row replacement, loaded-count
accounting, and per-path failure collection. The controls module chooses paths
and renders dialogs; this module changes the recording session.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from twopy.napari.errors import exception_message_for_user
from twopy.napari.loading import (
    ResolvedNapariRecording,
    reconvert_recording_to_output,
    resolve_or_convert_recording,
)
from twopy.napari.output_routing import NapariOutputRoute
from twopy.napari.paths import (
    DEFAULT_PATH_TEXT,
    PathInput,
    is_default_path,
)
from twopy.napari.session import (
    NapariSessionState,
    record_loaded_view,
    select_loaded_recording,
    selected_loaded_recording,
)
from twopy.napari.state import recording_folder_for_state, write_last_recording_folder

if TYPE_CHECKING:
    from twopy.napari.viewer import PreparedNapariRecordingViewData

__all__ = [
    "RecordingLoadFailure",
    "RecordingLoadResult",
    "AppliedRecordingLoad",
    "PreparedRecordingLoad",
    "ResolvedRecordingLoad",
    "SourceUnavailableWarning",
    "apply_prepared_recording_load",
    "load_recording_paths",
    "prepare_resolved_recording_load",
    "reconvert_selected_recording",
    "resolve_recording_load",
    "select_duplicate_recording_load",
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
class ResolvedRecordingLoad:
    """Resolved paths for one selected recording before movie reads.

    Inputs: one selected source or converted path.
    Outputs: converted twopy paths and cache warning details.
    """

    selected_path: Path
    resolved_recording: ResolvedNapariRecording

    @property
    def source_warnings(self) -> tuple[SourceUnavailableWarning, ...]:
        """Return cache-backed source warnings for this resolved load."""
        if not self.resolved_recording.source_unavailable:
            return ()
        return (
            SourceUnavailableWarning(
                source_path=self.selected_path,
                cache_path=self.resolved_recording.paths.recording_data_path,
            ),
        )


@dataclass(frozen=True)
class PreparedRecordingLoad:
    """Worker-prepared data for one recording before Qt layer creation.

    Inputs: resolved recording paths.
    Outputs: resolved paths plus arrays that can be applied on the Qt thread.
    """

    resolved_load: ResolvedRecordingLoad
    view_data: PreparedNapariRecordingViewData

    @property
    def selected_path(self) -> Path:
        """Return the originally selected source or converted path."""
        return self.resolved_load.selected_path

    @property
    def resolved_recording(self) -> ResolvedNapariRecording:
        """Return resolved converted paths and conversion status."""
        return self.resolved_load.resolved_recording

    @property
    def source_warnings(self) -> tuple[SourceUnavailableWarning, ...]:
        """Return source warnings inherited from the resolved load."""
        return self.resolved_load.source_warnings


@dataclass(frozen=True)
class AppliedRecordingLoad:
    """Result for one applied recording before batch failure aggregation."""

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
    prepare_view_data: Callable[..., PreparedNapariRecordingViewData] | None = None,
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
        prepare_view_data: Worker-safe view-data reader. Tests can provide a
            fake; production reads converted HDF5 files.

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
                resolved = resolve_recording_load(
                    path,
                    resolve_recording=resolve_recording,
                )
                single = select_duplicate_recording_load(
                    state,
                    resolved,
                    replace_selected=replace_selected if len(paths) == 1 else False,
                    remember_selected_folder=remember_selected_folder,
                )
                if single is None:
                    prepared = prepare_resolved_recording_load(
                        resolved,
                        roi_file_to_load=(
                            roi_file_to_load
                            if len(paths) == 1
                            else Path(DEFAULT_PATH_TEXT)
                        ),
                        prepare_view_data=prepare_view_data,
                    )
                    single = apply_prepared_recording_load(
                        state,
                        prepared,
                        replace_selected=replace_selected if len(paths) == 1 else False,
                        remember_selected_folder=remember_selected_folder,
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


def resolve_recording_load(
    selected_recording_path: Path,
    *,
    resolve_recording: Callable[[PathInput], ResolvedNapariRecording] = (
        resolve_or_convert_recording
    ),
) -> ResolvedRecordingLoad:
    """Resolve one selected recording path without reading movie arrays."""
    resolved_recording = resolve_recording(selected_recording_path)
    return ResolvedRecordingLoad(
        selected_path=selected_recording_path,
        resolved_recording=resolved_recording,
    )


def select_duplicate_recording_load(
    state: RecordingLoadState,
    resolved: ResolvedRecordingLoad,
    *,
    replace_selected: bool,
    remember_selected_folder: bool,
) -> AppliedRecordingLoad | None:
    """Select an existing loaded recording when the resolved path is a duplicate.

    Args:
        state: Mutable napari session state.
        resolved: Resolved recording paths.
        replace_selected: Whether a selected duplicate should be reloaded.
        remember_selected_folder: Whether to update manual picker memory.

    Returns:
        Applied result for a duplicate selection, or ``None`` when the caller
        should continue preparing view data.
    """
    paths = resolved.resolved_recording.paths
    duplicate_index = _loaded_recording_index_for_path(
        state,
        paths.recording_data_path,
    )
    is_selected_replacement = (
        replace_selected and duplicate_index == state.selected_recording_index
    )
    if duplicate_index is None or is_selected_replacement:
        return None

    select_loaded_recording(state, duplicate_index)
    if remember_selected_folder:
        _remember_recording_folder(
            resolved.selected_path,
            paths.recording_data_path,
        )
    return AppliedRecordingLoad(
        status_text=f"Already loaded {paths.recording_data_path}",
        source_warnings=resolved.source_warnings,
    )


def prepare_resolved_recording_load(
    resolved: ResolvedRecordingLoad,
    *,
    roi_file_to_load: PathInput = Path(DEFAULT_PATH_TEXT),
    prepare_view_data: Callable[..., PreparedNapariRecordingViewData] | None = None,
) -> PreparedRecordingLoad:
    """Read recording arrays for one already-resolved recording.

    Args:
        resolved: Resolved converted paths from ``resolve_recording_load``.
        roi_file_to_load: Optional ROI HDF5 file. ``default`` uses the resolved
            recording's saved ROI path when it exists.
        prepare_view_data: Worker-safe recording view data reader.

    Returns:
        Prepared load data ready for ``apply_prepared_recording_load``.
    """
    resolved_prepare_view_data = prepare_view_data
    if resolved_prepare_view_data is None:
        from twopy.napari.viewer import prepare_recording_view_data

        resolved_prepare_view_data = prepare_recording_view_data
    paths = resolved.resolved_recording.paths
    roi_path = (
        paths.roi_file_to_load
        if is_default_path(roi_file_to_load)
        else _resolve_optional_roi_path(roi_file_to_load)
    )
    view_data = resolved_prepare_view_data(
        paths.recording_data_path,
        roi_set=roi_path,
        movie_path=paths.movie_path,
        movie_frame_range=(0, None),
    )
    return PreparedRecordingLoad(
        resolved_load=resolved,
        view_data=view_data,
    )


def apply_prepared_recording_load(
    state: RecordingLoadState,
    prepared: PreparedRecordingLoad,
    *,
    replace_selected: bool,
    remember_selected_folder: bool,
) -> AppliedRecordingLoad:
    """Apply prepared recording data to napari on the Qt thread.

    Args:
        state: Mutable napari session state.
        prepared: Worker-prepared recording arrays and paths.
        replace_selected: Whether to replace the selected loaded row.
        remember_selected_folder: Whether to update the manual picker folder.

    Returns:
        Status text plus any source-cache warning.
    """
    paths = prepared.resolved_recording.paths
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
                prepared.selected_path,
                paths.recording_data_path,
            )
        return AppliedRecordingLoad(
            status_text=f"Already loaded {paths.recording_data_path}",
            source_warnings=prepared.source_warnings,
        )

    from twopy.napari.viewer import add_prepared_recording_to_viewer

    view = add_prepared_recording_to_viewer(prepared.view_data, viewer=state.viewer)
    record_loaded_view(
        state,
        view=view,
        output_route=prepared.resolved_recording.output_route,
        roi_save_file=paths.roi_save_file,
        replace_selected=replace_selected,
    )
    if remember_selected_folder:
        _remember_recording_folder(
            prepared.selected_path,
            paths.recording_data_path,
        )
    status = (
        "Converted and loaded"
        if prepared.resolved_recording.was_converted
        else "Loaded"
    )
    return AppliedRecordingLoad(
        status_text=f"{status} {paths.recording_data_path}",
        source_warnings=prepared.source_warnings,
    )


def reconvert_selected_recording(
    state: RecordingLoadState,
    *,
    reconvert_recording: Callable[
        [Path, Path, NapariOutputRoute],
        ResolvedNapariRecording,
    ] = reconvert_recording_to_output,
    prepare_view_data: Callable[..., PreparedNapariRecordingViewData] | None = None,
) -> RecordingLoadResult:
    """Reconvert the selected loaded recording and reload it in place.

    Args:
        state: Mutable napari session state.
        reconvert_recording: Conversion callback. Tests can provide a fake;
            production rewrites the selected output folder from source data
            while preserving the selected recording's output route.
        prepare_view_data: Worker-safe view-data reader. Tests can provide a
            fake; production reads converted HDF5 files.

    Returns:
        Structured result with status text or one failure.

    The action is intentionally tied to the selected loaded row. That keeps the
    overwrite target auditable: source data comes from the recording metadata
    and converted files are rewritten in the same folder currently being read.
    """
    if state.is_loading:
        message = "Recording load already in progress."
        return RecordingLoadResult(
            loaded_count=0,
            failures=(RecordingLoadFailure(path=None, message=message),),
            status_text=message,
        )

    selected = selected_loaded_recording(state)
    if selected is None:
        message = "No recording selected."
        return RecordingLoadResult(
            loaded_count=0,
            failures=(RecordingLoadFailure(path=None, message=message),),
            status_text=message,
        )

    source_dir = selected.recording.source_session_dir.expanduser()
    output_dir = selected.recording.path.parent.expanduser()
    state.is_loading = True
    state.defer_timeline_updates = True
    try:
        try:
            resolved = reconvert_recording(
                source_dir,
                output_dir,
                selected.output_route,
            )
            prepared = prepare_resolved_recording_load(
                ResolvedRecordingLoad(
                    selected_path=source_dir,
                    resolved_recording=resolved,
                ),
                prepare_view_data=prepare_view_data,
            )
            apply_prepared_recording_load(
                state,
                prepared,
                replace_selected=True,
                remember_selected_folder=False,
            )
        except Exception as error:
            message = exception_message_for_user(error)
            return RecordingLoadResult(
                loaded_count=0,
                failures=(RecordingLoadFailure(path=source_dir, message=message),),
                status_text=message,
            )
    finally:
        state.is_loading = False
        state.defer_timeline_updates = False
        if state.selected_recording_index is not None:
            select_loaded_recording(state, state.selected_recording_index)

    return RecordingLoadResult(
        loaded_count=0,
        status_text=f"Reconverted and loaded {resolved.paths.recording_data_path}",
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
