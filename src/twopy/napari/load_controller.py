"""Asynchronous controller for napari recording loads.

Inputs: selected recording paths and the mutable napari control state.
Outputs: prepared recordings loaded into napari without blocking the Qt event
loop during conversion, cache refresh, or HDF5 reads.

The controller keeps Qt ownership simple: worker threads only prepare plain
data, and the Qt thread applies each finished recording to napari.
"""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

from twopy.napari.errors import exception_message_for_user
from twopy.napari.load_progress import RecordingLoadProgressDialog
from twopy.napari.load_workflow import (
    AppliedRecordingLoad,
    PreparedRecordingLoad,
    RecordingLoadFailure,
    RecordingLoadResult,
    RecordingLoadState,
    ResolvedRecordingLoad,
    SourceUnavailableWarning,
    apply_prepared_recording_load,
    prepare_resolved_recording_load,
    resolve_recording_load,
    select_duplicate_recording_load,
)
from twopy.napari.paths import DEFAULT_PATH_TEXT, PathInput
from twopy.napari.session import loaded_recording_cache_roots, select_loaded_recording

__all__ = ["RecordingLoadController"]


@dataclass(frozen=True)
class _QueuedRecordingLoad:
    """One selected recording waiting for worker preparation."""

    path: Path
    roi_file_to_load: PathInput
    replace_selected: bool


class RecordingLoadController:
    """Queue recording loads through a worker and apply them on the Qt thread.

    Args:
        state: Mutable napari control state.
        on_finished: Callback receiving the final structured result.

    Outputs:
        Started load batches, progress dialog updates, and final load results.
    """

    def __init__(
        self,
        state: RecordingLoadState,
        *,
        on_finished: Callable[[RecordingLoadResult], None],
    ) -> None:
        """Create a queued load controller."""
        self._state = state
        self._on_finished = on_finished
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="twopy-recording-load",
        )
        self._future: Future[ResolvedRecordingLoad | PreparedRecordingLoad] | None = (
            None
        )
        self._queue: list[_QueuedRecordingLoad] = []
        self._active: _QueuedRecordingLoad | None = None
        self._active_stage: str | None = None
        self._progress_dialog: RecordingLoadProgressDialog | None = None
        self._poll_timer = QTimer()
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._collect_finished_job)
        self._total_count = 0
        self._completed_count = 0
        self._loaded_count = 0
        self._failures: list[RecordingLoadFailure] = []
        self._source_warnings: list[SourceUnavailableWarning] = []
        self._status_text = ""
        self._remember_selected_folder = False
        self._cancel_pending = False
        self._after_finished: Callable[[], None] | None = None
        self._quit_application = QApplication.instance()
        if self._quit_application is not None:
            self._quit_application.aboutToQuit.connect(self.shutdown)

    def load_paths(
        self,
        paths: tuple[Path, ...],
        *,
        remember_selected_folder: bool,
        roi_file_to_load: PathInput = Path(DEFAULT_PATH_TEXT),
        replace_selected: bool = False,
        after_finished: Callable[[], None] | None = None,
    ) -> RecordingLoadResult | None:
        """Start loading selected paths asynchronously.

        Args:
            paths: Recording folders or converted recording paths to load.
            remember_selected_folder: Whether successful loads update the next
                manual picker starting folder.
            roi_file_to_load: Optional ROI HDF5 file for a single load.
            replace_selected: Whether a single load replaces the selected row.
            after_finished: Optional callback run after the final result is
                delivered. It is used for CSV-adjacent group-matching defaults.

        Returns:
            Immediate failure result, or ``None`` when a batch started.
        """
        if self._state.is_loading:
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

        self._state.is_loading = True
        self._state.defer_timeline_updates = True
        self._remember_selected_folder = remember_selected_folder
        self._cancel_pending = False
        self._total_count = len(paths)
        self._completed_count = 0
        self._loaded_count = 0
        self._failures = []
        self._source_warnings = []
        self._status_text = ""
        self._after_finished = after_finished
        self._queue = [
            _QueuedRecordingLoad(
                path=path,
                roi_file_to_load=(
                    roi_file_to_load if len(paths) == 1 else Path(DEFAULT_PATH_TEXT)
                ),
                replace_selected=replace_selected if len(paths) == 1 else False,
            )
            for path in paths
        ]
        self._show_progress_dialog()
        self._start_next_job()
        return None

    def cancel_pending(self) -> None:
        """Cancel queued recordings after the active worker job finishes."""
        self._cancel_pending = True
        self._queue.clear()

    def shutdown(self) -> None:
        """Stop polling and release worker resources."""
        self.cancel_pending()
        self._poll_timer.stop()
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _show_progress_dialog(self) -> None:
        """Open or reset the modeless progress dialog."""
        dialog = RecordingLoadProgressDialog(
            total_count=self._total_count,
            on_cancel_pending=self.cancel_pending,
        )
        self._progress_dialog = dialog
        dialog.update_progress(
            completed_count=0,
            total_count=self._total_count,
            current_path=None,
            phase="Preparing load queue...",
        )
        dialog.show()
        dialog.raise_()

    def _start_next_job(self) -> None:
        """Start preparing the next queued recording."""
        if self._future is not None:
            return
        if self._cancel_pending or len(self._queue) == 0:
            self._finish()
            return

        self._active = self._queue.pop(0)
        self._update_progress(
            current_path=self._active.path,
            phase="Preparing recording data...",
        )
        active = self._active
        protected_roots = loaded_recording_cache_roots(self._state.loaded_recordings)
        self._future = self._executor.submit(
            resolve_recording_load,
            active.path,
            protected_cache_roots=protected_roots,
        )
        self._active_stage = "resolve"
        self._poll_timer.start()

    def _collect_finished_job(self) -> None:
        """Apply a finished prepared recording on the Qt thread."""
        future = self._future
        if future is None or not future.done():
            return

        active = self._active
        stage = self._active_stage
        self._future = None
        self._active_stage = None
        self._poll_timer.stop()
        if active is None or stage is None:
            self._active = None
            self._finish()
            return

        try:
            result = future.result()
        except Exception as error:
            self._failures.append(
                RecordingLoadFailure(
                    path=active.path,
                    message=exception_message_for_user(error),
                ),
            )
            self._complete_active(active)
            return

        if stage == "resolve":
            self._handle_resolved_recording(active, result)
            return
        if stage == "prepare":
            self._handle_prepared_recording(active, result)
            return

        self._failures.append(
            RecordingLoadFailure(
                path=active.path,
                message=f"Unknown recording-load stage: {stage}",
            ),
        )
        self._complete_active(active)

    def _handle_resolved_recording(
        self,
        active: _QueuedRecordingLoad,
        result: ResolvedRecordingLoad | PreparedRecordingLoad,
    ) -> None:
        """Select a duplicate or start array preparation for a resolved path."""
        if not isinstance(result, ResolvedRecordingLoad):
            self._failures.append(
                RecordingLoadFailure(
                    path=active.path,
                    message=f"Expected resolved recording, got {type(result).__name__}",
                ),
            )
            self._complete_active(active)
            return

        applied = select_duplicate_recording_load(
            self._state,
            result,
            replace_selected=active.replace_selected,
            remember_selected_folder=self._remember_selected_folder,
        )
        if applied is not None:
            self._record_applied_load(applied, previous_loaded_count=None)
            self._complete_active(active)
            return

        self._update_progress(
            current_path=active.path,
            phase="Reading recording data...",
        )
        self._future = self._executor.submit(
            prepare_resolved_recording_load,
            result,
            roi_file_to_load=active.roi_file_to_load,
        )
        self._active_stage = "prepare"
        self._poll_timer.start()

    def _handle_prepared_recording(
        self,
        active: _QueuedRecordingLoad,
        result: ResolvedRecordingLoad | PreparedRecordingLoad,
    ) -> None:
        """Apply one prepared recording to napari."""
        if not isinstance(result, PreparedRecordingLoad):
            self._failures.append(
                RecordingLoadFailure(
                    path=active.path,
                    message=f"Expected prepared recording, got {type(result).__name__}",
                ),
            )
            self._complete_active(active)
            return

        self._update_progress(
            current_path=active.path,
            phase="Adding recording layers...",
        )
        try:
            previous_loaded_count = len(self._state.loaded_recordings)
            applied = apply_prepared_recording_load(
                self._state,
                result,
                replace_selected=active.replace_selected,
                remember_selected_folder=self._remember_selected_folder,
            )
        except Exception as error:
            self._failures.append(
                RecordingLoadFailure(
                    path=active.path,
                    message=exception_message_for_user(error),
                ),
            )
        else:
            self._record_applied_load(
                applied,
                previous_loaded_count=previous_loaded_count,
            )
        self._complete_active(active)

    def _record_applied_load(
        self,
        applied: AppliedRecordingLoad,
        *,
        previous_loaded_count: int | None,
    ) -> None:
        """Merge one applied load into the current batch result."""
        self._status_text = applied.status_text
        self._source_warnings.extend(applied.source_warnings)
        if previous_loaded_count is None:
            return
        if len(self._state.loaded_recordings) > previous_loaded_count:
            self._loaded_count += 1

    def _complete_active(self, active: _QueuedRecordingLoad) -> None:
        """Mark the current queue item complete and continue the batch."""
        self._active = None
        self._completed_count += 1
        self._update_progress(
            current_path=active.path,
            phase="Recording finished.",
        )
        self._start_next_job()

    def _finish(self) -> None:
        """Close the batch and report the final result."""
        self._poll_timer.stop()
        self._state.is_loading = False
        self._state.defer_timeline_updates = False
        if self._state.selected_recording_index is not None:
            select_loaded_recording(self._state, self._state.selected_recording_index)

        if self._status_text == "" and self._failures:
            self._status_text = self._failures[0].message
        result = RecordingLoadResult(
            loaded_count=self._loaded_count,
            failures=tuple(self._failures),
            source_warnings=tuple(self._source_warnings),
            status_text=self._status_text,
        )
        dialog = self._progress_dialog
        if dialog is not None:
            phase = "Load complete." if not self._cancel_pending else "Load stopped."
            dialog.update_progress(
                completed_count=self._completed_count,
                total_count=self._total_count,
                current_path=None,
                phase=phase,
            )
            dialog.mark_finishing(phase=phase)
            QTimer.singleShot(700, dialog.close)
            self._progress_dialog = None
        self._on_finished(result)
        if self._after_finished is not None:
            self._after_finished()
        self._after_finished = None

    def _update_progress(self, *, current_path: Path | None, phase: str) -> None:
        """Refresh the progress dialog when it exists."""
        if self._progress_dialog is None:
            return
        self._progress_dialog.update_progress(
            completed_count=self._completed_count,
            total_count=self._total_count,
            current_path=current_path,
            phase=phase,
        )
