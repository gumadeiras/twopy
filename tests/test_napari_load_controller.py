"""Tests for the asynchronous napari recording-load controller.

Inputs: fake resolved recordings and patched worker/apply stages.
Outputs: assertions that the controller queues work, reports progress results,
and keeps the GUI load state coherent.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import cast
from unittest.mock import patch

import twopy.napari.load_controller as load_controller
from tests.napari_support import QApplication, process_qt_events_until, unittest
from twopy.converted import RecordingData
from twopy.napari.load_workflow import (
    AppliedRecordingLoad,
    PreparedRecordingLoad,
    ResolvedRecordingLoad,
)
from twopy.napari.loading import ResolvedNapariRecording
from twopy.napari.output_routing import NapariOutputRoute
from twopy.napari.paths import NapariRecordingPaths
from twopy.napari.protocols import NapariViewer
from twopy.napari.session import LoadedNapariRecording, LoadedRecordingsPanel
from twopy.napari.viewer import PreparedNapariRecordingViewData


@dataclass
class _State:
    """Small session-state double used by ``RecordingLoadController`` tests.

    Inputs: load-state flags and loaded-recording rows.
    Outputs: mutable fields the controller reads and resets.
    """

    viewer: NapariViewer = field(default_factory=lambda: cast(NapariViewer, object()))
    roi_labels_layer: object | None = None
    roi_save_file: Path = Path("rois.h5")
    recording: RecordingData | None = None
    output_route: NapariOutputRoute | None = None
    response_plot_widget: object | None = None
    trial_timeline_controller: object | None = None
    is_loading: bool = False
    defer_timeline_updates: bool = False
    loaded_recordings: list[LoadedNapariRecording] = field(default_factory=list)
    selected_recording_index: int | None = None
    loaded_recordings_panel: LoadedRecordingsPanel | None = None
    group_matching_panel: object | None = None
    recording_required_widgets: list[object] = field(default_factory=list)


class RecordingLoadControllerTest(unittest.TestCase):
    """Tests for queued asynchronous recording loads."""

    def test_batch_applies_paths_and_resets_state(self) -> None:
        """Confirm a batch resolves, prepares, applies, and reports results.

        Inputs: two selected paths and patched load stages.
        Outputs: two applied rows, one final result, and reset load flags.
        """
        _ = QApplication.instance() or QApplication([])
        state = _State()
        applied_paths: list[Path] = []
        results = []
        controller = load_controller.RecordingLoadController(
            state,
            on_finished=results.append,
        )

        with _patched_load_stages(applied_paths):
            started = controller.load_paths(
                (Path("/source/first"), Path("/source/second")),
                remember_selected_folder=False,
            )
            self.assertIsNone(started)
            self.assertTrue(state.is_loading)
            process_qt_events_until(lambda: len(results) == 1)

        self.assertEqual(
            applied_paths,
            [Path("/source/first"), Path("/source/second")],
        )
        self.assertEqual(results[0].loaded_count, 2)
        self.assertEqual(results[0].failures, ())
        self.assertEqual(results[0].status_text, "Loaded /source/second")
        self.assertFalse(state.is_loading)
        self.assertFalse(state.defer_timeline_updates)
        controller.shutdown()

    def test_active_load_rejects_second_batch(self) -> None:
        """Confirm one active batch blocks another load request.

        Inputs: one blocked worker job and a second load request.
        Outputs: the second request returns a structured in-progress failure.
        """
        _ = QApplication.instance() or QApplication([])
        state = _State()
        applied_paths: list[Path] = []
        results = []
        resolve_started = Event()
        release_resolve = Event()
        controller = load_controller.RecordingLoadController(
            state,
            on_finished=results.append,
        )

        with _patched_load_stages(
            applied_paths,
            resolve_started=resolve_started,
            release_resolve=release_resolve,
        ):
            controller.load_paths(
                (Path("/source/first"),),
                remember_selected_folder=False,
            )
            process_qt_events_until(resolve_started.is_set)

            rejected = controller.load_paths(
                (Path("/source/second"),),
                remember_selected_folder=False,
            )
            release_resolve.set()
            process_qt_events_until(lambda: len(results) == 1)

        assert rejected is not None
        self.assertEqual(rejected.loaded_count, 0)
        self.assertEqual(len(rejected.failures), 1)
        self.assertEqual(
            rejected.failures[0].message,
            "Recording load already in progress.",
        )
        self.assertEqual(applied_paths, [Path("/source/first")])
        self.assertFalse(state.is_loading)
        controller.shutdown()

    def test_progress_uses_resolved_source_path_after_resolve(self) -> None:
        """Confirm the progress dialog shows source identity after resolution.

        Inputs: a selected converted path whose resolved load names a source
        recording folder.
        Outputs: progress updates keep the queued row index but switch active
        path fields to the source recording.
        """
        _ = QApplication.instance() or QApplication([])
        state = _State()
        applied_paths: list[Path] = []
        results = []
        updates: list[dict[str, object]] = []
        selected_path = Path("/analysis-cache/genotype/stim/2026/06_12/18_42_11")
        source_path = Path(
            "/Volumes/DataNAS_2/2p_microscope_data/genotype/stim/2026/06_12/18_42_11",
        )
        controller = load_controller.RecordingLoadController(
            state,
            on_finished=results.append,
        )

        with (
            _patched_load_stages(applied_paths, source_session_dir=source_path),
            patch.object(
                load_controller,
                "RecordingLoadProgressDialog",
                lambda **kwargs: _ProgressDialogSpy(updates, **kwargs),
            ),
        ):
            controller.load_paths(
                (selected_path,),
                remember_selected_folder=False,
            )
            process_qt_events_until(lambda: len(results) == 1)

        resolved_updates = [
            update
            for update in updates
            if update["phase"]
            in {"Reading recording data...", "Adding recording layers..."}
        ]
        self.assertGreaterEqual(len(resolved_updates), 2)
        self.assertTrue(
            all(update["current_path"] == source_path for update in resolved_updates),
        )
        self.assertTrue(all(update["active_index"] == 0 for update in resolved_updates))
        controller.shutdown()

    def test_cancel_pending_finishes_active_and_skips_queue(self) -> None:
        """Confirm cancellation skips queued paths after the active load.

        Inputs: two selected paths and a blocked first resolve stage.
        Outputs: only the active path is applied and the final result reports it.
        """
        _ = QApplication.instance() or QApplication([])
        state = _State()
        applied_paths: list[Path] = []
        results = []
        resolve_started = Event()
        release_resolve = Event()
        controller = load_controller.RecordingLoadController(
            state,
            on_finished=results.append,
        )

        with _patched_load_stages(
            applied_paths,
            resolve_started=resolve_started,
            release_resolve=release_resolve,
        ):
            controller.load_paths(
                (Path("/source/first"), Path("/source/second")),
                remember_selected_folder=False,
            )
            process_qt_events_until(resolve_started.is_set)

            controller.cancel_pending()
            release_resolve.set()
            process_qt_events_until(lambda: len(results) == 1)

        self.assertEqual(applied_paths, [Path("/source/first")])
        self.assertEqual(results[0].loaded_count, 1)
        self.assertEqual(results[0].failures, ())
        self.assertFalse(state.is_loading)
        controller.shutdown()


def _patched_load_stages(
    applied_paths: list[Path],
    *,
    resolve_started: Event | None = None,
    release_resolve: Event | None = None,
    source_session_dir: Path | None = None,
) -> AbstractContextManager[dict[str, object]]:
    """Patch controller load stages with deterministic test doubles.

    Args:
        applied_paths: List that receives paths applied on the Qt thread.
        resolve_started: Optional event set when the first resolve begins.
        release_resolve: Optional event that lets the first resolve continue.
        source_session_dir: Optional source path to attach to each resolved
            fake.

    Returns:
        Context manager that patches resolve, prepare, and apply stages.
    """

    def resolve(
        path: Path,
        *,
        protected_cache_roots: tuple[Path, ...] = (),
    ) -> ResolvedRecordingLoad:
        del protected_cache_roots
        if resolve_started is not None and not resolve_started.is_set():
            resolve_started.set()
            assert release_resolve is not None
            release_resolve.wait(timeout=2.0)
        return _resolved_load(path, source_session_dir=source_session_dir)

    def prepare(
        resolved: ResolvedRecordingLoad,
        *,
        roi_file_to_load: object,
    ) -> PreparedRecordingLoad:
        del roi_file_to_load
        return PreparedRecordingLoad(
            resolved,
            cast(PreparedNapariRecordingViewData, object()),
        )

    def apply(
        state: _State,
        prepared: PreparedRecordingLoad,
        *,
        replace_selected: bool,
        remember_selected_folder: bool,
    ) -> AppliedRecordingLoad:
        del replace_selected, remember_selected_folder
        state.loaded_recordings.append(_loaded_recording(prepared.selected_path))
        applied_paths.append(prepared.selected_path)
        return AppliedRecordingLoad(status_text=f"Loaded {prepared.selected_path}")

    def select_duplicate(*_args: object, **_kwargs: object) -> None:
        """Return no duplicate so the controller always prepares the path."""
        return None

    return patch.multiple(
        load_controller,
        resolve_recording_load=resolve,
        select_duplicate_recording_load=select_duplicate,
        prepare_resolved_recording_load=prepare,
        apply_prepared_recording_load=apply,
    )


def _resolved_load(
    selected_path: Path,
    *,
    source_session_dir: Path | None = None,
) -> ResolvedRecordingLoad:
    """Return a resolved-load fake for one selected path."""
    converted_path = selected_path / "recording_data.h5"
    paths = NapariRecordingPaths(
        recording_data_path=converted_path,
        movie_path=selected_path / "aligned_movie.h5",
        roi_file_to_load=None,
        roi_save_file=selected_path / "rois.h5",
    )
    return ResolvedRecordingLoad(
        selected_path=selected_path,
        resolved_recording=ResolvedNapariRecording(
            paths=paths,
            output_route=NapariOutputRoute(
                local_root=selected_path,
                publish_root=selected_path,
            ),
            was_converted=False,
            source_session_dir=source_session_dir,
        ),
    )


class _ProgressDialogSpy:
    """Capture progress-dialog updates without showing a Qt dialog."""

    def __init__(
        self,
        updates: list[dict[str, object]],
        *,
        paths: tuple[Path, ...],
        on_cancel_pending: object,
    ) -> None:
        """Store constructor inputs and a mutable update list."""
        del on_cancel_pending
        self.paths = paths
        self._updates = updates

    def update_progress(self, **kwargs: object) -> None:
        """Record one progress update."""
        self._updates.append(dict(kwargs))

    def show(self) -> None:
        """Match the dialog API used by the controller."""

    def raise_(self) -> None:
        """Match the dialog API used by the controller."""

    def mark_finishing(self, *, phase: str) -> None:
        """Record final progress state."""
        self._updates.append({"phase": phase, "mark_finishing": True})

    def close(self) -> None:
        """Match the QTimer close callback target."""


def _loaded_recording(path: Path) -> LoadedNapariRecording:
    """Return a loaded-recording row with the path as its local root."""
    return LoadedNapariRecording(
        recording=cast(RecordingData, object()),
        output_route=NapariOutputRoute(
            local_root=path,
            publish_root=path,
        ),
        roi_save_file=path / "rois.h5",
        mean_image_layer=object(),
        movie_layer=None,
        roi_labels_layer=None,
    )


if __name__ == "__main__":
    unittest.main()
