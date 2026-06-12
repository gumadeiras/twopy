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
) -> AbstractContextManager[dict[str, object]]:
    """Patch controller load stages with deterministic test doubles.

    Args:
        applied_paths: List that receives paths applied on the Qt thread.
        resolve_started: Optional event set when the first resolve begins.
        release_resolve: Optional event that lets the first resolve continue.

    Returns:
        Context manager that patches resolve, prepare, and apply stages.
    """

    def resolve(path: Path) -> ResolvedRecordingLoad:
        if resolve_started is not None and not resolve_started.is_set():
            resolve_started.set()
            assert release_resolve is not None
            release_resolve.wait(timeout=2.0)
        return _resolved_load(path)

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
        state.loaded_recordings.append(cast(LoadedNapariRecording, object()))
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


def _resolved_load(selected_path: Path) -> ResolvedRecordingLoad:
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
        ),
    )


if __name__ == "__main__":
    unittest.main()
