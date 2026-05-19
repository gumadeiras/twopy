"""Tests for the napari recording-load workflow.

Inputs: fake converted recording paths, resolver outcomes, and a small napari
state double.
Outputs: loaded-recording session state, structured failures, and source-cache
warnings.
"""

from __future__ import annotations

import unittest
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from tests.tempdir import temporary_directory

from twopy.conversion.types import FrameCountAudit
from twopy.converted import ConvertedMovie, RecordingData
from twopy.napari.load_workflow import (
    load_recording_paths,
    reconvert_selected_recording,
)
from twopy.napari.loading import ResolvedNapariRecording
from twopy.napari.paths import NapariRecordingPaths, PathInput
from twopy.napari.protocols import NapariViewer
from twopy.napari.session import LoadedNapariRecording, LoadedRecordingsPanel
from twopy.napari.types import NapariRecordingView
from twopy.spatial import SpatialCrop


@dataclass
class _Layer:
    """Small layer double with the visibility attribute session code toggles.

    Inputs: layer name.
    Outputs: mutable layer object tracked by the fake viewer.
    """

    name: str
    visible: bool = True


class _Window:
    """Small napari window double.

    Inputs: dock-widget requests.
    Outputs: inert dock placeholders.
    """

    def add_dock_widget(
        self,
        widget: object,
        *,
        name: str,
        area: str,
    ) -> object:
        """Return a simple dock placeholder.

        Args:
            widget: Widget object being docked.
            name: Dock title.
            area: Dock area.

        Returns:
            A tuple that preserves the dock request.
        """
        return (name, area, widget)


@dataclass
class _Viewer:
    """Small napari viewer double used by recording-load tests.

    Inputs: image and Labels layer additions.
    Outputs: lists that session helpers can mutate.
    """

    images: list[_Layer] = field(default_factory=list)
    labels: list[_Layer] = field(default_factory=list)
    window: _Window = field(default_factory=_Window)

    def add_image(self, data: object, *, name: str, **kwargs: object) -> object:
        """Record an image-layer request.

        Args:
            data: Image payload.
            name: Layer name.
            kwargs: Napari options.

        Returns:
            Created fake layer.
        """
        del data, kwargs
        layer = _Layer(name=name)
        self.images.append(layer)
        return layer

    def add_labels(self, data: object, *, name: str, **kwargs: object) -> object:
        """Record a Labels-layer request.

        Args:
            data: Label image.
            name: Layer name.
            kwargs: Napari options.

        Returns:
            Created fake layer.
        """
        del data, kwargs
        layer = _Layer(name=name)
        self.labels.append(layer)
        return layer


@dataclass
class _State:
    """Mutable subset of napari control state required by load workflow.

    Inputs: fake viewer plus session bookkeeping fields.
    Outputs: state mutated by ``load_recording_paths``.
    """

    viewer: NapariViewer = field(default_factory=_Viewer)
    roi_labels_layer: object | None = None
    roi_save_file: Path = Path("rois.h5")
    recording: RecordingData | None = None
    response_plot_widget: object | None = None
    trial_timeline_controller: object | None = None
    defer_timeline_updates: bool = False
    loaded_recordings: list[LoadedNapariRecording] = field(default_factory=list)
    selected_recording_index: int | None = None
    loaded_recordings_panel: LoadedRecordingsPanel | None = None
    group_matching_panel: object | None = None
    recording_required_widgets: list[object] = field(default_factory=list)
    is_loading: bool = False


class LoadWorkflowTest(unittest.TestCase):
    """Tests for the GUI-independent recording-load coordinator."""

    def test_empty_selection_returns_structured_failure(self) -> None:
        """Confirm empty path selections fail without changing load state.

        Inputs: no selected paths.
        Outputs: one structured failure and ``is_loading`` reset to ``False``.
        """
        state = _State()

        result = load_recording_paths(
            state,
            (),
            remember_selected_folder=False,
            open_recording=_open_recording,
        )

        self.assertEqual(result.loaded_count, 0)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].message, "No recordings selected.")
        self.assertFalse(state.is_loading)

    def test_loads_one_recording_and_selects_it(self) -> None:
        """Confirm one resolved recording becomes the active session row.

        Inputs: one selected source path and a resolver mapping.
        Outputs: one loaded row, selected recording state, and viewer layers.
        """
        state = _State()
        viewer = cast(_Viewer, state.viewer)
        source = Path("/source/first")
        resolved = _resolved_recording(Path("/cache/first/recording_data.h5"))

        result = load_recording_paths(
            state,
            (source,),
            remember_selected_folder=False,
            resolve_recording=_resolver({source: resolved}),
            open_recording=_open_recording,
        )

        self.assertEqual(result.loaded_count, 1)
        self.assertEqual(result.failures, ())
        self.assertIn("/cache/first/recording_data.h5", result.status_text)
        self.assertEqual(state.selected_recording_index, 0)
        self.assertIsNotNone(state.recording)
        assert state.recording is not None
        self.assertEqual(state.recording.path, Path("/cache/first/recording_data.h5"))
        self.assertEqual(len(state.loaded_recordings), 1)
        self.assertEqual(len(viewer.images), 2)
        self.assertEqual(len(viewer.labels), 1)

    def test_duplicate_load_selects_existing_recording(self) -> None:
        """Confirm an already loaded recording is selected instead of reopened.

        Inputs: two loaded recordings and a second request for the first path.
        Outputs: unchanged loaded rows with the duplicate row selected.
        """
        state = _State()
        first = Path("/source/first")
        second = Path("/source/second")
        resolver = _resolver(
            {
                first: _resolved_recording(Path("/cache/first/recording_data.h5")),
                second: _resolved_recording(Path("/cache/second/recording_data.h5")),
            },
        )
        for path in (first, second):
            load_recording_paths(
                state,
                (path,),
                remember_selected_folder=False,
                resolve_recording=resolver,
                open_recording=_open_recording,
            )

        result = load_recording_paths(
            state,
            (first,),
            remember_selected_folder=False,
            resolve_recording=resolver,
            open_recording=_open_recording,
        )

        self.assertEqual(result.loaded_count, 0)
        self.assertEqual(len(state.loaded_recordings), 2)
        self.assertEqual(state.selected_recording_index, 0)
        self.assertTrue(
            cast(_Layer, state.loaded_recordings[0].mean_image_layer).visible,
        )
        self.assertFalse(
            cast(_Layer, state.loaded_recordings[1].mean_image_layer).visible,
        )
        self.assertIn("Already loaded", result.status_text)

    def test_replace_selected_recording_removes_old_layers(self) -> None:
        """Confirm single-load replacement swaps state and removes old layers.

        Inputs: one loaded recording and a replacement path.
        Outputs: one loaded row for the new recording and no old viewer layers.
        """
        state = _State()
        viewer = cast(_Viewer, state.viewer)
        first = Path("/source/first")
        second = Path("/source/second")
        resolver = _resolver(
            {
                first: _resolved_recording(Path("/cache/first/recording_data.h5")),
                second: _resolved_recording(Path("/cache/second/recording_data.h5")),
            },
        )
        load_recording_paths(
            state,
            (first,),
            remember_selected_folder=False,
            resolve_recording=resolver,
            open_recording=_open_recording,
        )
        old_layers = (
            *viewer.images,
            *viewer.labels,
        )

        result = load_recording_paths(
            state,
            (second,),
            remember_selected_folder=False,
            replace_selected=True,
            resolve_recording=resolver,
            open_recording=_open_recording,
        )

        self.assertEqual(result.loaded_count, 0)
        self.assertEqual(len(state.loaded_recordings), 1)
        self.assertEqual(state.selected_recording_index, 0)
        self.assertIsNotNone(state.recording)
        assert state.recording is not None
        self.assertEqual(state.recording.path, Path("/cache/second/recording_data.h5"))
        self.assertEqual(len(viewer.images), 2)
        self.assertEqual(len(viewer.labels), 1)
        for layer in old_layers:
            for remaining_layer in (*viewer.images, *viewer.labels):
                self.assertIsNot(layer, remaining_layer)

    def test_roi_reload_replaces_selected_duplicate_with_requested_roi(self) -> None:
        """Confirm ROI reload opens the selected recording instead of deduping.

        Inputs: one loaded recording, an alternate ROI file, and a replacement
        request for the same recording path.
        Outputs: one loaded row remains and the opener receives the explicit
        ROI path.
        """
        with temporary_directory() as temp_dir:
            state = _State()
            source = Path("/source/first")
            recording_path = Path("/cache/first/recording_data.h5")
            roi_path = Path(temp_dir) / "alternate_rois.h5"
            roi_path.touch()
            opened_roi_paths: list[Path | None] = []

            def open_recording(
                recording_data_path: Path,
                *,
                viewer: _Viewer,
                roi_set: Path | None,
                roi_save_file: Path,
                movie_path: Path | None,
                movie_frame_range: tuple[int, int | None],
                add_controls: bool,
            ) -> NapariRecordingView:
                """Record the ROI path and delegate to the fake opener."""
                opened_roi_paths.append(roi_set)
                return _open_recording(
                    recording_data_path,
                    viewer=viewer,
                    roi_set=roi_set,
                    roi_save_file=roi_save_file,
                    movie_path=movie_path,
                    movie_frame_range=movie_frame_range,
                    add_controls=add_controls,
                )

            resolver = _resolver({source: _resolved_recording(recording_path)})
            load_recording_paths(
                state,
                (source,),
                remember_selected_folder=False,
                resolve_recording=resolver,
                open_recording=open_recording,
            )

            result = load_recording_paths(
                state,
                (source,),
                remember_selected_folder=False,
                roi_file_to_load=roi_path,
                replace_selected=True,
                resolve_recording=resolver,
                open_recording=open_recording,
            )

            self.assertEqual(result.loaded_count, 0)
            self.assertEqual(len(state.loaded_recordings), 1)
            self.assertEqual(state.selected_recording_index, 0)
            self.assertEqual(opened_roi_paths, [None, roi_path.resolve()])

    def test_partial_failure_records_path_and_keeps_success(self) -> None:
        """Confirm one bad path does not hide a successful sibling load.

        Inputs: one valid resolver mapping and one resolver ``KeyError``.
        Outputs: one loaded row plus one path-scoped failure.
        """
        state = _State()
        good = Path("/source/good")
        bad = Path("/source/bad")

        result = load_recording_paths(
            state,
            (good, bad),
            remember_selected_folder=False,
            resolve_recording=_resolver(
                {good: _resolved_recording(Path("/cache/good/recording_data.h5"))},
            ),
            open_recording=_open_recording,
        )

        self.assertEqual(result.loaded_count, 1)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].path, bad)
        self.assertIn("KeyError", result.failures[0].message)
        self.assertEqual(len(state.loaded_recordings), 1)
        self.assertFalse(state.is_loading)

    def test_source_unavailable_warning_is_returned_to_qt_layer(self) -> None:
        """Confirm cache-backed unavailable sources report a warning item.

        Inputs: one resolver result marked as ``source_unavailable``.
        Outputs: the source path and cache path needed for the warning dialog.
        """
        state = _State()
        source = Path("/missing/source")
        cache = Path("/cache/source/recording_data.h5")

        result = load_recording_paths(
            state,
            (source,),
            remember_selected_folder=False,
            resolve_recording=_resolver(
                {source: _resolved_recording(cache, source_unavailable=True)},
            ),
            open_recording=_open_recording,
        )

        self.assertEqual(result.loaded_count, 1)
        self.assertEqual(len(result.source_warnings), 1)
        self.assertEqual(result.source_warnings[0].source_path, source)
        self.assertEqual(result.source_warnings[0].cache_path, cache)

    def test_reconvert_selected_recording_reloads_row_in_place(self) -> None:
        """Confirm reconversion overwrites and reloads the selected row.

        Inputs: one loaded recording and a reconverter fake returning the same
        output paths.
        Outputs: old layers are replaced and conversion receives source and
        output paths from the loaded recording.
        """
        state = _State()
        viewer = cast(_Viewer, state.viewer)
        source = Path("/source/first")
        recording_path = Path("/cache/first/recording_data.h5")
        load_recording_paths(
            state,
            (source,),
            remember_selected_folder=False,
            resolve_recording=_resolver({source: _resolved_recording(recording_path)}),
            open_recording=_open_recording,
        )
        old_layers = (*viewer.images, *viewer.labels)
        reconvert_calls: list[tuple[Path, Path]] = []

        def reconvert(
            source_dir: Path,
            output_dir: Path,
        ) -> ResolvedNapariRecording:
            """Record reconversion arguments and return fresh converted paths."""
            reconvert_calls.append((source_dir, output_dir))
            return _resolved_recording(output_dir / "recording_data.h5")

        result = reconvert_selected_recording(
            state,
            reconvert_recording=reconvert,
            open_recording=_open_recording,
        )

        self.assertEqual(result.failures, ())
        self.assertEqual(
            result.status_text,
            "Reconverted and loaded /cache/first/recording_data.h5",
        )
        self.assertEqual(
            reconvert_calls, [(Path("/cache/first"), Path("/cache/first"))]
        )
        self.assertEqual(len(state.loaded_recordings), 1)
        self.assertEqual(state.selected_recording_index, 0)
        self.assertEqual(len(viewer.images), 2)
        self.assertEqual(len(viewer.labels), 1)
        for layer in old_layers:
            for remaining_layer in (*viewer.images, *viewer.labels):
                self.assertIsNot(layer, remaining_layer)

    def test_reconvert_selected_recording_reports_missing_selection(self) -> None:
        """Confirm reconversion needs a selected loaded recording.

        Inputs: empty session state.
        Outputs: structured failure and load flag reset.
        """
        state = _State()

        result = reconvert_selected_recording(state, open_recording=_open_recording)

        self.assertEqual(result.loaded_count, 0)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].message, "No recording selected.")
        self.assertFalse(state.is_loading)


def _resolver(
    recordings: dict[Path, ResolvedNapariRecording],
) -> Callable[[PathInput], ResolvedNapariRecording]:
    """Create a resolver fake for selected recording paths.

    Args:
        recordings: Mapping from selected path to resolver result.

    Returns:
        Callable matching ``resolve_or_convert_recording``.
    """

    def resolve(path: PathInput) -> ResolvedNapariRecording:
        """Resolve one fake recording path or raise a path-level error."""
        selected = Path(path)
        if selected not in recordings:
            raise KeyError("missing stimtype")
        return recordings[selected]

    return resolve


def _resolved_recording(
    recording_data_path: Path,
    *,
    source_unavailable: bool = False,
) -> ResolvedNapariRecording:
    """Build a resolved-recording fake for the workflow coordinator.

    Args:
        recording_data_path: Converted recording data path.
        source_unavailable: Whether the resolver used cache because source was
            unreachable.

    Returns:
        Resolved recording paths consumed by ``load_recording_paths``.
    """
    recording_dir = recording_data_path.parent
    paths = NapariRecordingPaths(
        recording_data_path=recording_data_path,
        movie_path=recording_dir / "aligned_movie.h5",
        roi_file_to_load=None,
        roi_save_file=recording_dir / "rois.h5",
    )
    return ResolvedNapariRecording(
        paths=paths,
        was_converted=False,
        source_unavailable=source_unavailable,
    )


def _open_recording(
    recording_data_path: Path,
    *,
    viewer: _Viewer,
    roi_set: Path | None,
    roi_save_file: Path,
    movie_path: Path | None,
    movie_frame_range: tuple[int, int | None],
    add_controls: bool,
) -> NapariRecordingView:
    """Open one fake converted recording into fake viewer layers.

    Args:
        recording_data_path: Converted recording path to attach to the view.
        viewer: Fake napari viewer.
        roi_set: Optional ROI set path.
        roi_save_file: Default ROI save path.
        movie_path: Optional movie file path.
        movie_frame_range: Requested frame range.
        add_controls: Whether controls should be added.

    Returns:
        Fake napari recording view suitable for session bookkeeping.
    """
    del roi_set, roi_save_file, movie_path, movie_frame_range, add_controls
    recording = _recording(recording_data_path)
    mean_image_layer = viewer.add_image(recording.mean_image, name="Mean image")
    movie_layer = viewer.add_image(np.zeros((3, 2, 2)), name="Movie")
    roi_labels_layer = viewer.add_labels(np.zeros((2, 2), dtype=np.uint16), name="ROIs")
    return NapariRecordingView(
        viewer=viewer,
        recording=recording,
        mean_image_layer=mean_image_layer,
        movie_layer=movie_layer,
        roi_labels_layer=roi_labels_layer,
        load_widget=None,
        loaded_recordings_widget=None,
        twopy_sidebar_widget=None,
        twopy_sidebar_dock_widget=None,
        response_plot_widget=None,
        response_plot_dock_widget=None,
        response_options_widget=None,
        trial_timeline_controller=None,
    )


def _recording(recording_data_path: Path) -> RecordingData:
    """Create a minimal converted recording object for session tests.

    Args:
        recording_data_path: Path stored on the recording.

    Returns:
        Recording data object with tiny arrays and valid metadata fields.
    """
    movie = ConvertedMovie(
        path=recording_data_path.parent / "aligned_movie.h5",
        dataset_name="movie/aligned",
        shape=(3, 2, 2),
        dtype="float64",
    )
    crop = SpatialCrop(
        axis0_start=0,
        axis0_stop=2,
        axis1_start=0,
        axis1_stop=2,
        original_shape=(2, 2),
        source="test",
    )
    return RecordingData(
        path=recording_data_path,
        movie=movie,
        source_session_dir=recording_data_path.parent,
        acquisition_metadata={},
        run_metadata={},
        synchronization_metadata={},
        stimulus_data=np.zeros((1, 2), dtype=np.float64),
        stimulus_data_column_names=("time_seconds", "epoch_number"),
        stimulus_parameters=(),
        stimulus_function_lookup={},
        stimulus_specific_columns={},
        imaging_res_pd=np.zeros(3, dtype=np.float64),
        high_res_pd=np.zeros(3, dtype=np.float64),
        mean_image=np.zeros((2, 2), dtype=np.float64),
        alignment_valid_crop=crop,
        alignment_shift_pixels=np.zeros(3, dtype=np.float64),
        motion_artifact_mask=np.zeros(3, dtype=np.bool_),
        frame_counts=FrameCountAudit(
            aligned_movie_frames=3,
            imaging_res_pd_samples=3,
            acquisition_number_of_frames=3,
            imaging_res_pd_minus_movie=0,
            acquisition_minus_movie=0,
        ),
    )


if __name__ == "__main__":
    unittest.main()
