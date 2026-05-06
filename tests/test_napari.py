"""Tests for the thin napari adapter.

Inputs: a tiny converted recording, ROI labels, and a fake viewer.
Outputs: layer data sent to napari-shaped methods and saved ROI HDF5 files.
"""

import tempfile
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from os import chdir, environ
from pathlib import Path
from typing import Any, Protocol, cast

import h5py
import numpy as np

from twopy import (
    add_twopy_magicgui_controls,
    load_roi_set,
    make_roi_set,
    open_recording_in_napari,
    roi_label_image_from_layer,
    save_napari_label_rois,
    save_roi_set,
)
from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.responses import group_delta_f_over_f_by_epoch
from twopy.analysis.trials import EpochFrameWindow, FrameWindow
from twopy.napari.movie import resolve_movie_frame_range
from twopy.napari.paths import resolve_launch_recording_path, resolve_recording_paths
from twopy.napari.plot_data import response_plot_data_from_grouped
from twopy.napari.plot_widgets import (
    global_time_bounds,
    global_value_bounds,
    roi_colors_from_layer,
)
from twopy.napari.state import write_last_recording_folder


class _ControlWidget(Protocol):
    """Small protocol for the magicgui control container in tests.

    Inputs: integer widget index.
    Outputs: callable magicgui function widget.
    """

    def __getitem__(self, index: int) -> Callable[..., object]:
        """Return one child widget by index.

        Args:
            index: Child widget index.

        Returns:
            Callable widget.
        """
        ...


@dataclass
class _FakeLayer:
    """Layer object returned by the fake viewer.

    Inputs: layer name and data.
    Outputs: inspectable test object.
    """

    name: str
    data: object
    options: dict[str, object]


@dataclass(frozen=True)
class _FakeDockWidget:
    """Dock widget object returned by the fake viewer window.

    Inputs: dock name, area, and widget.
    Outputs: inspectable test object.
    """

    name: str
    area: str
    widget: object


class _FakeWindow:
    """Small napari-shaped window for dock-widget tests."""

    dock_widgets: list[_FakeDockWidget]

    def __init__(self) -> None:
        """Create an empty fake window.

        Inputs: none.
        Outputs: window with no dock widgets.
        """
        self.dock_widgets: list[_FakeDockWidget] = []

    def add_dock_widget(
        self,
        widget: object,
        *,
        name: str,
        area: str,
    ) -> object:
        """Record a dock-widget request.

        Args:
            widget: Widget object added by the adapter.
            name: Dock name.
            area: Dock area.

        Returns:
            Fake dock widget.
        """
        dock_widget = _FakeDockWidget(name=name, area=area, widget=widget)
        self.dock_widgets.append(dock_widget)
        return dock_widget


class _FakeViewer:
    """Small napari-shaped viewer for adapter tests."""

    images: list[_FakeLayer]
    labels: list[_FakeLayer]
    window: _FakeWindow

    def __init__(self) -> None:
        """Create an empty fake viewer.

        Inputs: none.
        Outputs: viewer with no recorded layers.
        """
        self.images: list[_FakeLayer] = []
        self.labels: list[_FakeLayer] = []
        self.window = _FakeWindow()

    def add_image(self, data: object, *, name: str, **kwargs: object) -> object:
        """Record an image layer request.

        Args:
            data: Image data sent by the adapter.
            name: Layer name.
            kwargs: Napari image options.

        Returns:
            Fake image layer.
        """
        layer = _FakeLayer(name=name, data=data, options=dict(kwargs))
        self.images.append(layer)
        return layer

    def add_labels(self, data: object, *, name: str, **kwargs: object) -> object:
        """Record a labels layer request.

        Args:
            data: Label image sent by the adapter.
            name: Layer name.
            kwargs: Napari label options.

        Returns:
            Fake labels layer.
        """
        layer = _FakeLayer(name=name, data=data, options=dict(kwargs))
        self.labels.append(layer)
        return layer


class _FakeColorLayer:
    """Small Labels-layer stand-in with deterministic colors."""

    def get_color(self, label: int) -> tuple[float, float, float, float]:
        """Return one RGBA color for a positive label.

        Args:
            label: Labels-layer integer value.

        Returns:
            RGBA tuple in napari's usual float range.
        """
        if label == 1:
            return (1.0, 0.0, 0.0, 1.0)
        return (0.0, 0.0, 1.0, 1.0)


class NapariAdapterTest(unittest.TestCase):
    """Tests napari loading and label saving without starting a GUI."""

    def setUp(self) -> None:
        """Route napari UI state to a temporary file during tests.

        Inputs: none.
        Outputs: isolated state path in the test environment.
        """
        self._state_temp_dir = tempfile.TemporaryDirectory()
        self._previous_state_file = environ.get("TWOPY_NAPARI_STATE_FILE")
        environ["TWOPY_NAPARI_STATE_FILE"] = str(
            Path(self._state_temp_dir.name) / "napari_state.json",
        )

    def tearDown(self) -> None:
        """Restore the caller's napari state-file environment.

        Inputs: none.
        Outputs: original environment value restored.
        """
        if self._previous_state_file is None:
            environ.pop("TWOPY_NAPARI_STATE_FILE", None)
        else:
            environ["TWOPY_NAPARI_STATE_FILE"] = self._previous_state_file
        self._state_temp_dir.cleanup()

    def test_opens_empty_roi_labels_layer_by_default(self) -> None:
        """Confirm new recordings open with an editable empty ROI layer.

        Inputs: tiny converted recording and fake viewer.
        Outputs: one zero-valued labels layer matching the movie frame shape.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()

            opened = open_recording_in_napari(recording_path, viewer=viewer)

            self.assertIsNotNone(opened.roi_labels_layer)
            self.assertIsNotNone(opened.controls_widget)
            self.assertIsNotNone(opened.controls_dock_widget)
            self.assertIsNotNone(opened.response_plot_widget)
            self.assertIsNotNone(opened.response_plot_dock_widget)
            self.assertEqual(len(viewer.labels), 1)
            self.assertEqual(viewer.labels[0].options["opacity"], 0.5)
            self.assertEqual(viewer.labels[0].options["blending"], "additive")
            self.assertEqual(len(viewer.window.dock_widgets), 2)
            self.assertEqual(viewer.window.dock_widgets[0].name, "twopy responses")
            self.assertEqual(viewer.window.dock_widgets[1].name, "twopy")
            np.testing.assert_array_equal(
                roi_label_image_from_layer(viewer.labels[0]),
                np.zeros((2, 2), dtype=np.int64),
            )

    def test_opens_converted_recording_with_roi_labels(self) -> None:
        """Confirm the adapter sends mean image, movie preview, and ROIs.

        Inputs: tiny converted recording, ROI HDF5 file, and fake viewer.
        Outputs: fake napari layers with expected shapes.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            roi_path = root / "rois.h5"
            save_roi_set(
                make_roi_set(
                    np.array(
                        [
                            [[True, False], [False, False]],
                            [[False, False], [False, True]],
                        ],
                    ),
                ),
                roi_path,
            )
            viewer = _FakeViewer()

            opened = open_recording_in_napari(
                recording_path,
                roi_set=roi_path,
                viewer=viewer,
                movie_frame_range=(0, 1),
            )

            self.assertIs(opened.viewer, viewer)
            self.assertEqual(opened.recording.movie.shape, (3, 2, 2))
            self.assertEqual(len(viewer.images), 2)
            self.assertEqual(viewer.images[0].name, "mean image")
            self.assertEqual(np.asarray(viewer.images[0].data).shape, (2, 2))
            self.assertEqual(viewer.images[1].name, "aligned movie")
            self.assertEqual(np.asarray(viewer.images[1].data).shape, (2, 2, 2))
            self.assertEqual(len(viewer.labels), 1)
            np.testing.assert_array_equal(
                np.unique(np.asarray(viewer.labels[0].data)),
                np.array([0, 1, 2]),
            )

    def test_saves_napari_label_image_as_roi_file(self) -> None:
        """Confirm edited napari labels round-trip through core ROI storage.

        Inputs: one label image and a temporary output path.
        Outputs: saved ROI HDF5 file loaded by the normal ROI helper.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "drawn_rois.h5"

            saved = save_napari_label_rois(
                np.array([[0, 1], [2, 2]]),
                output_path,
                label_prefix="drawn",
            )
            loaded = load_roi_set(output_path)

            self.assertEqual(saved.labels, ("drawn_0001", "drawn_0002"))
            self.assertEqual(loaded.labels, saved.labels)
            np.testing.assert_array_equal(loaded.masks, saved.masks)

    def test_magicgui_save_button_writes_current_labels(self) -> None:
        """Confirm the dock widget saves the editable Labels layer.

        Inputs: tiny converted recording, fake viewer, and edited label data.
        Outputs: ROI HDF5 file written by the magicgui function.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            roi_save_file = root / "drawn_rois.h5"
            viewer = _FakeViewer()
            opened = open_recording_in_napari(
                recording_path,
                viewer=viewer,
                roi_save_file=roi_save_file,
            )

            viewer.labels[0].data = np.array([[0, 1], [2, 2]])
            controls = cast(_ControlWidget, opened.controls_widget)
            save_widget = cast(Any, controls[3])
            result = save_widget(roi_save_file=roi_save_file)

            self.assertIn("Saved 2 ROI", str(result))
            loaded = load_roi_set(roi_save_file)
            self.assertEqual(loaded.labels, ("roi_0001", "roi_0002"))

    def test_recording_folder_selection_loads_recording_layers(self) -> None:
        """Confirm selecting a recording folder populates an empty viewer.

        Inputs: fake viewer, control dock, and tiny converted recording folder.
        Outputs: mean image, movie preview, and ROI Labels layer.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            roi_save_file = root / "drawn_rois.h5"
            _write_converted_recording(root)
            viewer = _FakeViewer()

            controls_widget, _dock = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=roi_save_file,
            )
            controls = cast(_ControlWidget, controls_widget)
            load_widget = cast(Any, controls[1])

            load_widget.roi_save_file.value = roi_save_file
            load_widget.recording_folder.value = root

            self.assertEqual(load_widget.movie_frame_range.value, (0, 2))
            self.assertEqual(len(viewer.images), 2)
            self.assertEqual(len(viewer.labels), 1)
            np.testing.assert_array_equal(
                roi_label_image_from_layer(viewer.labels[0]),
                np.zeros((2, 2), dtype=np.int64),
            )

    def test_loaded_recording_updates_default_roi_save_path(self) -> None:
        """Confirm loading a recording makes Save ROIs write beside it.

        Inputs: empty viewer, loaded recording, and edited labels.
        Outputs: ``rois.h5`` beside the loaded ``recording_data.h5``.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            controls_widget, _dock = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            controls = cast(_ControlWidget, controls_widget)
            load_widget = cast(Any, controls[1])
            save_widget = cast(Any, controls[3])

            load_widget(recording_folder=recording_path)
            viewer.labels[0].data = np.array([[1, 0], [0, 0]])
            result = save_widget()

            self.assertIn("Saved 1 ROI", str(result))
            self.assertTrue((root / "rois.h5").exists())

    def test_controls_hide_remembered_recording_folder(self) -> None:
        """Confirm the Load Recording control keeps long remembered paths hidden.

        Inputs: one folder stored in the napari state file.
        Outputs: recording-folder widget displays ``default`` instead of the
        remembered path.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_last_recording_folder(root)
            viewer = _FakeViewer()

            controls_widget, _dock = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            controls = cast(_ControlWidget, controls_widget)
            load_widget = cast(Any, controls[1])

            self.assertEqual(load_widget.recording_folder.value, Path("default"))

    def test_recording_folder_resolves_recording_defaults(self) -> None:
        """Confirm folder loading finds recording, movie, and ROI files.

        Inputs: converted output folder containing recording/movie/ROI files.
        Outputs: loaded layers and default ROI save path from that folder.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            roi_path = root / "rois.h5"
            save_roi_set(
                make_roi_set(np.array([[[True, False], [False, False]]])),
                roi_path,
            )
            viewer = _FakeViewer()
            controls_widget, _dock = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            controls = cast(_ControlWidget, controls_widget)
            load_widget = cast(Any, controls[1])

            result = load_widget(recording_folder=root)

            self.assertIn(str(recording_path), str(result))
            self.assertEqual(len(viewer.images), 2)
            np.testing.assert_array_equal(
                np.unique(roi_label_image_from_layer(viewer.labels[0])),
                np.array([0, 1]),
            )

    def test_recording_path_resolution_reports_available_files(self) -> None:
        """Confirm folder resolution finds optional movie and ROI paths.

        Inputs: converted output folder with ``aligned_movie.h5`` and
            ``rois.h5``.
        Outputs: concrete paths for the viewer and controls.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            roi_path = root / "rois.h5"
            roi_path.touch()

            paths = resolve_recording_paths(root)

            self.assertEqual(paths.recording_data_path, recording_path.resolve())
            self.assertEqual(paths.movie_path, (root / "aligned_movie.h5").resolve())
            self.assertEqual(paths.roi_file_to_load, roi_path.resolve())
            self.assertEqual(paths.roi_save_file, roi_path.resolve())

    def test_recording_path_resolution_accepts_widget_strings(self) -> None:
        """Confirm magicgui string paths resolve the same way as ``Path`` values.

        Inputs: converted output folder passed as text, matching magicgui
            ``FileEdit`` callback values.
        Outputs: concrete recording, movie, and ROI paths.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            roi_path = root / "rois.h5"
            roi_path.touch()

            paths = resolve_recording_paths(str(root))

            self.assertEqual(paths.recording_data_path, recording_path.resolve())
            self.assertEqual(paths.movie_path, (root / "aligned_movie.h5").resolve())
            self.assertEqual(paths.roi_file_to_load, roi_path.resolve())
            self.assertEqual(paths.roi_save_file, roi_path.resolve())

    def test_movie_frame_range_accepts_last_default(self) -> None:
        """Confirm empty-launch widget defaults request the full movie.

        Inputs: start/end frame zero before a recording is selected.
        Outputs: the actual final frame after frame count is known.
        """
        self.assertEqual(
            resolve_movie_frame_range(start_frame=0, end_frame=0, frame_count=3),
            (0, 0),
        )
        self.assertEqual(
            resolve_movie_frame_range(
                start_frame=0,
                end_frame=0,
                frame_count=3,
                zero_end_means_last=True,
            ),
            (0, 2),
        )

    def test_response_plot_data_summarizes_epoch_roi_mean_and_sem(self) -> None:
        """Confirm plotting data has one mean and SEM trace per ROI.

        Inputs: two repeated trials for one epoch and two ROIs.
        Outputs: plot-ready arrays shaped as ``(roi, frame)``.
        """
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((4, 2), dtype=np.float64),
            baseline=np.ones((4, 2), dtype=np.float64),
            values=np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 8.0],
                    [7.0, 10.0],
                ],
                dtype=np.float64,
            ),
            labels=("roi_1", "roi_2"),
            start_frame=0,
            stop_frame=4,
            tau=0.0,
            amplitudes=np.ones(2, dtype=np.float64),
            interleave_frame_numbers=np.array([0.0]),
            interleave_fluorescence=np.ones((1, 2), dtype=np.float64),
            metadata={"method": "test"},
        )
        grouped = group_delta_f_over_f_by_epoch(
            dff,
            (
                EpochFrameWindow(FrameWindow(0, 0, 2, "odor_1"), 2, "Odor"),
                EpochFrameWindow(FrameWindow(1, 2, 4, "odor_2"), 2, "Odor"),
            ),
            data_rate_hz=2.0,
        )

        plot_data = response_plot_data_from_grouped(grouped)

        self.assertEqual(len(plot_data.epochs), 1)
        epoch = plot_data.epochs[0]
        self.assertEqual(epoch.epoch_name, "Odor")
        np.testing.assert_allclose(epoch.time_seconds, np.array([0.0, 0.5]))
        np.testing.assert_allclose(
            epoch.mean_values,
            np.array([[3.0, 5.0], [5.0, 7.0]]),
        )
        np.testing.assert_allclose(
            epoch.sem_values,
            np.array([[2.0, 2.0], [3.0, 3.0]]),
        )

    def test_response_plot_data_uses_prestimulus_time_axis(self) -> None:
        """Confirm response plots preserve negative prestimulus times.

        Inputs: dF/F grouped with one second before stimulus onset.
        Outputs: plot-ready time axis starts at ``-1`` seconds.
        """
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((4, 1), dtype=np.float64),
            baseline=np.ones((4, 1), dtype=np.float64),
            values=np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64),
            labels=("roi_1",),
            start_frame=0,
            stop_frame=4,
            tau=0.0,
            amplitudes=np.ones(1, dtype=np.float64),
            interleave_frame_numbers=np.array([0.0]),
            interleave_fluorescence=np.ones((1, 1), dtype=np.float64),
            metadata={"method": "test"},
        )
        grouped = group_delta_f_over_f_by_epoch(
            dff,
            (EpochFrameWindow(FrameWindow(0, 2, 4, "odor"), 2, "Odor"),),
            data_rate_hz=2.0,
            pre_window_seconds=1.0,
        )

        plot_data = response_plot_data_from_grouped(grouped)

        np.testing.assert_allclose(
            plot_data.epochs[0].time_seconds,
            np.array([-1.0, -0.5, 0.0, 0.5]),
        )
        np.testing.assert_allclose(
            plot_data.epochs[0].mean_values,
            np.array([[0.0, 1.0, 2.0, 3.0]]),
        )

    def test_response_plot_bounds_use_selected_epochs_and_rois(self) -> None:
        """Confirm plot bounds follow selected ROI and epoch visibility.

        Inputs: two epoch plots and two ROIs with different ranges.
        Outputs: bounds for only the selected epoch and ROI.
        """
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((4, 2), dtype=np.float64),
            baseline=np.ones((4, 2), dtype=np.float64),
            values=np.array(
                [[0.0, 10.0], [1.0, 11.0], [2.0, 20.0], [3.0, 21.0]],
                dtype=np.float64,
            ),
            labels=("roi_1", "roi_2"),
            start_frame=0,
            stop_frame=4,
            tau=0.0,
            amplitudes=np.ones(2, dtype=np.float64),
            interleave_frame_numbers=np.array([0.0]),
            interleave_fluorescence=np.ones((1, 2), dtype=np.float64),
            metadata={"method": "test"},
        )
        grouped = group_delta_f_over_f_by_epoch(
            dff,
            (
                EpochFrameWindow(FrameWindow(0, 0, 2, "gray"), 1, "Gray"),
                EpochFrameWindow(FrameWindow(1, 2, 4, "odor"), 2, "Odor"),
            ),
            data_rate_hz=1.0,
        )
        plot_data = response_plot_data_from_grouped(grouped)

        self.assertEqual(global_time_bounds(plot_data, ((2, "Odor"),)), (0.0, 1.0))
        self.assertEqual(
            global_value_bounds(plot_data, (0,), ((2, "Odor"),)),
            (1.6, 3.6),
        )

    def test_response_plot_colors_can_match_labels_layer(self) -> None:
        """Confirm plot colors can come from the napari Labels layer.

        Inputs: fake Labels layer with deterministic per-label RGBA values.
        Outputs: Qt colors matching those values.
        """
        colors = roi_colors_from_layer(_FakeColorLayer(), 2)

        self.assertEqual(colors[0].red(), 255)
        self.assertEqual(colors[0].blue(), 0)
        self.assertEqual(colors[1].red(), 0)
        self.assertEqual(colors[1].blue(), 255)

    def test_launch_recording_path_returns_none_when_no_default_exists(self) -> None:
        """Confirm no-path app launch can start empty instead of failing.

        Inputs: temporary directory without converted recording files.
        Outputs: ``None`` so the launcher can open an empty viewer.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                chdir(temp_dir)

                resolved = resolve_launch_recording_path(None)
            finally:
                chdir(original_cwd)

            self.assertIsNone(resolved)

    def test_launch_recording_path_resolves_from_current_directory(self) -> None:
        """Confirm launcher can run with no path from a converted folder.

        Inputs: temporary directory containing ``recording_data.h5``.
        Outputs: resolved path to that file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = root / "recording_data.h5"
            recording_path.touch()
            original_cwd = Path.cwd()
            try:
                chdir(root)

                resolved = resolve_launch_recording_path(None)
            finally:
                chdir(original_cwd)

            self.assertEqual(resolved, recording_path.resolve())

    def test_launch_recording_path_resolves_from_source_recording_directory(
        self,
    ) -> None:
        """Confirm launcher can run from a source recording with twopy output.

        Inputs: temporary directory containing ``twopy/recording_data.h5``.
        Outputs: resolved path to the converted recording file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = root / "twopy" / "recording_data.h5"
            recording_path.parent.mkdir()
            recording_path.touch()
            original_cwd = Path.cwd()
            try:
                chdir(root)

                resolved = resolve_launch_recording_path(None)
            finally:
                chdir(original_cwd)

            self.assertEqual(resolved, recording_path.resolve())


def _write_converted_recording(root: Path) -> Path:
    """Write a tiny converted recording for adapter tests.

    Args:
        root: Temporary directory receiving HDF5 files.

    Returns:
        Path to ``recording_data.h5``.
    """
    movie_values = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
    movie_path = root / "aligned_movie.h5"
    with h5py.File(movie_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = "aligned-movie"
        h5_file.create_dataset("movie/aligned", data=movie_values)

    recording_path = root / "recording_data.h5"
    with h5py.File(recording_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = "converted-recording"
        h5_file.attrs["source_session_dir"] = str(root / "source")
        movie_group = h5_file.create_group("movie")
        movie_group.attrs["aligned_movie_file"] = "aligned_movie.h5"
        movie_group.attrs["aligned_movie_dataset"] = "movie/aligned"
        movie_group.attrs["aligned_movie_shape"] = movie_values.shape
        movie_group.attrs["aligned_movie_dtype"] = "float64"
        movie_group.create_dataset("mean_image", data=movie_values.mean(axis=0))
        crop_group = movie_group.create_group("alignment_valid_crop")
        crop_group.attrs["source"] = "full_frame"
        crop_group.attrs["axis0_start"] = 0
        crop_group.attrs["axis0_stop"] = 2
        crop_group.attrs["axis1_start"] = 0
        crop_group.attrs["axis1_stop"] = 2
        crop_group.attrs["original_shape"] = (2, 2)
        crop_group.create_dataset(
            "alignment_shift_pixels",
            data=np.zeros(3, dtype=np.float64),
        )
        crop_group.create_dataset(
            "motion_artifact_mask",
            data=np.zeros(3, dtype=np.bool_),
        )
        metadata_group = h5_file.create_group("metadata")
        metadata_group.attrs["acq.frameRate"] = 10.0
        metadata_group.attrs["acq.zoomFactor"] = 2.0
        run_group = h5_file.create_group("run")
        run_group.attrs["rig_name"] = "TestRig"
        stimulus_group = h5_file.create_group("stimulus")
        stimulus_group.create_dataset("data", data=np.zeros((1, 2), dtype=np.float64))
        stimulus_group.create_dataset(
            "data_column_names",
            data=np.asarray(
                ("time_seconds", "epoch_number"),
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        stimulus_group.create_dataset("parameters_json", data="[]")
        stimulus_group.create_dataset("function_lookup_json", data="{}")
        stimulus_group.create_dataset("stimulus_specific_columns_json", data="{}")
        photodiode_group = h5_file.create_group("photodiode")
        photodiode_group.create_dataset(
            "imaging_res_pd",
            data=np.zeros(3, dtype=np.float64),
        )
        photodiode_group.create_dataset(
            "high_res_pd",
            data=np.zeros(3, dtype=np.float64),
        )
        frame_counts = h5_file.create_group("frame_counts")
        frame_counts.attrs["aligned_movie_frames"] = 3
        frame_counts.attrs["imaging_res_pd_samples"] = 3
        frame_counts.attrs["acquisition_number_of_frames"] = 3
        frame_counts.attrs["imaging_res_pd_minus_movie"] = 0
        frame_counts.attrs["acquisition_minus_movie"] = 0
    return recording_path


if __name__ == "__main__":
    unittest.main()
