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
from types import SimpleNamespace
from typing import Any, Protocol, cast
from unittest.mock import patch

import h5py
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from napari.layers import Labels
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QListWidget,
    QPushButton,
    QTabWidget,
    QWidget,
)

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
from twopy.analysis.responses import GroupedRoiResponses, group_delta_f_over_f_by_epoch
from twopy.analysis.trials import EpochFrameWindow, FrameWindow
from twopy.conversion.types import ConvertedRecording
from twopy.converted import load_converted_recording
from twopy.napari.display import (
    display_image_from_movie_image,
    display_labels_from_movie_labels,
    display_metadata_for_spatial_crop,
    movie_labels_from_display_layer,
)
from twopy.napari.interactive import LiveResponseController
from twopy.napari.loading import resolve_or_convert_recording
from twopy.napari.movie import resolve_movie_frame_range
from twopy.napari.paths import resolve_launch_recording_path, resolve_recording_paths
from twopy.napari.plotting.data import ResponsePlotData, response_plot_data_from_grouped
from twopy.napari.plotting.docks import create_response_plot_widget
from twopy.napari.plotting.export import (
    draw_roi_contours,
    export_epoch_plots,
    labels_for_recording_image,
    roi_boundary_segments,
)
from twopy.napari.plotting.label_visibility import apply_roi_visibility_to_labels_layer
from twopy.napari.plotting.options import visibility_options_widget
from twopy.napari.plotting.widgets import (
    EpochPlotWidget,
    global_time_bounds,
    global_value_bounds,
    roi_colors_from_layer,
)
from twopy.napari.state import write_last_recording_folder
from twopy.napari.viewer import interior_random_frame_indices
from twopy.spatial import SpatialCrop


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
    visible: bool = True
    brush_size: int | None = None
    contrast_limits_range: tuple[float, float] | None = None
    events: object | None = None


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


class _FakeEmitter:
    """Small event emitter used to test live ROI update wiring."""

    def __init__(self) -> None:
        """Create an emitter with no connected callbacks."""
        self._callbacks: list[Callable[..., None]] = []

    def connect(self, callback: Callable[..., None]) -> None:
        """Register one callback.

        Args:
            callback: Function called when ``emit`` runs.

        Returns:
            None.
        """
        self._callbacks.append(callback)

    def disconnect(self, callback: Callable[..., None]) -> None:
        """Remove one callback.

        Args:
            callback: Previously connected callback.

        Returns:
            None.
        """
        self._callbacks.remove(callback)

    def emit(self) -> None:
        """Call all connected callbacks."""
        for callback in tuple(self._callbacks):
            callback()


class _FakeLabelEvents:
    """Event group with the Labels event names watched by twopy."""

    def __init__(self) -> None:
        """Create data-change emitters."""
        self.data = _FakeEmitter()
        self.labels_update = _FakeEmitter()
        self.paint = _FakeEmitter()


class _FakePlotReceiver:
    """Response plot receiver used by live-controller tests."""

    def __init__(self) -> None:
        """Create an empty receiver."""
        self.plot_data: ResponsePlotData | None = None
        self.status: str | None = None

    def set_response_plot_data(
        self,
        plot_data: ResponsePlotData,
        *,
        reset_axes: bool,
    ) -> None:
        """Store plot data received from the controller."""
        del reset_axes
        self.plot_data = plot_data

    def show_response_status(self, text: str) -> None:
        """Store status text received from the controller."""
        self.status = text


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
            self.assertIsNotNone(opened.loaded_recordings_widget)
            self.assertIsNotNone(opened.loaded_recordings_dock_widget)
            self.assertIsNotNone(opened.save_rois_widget)
            self.assertIsNotNone(opened.save_rois_dock_widget)
            self.assertIsNotNone(opened.response_plot_widget)
            self.assertIsNotNone(opened.response_plot_dock_widget)
            self.assertIsNotNone(opened.response_options_widget)
            self.assertIsNotNone(opened.response_options_dock_widget)
            self.assertEqual(len(viewer.labels), 1)
            self.assertEqual(viewer.labels[0].options["opacity"], 0.5)
            self.assertEqual(viewer.labels[0].options["blending"], "additive")
            self.assertEqual(viewer.labels[0].brush_size, 6)
            self.assertEqual(viewer.images[0].options["opacity"], 0.5)
            self.assertEqual(viewer.images[0].options["gamma"], 1.3)
            self.assertEqual(viewer.images[0].options["contrast_limits"], (4.3, 7.0))
            self.assertEqual(viewer.images[0].contrast_limits_range, (4.0, 7.0))
            self.assertEqual(len(viewer.window.dock_widgets), 5)
            self.assertEqual(viewer.window.dock_widgets[0].name, "twopy responses")
            self.assertEqual(viewer.window.dock_widgets[0].area, "top")
            self.assertEqual(viewer.window.dock_widgets[1].name, "twopy")
            self.assertEqual(
                viewer.window.dock_widgets[2].name,
                "twopy loaded recordings",
            )
            self.assertEqual(viewer.window.dock_widgets[2].area, "right")
            self.assertEqual(viewer.window.dock_widgets[3].name, "twopy save ROIs")
            self.assertEqual(viewer.window.dock_widgets[3].area, "left")
            self.assertEqual(
                viewer.window.dock_widgets[4].name,
                "twopy response options",
            )
            self.assertEqual(viewer.window.dock_widgets[4].area, "right")
            options_widget = cast(QTabWidget, opened.response_options_widget)
            self.assertEqual(
                tuple(
                    options_widget.tabText(index)
                    for index in range(options_widget.count())
                ),
                ("Update", "Plot", "ROIs", "Epochs", "Export"),
            )
            update_buttons = {
                button.text() for button in options_widget.findChildren(QPushButton)
            }
            self.assertIn("Save analysis + ROIs", update_buttons)
            self.assertIn("Recompute preview now", update_buttons)
            self.assertIn("Reload saved analysis", update_buttons)
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
            self.assertEqual(viewer.images[0].options["contrast_limits"], (4.3, 7.0))
            self.assertEqual(viewer.images[0].contrast_limits_range, (4.0, 7.0))
            self.assertEqual(np.asarray(viewer.images[0].data).shape, (2, 2))
            self.assertEqual(viewer.images[1].name, "aligned movie")
            self.assertEqual(viewer.images[1].options["blending"], "additive")
            self.assertEqual(
                viewer.images[1].options["contrast_limits"],
                (4.0, 7.0),
            )
            self.assertEqual(viewer.images[1].contrast_limits_range, (0.0, 7.0))
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
            save_widget = cast(Any, opened.save_rois_widget)
            result = save_widget(roi_save_file=roi_save_file)

            self.assertIn("Saved 2 ROI", str(result))
            loaded = load_roi_set(roi_save_file)
            self.assertEqual(loaded.labels, ("roi_0001", "roi_0002"))

    def test_napari_roi_save_uses_cropped_view_and_full_frame_storage(self) -> None:
        """Confirm napari saves crop-native labels as full-frame ROI masks.

        Inputs: recording whose valid display crop is smaller than the movie
        frame, and one drawn ROI in the cropped Labels layer.
        Outputs: saved ROI masks keep the full movie shape with zeros outside
        the displayed crop.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            roi_save_file = root / "rois.h5"
            recording_path = _write_converted_recording(
                root,
                movie_values=np.arange(27, dtype=np.float64).reshape(3, 3, 3),
                alignment_valid_crop=SpatialCrop(
                    axis0_start=1,
                    axis0_stop=3,
                    axis1_start=0,
                    axis1_stop=2,
                    original_shape=(3, 3),
                    source="alignment_valid_crop",
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(
                recording_path,
                viewer=viewer,
                roi_save_file=roi_save_file,
            )
            viewer.labels[0].data = np.array([[0, 1], [0, 1]], dtype=np.int64)
            save_widget = cast(Any, opened.save_rois_widget)

            result = save_widget()

            self.assertIn("Saved 1 ROI", str(result))
            loaded = load_roi_set(roi_save_file)
            self.assertEqual(loaded.masks.shape, (1, 3, 3))
            self.assertFalse(np.any(loaded.masks[:, 0, :]))
            self.assertFalse(np.any(loaded.masks[:, :, 2]))
            self.assertTrue(np.any(loaded.masks[:, 1:3, 0:2]))

    def test_napari_roi_save_rejects_full_frame_labels_layer(self) -> None:
        """Confirm Save ROIs fails when the active layer is not crop-native.

        Inputs: cropped recording display plus a stale full-frame Labels image.
        Outputs: clear status instead of silently saving an invalid GUI mask.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                movie_values=np.arange(27, dtype=np.float64).reshape(3, 3, 3),
                alignment_valid_crop=SpatialCrop(
                    axis0_start=1,
                    axis0_stop=3,
                    axis1_start=0,
                    axis1_stop=2,
                    original_shape=(3, 3),
                    source="alignment_valid_crop",
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            viewer.labels[0].data = np.ones((3, 3), dtype=np.int64)
            save_widget = cast(Any, opened.save_rois_widget)

            result = save_widget()

            self.assertIn(
                "ROI Labels layer must use the cropped recording view",
                str(result),
            )

    def test_response_update_rejects_full_frame_labels_layer(self) -> None:
        """Confirm response updates use the same crop-native ROI contract.

        Inputs: cropped recording display plus a stale full-frame Labels image.
        Outputs: response widget status explains that the layer shape is
        invalid before analysis runs.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                movie_values=np.arange(27, dtype=np.float64).reshape(3, 3, 3),
                alignment_valid_crop=SpatialCrop(
                    axis0_start=1,
                    axis0_stop=3,
                    axis1_start=0,
                    axis1_stop=2,
                    original_shape=(3, 3),
                    source="alignment_valid_crop",
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            viewer.labels[0].data = np.ones((3, 3), dtype=np.int64)
            response_widget = cast(Any, opened.response_plot_widget)

            response_widget.update_from_current_rois()

            status_text = "\n".join(
                label.text() for label in response_widget.findChildren(QLabel)
            )
            self.assertIn(
                "ROI Labels layer must use the cropped recording view",
                status_text,
            )

    def test_response_update_from_rois_does_not_write_analysis_file(self) -> None:
        """Confirm napari plot previews compute in memory only.

        Inputs: edited Labels layer and a patched response calculation.
        Outputs: plot data shown in the widget without creating analysis HDF5.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            viewer.labels[0].data = np.array([[1, 0], [0, 0]], dtype=np.int64)
            response_widget = cast(Any, opened.response_plot_widget)

            with patch(
                "twopy.napari.interactive.compute_response_plot_data_from_roi_set",
                return_value=_tiny_response_plot_data(),
            ):
                response_widget.update_from_current_rois()

            self.assertFalse((root / "analysis_outputs.h5").exists())
            self.assertFalse((root / "response_summary.csv").exists())
            self.assertIsNotNone(response_widget._plot_data)

    def test_save_analysis_button_writes_roi_and_analysis_outputs(self) -> None:
        """Confirm the Update tab can persist current ROI analysis.

        Inputs: edited Labels layer and a patched analysis workflow.
        Outputs: ROI HDF5 file written beside the recording without replacing
        the current in-memory plot preview.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            analysis_path = root / "analysis_outputs.h5"
            summary_path = root / "response_summary.csv"
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            viewer.labels[0].data = np.array([[1, 0], [0, 0]], dtype=np.int64)
            response_widget = cast(Any, opened.response_plot_widget)
            preview_plot_data = _tiny_response_plot_data()
            response_widget._plot_data = preview_plot_data

            with patch(
                "twopy.napari.plotting.docks.analyze_recording_responses",
                return_value=SimpleNamespace(
                    output_path=analysis_path,
                    response_summary_csv_path=summary_path,
                    grouped_responses=_tiny_grouped_responses(),
                ),
            ) as analyze:
                response_widget.save_analysis_and_rois()

            roi_path = root / "rois.h5"
            self.assertTrue(roi_path.is_file())
            analyze.assert_called_once()
            _, roi_set = analyze.call_args.args
            self.assertEqual(roi_set.labels, ("roi_0001",))
            self.assertEqual(analyze.call_args.kwargs["output_path"], analysis_path)
            self.assertIs(response_widget._plot_data, preview_plot_data)

    def test_live_response_controller_updates_after_paint_event(self) -> None:
        """Confirm committed Labels painting triggers a response plot refresh.

        Inputs: selected recording, fake Labels paint emitter, and patched
        response calculation.
        Outputs: the plot receiver gets new plot data after an event.
        """
        _ = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(_write_converted_recording(root))
            events = _FakeLabelEvents()
            layer = _FakeLayer(
                name="rois",
                data=np.array([[1, 0], [0, 0]], dtype=np.int64),
                options={},
                events=events,
            )
            receiver = _FakePlotReceiver()
            controller = LiveResponseController(
                receiver,
                debounce_ms=0,
                run_async=False,
            )
            controller.set_context(recording, layer)

            with patch(
                "twopy.napari.interactive.compute_response_plot_data_from_roi_set",
                return_value=_tiny_response_plot_data(),
            ):
                events.paint.emit()

            self.assertIsNotNone(receiver.plot_data)
            self.assertIsNone(receiver.status)

    def test_live_response_controller_ignores_labels_update_event(self) -> None:
        """Confirm display refresh events do not recompute responses.

        Inputs: selected recording and fake Labels events.
        Outputs: ``labels_update`` emits no plot data because napari can emit it
        during mouse movement without a committed ROI edit.
        """
        _ = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(_write_converted_recording(root))
            events = _FakeLabelEvents()
            layer = _FakeLayer(
                name="rois",
                data=np.array([[1, 0], [0, 0]], dtype=np.int64),
                options={},
                events=events,
            )
            receiver = _FakePlotReceiver()
            controller = LiveResponseController(
                receiver,
                debounce_ms=0,
                run_async=False,
            )
            controller.set_context(recording, layer)

            with patch(
                "twopy.napari.interactive.compute_response_plot_data_from_roi_set",
                return_value=_tiny_response_plot_data(),
            ) as compute:
                events.labels_update.emit()

            compute.assert_not_called()
            self.assertIsNone(receiver.plot_data)

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

            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=roi_save_file,
            )
            controls = cast(_ControlWidget, control_docks.load_widget)
            load_widget = cast(Any, controls[1])

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
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            controls = cast(_ControlWidget, control_docks.load_widget)
            load_widget = cast(Any, controls[1])
            save_widget = cast(Any, control_docks.save_rois_widget)

            load_widget(recording_folder=recording_path)
            viewer.labels[0].data = np.array([[1, 0], [0, 0]])
            result = save_widget()

            self.assertIn("Saved 1 ROI", str(result))
            self.assertTrue((root / "rois.h5").exists())

    def test_loaded_recordings_panel_selects_and_unloads_layers(self) -> None:
        """Confirm multi-recording selection controls save and layer ownership.

        Inputs: two converted folders loaded into one fake viewer.
        Outputs: selected recording controls the ROI save default, and unload
        removes only that recording's layers.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            _write_converted_recording(first)
            _write_converted_recording(second)
            viewer = _FakeViewer()
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            controls = cast(_ControlWidget, control_docks.load_widget)
            load_widget = cast(Any, controls[1])
            save_widget = cast(Any, control_docks.save_rois_widget)
            panel = cast(QWidget, control_docks.loaded_recordings_widget)
            loaded_list = panel.findChild(QListWidget)
            unload_button = next(
                button
                for button in panel.findChildren(QPushButton)
                if button.text() == "Unload Recording"
            )

            load_widget(recording_folder=first)
            load_widget(recording_folder=second)

            self.assertEqual(loaded_list.count(), 2)
            self.assertEqual(loaded_list.currentRow(), 1)
            self.assertEqual(len(viewer.images), 4)
            self.assertEqual(len(viewer.labels), 2)
            self.assertFalse(viewer.images[0].visible)
            self.assertFalse(viewer.images[1].visible)
            self.assertFalse(viewer.labels[0].visible)
            self.assertTrue(viewer.images[2].visible)
            self.assertTrue(viewer.images[3].visible)
            self.assertTrue(viewer.labels[1].visible)

            loaded_list.setCurrentRow(0)
            self.assertTrue(viewer.images[0].visible)
            self.assertTrue(viewer.images[1].visible)
            self.assertTrue(viewer.labels[0].visible)
            self.assertFalse(viewer.images[2].visible)
            self.assertFalse(viewer.images[3].visible)
            self.assertFalse(viewer.labels[1].visible)
            viewer.labels[0].data = np.array([[1, 0], [0, 0]])
            result = save_widget()
            self.assertIn("Saved 1 ROI", str(result))
            self.assertTrue((first / "rois.h5").is_file())

            unload_button.click()

            self.assertEqual(loaded_list.count(), 1)
            self.assertEqual(len(viewer.images), 2)
            self.assertEqual(len(viewer.labels), 1)
            self.assertTrue(viewer.images[0].visible)
            self.assertTrue(viewer.images[1].visible)
            self.assertTrue(viewer.labels[0].visible)
            remaining_item = loaded_list.item(0)
            assert remaining_item is not None
            self.assertIn(str(second), remaining_item.text())

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

            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            controls = cast(_ControlWidget, control_docks.load_widget)
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
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            controls = cast(_ControlWidget, control_docks.load_widget)
            load_widget = cast(Any, controls[1])

            result = load_widget(recording_folder=root)

            self.assertIn(str(recording_path), str(result))
            self.assertEqual(len(viewer.images), 2)
            self.assertIn(root.name, str(load_widget.recording_folder.line_edit.value))
            np.testing.assert_array_equal(
                np.unique(roi_label_image_from_layer(viewer.labels[0])),
                np.array([0, 1]),
            )

    def test_recording_folder_picker_shows_short_loaded_path(self) -> None:
        """Confirm loaded recording paths show the useful folder tail.

        Inputs: long converted output path selected through the load widget.
        Outputs: picker text starts with ``...`` and ends with the recording
            folder tail.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = (
                Path(temp_dir)
                / "very_long_project_name"
                / "fly"
                / "stimulus"
                / "2025"
                / "10_03"
                / "twopy"
            )
            root.mkdir(parents=True)
            _write_converted_recording(root)
            viewer = _FakeViewer()
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            controls = cast(_ControlWidget, control_docks.load_widget)
            load_widget = cast(Any, controls[1])

            load_widget(recording_folder=root)

            display_text = str(load_widget.recording_folder.line_edit.value)
            self.assertTrue(display_text.startswith("..."))
            self.assertTrue(display_text.endswith("/2025/10_03/twopy/"))

    def test_movie_frame_slider_reuses_resolved_recording_path(self) -> None:
        """Confirm movie range changes work after the path field is shortened.

        Inputs: long converted output path loaded through the control widget.
        Outputs: changing the frame slider reloads without resolving the
        shortened display text as a filesystem path.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / ("long_" * 20) / "twopy"
            root.mkdir(parents=True)
            _write_converted_recording(root)
            viewer = _FakeViewer()
            control_docks = add_twopy_magicgui_controls(
                viewer,
                roi_labels_layer=None,
                roi_save_file=Path("unused.h5"),
            )
            controls = cast(_ControlWidget, control_docks.load_widget)
            load_widget = cast(Any, controls[1])

            load_widget(recording_folder=root)
            load_widget.movie_frame_range.value = (0, 1)

            self.assertEqual(load_widget.movie_frame_range.value, (0, 1))
            self.assertEqual(len(viewer.images), 2)

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

    def test_recording_path_resolution_converts_source_folder_when_output_missing(
        self,
    ) -> None:
        """Confirm source folders are converted before napari opens them.

        Inputs: source-shaped recording folder without twopy output.
        Outputs: converted paths plus a flag saying conversion ran.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            _write_source_recording_shape(source_dir)

            with patch(
                "twopy.napari.loading.convert_recording_to_twopy",
                side_effect=_fake_convert_recording,
            ) as convert:
                resolved = resolve_or_convert_recording(source_dir)

            self.assertTrue(resolved.was_converted)
            convert.assert_called_once_with(source_dir.resolve())
            self.assertEqual(
                resolved.paths.recording_data_path,
                (source_dir / "twopy" / "recording_data.h5").resolve(),
            )
            self.assertEqual(
                resolved.paths.movie_path,
                (source_dir / "twopy" / "aligned_movie.h5").resolve(),
            )

    def test_recording_path_resolution_repairs_missing_converted_movie(self) -> None:
        """Confirm incomplete source-local twopy output is regenerated in place.

        Inputs: source-shaped folder with ``twopy/recording_data.h5`` but no
            ``aligned_movie.h5``.
        Outputs: conversion called with that existing output directory.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            output_dir = source_dir / "twopy"
            _write_source_recording_shape(source_dir)
            output_dir.mkdir()
            (output_dir / "recording_data.h5").touch()

            with patch(
                "twopy.napari.loading.convert_recording_to_twopy",
                side_effect=_fake_convert_recording,
            ) as convert:
                resolved = resolve_or_convert_recording(source_dir)

            self.assertTrue(resolved.was_converted)
            convert.assert_called_once_with(
                source_dir.resolve(),
                output_dir=output_dir.resolve(),
            )
            self.assertEqual(
                resolved.paths.movie_path,
                (output_dir / "aligned_movie.h5").resolve(),
            )

    def test_recording_path_resolution_repairs_mirrored_output_from_source_attr(
        self,
    ) -> None:
        """Confirm missing movie repair works outside the source folder.

        Inputs: converted ``recording_data.h5`` with a ``source_session_dir``
            attribute and no ``aligned_movie.h5``.
        Outputs: conversion called with the selected converted output folder.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            output_dir = root / "converted"
            _write_source_recording_shape(source_dir)
            output_dir.mkdir()
            with h5py.File(output_dir / "recording_data.h5", "w") as h5_file:
                h5_file.attrs["source_session_dir"] = str(source_dir)

            with patch(
                "twopy.napari.loading.convert_recording_to_twopy",
                side_effect=_fake_convert_recording,
            ) as convert:
                resolved = resolve_or_convert_recording(output_dir)

            self.assertTrue(resolved.was_converted)
            convert.assert_called_once_with(
                source_dir.resolve(),
                output_dir=output_dir.resolve(),
            )
            self.assertEqual(
                resolved.paths.movie_path,
                (output_dir / "aligned_movie.h5").resolve(),
            )

    def test_roi_visibility_options_show_color_swatches(self) -> None:
        """Confirm ROI visibility rows include color squares.

        Inputs: two ROI labels and two Qt colors.
        Outputs: option widget with swatch styles for both colors.
        """
        _ = QApplication.instance() or QApplication([])
        widget = visibility_options_widget(
            title="ROIs",
            labels=("roi_1", "roi_2"),
            visibility={"roi_1": True, "roi_2": False},
            on_change=lambda _label, _visible: None,
            colors=(QColor("#ff0000"), QColor("#0000ff")),
        )

        styles = "\n".join(child.styleSheet() for child in widget.findChildren(QWidget))
        self.assertIn("#ff0000", styles)
        self.assertIn("#0000ff", styles)

    def test_visibility_select_none_uses_one_batch_callback(self) -> None:
        """Confirm bulk visibility changes avoid one redraw per checkbox.

        Inputs: three ROI labels and a Select-none button click.
        Outputs: no per-checkbox callback and one batch visibility update.
        """
        _ = QApplication.instance() or QApplication([])
        single_calls: list[tuple[str, bool]] = []
        batch_calls: list[dict[str, bool]] = []
        widget = visibility_options_widget(
            title="ROIs",
            labels=("roi_1", "roi_2", "roi_3"),
            visibility={"roi_1": True, "roi_2": True, "roi_3": True},
            on_change=lambda label, visible: single_calls.append((label, visible)),
            on_change_batch=lambda visibility: batch_calls.append(visibility),
        )
        none_button = next(
            button
            for button in widget.findChildren(QPushButton)
            if button.text() == "None"
        )

        none_button.click()

        self.assertEqual(single_calls, [])
        self.assertEqual(
            batch_calls,
            [{"roi_1": False, "roi_2": False, "roi_3": False}],
        )

    def test_visibility_options_use_stable_keys_for_duplicate_labels(self) -> None:
        """Confirm duplicate display labels can still toggle distinct items.

        Inputs: two checkboxes with the same visible label but separate keys.
        Outputs: toggling the second checkbox reports the second key.
        """
        _ = QApplication.instance() or QApplication([])
        calls: list[tuple[object, bool]] = []
        widget = visibility_options_widget(
            title="Epochs",
            labels=("same epoch", "same epoch"),
            keys=((1, "same epoch"), (2, "same epoch")),
            visibility={"same epoch": True},
            on_change=lambda key, visible: calls.append((key, visible)),
        )
        checkboxes = widget.findChildren(QCheckBox)

        checkboxes[1].setChecked(False)

        self.assertEqual(calls, [((2, "same epoch"), False)])

    def test_visibility_options_read_initial_state_from_keys(self) -> None:
        """Confirm keyed rows do not read visibility from duplicate labels.

        Inputs: two rows with the same display label and different keys.
        Outputs: each checkbox uses the matching key state.
        """
        _ = QApplication.instance() or QApplication([])
        widget = visibility_options_widget(
            title="ROIs",
            labels=("same roi", "same roi"),
            keys=((0, "same roi"), (1, "same roi")),
            visibility={(0, "same roi"): True, (1, "same roi"): False},
            on_change=lambda _key, _visible: None,
        )
        checkboxes = widget.findChildren(QCheckBox)

        self.assertTrue(checkboxes[0].isChecked())
        self.assertFalse(checkboxes[1].isChecked())

    def test_roi_visibility_can_hide_labels_layer_rois(self) -> None:
        """Confirm plot ROI visibility can hide ROI labels in napari.

        Inputs: one Labels layer with two ROI labels and one hidden ROI.
        Outputs: hidden ROI color becomes transparent while label pixels remain.
        """
        label_image = np.array([[0, 1], [2, 0]], dtype=np.int64)
        layer = Labels(label_image)

        apply_roi_visibility_to_labels_layer(
            layer,
            roi_labels=("roi_1", "roi_2"),
            visibility={"roi_1": False, "roi_2": True},
            colors=(QColor("#ff0000"), QColor("#0000ff")),
        )

        np.testing.assert_array_equal(layer.data, label_image)
        self.assertEqual(layer.get_color(1)[3], 0.0)
        self.assertEqual(layer.get_color(2)[3], 1.0)

    def test_roi_visibility_keeps_new_labels_drawable(self) -> None:
        """Confirm hiding ROIs does not make future Labels values transparent.

        Inputs: Labels layer with one hidden high-numbered ROI label.
        Outputs: that ROI is hidden, but a new unseen label keeps visible color.
        """
        label_image = np.zeros((2, 2), dtype=np.int64)
        layer = Labels(label_image)

        apply_roi_visibility_to_labels_layer(
            layer,
            roi_labels=("roi_0010",),
            visibility={"roi_0010": False},
            colors=(QColor("#ff0000"),),
        )

        self.assertEqual(layer.get_color(10)[3], 0.0)
        self.assertEqual(layer.get_color(11)[3], 1.0)
        self.assertEqual(layer.get_color(12)[3], 1.0)
        self.assertFalse(np.array_equal(layer.get_color(11), layer.get_color(12)))

    def test_roi_visibility_layer_uses_stable_keys_for_duplicate_names(self) -> None:
        """Confirm duplicate ROI names can hide distinct label values.

        Inputs: two label values with the same displayed ROI name.
        Outputs: hiding the second key makes only the second label transparent.
        """
        layer = Labels(np.array([[1, 2]], dtype=np.int64))

        apply_roi_visibility_to_labels_layer(
            layer,
            roi_labels=("cell", "cell"),
            visibility={(0, "cell"): True, (1, "cell"): False},
            keys=((0, "cell"), (1, "cell")),
            colors=(QColor("#ff0000"), QColor("#0000ff")),
        )

        self.assertEqual(layer.get_color(1)[3], 1.0)
        self.assertEqual(layer.get_color(2)[3], 0.0)

    def test_display_helpers_transpose_movie_axes_for_napari(self) -> None:
        """Confirm movie axes are transposed only at the napari boundary.

        Inputs: non-square movie-coordinate image.
        Outputs: display image with swapped spatial axes.
        """
        movie_image = np.arange(6, dtype=np.float64).reshape(2, 3)

        display_image = display_image_from_movie_image(movie_image)

        np.testing.assert_array_equal(display_image, movie_image.T)

    def test_display_labels_round_trip_through_alignment_crop(self) -> None:
        """Confirm napari labels return to full-frame movie coordinates.

        Inputs: full-frame labels plus an alignment-valid crop.
        Outputs: cropped/transposed display labels and restored full-frame
        labels with zeros outside the displayed crop.
        """
        crop = SpatialCrop(
            axis0_start=1,
            axis0_stop=3,
            axis1_start=0,
            axis1_stop=3,
            original_shape=(3, 4),
            source="alignment_valid_crop",
        )
        movie_labels = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 2, 0],
                [0, 3, 0, 0],
            ],
            dtype=np.int64,
        )

        display_labels = display_labels_from_movie_labels(movie_labels, crop)
        layer = _FakeLayer(
            name="rois",
            data=display_labels,
            options={"metadata": display_metadata_for_spatial_crop(crop)},
        )

        self.assertEqual(display_labels.shape, (3, 2))
        np.testing.assert_array_equal(roi_label_image_from_layer(layer), movie_labels)
        np.testing.assert_array_equal(
            movie_labels_from_display_layer(layer),
            np.array(
                [
                    [0, 0, 0, 0],
                    [1, 0, 2, 0],
                    [0, 3, 0, 0],
                ],
                dtype=np.int64,
            ),
        )

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

    def test_movie_contrast_sampling_skips_recording_edges(self) -> None:
        """Confirm movie display sampling avoids the first and last 10 percent.

        Inputs: 100-frame recording contract and deterministic random seed.
        Outputs: 10 sampled frames, none from the first or last 10 frames.
        """
        frame_indices = interior_random_frame_indices(
            frame_count=100,
            sample_count=10,
            edge_fraction=0.10,
            random_seed=0,
        )

        np.testing.assert_array_equal(
            frame_indices,
            np.array([11, 13, 15, 23, 29, 33, 47, 55, 70, 75]),
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

    def test_response_plot_data_sorts_epochs_by_number(self) -> None:
        """Confirm epoch plots and option lists use epoch-number order.

        Inputs: grouped responses whose first trial is epoch 3 before epoch 1.
        Outputs: plot data ordered as epoch 1, then epoch 3.
        """
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((4, 1), dtype=np.float64),
            baseline=np.ones((4, 1), dtype=np.float64),
            values=np.arange(4, dtype=np.float64).reshape(4, 1),
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
            (
                EpochFrameWindow(FrameWindow(0, 0, 2, "epoch_3"), 3, "Third"),
                EpochFrameWindow(FrameWindow(1, 2, 4, "epoch_1"), 1, "First"),
            ),
            data_rate_hz=1.0,
        )

        plot_data = response_plot_data_from_grouped(grouped)

        self.assertEqual(
            tuple(epoch.epoch_number for epoch in plot_data.epochs),
            (1, 3),
        )

    def test_epoch_visibility_toggle_is_idempotent_by_key(self) -> None:
        """Confirm hiding and showing an epoch restores the same epoch.

        Inputs: two epoch plots and direct stable epoch-key toggles.
        Outputs: the selected epoch set returns to the original keys.
        """
        _ = QApplication.instance() or QApplication([])
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((4, 1), dtype=np.float64),
            baseline=np.ones((4, 1), dtype=np.float64),
            values=np.arange(4, dtype=np.float64).reshape(4, 1),
            labels=("roi_1",),
            start_frame=0,
            stop_frame=4,
            tau=0.0,
            amplitudes=np.ones(1, dtype=np.float64),
            interleave_frame_numbers=np.array([0.0]),
            interleave_fluorescence=np.ones((1, 1), dtype=np.float64),
            metadata={"method": "test"},
        )
        plot_data = response_plot_data_from_grouped(
            group_delta_f_over_f_by_epoch(
                dff,
                (
                    EpochFrameWindow(FrameWindow(0, 0, 2, "first"), 1, "First"),
                    EpochFrameWindow(FrameWindow(1, 2, 4, "second"), 2, "Second"),
                ),
                data_rate_hz=1.0,
            ),
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(plot_data, reset_axes=True)

        response_widget._set_epoch_visibility((1, "First"), False)
        response_widget._set_epoch_visibility((1, "First"), True)

        self.assertEqual(
            response_widget._visible_epoch_keys(),
            ((1, "First"), (2, "Second")),
        )

    def test_roi_visibility_toggle_is_idempotent_by_key(self) -> None:
        """Confirm ROI visibility does not depend on unique display labels.

        Inputs: one plot data object with duplicate ROI names.
        Outputs: hiding and showing the second ROI restores both ROI indices.
        """
        _ = QApplication.instance() or QApplication([])
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((4, 2), dtype=np.float64),
            baseline=np.ones((4, 2), dtype=np.float64),
            values=np.arange(8, dtype=np.float64).reshape(4, 2),
            labels=("same", "same"),
            start_frame=0,
            stop_frame=4,
            tau=0.0,
            amplitudes=np.ones(2, dtype=np.float64),
            interleave_frame_numbers=np.array([0.0]),
            interleave_fluorescence=np.ones((1, 2), dtype=np.float64),
            metadata={"method": "test"},
        )
        plot_data = response_plot_data_from_grouped(
            group_delta_f_over_f_by_epoch(
                dff,
                (EpochFrameWindow(FrameWindow(0, 0, 4, "first"), 1, "First"),),
                data_rate_hz=1.0,
            ),
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(plot_data, reset_axes=True)

        response_widget._set_roi_visibility((1, "same"), False)
        self.assertEqual(response_widget._visible_roi_indices(), (0,))

        response_widget._set_roi_visibility((1, "same"), True)
        self.assertEqual(response_widget._visible_roi_indices(), (0, 1))

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

    def test_epoch_plot_widget_prefers_square_size(self) -> None:
        """Confirm response plots ask Qt for a square panel.

        Inputs: plot-ready data for one epoch.
        Outputs: equal preferred width and height.
        """
        _ = QApplication.instance() or QApplication([])
        plot_data = _tiny_response_plot_data()

        widget = EpochPlotWidget(
            plot_data.epochs[0],
            show_sem=True,
            roi_indices=(0,),
            roi_colors=(QColor("#ff0000"),),
            time_min=0.0,
            time_max=1.0,
            value_min=0.0,
            value_max=1.0,
            plot_size=480,
        )

        self.assertEqual(widget.sizeHint().width(), widget.sizeHint().height())
        self.assertEqual(widget.sizeHint().width(), 480)

        widget.update_display(
            show_sem=False,
            roi_indices=(0,),
            roi_colors=(QColor("#ff0000"),),
            time_min=-1.0,
            time_max=2.0,
            value_min=-0.5,
            value_max=1.5,
            plot_size=320,
        )

        self.assertEqual(widget.sizeHint().width(), 320)

    def test_response_plot_export_writes_editable_figure_bundle(self) -> None:
        """Confirm response plots export in the expected file formats.

        Inputs: one tiny plot-ready epoch and a temporary output folder.
        Outputs: PDF and PNG files under a plot-specific export folder.
        """
        plot_data = _tiny_response_plot_data()
        with tempfile.TemporaryDirectory() as temp_dir:
            written = export_epoch_plots(
                plot_data=plot_data,
                output_dir=Path(temp_dir),
                epoch_keys=((1, "Odor"),),
                roi_indices=(0,),
                roi_colors=("#ff0000",),
                show_sem=True,
                time_bounds=(0.0, 1.0),
                value_bounds=(0.0, 1.0),
            )

            self.assertEqual({path.suffix for path in written}, {".pdf", ".png"})
            self.assertEqual({path.parent.name for path in written}, {"plots"})
            self.assertTrue(all(path.is_file() for path in written))

    def test_export_roi_contours_keep_display_orientation(self) -> None:
        """Confirm exported ROI contours keep the same top-origin view as napari.

        Inputs: one ROI in the top rows of a display-coordinate label image.
        Outputs: contour vertices stay near the top rows, not vertically flipped.
        """
        labels = np.zeros((6, 6), dtype=np.int64)
        labels[0:2, 1:3] = 1
        fig = Figure()
        ax = fig.add_subplot(111)

        draw_roi_contours(
            ax,
            labels=labels,
            roi_label_values=(1,),
            roi_indices=(0,),
            roi_colors=("#ff0000",),
        )
        contour_lines = cast(LineCollection, ax.collections[0])
        vertices = np.asarray(
            contour_lines.get_segments()[0],
            dtype=np.float64,
        )

        self.assertLess(float(np.max(vertices[:, 1])), 2.0)

    def test_export_roi_contours_follow_pixel_edges_at_image_boundary(self) -> None:
        """Confirm edge-touching ROIs export outlines on the image boundary.

        Inputs: one ROI pixel at the top-left image corner.
        Outputs: boundary segments include the top and left external pixel
        edges instead of shifting inward.
        """
        segments = roi_boundary_segments(
            np.array([[True, False], [False, False]], dtype=np.bool_),
        )

        self.assertIn(((-0.5, -0.5), (0.5, -0.5)), segments)
        self.assertIn(((-0.5, -0.5), (-0.5, 0.5)), segments)

    def test_export_crops_full_frame_display_labels_to_recording_view(self) -> None:
        """Confirm stale full-frame Labels data exports only the displayed crop.

        Inputs: converted recording with an alignment-valid crop and one
        full-frame display label image.
        Outputs: labels cropped to the same display shape as the recording
        image, excluding ROI pixels outside the valid crop.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                movie_values=np.arange(27, dtype=np.float64).reshape(3, 3, 3),
                alignment_valid_crop=SpatialCrop(
                    axis0_start=1,
                    axis0_stop=3,
                    axis1_start=0,
                    axis1_stop=2,
                    original_shape=(3, 3),
                    source="alignment_valid_crop",
                ),
            )
            opened = open_recording_in_napari(
                recording_path,
                viewer=_FakeViewer(),
                add_controls=False,
            )
            full_frame_display_labels = np.zeros((3, 3), dtype=np.int64)
            full_frame_display_labels[0, 0] = 1
            full_frame_display_labels[1, 2] = 2

            labels = labels_for_recording_image(
                full_frame_display_labels,
                recording=opened.recording,
            )

            self.assertEqual(labels.shape, (2, 2))
            np.testing.assert_array_equal(np.unique(labels), np.array([0, 2]))

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


def _write_converted_recording(
    root: Path,
    *,
    movie_values: npt.NDArray[np.float64] | None = None,
    alignment_valid_crop: SpatialCrop | None = None,
) -> Path:
    """Write a tiny converted recording for adapter tests.

    Args:
        root: Temporary directory receiving HDF5 files.
        movie_values: Optional movie array shaped ``(frames, axis0, axis1)``.
        alignment_valid_crop: Optional spatial crop metadata.

    Returns:
        Path to ``recording_data.h5``.
    """
    resolved_movie_values = movie_values
    if resolved_movie_values is None:
        resolved_movie_values = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
    resolved_crop = alignment_valid_crop
    if resolved_crop is None:
        resolved_crop = SpatialCrop(
            axis0_start=0,
            axis0_stop=resolved_movie_values.shape[1],
            axis1_start=0,
            axis1_stop=resolved_movie_values.shape[2],
            original_shape=resolved_movie_values.shape[1:],
            source="full_frame",
        )
    movie_path = root / "aligned_movie.h5"
    with h5py.File(movie_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = "aligned-movie"
        h5_file.create_dataset("movie/aligned", data=resolved_movie_values)

    recording_path = root / "recording_data.h5"
    with h5py.File(recording_path, "w") as h5_file:
        h5_file.attrs["twopy_format"] = "converted-recording"
        h5_file.attrs["source_session_dir"] = str(root / "source")
        movie_group = h5_file.create_group("movie")
        movie_group.attrs["aligned_movie_file"] = "aligned_movie.h5"
        movie_group.attrs["aligned_movie_dataset"] = "movie/aligned"
        movie_group.attrs["aligned_movie_shape"] = resolved_movie_values.shape
        movie_group.attrs["aligned_movie_dtype"] = "float64"
        movie_group.create_dataset(
            "mean_image",
            data=resolved_movie_values.mean(axis=0),
        )
        crop_group = movie_group.create_group("alignment_valid_crop")
        crop_group.attrs["source"] = resolved_crop.source
        crop_group.attrs["axis0_start"] = resolved_crop.axis0_start
        crop_group.attrs["axis0_stop"] = resolved_crop.axis0_stop
        crop_group.attrs["axis1_start"] = resolved_crop.axis1_start
        crop_group.attrs["axis1_stop"] = resolved_crop.axis1_stop
        crop_group.attrs["original_shape"] = resolved_crop.original_shape
        crop_group.create_dataset(
            "alignment_shift_pixels",
            data=np.zeros(resolved_movie_values.shape[0], dtype=np.float64),
        )
        crop_group.create_dataset(
            "motion_artifact_mask",
            data=np.zeros(resolved_movie_values.shape[0], dtype=np.bool_),
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
            data=np.zeros(resolved_movie_values.shape[0], dtype=np.float64),
        )
        photodiode_group.create_dataset(
            "high_res_pd",
            data=np.zeros(resolved_movie_values.shape[0], dtype=np.float64),
        )
        frame_counts = h5_file.create_group("frame_counts")
        frame_counts.attrs["aligned_movie_frames"] = resolved_movie_values.shape[0]
        frame_counts.attrs["imaging_res_pd_samples"] = resolved_movie_values.shape[0]
        frame_counts.attrs["acquisition_number_of_frames"] = (
            resolved_movie_values.shape[0]
        )
        frame_counts.attrs["imaging_res_pd_minus_movie"] = 0
        frame_counts.attrs["acquisition_minus_movie"] = 0
    return recording_path


def _tiny_response_plot_data() -> ResponsePlotData:
    """Build one tiny response plot data object for napari plot tests.

    Args:
        None.

    Returns:
        Plot-ready response data with one ROI and one epoch.
    """
    return response_plot_data_from_grouped(_tiny_grouped_responses())


def _tiny_grouped_responses() -> GroupedRoiResponses:
    """Build one tiny grouped response object for napari plot tests.

    Args:
        None.

    Returns:
        Grouped response data with one ROI and one epoch.
    """
    dff = RoiDeltaFOverF(
        fluorescence=np.ones((2, 1), dtype=np.float64),
        baseline=np.ones((2, 1), dtype=np.float64),
        values=np.array([[0.0], [1.0]], dtype=np.float64),
        labels=("roi_1",),
        start_frame=0,
        stop_frame=2,
        tau=0.0,
        amplitudes=np.ones(1, dtype=np.float64),
        interleave_frame_numbers=np.array([0.0]),
        interleave_fluorescence=np.ones((1, 1), dtype=np.float64),
        metadata={"method": "test"},
    )
    return group_delta_f_over_f_by_epoch(
        dff,
        (EpochFrameWindow(FrameWindow(0, 0, 2, "odor"), 1, "Odor"),),
        data_rate_hz=1.0,
    )


def _write_source_recording_shape(root: Path) -> None:
    """Write the required source file names without real imaging contents.

    Args:
        root: Temporary source recording folder.

    Returns:
        None. The folder shape is enough for napari conversion routing tests
        because the actual converter is patched there.
    """
    root.mkdir(parents=True, exist_ok=True)
    stimulus_dir = root / "stimulusData"
    stimulus_dir.mkdir()
    for path in (
        root / "alignedMovie.mat",
        root / "stimulus_name_changes_001.tif",
        root / "stimulus_name_changes_001_ch1_disinterleaved_alignment.txt",
        root / "defaultAlignChannel.txt",
        root / "highResPd.mat",
        root / "imageDescription.mat",
        root / "imagingResPd.mat",
        stimulus_dir / "filebackup.zip",
    ):
        path.touch()


def _fake_convert_recording(
    source_dir: Path,
    output_dir: Path | None = None,
    **_kwargs: object,
) -> ConvertedRecording:
    """Create tiny placeholder converted files for routing tests.

    Args:
        source_dir: Source recording folder passed by napari loading.
        output_dir: Optional output folder requested by napari loading.
        _kwargs: Ignored converter keyword arguments.

    Returns:
        Converted recording summary pointing at placeholder files.
    """
    destination = output_dir or source_dir / "twopy"
    destination.mkdir(parents=True, exist_ok=True)
    recording_path = destination / "recording_data.h5"
    movie_path = destination / "aligned_movie.h5"
    recording_path.touch()
    movie_path.touch()
    return ConvertedRecording(
        path=recording_path,
        movie_path=movie_path,
        source_session_dir=source_dir,
        movie_shape=(1, 1, 1),
        mean_image_start_frame=0,
        mean_image_stop_frame=1,
    )


if __name__ == "__main__":
    unittest.main()
