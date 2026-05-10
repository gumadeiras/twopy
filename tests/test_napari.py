"""Tests for the thin napari adapter.

Inputs: a tiny converted recording, ROI labels, and a fake viewer.
Outputs: layer data sent to napari-shaped methods and saved ROI HDF5 files.
"""

import sqlite3
import tempfile
import unittest
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from os import chdir, environ
from pathlib import Path
from time import monotonic, sleep
from types import SimpleNamespace
from typing import Any, Protocol, cast
from unittest.mock import patch

import h5py
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from napari.layers import Labels
from qtpy.QtCore import Qt
from qtpy.QtGui import QCloseEvent, QColor
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QWidget,
)

import twopy.napari.controls as napari_controls
from twopy import (
    add_twopy_magicgui_controls,
    load_roi_set,
    make_roi_set,
    open_recording_in_napari,
    roi_label_image_from_layer,
    save_napari_label_rois,
    save_roi_set,
)
from twopy._version import __version__
from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.persistence import save_analysis_outputs
from twopy.analysis.response_processing import (
    ResponseProcessingOptions,
    SmoothingOptions,
)
from twopy.analysis.response_window_options import ResponseWindowOptions
from twopy.analysis.responses import GroupedRoiResponses, group_delta_f_over_f_by_epoch
from twopy.analysis.trials import EpochFrameWindow, FrameWindow
from twopy.conversion.types import ConvertedRecording
from twopy.converted import load_converted_recording
from twopy.napari.database_search import (
    ExperimentLoadErrorDialog,
    ExperimentLoadFailure,
    ExperimentLoadResult,
    ExperimentSearchDialog,
    ExperimentSearchErrorDialog,
)
from twopy.napari.display import (
    display_image_from_movie_image,
    display_labels_from_movie_labels,
    display_metadata_for_spatial_crop,
    movie_labels_from_display_layer,
)
from twopy.napari.display_paths import (
    format_output_folder,
    format_twopy_h5_output,
    recording_display_summary,
)
from twopy.napari.interactive import LiveResponseController
from twopy.napari.loading import resolve_or_convert_recording
from twopy.napari.movie import resolve_movie_frame_range
from twopy.napari.paths import resolve_launch_recording_path, resolve_recording_paths
from twopy.napari.plotting.data import (
    EpochResponsePlotData,
    ResponsePlotData,
    filter_response_plot_data_rois,
    load_response_plot_data,
    response_plot_data_from_grouped,
)
from twopy.napari.plotting.dff_options import DeltaFOverFOptionsWidget
from twopy.napari.plotting.docks import create_response_plot_widget
from twopy.napari.plotting.export import (
    draw_roi_contours,
    export_epoch_plots,
    labels_for_recording_image,
    roi_boundary_segments,
)
from twopy.napari.plotting.form_controls import (
    PLOT_CONTROL_WIDTH,
    PLOT_DROPDOWN_WIDTH,
)
from twopy.napari.plotting.label_visibility import apply_roi_visibility_to_labels_layer
from twopy.napari.plotting.options import (
    plot_display_options_group,
    visibility_options_widget,
)
from twopy.napari.plotting.panels import epoch_plot_panel
from twopy.napari.plotting.processing_options import ResponseProcessingOptionsWidget
from twopy.napari.plotting.response_window_options import ResponseWindowOptionsWidget
from twopy.napari.plotting.widgets import (
    EpochPlotWidget,
    global_time_bounds,
    global_value_bounds,
    roi_colors_from_layer,
)
from twopy.napari.responses import compute_response_plot_data_from_roi_set
from twopy.napari.roi import remove_roi_label_values_from_layer
from twopy.napari.sidebar import TWOPY_SIDEBAR_MINIMUM_WIDTH
from twopy.napari.state import write_last_recording_folder
from twopy.napari.text import counted_noun
from twopy.napari.viewer import (
    APPLICATION_TITLE,
    create_viewer,
    interior_random_frame_indices,
)
from twopy.spatial import SpatialCrop


def _load_recording_widget(panel: object) -> "_RecordingLoadWidget":
    """Return the callable magicgui loader from the compact Load panel.

    Args:
        panel: Load panel returned by ``add_twopy_magicgui_controls``.

    Returns:
        Magicgui callable used by tests to simulate user loading actions.
    """
    return cast(_RecordingLoadWidget, cast(_LoadRecordingPanel, panel).load_recording)


class _LineEditControl(Protocol):
    """Small protocol for a magicgui line-edit child used in tests."""

    value: object


class _RecordingFolderControl(Protocol):
    """Small protocol for the recording-folder picker used in tests."""

    value: object
    line_edit: _LineEditControl


class _RecordingLoadWidget(Protocol):
    """Small protocol for the magicgui recording-load callable in tests."""

    recording_folder: _RecordingFolderControl
    roi_file_to_load: _RecordingFolderControl

    def __call__(self, **kwargs: object) -> object:
        """Call the recording loader with optional magicgui keyword values."""
        ...


class _LoadRecordingPanel(Protocol):
    """Small protocol for the compact load-recording panel in tests.

    Inputs: load-recording panel.
    Outputs: callable magicgui function widget.
    """

    load_recording: Callable[..., object]


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


@dataclass
class _FakeResizeCall:
    """Dock resize request recorded by the fake Qt window.

    Inputs: dock widgets, target sizes, and orientation.
    Outputs: inspectable resize request for dock-layout tests.
    """

    docks: list[object]
    sizes: list[int]
    orientation: object


class _FakeQtWindow:
    """Small Qt-window-shaped object for dock resize tests."""

    resize_calls: list[_FakeResizeCall]

    def __init__(self) -> None:
        """Create an empty fake Qt window.

        Inputs: none.
        Outputs: fake window with no resize calls.
        """
        self.resize_calls: list[_FakeResizeCall] = []

    def resizeDocks(
        self,
        docks: list[object],
        sizes: list[int],
        orientation: object,
    ) -> None:
        """Record a Qt dock resize request.

        Args:
            docks: Dock widgets to resize.
            sizes: Target sizes.
            orientation: Resize orientation.

        Returns:
            None.
        """
        self.resize_calls.append(
            _FakeResizeCall(docks=docks, sizes=sizes, orientation=orientation)
        )


class _FakeWindow:
    """Small napari-shaped window for dock-widget tests."""

    dock_widgets: list[_FakeDockWidget]
    _qt_window: _FakeQtWindow

    def __init__(self) -> None:
        """Create an empty fake window.

        Inputs: none.
        Outputs: window with no dock widgets.
        """
        self.dock_widgets: list[_FakeDockWidget] = []
        self._qt_window = _FakeQtWindow()

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


class _ProcessingEchoPlotReceiver(_FakePlotReceiver):
    """Receiver that mimics plot widgets loading result processing options."""

    def __init__(self) -> None:
        """Create a receiver whose controller is attached by each test."""
        super().__init__()
        self.controller: LiveResponseController | None = None

    def set_response_plot_data(
        self,
        plot_data: ResponsePlotData,
        *,
        reset_axes: bool,
    ) -> None:
        """Store plot data and echo its processing options into the controller."""
        super().set_response_plot_data(plot_data, reset_axes=reset_axes)
        if (
            self.controller is not None
            and plot_data.response_processing_options is not None
        ):
            self.controller.set_response_processing_options(
                plot_data.response_processing_options,
            )


def _wait_for_live_response_job(
    controller: LiveResponseController,
    *,
    timeout_seconds: float = 2.0,
) -> None:
    """Wait until one async live-response job is ready to collect.

    Args:
        controller: Live response controller with a submitted worker job.
        timeout_seconds: Maximum wait before failing the test.

    Returns:
        None.

    Raises:
        AssertionError: If no worker job finishes within the timeout.
    """
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        future = controller._future
        if future is not None and future.done():
            return
        sleep(0.01)
    raise AssertionError("live response worker did not finish")


def _record_loaded_paths(
    loaded_paths: list[tuple[Path, ...]],
    paths: tuple[Path, ...],
) -> ExperimentLoadResult:
    """Record database-search paths received by a test callback.

    Args:
        loaded_paths: Mutable list of callback payloads.
        paths: Paths sent by the dialog.

    Returns:
        Structured load result for the dialog.
    """
    loaded_paths.append(paths)
    return ExperimentLoadResult(loaded_count=len(paths))


def _write_database(path: Path) -> None:
    """Create the smallest SQLite DB needed by napari DB-search tests.

    Args:
        path: SQLite database path to create.

    Returns:
        None. The function writes one fly and one stimulus row.
    """
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE fly (
                flyId INTEGER PRIMARY KEY,
                genotype TEXT,
                cellType TEXT,
                fluorescentProtein TEXT,
                eye TEXT,
                surgeon TEXT
            )
            """,
        )
        connection.execute(
            """
            CREATE TABLE stimulusPresentation (
                stimulusPresentationId INTEGER PRIMARY KEY,
                fly INT,
                relativeDataPath TEXT,
                stimulusFunction TEXT,
                date TEXT,
                dataQuality INT
            )
            """,
        )
        connection.execute(
            """
            INSERT INTO fly
                (flyId, genotype, cellType, fluorescentProtein, eye, surgeon)
            VALUES
                (10923, 'gh146gal4', 'ALPN', 'g6f', 'left', 'Gustavo')
            """,
        )
        connection.execute(
            """
            INSERT INTO stimulusPresentation
                (
                    stimulusPresentationId,
                    fly,
                    relativeDataPath,
                    stimulusFunction,
                    date,
                    dataQuality
                )
            VALUES
                (
                    20005,
                    10923,
                    'genotype\\stimulus\\2023\\10_17\\10_02_49',
                    'combo_stim_singles',
                    '2023-10-17 10:03:14',
                    1
                )
            """,
        )
        connection.commit()
    finally:
        connection.close()


class NapariAdapterTest(unittest.TestCase):
    """Tests napari loading and label saving without starting a GUI."""

    def test_counted_noun_formats_status_text(self) -> None:
        """Confirm napari status text uses normal singular and plural words.

        Inputs: singular and plural counts.
        Outputs: phrases with full singular or plural nouns.
        """
        self.assertEqual(counted_noun(1, "file"), "1 file")
        self.assertEqual(counted_noun(2, "file"), "2 files")
        self.assertEqual(counted_noun(1, "ROI", "ROIs"), "1 ROI")
        self.assertEqual(counted_noun(2, "ROI", "ROIs"), "2 ROIs")

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

    def test_create_viewer_uses_twopy_version_window_title(self) -> None:
        """Confirm the launcher brands the top-level napari window.

        Inputs: patched napari Viewer constructor.
        Outputs: viewer construction receives the twopy name and version.
        """
        viewer = _FakeViewer()

        with patch("napari.Viewer", return_value=viewer) as viewer_constructor:
            created = create_viewer()

        self.assertIs(created, viewer)
        self.assertEqual(APPLICATION_TITLE, f"twopy {__version__}")
        viewer_constructor.assert_called_once_with(title=APPLICATION_TITLE)

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
            self.assertIsNotNone(opened.load_widget)
            self.assertIsNotNone(opened.loaded_recordings_widget)
            self.assertIsNotNone(opened.twopy_sidebar_widget)
            self.assertIsNotNone(opened.twopy_sidebar_dock_widget)
            self.assertIsNotNone(opened.response_plot_widget)
            self.assertIsNotNone(opened.response_plot_dock_widget)
            self.assertIsNotNone(opened.response_options_widget)
            self.assertEqual(len(viewer.labels), 1)
            self.assertEqual(viewer.labels[0].options["opacity"], 0.5)
            self.assertEqual(viewer.labels[0].options["blending"], "additive")
            self.assertEqual(viewer.labels[0].brush_size, 6)
            self.assertEqual(viewer.images[0].options["opacity"], 0.5)
            self.assertEqual(viewer.images[0].options["gamma"], 1.3)
            self.assertEqual(viewer.images[0].options["contrast_limits"], (4.3, 7.0))
            self.assertEqual(viewer.images[0].contrast_limits_range, (4.0, 7.0))
            self.assertEqual(len(viewer.window.dock_widgets), 2)
            self.assertEqual(viewer.window.dock_widgets[0].name, "twopy responses")
            self.assertEqual(viewer.window.dock_widgets[0].area, "top")
            self.assertEqual(viewer.window._qt_window.resize_calls[0].sizes, [345])
            self.assertEqual(
                viewer.window._qt_window.resize_calls[0].docks,
                [viewer.window.dock_widgets[0]],
            )
            self.assertEqual(viewer.window.dock_widgets[1].name, "twopy")
            self.assertEqual(viewer.window.dock_widgets[1].area, "right")
            load_panel = cast(QWidget, opened.load_widget)
            load_labels = {label.text() for label in load_panel.findChildren(QLabel)}
            self.assertIn("Recording", load_labels)
            self.assertIn("ROI file", load_labels)
            browse_buttons = [
                button
                for button in load_panel.findChildren(QPushButton)
                if button.text() == "Browse"
            ]
            self.assertEqual(len(browse_buttons), 2)
            sidebar_tabs = cast(QTabWidget, opened.twopy_sidebar_widget)
            self.assertIs(sidebar_tabs, opened.response_options_widget)
            self.assertEqual(
                sidebar_tabs.minimumWidth(),
                TWOPY_SIDEBAR_MINIMUM_WIDTH,
            )
            self.assertEqual(sidebar_tabs.currentIndex(), 0)
            options_widget = cast(QTabWidget, opened.response_options_widget)
            self.assertEqual(
                tuple(
                    options_widget.tabText(index)
                    for index in range(options_widget.count())
                ),
                ("Load", "Update", "Plot", "ROIs", "Epochs", "Export"),
            )
            self.assertIsInstance(options_widget.widget(1), QScrollArea)
            update_buttons = {
                button.text() for button in options_widget.findChildren(QPushButton)
            }
            self.assertIn("Search Database", update_buttons)
            self.assertIn("Save ROIs + analysis", update_buttons)
            self.assertIn("Recompute preview now", update_buttons)
            self.assertIn("Reload saved analysis", update_buttons)
            group_titles = {
                group.title() for group in options_widget.findChildren(QGroupBox)
            }
            self.assertIn("Plot", group_titles)
            self.assertIn("Smoothing", group_titles)
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
                    labels=("roi_0003", "roi_0004"),
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
                np.array([0, 3, 4]),
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
            self.assertFalse((root / "response_summary_trials.csv").exists())
            self.assertFalse((root / "response_summary_grouped.csv").exists())
            self.assertFalse((root / "exports" / "csvs").exists())
            self.assertIsNotNone(response_widget._plot_data)

    def test_live_response_plot_data_preserves_gray_post_window(self) -> None:
        """Confirm live Plot-tab recompute keeps post-stimulus plot context.

        Inputs: converted recording with a gray epoch and patched analysis
        computation.
        Outputs: the live plot path requests the same two-second post window
        used when saved analysis outputs are loaded for plotting.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(
                _write_converted_recording(
                    root,
                    stimulus_parameters_json=(
                        '[{"epochName": "Gray"}, {"epochName": "Odor A"}]'
                    ),
                ),
            )
            roi_set = make_roi_set(
                np.array([[[True, False], [False, False]]]),
                labels=("roi_1",),
            )
            computation = SimpleNamespace(
                grouped_responses=_tiny_grouped_responses(),
                response_processing_options=ResponseProcessingOptions(),
            )

            with patch(
                "twopy.napari.responses.compute_recording_responses",
                return_value=computation,
            ) as compute:
                plot_data = compute_response_plot_data_from_roi_set(recording, roi_set)

            self.assertIsNotNone(plot_data)
            self.assertEqual(
                compute.call_args.kwargs["response_pre_window_seconds"],
                2.0,
            )
            self.assertEqual(
                compute.call_args.kwargs["response_post_window_seconds"],
                2.0,
            )

    def test_live_response_plot_data_omits_post_window_without_gray(self) -> None:
        """Confirm non-interleave recordings keep epoch-bounded live plots.

        Inputs: converted recording without gray/grey/interleave epoch names.
        Outputs: the live plot path requests no post-epoch plotting context.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(
                _write_converted_recording(
                    root,
                    stimulus_parameters_json=(
                        '[{"epochName": "Odor A"}, {"epochName": "Odor B"}]'
                    ),
                ),
            )
            roi_set = make_roi_set(
                np.array([[[True, False], [False, False]]]),
                labels=("roi_1",),
            )
            computation = SimpleNamespace(
                grouped_responses=_tiny_grouped_responses(),
                response_processing_options=ResponseProcessingOptions(),
            )

            with patch(
                "twopy.napari.responses.compute_recording_responses",
                return_value=computation,
            ) as compute:
                plot_data = compute_response_plot_data_from_roi_set(recording, roi_set)

            self.assertIsNotNone(plot_data)
            self.assertEqual(
                compute.call_args.kwargs["response_pre_window_seconds"],
                2.0,
            )
            self.assertEqual(
                compute.call_args.kwargs["response_post_window_seconds"],
                0.0,
            )

    def test_live_response_plot_data_uses_manual_response_window(self) -> None:
        """Confirm manual Plot-tab response windows reach live analysis.

        Inputs: converted recording, manual response-window options, and
        patched analysis computation.
        Outputs: live analysis receives the requested pre/post seconds.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(_write_converted_recording(root))
            roi_set = make_roi_set(
                np.array([[[True, False], [False, False]]]),
                labels=("roi_1",),
            )
            computation = SimpleNamespace(
                grouped_responses=_tiny_grouped_responses(),
                response_processing_options=ResponseProcessingOptions(),
            )

            with patch(
                "twopy.napari.responses.compute_recording_responses",
                return_value=computation,
            ) as compute:
                plot_data = compute_response_plot_data_from_roi_set(
                    recording,
                    roi_set,
                    response_window_options=ResponseWindowOptions(
                        auto=False,
                        pre_window_seconds=0.5,
                        post_window_seconds=1.5,
                    ),
                )

            self.assertIsNotNone(plot_data)
            self.assertEqual(
                compute.call_args.kwargs["response_pre_window_seconds"],
                0.5,
            )
            self.assertEqual(
                compute.call_args.kwargs["response_post_window_seconds"],
                1.5,
            )

    def test_save_analysis_button_writes_roi_and_analysis_outputs(self) -> None:
        """Confirm the Update tab can persist current ROI analysis.

        Inputs: edited Labels layer and a patched analysis workflow.
        Outputs: ROI HDF5 file written beside the recording without replacing
        the current in-memory plot preview.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            analysis_path = root / "analysis_outputs.h5"
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            viewer.labels[0].data = np.array([[1, 0], [0, 0]], dtype=np.int64)
            response_widget = cast(Any, opened.response_plot_widget)
            preview_plot_data = _tiny_response_plot_data()
            response_widget._plot_data = preview_plot_data
            response_widget._set_response_window_options(
                ResponseWindowOptions(
                    auto=False,
                    pre_window_seconds=0.5,
                    post_window_seconds=1.5,
                ),
            )

            with patch(
                (
                    "twopy.napari.plotting.docks.save_actions."
                    "analyze_recording_responses"
                ),
                return_value=SimpleNamespace(
                    output_path=analysis_path,
                    grouped_responses=_tiny_grouped_responses(),
                    response_summary_trials_csv_path=None,
                    response_summary_grouped_csv_path=None,
                ),
            ) as analyze:
                response_widget.save_analysis_and_rois()

            roi_path = root / "rois.h5"
            self.assertTrue(roi_path.is_file())
            analyze.assert_called_once()
            _, roi_set = analyze.call_args.args
            self.assertEqual(roi_set.labels, ("roi_0001",))
            self.assertEqual(analyze.call_args.kwargs["output_path"], analysis_path)
            self.assertEqual(
                analyze.call_args.kwargs["background_method"],
                "movie_global_percentile",
            )
            self.assertEqual(
                analyze.call_args.kwargs["baseline_sample_seconds"],
                1.0,
            )
            self.assertIsNone(analyze.call_args.kwargs["baseline_epoch_name"])
            self.assertEqual(
                analyze.call_args.kwargs["fit_mode"],
                "direct_bounded_tau",
            )
            self.assertTrue(analyze.call_args.kwargs["apply_motion_mask"])
            self.assertEqual(
                analyze.call_args.kwargs["response_pre_window_seconds"],
                0.5,
            )
            self.assertEqual(
                analyze.call_args.kwargs["response_post_window_seconds"],
                1.5,
            )
            self.assertFalse(analyze.call_args.kwargs["response_window_auto"])
            self.assertIsInstance(
                analyze.call_args.kwargs["response_processing_options"],
                ResponseProcessingOptions,
            )
            self.assertIs(response_widget._plot_data, preview_plot_data)

            recording_summary_text = response_widget._recording_summary_label.text()
            self.assertEqual(len(recording_summary_text.split("\n\n")), 4)
            labels_text = "\n".join(
                (
                    response_widget._analysis_path_label.text(),
                    response_widget._roi_save_path_label.text(),
                    response_widget._update_status_label.text(),
                ),
            )
            self.assertIn("Analysis output: ./twopy/analysis_outputs.h5", labels_text)
            self.assertIn("ROI output: ./twopy/rois.h5", labels_text)
            self.assertIn("Saved 1 ROI to ./twopy", labels_text)
            response_widget.shutdown()

    def test_save_analysis_button_syncs_cached_outputs_in_background(self) -> None:
        """Confirm cached saves publish changed outputs without blocking save.

        Inputs: local cached recording with a source publish destination.
        Outputs: background sync copies ROI and analysis files to publish path.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            local_dir = root / "cache" / "fly" / "stim" / "2023" / "10_17"
            publish_dir = source_dir / "twopy"
            local_dir.mkdir(parents=True)
            _write_source_recording_shape(source_dir)
            recording_path = _write_converted_recording(
                local_dir,
                source_session_dir=source_dir,
            )
            (root / "config.yml").write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {root / 'cache'}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            viewer = _FakeViewer()
            original_cwd = Path.cwd()
            response_widget: Any | None = None
            try:
                chdir(root)
                opened = open_recording_in_napari(recording_path, viewer=viewer)
                viewer.labels[0].data = np.array([[1, 0], [0, 0]], dtype=np.int64)
                response_widget = cast(Any, opened.response_plot_widget)
                analysis_path = local_dir / "analysis_outputs.h5"

                def fake_analyze(*_args: object, **_kwargs: object) -> object:
                    analysis_path.write_text("analysis", encoding="utf-8")
                    return SimpleNamespace(
                        output_path=analysis_path,
                        grouped_responses=_tiny_grouped_responses(),
                        response_summary_trials_csv_path=None,
                        response_summary_grouped_csv_path=None,
                    )

                with patch(
                    (
                        "twopy.napari.plotting.docks.save_actions."
                        "analyze_recording_responses"
                    ),
                    side_effect=fake_analyze,
                ):
                    response_widget.save_analysis_and_rois()

                future = response_widget._sync_future
                self.assertIsNotNone(future)
                future.result(timeout=2)
                response_widget._collect_finished_sync()
            finally:
                chdir(original_cwd)
                if response_widget is not None:
                    response_widget.shutdown()

            self.assertTrue((publish_dir / "rois.h5").is_file())
            self.assertEqual(
                (publish_dir / "analysis_outputs.h5").read_text(encoding="utf-8"),
                "analysis",
            )

    def test_processing_option_changes_request_preview_update(self) -> None:
        """Confirm processing controls trigger debounced preview recompute.

        Inputs: loaded recording, active ROI Labels layer, and a processing
        settings change.
        Outputs: the live response controller receives an update request.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            requests: list[str] = []
            response_widget._live_controller.request_update = lambda: requests.append(
                "requested"
            )

            response_widget._set_response_processing_options(
                ResponseProcessingOptions(),
            )

            self.assertEqual(requests, ["requested"])

    def test_delta_f_over_f_option_changes_request_preview_update(self) -> None:
        """Confirm dF/F controls trigger debounced preview recompute.

        Inputs: loaded recording, active ROI Labels layer, and a dF/F settings
        change.
        Outputs: the live response controller receives an update request.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(root)
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            requests: list[str] = []
            response_widget._live_controller.request_update = lambda: requests.append(
                "requested"
            )

            response_widget._set_delta_f_over_f_options(
                DeltaFOverFOptions(
                    background_method="none",
                    baseline_sample_seconds=None,
                    fit_mode="direct_bounded_tau_and_log_amplitude",
                    apply_motion_mask=False,
                ),
            )

            self.assertEqual(requests, ["requested"])

    def test_plot_tab_shows_response_window_before_delta_f_over_f(self) -> None:
        """Confirm the Plot tab presents response window before dF/F.

        Inputs: a newly created response widget.
        Outputs: the Plot tab orders Plot, Response window, dF/F, then
        processing controls.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))

        self.assertIsInstance(
            response_widget._response_window_options_widget,
            ResponseWindowOptionsWidget,
        )
        self.assertIsInstance(
            response_widget._delta_f_over_f_options_widget,
            DeltaFOverFOptionsWidget,
        )
        self.assertLess(
            response_widget._plot_options_layout.indexOf(
                response_widget._response_window_options_widget,
            ),
            response_widget._plot_options_layout.indexOf(
                response_widget._delta_f_over_f_options_widget,
            ),
        )
        self.assertLess(
            response_widget._plot_options_layout.indexOf(
                response_widget._delta_f_over_f_options_widget,
            ),
            response_widget._plot_options_layout.indexOf(
                response_widget._processing_options_widget,
            ),
        )

    def test_response_window_options_auto_default_and_manual_bounds(self) -> None:
        """Confirm response-window controls default to auto and cap manual values.

        Inputs: response-window widget with a two-second interleave limit.
        Outputs: auto starts checked, manual fields are disabled, and manual
        spin boxes cap at the supplied limit when auto is unchecked.
        """
        _ = QApplication.instance() or QApplication([])
        widget = cast(
            Any,
            ResponseWindowOptionsWidget(
                ResponseWindowOptions(),
                max_window_seconds=2.0,
            ),
        )

        self.assertTrue(widget._auto.isChecked())
        self.assertFalse(widget._pre_seconds.isEnabled())
        self.assertFalse(widget._post_seconds.isEnabled())
        self.assertEqual(widget._pre_seconds.maximum(), 2.0)
        self.assertEqual(widget._post_seconds.maximum(), 2.0)

        widget._auto.setChecked(False)

        self.assertTrue(widget._pre_seconds.isEnabled())
        self.assertTrue(widget._post_seconds.isEnabled())

    def test_delta_f_over_f_controls_use_readable_menu_labels(self) -> None:
        """Confirm Plot-tab menus show readable labels, not stored values.

        Inputs: dF/F and processing option widgets with default settings.
        Outputs: dropdowns hide internal option names.
        """
        _ = QApplication.instance() or QApplication([])
        dff_widget = cast(Any, DeltaFOverFOptionsWidget(DeltaFOverFOptions()))
        processing_widget = cast(
            Any,
            ResponseProcessingOptionsWidget(ResponseProcessingOptions()),
        )

        background_labels = tuple(
            dff_widget._background_method.itemText(index)
            for index in range(dff_widget._background_method.count())
        )
        fit_labels = tuple(
            dff_widget._fit_mode.itemText(index)
            for index in range(dff_widget._fit_mode.count())
        )
        smoothing_labels = tuple(
            processing_widget._smoothing_method.itemText(index)
            for index in range(processing_widget._smoothing_method.count())
        )
        correlation_labels = tuple(
            processing_widget._correlation_reference.itemText(index)
            for index in range(processing_widget._correlation_reference.count())
        )

        self.assertIn("global percentile", background_labels)
        self.assertIn("shared y-stripe P%", background_labels)
        self.assertIn("ROI y-stripe P%", background_labels)
        self.assertNotIn("roi_y_stripe_percentile", background_labels)
        rich_label_role = int(Qt.ItemDataRole.UserRole) + 1
        self.assertEqual(
            dff_widget._background_method.itemData(2, rich_label_role),
            "shared y-stripe P<sub>%</sub>",
        )
        self.assertEqual(
            dff_widget._background_method.itemData(3, rich_label_role),
            "ROI y-stripe P<sub>%</sub>",
        )
        self.assertIn("bounded tau", fit_labels)
        self.assertIn("bounded tau/amplitude", fit_labels)
        self.assertIn("log-linear", fit_labels)
        self.assertNotIn("direct_bounded_tau", fit_labels)
        self.assertIn("moving average", smoothing_labels)
        self.assertNotIn("moving_average", smoothing_labels)
        self.assertIn("epoch mean", correlation_labels)
        self.assertNotIn("epoch_mean", correlation_labels)

    def test_plot_display_size_label_preserves_ui_text(self) -> None:
        """Confirm the Plot subsection puts show SEM before size.

        Inputs: plot display options group with default numeric bounds.
        Outputs: first field row is the SEM toggle, then the configured size label.
        """
        _ = QApplication.instance() or QApplication([])
        group = plot_display_options_group(
            show_sem_checkbox=QCheckBox("show SEM"),
            plot_size_spin=QSpinBox(),
            x_min=0.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            on_change=lambda **_kwargs: None,
        )
        layout = cast(QFormLayout, group.layout())
        sem_item = layout.itemAt(0, QFormLayout.ItemRole.FieldRole)
        if sem_item is None:
            self.fail("Plot display show SEM checkbox is missing")
        sem_widget = sem_item.widget()
        if not isinstance(sem_widget, QCheckBox):
            self.fail("Plot display show SEM row is not a QCheckBox")
        self.assertEqual(sem_widget.text(), "show SEM")
        self.assertEqual(sem_widget.minimumWidth(), PLOT_DROPDOWN_WIDTH)
        self.assertEqual(sem_widget.maximumWidth(), PLOT_DROPDOWN_WIDTH)

        label_item = layout.itemAt(1, QFormLayout.ItemRole.LabelRole)
        if label_item is None:
            self.fail("Plot display size label is missing")
        label_widget = label_item.widget()
        if not isinstance(label_widget, QLabel):
            self.fail("Plot display size label is not a QLabel")

        self.assertEqual(label_widget.text(), "Size")
        self.assertTrue(layout.formAlignment() & Qt.AlignmentFlag.AlignHCenter)

    def test_plot_display_numeric_controls_share_width_and_rounding(self) -> None:
        """Confirm Plot numeric controls use one width and two-decimal bounds.

        Inputs: plot display options group with fractional axis bounds.
        Outputs: size and axis limit controls share width, and limits round to
        two decimal places.
        """
        _ = QApplication.instance() or QApplication([])
        group = plot_display_options_group(
            show_sem_checkbox=QCheckBox("show SEM"),
            plot_size_spin=QSpinBox(),
            x_min=1.123,
            x_max=1.456,
            y_min=-1.123,
            y_max=-1.456,
            on_change=lambda **_kwargs: None,
        )
        layout = cast(QFormLayout, group.layout())
        size_item = layout.itemAt(1, QFormLayout.ItemRole.FieldRole)
        if size_item is None:
            self.fail("Plot display size control is missing")
        size_spin = size_item.widget()
        if not isinstance(size_spin, QSpinBox):
            self.fail("Plot display size control is not a QSpinBox")

        axis_spins: list[QDoubleSpinBox] = []
        for row in range(2, 6):
            axis_item = layout.itemAt(row, QFormLayout.ItemRole.FieldRole)
            if axis_item is None:
                self.fail("Plot display axis control is missing")
            axis_spin = axis_item.widget()
            if not isinstance(axis_spin, QDoubleSpinBox):
                self.fail("Plot display axis control is not a QDoubleSpinBox")
            axis_spins.append(axis_spin)

        widths = {
            size_spin.minimumWidth(),
            *(spin.minimumWidth() for spin in axis_spins),
        }
        max_widths = {
            size_spin.maximumWidth(),
            *(spin.maximumWidth() for spin in axis_spins),
        }
        self.assertEqual(len(widths), 1)
        self.assertEqual(len(max_widths), 1)
        self.assertEqual([spin.decimals() for spin in axis_spins], [2, 2, 2, 2])
        self.assertEqual(
            [spin.value() for spin in axis_spins],
            [1.12, 1.46, -1.12, -1.46],
        )

    def test_plot_tab_option_controls_use_shared_widths(self) -> None:
        """Confirm Plot-tab fields use shared widths by control type.

        Inputs: response-window, dF/F, and processing option widgets.
        Outputs: dropdowns and checkboxes use the wider width, while spin boxes
        use the compact control width.
        """
        _ = QApplication.instance() or QApplication([])
        response_window_widget = cast(
            Any,
            ResponseWindowOptionsWidget(ResponseWindowOptions()),
        )
        dff_widget = cast(Any, DeltaFOverFOptionsWidget(DeltaFOverFOptions()))
        processing_widget = cast(
            Any,
            ResponseProcessingOptionsWidget(ResponseProcessingOptions()),
        )
        for widget in (response_window_widget, dff_widget, processing_widget):
            widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            widget.ensurePolished()
            widget.adjustSize()
            widget.show()
        QApplication.processEvents()

        wide_controls: list[QWidget] = [
            response_window_widget._auto,
            dff_widget._background_method,
            dff_widget._baseline_epoch,
            dff_widget._use_full_baseline,
            dff_widget._fit_mode,
            dff_widget._apply_motion_mask,
            processing_widget._smoothing_method,
            processing_widget._low_pass_method,
            processing_widget._correlation_reference,
            processing_widget._correlation_window_has_stop,
        ]
        dropdowns: list[QComboBox] = [
            dff_widget._background_method,
            dff_widget._baseline_epoch,
            dff_widget._fit_mode,
            processing_widget._smoothing_method,
            processing_widget._low_pass_method,
            processing_widget._correlation_reference,
        ]
        compact_controls: list[QWidget] = [
            response_window_widget._pre_seconds,
            response_window_widget._post_seconds,
            dff_widget._baseline_seconds,
            processing_widget._smoothing_window_frames,
            processing_widget._smoothing_polynomial_order,
            processing_widget._low_pass_cutoff_hz,
            processing_widget._low_pass_order,
            processing_widget._minimum_correlation,
            processing_widget._correlation_window_start,
            processing_widget._correlation_window_stop,
        ]

        self.assertTrue(
            all(
                isinstance(
                    control,
                    QSpinBox | QDoubleSpinBox,
                )
                for control in compact_controls
            ),
        )
        self.assertEqual(
            {control.width() for control in compact_controls},
            {PLOT_CONTROL_WIDTH},
        )
        self.assertEqual(
            {control.minimumWidth() for control in wide_controls},
            {PLOT_DROPDOWN_WIDTH},
        )
        self.assertEqual(
            {control.maximumWidth() for control in wide_controls},
            {PLOT_DROPDOWN_WIDTH},
        )
        self.assertEqual(
            {control.width() for control in wide_controls},
            {PLOT_DROPDOWN_WIDTH},
        )
        popup_view_widths: set[int] = set()
        popup_window_widths: set[int] = set()
        for control in dropdowns:
            view = control.view()
            self.assertIsNotNone(view)
            if view is None:
                continue
            popup_view_widths.add(view.width())
            popup = view.window()
            self.assertIsNotNone(popup)
            if popup is not None:
                popup_window_widths.add(popup.width())
        self.assertEqual(popup_view_widths, {PLOT_DROPDOWN_WIDTH})
        self.assertEqual(popup_window_widths, {PLOT_DROPDOWN_WIDTH})

    def test_plot_tab_decimal_controls_show_two_places(self) -> None:
        """Confirm Plot-tab dF/F and processing decimals show two places.

        Inputs: dF/F and processing option widgets.
        Outputs: every decimal spin box in those subsections has two decimals.
        """
        _ = QApplication.instance() or QApplication([])
        dff_widget = DeltaFOverFOptionsWidget(DeltaFOverFOptions())
        processing_widget = ResponseProcessingOptionsWidget(
            ResponseProcessingOptions(),
        )

        self.assertEqual(dff_widget._baseline_seconds.decimals(), 2)
        self.assertEqual(processing_widget._low_pass_cutoff_hz.decimals(), 2)
        self.assertEqual(processing_widget._minimum_correlation.decimals(), 2)
        self.assertEqual(processing_widget._correlation_window_start.decimals(), 2)
        self.assertEqual(processing_widget._correlation_window_stop.decimals(), 2)

    def test_baseline_epoch_dropdown_uses_recording_epoch_names(self) -> None:
        """Confirm baseline selection shows actual recording epoch names.

        Inputs: converted recording with stimulus parameter epoch names.
        Outputs: the dF/F baseline dropdown lists those names and stores the
        selected epoch selector.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                stimulus_parameters_json=(
                    '[{"epochName": "Gray Interleave"}, {"epochName": "Odor A"}]'
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            dff_widget = response_widget._delta_f_over_f_options_widget

            labels = tuple(
                dff_widget._baseline_epoch.itemText(index)
                for index in range(dff_widget._baseline_epoch.count())
            )
            dff_widget._baseline_epoch.setCurrentIndex(1)

            self.assertEqual(labels, ("1: Gray Interleave", "2: Odor A"))
            self.assertEqual(
                dff_widget.options().baseline_epoch_name,
                "Odor A",
            )

    def test_baseline_epoch_dropdown_defaults_to_gray_like_epoch(self) -> None:
        """Confirm the dF/F baseline defaults to the gray/interleave epoch.

        Inputs: converted recording where the gray-like epoch is not epoch one.
        Outputs: the baseline dropdown selects that named epoch by default.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                stimulus_parameters_json=(
                    '[{"epochName": "Odor A"}, {"epochName": "Grey screen"}]'
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            options = response_widget._delta_f_over_f_options_widget.options()

            self.assertEqual(options.baseline_epoch_number, 2)
            self.assertEqual(options.baseline_epoch_name, "Grey screen")

    def test_baseline_epoch_dropdown_accepts_interleave_without_gray(self) -> None:
        """Confirm interleave names are accepted even without gray spelling.

        Inputs: converted recording with an ``interleave`` baseline name.
        Outputs: the baseline dropdown selects that epoch by default.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording_path = _write_converted_recording(
                root,
                stimulus_parameters_json=(
                    '[{"epochName": "Odor A"}, {"epochName": "Baseline Interleave"}]'
                ),
            )
            viewer = _FakeViewer()
            opened = open_recording_in_napari(recording_path, viewer=viewer)
            response_widget = cast(Any, opened.response_plot_widget)
            options = response_widget._delta_f_over_f_options_widget.options()

            self.assertEqual(options.baseline_epoch_number, 2)
            self.assertEqual(options.baseline_epoch_name, "Baseline Interleave")

    def test_saved_processing_options_update_plot_tab_controls(self) -> None:
        """Confirm saved analysis settings hydrate Plot-tab controls.

        Inputs: plot data carrying persisted response-processing options.
        Outputs: response widget controls show the saved smoothing settings.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        options = ResponseProcessingOptions(
            smoothing=SmoothingOptions(
                method="savgol",
                window_frames=11,
                polynomial_order=3,
            ),
        )
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=_tiny_response_plot_data().epochs,
            response_processing_options=options,
        )

        response_widget.set_response_plot_data(plot_data, reset_axes=True)

        self.assertEqual(
            response_widget._processing_options_widget.options().smoothing,
            options.smoothing,
        )
        self.assertEqual(response_widget._response_processing_options, options)

    def test_saved_delta_f_over_f_options_update_plot_tab_controls(self) -> None:
        """Confirm saved dF/F settings hydrate Plot-tab controls.

        Inputs: plot data carrying persisted dF/F options.
        Outputs: response widget controls show the saved dF/F settings.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        options = DeltaFOverFOptions(
            baseline_epoch_number=3,
            baseline_epoch_name="Manual baseline",
            background_method="roi_y_stripe_percentile",
            baseline_sample_seconds=None,
            fit_mode="direct_bounded_tau_and_log_amplitude",
            apply_motion_mask=False,
        )
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=_tiny_response_plot_data().epochs,
            delta_f_over_f_options=options,
        )

        response_widget.set_response_plot_data(plot_data, reset_axes=True)

        self.assertEqual(
            response_widget._delta_f_over_f_options_widget.options(),
            options,
        )
        self.assertEqual(response_widget._delta_f_over_f_options, options)

    def test_savgol_window_control_steps_through_valid_values(self) -> None:
        """Confirm Savitzky-Golay window clicks stay on valid values.

        Inputs: processing controls configured for second-order Savitzky-Golay.
        Outputs: the window spinbox uses odd values above the polynomial order.
        """
        _ = QApplication.instance() or QApplication([])
        widget = cast(
            Any,
            ResponseProcessingOptionsWidget(
                ResponseProcessingOptions(
                    smoothing=SmoothingOptions(
                        method="savgol",
                        window_frames=7,
                        polynomial_order=2,
                    ),
                ),
            ),
        )
        window_spin_box = widget._smoothing_window_frames

        self.assertEqual(window_spin_box.minimum(), 3)
        self.assertEqual(window_spin_box.singleStep(), 2)

        window_spin_box.stepUp()
        self.assertEqual(window_spin_box.value(), 9)

        window_spin_box.stepDown()
        self.assertEqual(window_spin_box.value(), 7)

        window_spin_box.setValue(8)
        self.assertEqual(window_spin_box.value(), 9)

    def test_savgol_window_minimum_tracks_polynomial_order(self) -> None:
        """Confirm Savitzky-Golay window minimum follows polynomial order.

        Inputs: processing controls with a small Savitzky-Golay window.
        Outputs: increasing polynomial order raises the window to the next
        valid odd length.
        """
        _ = QApplication.instance() or QApplication([])
        widget = cast(
            Any,
            ResponseProcessingOptionsWidget(
                ResponseProcessingOptions(
                    smoothing=SmoothingOptions(
                        method="savgol",
                        window_frames=3,
                        polynomial_order=2,
                    ),
                ),
            ),
        )

        widget._smoothing_polynomial_order.setValue(4)

        self.assertEqual(widget._smoothing_window_frames.minimum(), 5)
        self.assertEqual(widget._smoothing_window_frames.value(), 5)
        self.assertEqual(widget.options().smoothing.window_frames, 5)

    def test_response_widget_close_shuts_down_live_controller(self) -> None:
        """Confirm closing the response widget releases live-update resources.

        Inputs: a response plot widget with its live controller patched.
        Outputs: Qt close handling calls the controller shutdown path.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        shutdowns: list[str] = []
        response_widget._live_controller.shutdown = lambda: shutdowns.append("shutdown")

        response_widget.closeEvent(QCloseEvent())

        self.assertEqual(shutdowns, ["shutdown"])
        self.assertTrue(response_widget._is_shutdown)

    def test_response_display_paths_use_recording_identity(self) -> None:
        """Confirm response widgets show compact recording and output paths.

        Inputs: converted recording whose source path follows the lab date
        layout.
        Outputs: parsed root, genotype, stimulus, recording time, and
        twopy-relative output paths.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = (
                root / "data" / "gh146" / "combo_stim" / "2025" / "12_21" / "17_42_22"
            )
            recording_path = _write_converted_recording(
                root,
                source_session_dir=source_dir,
            )
            recording = load_converted_recording(recording_path)

            summary = recording_display_summary(recording)

            self.assertEqual(summary.root, root / "data")
            self.assertEqual(summary.genotype, "gh146")
            self.assertEqual(summary.stimulus, "combo_stim")
            self.assertEqual(summary.recording, "2025-12-21 17:42:22")
            self.assertEqual(
                format_twopy_h5_output(recording.path.parent / "rois.h5"),
                "./twopy/rois.h5",
            )
            self.assertEqual(
                format_output_folder(
                    recording.path.parent / "exports" / "plots" / "plot.png",
                    recording,
                ),
                "./twopy/exports/plots",
            )

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

    def test_live_response_controller_defaults_to_short_update_delay(self) -> None:
        """Confirm live ROI updates use a short drawing debounce.

        Inputs: default live controller.
        Outputs: debounce and poll intervals stay low enough for responsive
        post-drawing plot updates.
        """
        _ = QApplication.instance() or QApplication([])
        receiver = _FakePlotReceiver()
        controller = LiveResponseController(receiver)

        self.assertEqual(controller._debounce_ms, 200)
        self.assertEqual(controller._poll_timer.interval(), 30)
        controller.shutdown()

    def test_live_response_controller_does_not_loop_on_echoed_options(self) -> None:
        """Confirm result option hydration does not schedule another recompute.

        Inputs: async live controller whose receiver echoes result processing
        options, matching the real response plot widget.
        Outputs: applying the finished result does not mark the job stale and
        start another background computation.
        """
        _ = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = load_converted_recording(_write_converted_recording(root))
            layer = _FakeLayer(
                name="rois",
                data=np.array([[1, 0], [0, 0]], dtype=np.int64),
                options={},
                events=_FakeLabelEvents(),
            )
            receiver = _ProcessingEchoPlotReceiver()
            controller = LiveResponseController(receiver, debounce_ms=0)
            receiver.controller = controller
            controller.set_context(recording, layer)
            plot_data = ResponsePlotData(
                source_path=None,
                epochs=_tiny_response_plot_data().epochs,
                response_processing_options=ResponseProcessingOptions(),
            )

            with patch(
                "twopy.napari.interactive.compute_response_plot_data_from_roi_set",
                return_value=plot_data,
            ) as compute:
                controller.request_update()
                _wait_for_live_response_job(controller)
                controller._collect_finished_job()

            self.assertIs(receiver.plot_data, plot_data)
            self.assertIsNone(controller._future)
            compute.assert_called_once()
            controller.shutdown()

    def test_live_response_controller_shutdown_disconnects_events(self) -> None:
        """Confirm shutdown breaks event callbacks and ignores later updates.

        Inputs: selected recording, fake Labels events, and a synchronous
        controller.
        Outputs: event emitters no longer hold callbacks or run analysis after
        shutdown.
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

            controller.shutdown()
            with patch(
                "twopy.napari.interactive.compute_response_plot_data_from_roi_set",
            ) as compute:
                events.paint.emit()
                controller.request_update()

            compute.assert_not_called()
            self.assertEqual(events.paint._callbacks, [])
            self.assertEqual(events.data._callbacks, [])

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
            load_widget = _load_recording_widget(control_docks.load_widget)

            load_widget.recording_folder.value = root

            self.assertEqual(len(viewer.images), 2)
            self.assertEqual(len(viewer.labels), 1)
            self.assertEqual(np.asarray(viewer.images[1].data).shape[0], 3)
            np.testing.assert_array_equal(
                roi_label_image_from_layer(viewer.labels[0]),
                np.zeros((2, 2), dtype=np.int64),
            )

    def test_database_search_dialog_loads_selected_hierarchy_paths(self) -> None:
        """Confirm DB search selections resolve to configured source paths.

        Inputs: a temporary config, Clark-lab-style DB, and dialog callback.
        Outputs: selecting a hierarchy node sends all descendant source paths to
        the loader callback.
        """
        _ = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_dir = root / "db"
            data_dir = root / "data"
            database_dir.mkdir()
            data_dir.mkdir()
            _write_database(database_dir / "experimentLog.db")
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {database_dir}\n"
                f"data_path: {data_dir}\n"
                "database_access: direct\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            loaded_paths: list[tuple[Path, ...]] = []
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: _record_loaded_paths(
                    loaded_paths,
                    paths,
                ),
                config_path=config_path,
            )

            dialog._user_filter.setText("Gus")
            dialog.search()

            self.assertIsNotNone(dialog.findChild(QSplitter))
            self.assertEqual(dialog._user_filter.placeholderText(), "gustavo")
            self.assertEqual(dialog._cell_type_filter.placeholderText(), "ALPN")
            self.assertEqual(dialog._sensor_filter.placeholderText(), "g6f")
            self.assertEqual(
                dialog._stimulus_filter.placeholderText(),
                "bars_singles_stim=5s_bars=3x400ms_dt=200ms_int=20",
            )
            self.assertEqual(dialog._date_filter.placeholderText(), "2025-10-22")
            self.assertEqual(dialog._tree.columnWidth(1), 72)
            self.assertEqual(dialog._user_filter.textMargins().left(), 6)
            self.assertEqual(dialog._user_filter.textMargins().right(), 6)
            self.assertFalse(dialog._user_filter.font().italic())
            self.assertTrue(dialog._cell_type_filter.font().italic())
            self.assertNotIn(
                "Typing...",
                {label.text() for label in dialog.findChildren(QLabel)},
            )
            dialog._cell_type_filter.setText("LN6")
            self.assertFalse(dialog._cell_type_filter.font().italic())
            dialog._cell_type_filter.clear()
            self.assertTrue(dialog._cell_type_filter.font().italic())
            bottom_buttons = [
                button.text()
                for button in dialog.findChildren(QPushButton)
                if button.text() in {"Load Selected", "Close"}
            ]
            self.assertEqual(bottom_buttons, ["Load Selected", "Close"])

            result = dialog._tree.topLevelItem(0)
            assert result is not None
            result.setSelected(True)
            dialog.load_selected()

            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
            self.assertEqual(
                loaded_paths,
                [(data_dir / "genotype" / "stimulus" / "2023" / "10_17" / "10_02_49",)],
            )

    def test_database_load_error_dialog_shows_scrollable_failed_paths(self) -> None:
        """Confirm load failures get a readable scrollable path list.

        Inputs: structured load result with two failed recording paths.
        Outputs: an error dialog with a clear summary and plain scrollable
        failure details.
        """
        _ = QApplication.instance() or QApplication([])
        result = ExperimentLoadResult(
            loaded_count=1,
            failures=(
                ExperimentLoadFailure(
                    path=Path("/missing/first"),
                    message="Could not find recording_data.h5.",
                ),
                ExperimentLoadFailure(
                    path=Path("/missing/second"),
                    message="Missing source files.",
                ),
            ),
        )

        dialog = ExperimentLoadErrorDialog(result)
        summary_text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        details = dialog.findChild(QPlainTextEdit)
        assert details is not None

        self.assertIn("Loaded 1 recordings", summary_text)
        self.assertIn("Failed to load 2 recordings", summary_text)
        self.assertTrue(details.isReadOnly())
        self.assertIn("/missing/first", details.toPlainText())
        self.assertIn("Could not find recording_data.h5.", details.toPlainText())
        self.assertIn("/missing/second", details.toPlainText())
        self.assertIn("Missing source files.", details.toPlainText())

    def test_database_search_failure_shows_error_dialog(self) -> None:
        """Confirm database search failures do not look like empty results.

        Inputs: dialog pointed at a missing config file.
        Outputs: search clears the tree and reports the actual error.
        """
        _ = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temp_dir:
            errors: list[Exception] = []
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: ExperimentLoadResult(
                    loaded_count=len(paths),
                ),
                config_path=Path(temp_dir) / "missing.yml",
            )

            with patch.object(
                dialog,
                "_show_search_error",
                side_effect=lambda error: errors.append(error),
            ):
                dialog.search()

            self.assertEqual(dialog._tree.topLevelItemCount(), 0)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], FileNotFoundError)

    def test_database_search_error_dialog_shows_details(self) -> None:
        """Confirm search errors get readable scrollable details.

        Inputs: a representative search exception.
        Outputs: a dialog with a clear summary and exception details.
        """
        _ = QApplication.instance() or QApplication([])
        dialog = ExperimentSearchErrorDialog(FileNotFoundError("missing config.yml"))
        summary_text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        details = dialog.findChild(QPlainTextEdit)
        assert details is not None

        self.assertIn("Could not search", summary_text)
        self.assertTrue(details.isReadOnly())
        self.assertIn("FileNotFoundError", details.toPlainText())
        self.assertIn("missing config.yml", details.toPlainText())

    def test_database_search_loads_valid_paths_when_one_row_is_bad(self) -> None:
        """Confirm one bad DB path does not block valid selected siblings.

        Inputs: one valid DB result and one unsafe relativeDataPath.
        Outputs: the valid source path is loaded and the bad row is reported.
        """
        _ = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_dir = root / "db"
            data_dir = root / "data"
            database_dir.mkdir()
            data_dir.mkdir()
            database_path = database_dir / "experimentLog.db"
            _write_database(database_path)
            connection = sqlite3.connect(database_path)
            try:
                connection.execute(
                    """
                    INSERT INTO stimulusPresentation
                        (
                            stimulusPresentationId,
                            fly,
                            relativeDataPath,
                            stimulusFunction,
                            date,
                            dataQuality
                        )
                    VALUES
                        (
                            20006,
                            10923,
                            '..\\escape\\2023\\10_17\\10_03_00',
                            'combo_stim_singles',
                            '2023-10-17 10:03:00',
                            1
                        )
                    """,
                )
                connection.commit()
            finally:
                connection.close()
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {database_dir}\n"
                f"data_path: {data_dir}\n"
                "database_access: direct\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            loaded_paths: list[tuple[Path, ...]] = []
            shown_results: list[ExperimentLoadResult] = []
            dialog = ExperimentSearchDialog(
                on_load_recording_paths=lambda paths: _record_loaded_paths(
                    loaded_paths,
                    paths,
                ),
                config_path=config_path,
            )
            dialog._user_filter.setText("Gus")
            dialog.search()

            result = dialog._tree.topLevelItem(0)
            assert result is not None
            result.setSelected(True)
            with patch.object(
                dialog,
                "_show_load_errors",
                side_effect=lambda load_result: shown_results.append(load_result),
            ):
                dialog.load_selected()

            self.assertEqual(
                loaded_paths,
                [(data_dir / "genotype" / "stimulus" / "2023" / "10_17" / "10_02_49",)],
            )
            self.assertEqual(len(shown_results), 1)
            self.assertEqual(shown_results[0].loaded_count, 1)
            self.assertEqual(len(shown_results[0].failures), 1)
            self.assertIn("relativeDataPath", shown_results[0].failures[0].message)

    def test_database_load_records_non_value_errors_per_path(self) -> None:
        """Confirm conversion KeyErrors become path-level load failures.

        Inputs: database load callback with a loader raising ``KeyError``.
        Outputs: structured failure result instead of an uncaught traceback.
        """
        state = napari_controls.NapariControlState(
            viewer=_FakeViewer(),
            roi_labels_layer=None,
            roi_save_file=Path("rois.h5"),
            recording=None,
            response_plot_widget=None,
        )

        with patch.object(
            napari_controls,
            "_load_recording_path",
            side_effect=KeyError("missing stimtype"),
        ):
            result = napari_controls._load_database_recording_paths(
                state,
                (Path("/bad/source"),),
            )

        self.assertEqual(result.loaded_count, 0)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].path, Path("/bad/source"))
        self.assertIn("KeyError", result.failures[0].message)
        self.assertFalse(state.is_loading)

    def test_loaded_recordings_panel_selects_and_unloads_layers(self) -> None:
        """Confirm multi-recording selection controls layer ownership.

        Inputs: two converted folders loaded into one fake viewer.
        Outputs: selected recording controls layer visibility, and unload
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
            load_widget = _load_recording_widget(control_docks.load_widget)
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
            self.assertIn(first.name, str(load_widget.recording_folder.line_edit.value))
            self.assertTrue(viewer.images[0].visible)
            self.assertTrue(viewer.images[1].visible)
            self.assertTrue(viewer.labels[0].visible)
            self.assertFalse(viewer.images[2].visible)
            self.assertFalse(viewer.images[3].visible)
            self.assertFalse(viewer.labels[1].visible)

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
            self.assertIn(
                second.name, str(load_widget.recording_folder.line_edit.value)
            )

            unload_button.click()

            self.assertEqual(loaded_list.count(), 0)
            self.assertEqual(load_widget.recording_folder.value, Path("default"))

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
            load_widget = _load_recording_widget(control_docks.load_widget)

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
            load_widget = _load_recording_widget(control_docks.load_widget)

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
            load_widget = _load_recording_widget(control_docks.load_widget)

            load_widget(recording_folder=root)

            display_text = str(load_widget.recording_folder.line_edit.value)
            self.assertTrue(display_text.startswith("..."))
            self.assertTrue(display_text.endswith("/2025/10_03/twopy/"))

    def test_roi_file_change_reuses_resolved_recording_path(self) -> None:
        """Confirm ROI file changes work after the path field is shortened.

        Inputs: long converted output path and ROI file loaded through the
            control widget.
        Outputs: changing the ROI file reloads without resolving the
            shortened display text as a filesystem path.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / ("long_" * 20) / "twopy"
            root.mkdir(parents=True)
            _write_converted_recording(root)
            roi_path = root / "alternate_rois.h5"
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
            load_widget = _load_recording_widget(control_docks.load_widget)

            load_widget(recording_folder=root)
            load_widget.roi_file_to_load.value = roi_path

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

    def test_recording_path_resolution_uses_local_analysis_cache(self) -> None:
        """Confirm source loading converts into local cache when enabled.

        Inputs: a source-shaped recording under configured ``data_path``.
        Outputs: converted paths under ``analysis_cache_dir``.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            cache_root = root / "cache"
            _write_source_recording_shape(source_dir)
            (root / "config.yml").write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {cache_root}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            original_cwd = Path.cwd()
            try:
                chdir(root)
                with patch(
                    "twopy.napari.loading.convert_recording_to_twopy",
                    side_effect=_fake_convert_recording,
                ) as convert:
                    resolved = resolve_or_convert_recording(source_dir)
            finally:
                chdir(original_cwd)

            expected_dir = cache_root / "fly" / "stim" / "2023" / "10_17"
            self.assertTrue(resolved.was_converted)
            convert.assert_called_once_with(
                source_dir.resolve(), output_dir=expected_dir
            )
            self.assertEqual(
                resolved.paths.recording_data_path,
                (expected_dir / "recording_data.h5").resolve(),
            )
            self.assertEqual(
                resolved.paths.movie_path,
                (expected_dir / "aligned_movie.h5").resolve(),
            )

    def test_recording_path_resolution_pulls_published_rois_into_cache(self) -> None:
        """Confirm cached source loads reuse published ROI files locally.

        Inputs: source recording, cache config, and a published ``rois.h5``.
        Outputs: resolved ROI file path points at the local cache copy.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            cache_root = root / "cache"
            publish_dir = source_dir / "twopy"
            _write_source_recording_shape(source_dir)
            publish_dir.mkdir()
            (publish_dir / "rois.h5").write_text("published", encoding="utf-8")
            (root / "config.yml").write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {cache_root}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )
            original_cwd = Path.cwd()
            try:
                chdir(root)
                with patch(
                    "twopy.napari.loading.convert_recording_to_twopy",
                    side_effect=_fake_convert_recording,
                ):
                    resolved = resolve_or_convert_recording(source_dir)
            finally:
                chdir(original_cwd)

            expected_roi = cache_root / "fly" / "stim" / "2023" / "10_17" / "rois.h5"
            self.assertEqual(resolved.paths.roi_file_to_load, expected_roi.resolve())
            self.assertEqual(expected_roi.read_text(encoding="utf-8"), "published")

    def test_recording_path_resolution_localizes_selected_converted_output(
        self,
    ) -> None:
        """Confirm direct converted selections are copied to cache before use.

        Inputs: a converted folder on a publish path with source metadata.
        Outputs: resolved paths point at the local analysis cache.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            cache_root = root / "cache"
            publish_dir = root / "publish"
            _write_source_recording_shape(source_dir)
            publish_dir.mkdir()
            _write_converted_recording(publish_dir, source_session_dir=source_dir)
            (root / "config.yml").write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {cache_root}\n"
                f"analysis_output: {publish_dir}\n",
                encoding="utf-8",
            )
            original_cwd = Path.cwd()
            try:
                chdir(root)
                resolved = resolve_or_convert_recording(publish_dir)
            finally:
                chdir(original_cwd)

            expected_dir = cache_root / "fly" / "stim" / "2023" / "10_17"
            self.assertFalse(resolved.was_converted)
            self.assertEqual(
                resolved.paths.recording_data_path,
                (expected_dir / "recording_data.h5").resolve(),
            )
            self.assertEqual(
                resolved.paths.movie_path,
                (expected_dir / "aligned_movie.h5").resolve(),
            )

    def test_recording_path_resolution_refreshes_stale_localized_output(
        self,
    ) -> None:
        """Confirm direct converted selections replace older cache copies.

        Inputs: a newer selected converted file and an older local cache copy.
        Outputs: local cached recording data is refreshed from the selected file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            source_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            cache_root = root / "cache"
            publish_dir = root / "publish"
            local_dir = cache_root / "fly" / "stim" / "2023" / "10_17"
            _write_source_recording_shape(source_dir)
            publish_dir.mkdir()
            local_dir.mkdir(parents=True)
            _write_converted_recording(publish_dir, source_session_dir=source_dir)
            _write_converted_recording(local_dir, source_session_dir=source_dir)
            source_recording = publish_dir / "recording_data.h5"
            local_recording = local_dir / "recording_data.h5"
            source_recording.touch()
            (root / "config.yml").write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root.resolve()}\n"
                "database_access: copy\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {cache_root}\n"
                f"analysis_output: {publish_dir}\n",
                encoding="utf-8",
            )
            original_cwd = Path.cwd()
            try:
                chdir(root)
                resolved = resolve_or_convert_recording(publish_dir)
            finally:
                chdir(original_cwd)

            self.assertEqual(
                resolved.paths.recording_data_path,
                local_recording.resolve(),
            )
            self.assertEqual(
                local_recording.stat().st_mtime_ns,
                source_recording.stat().st_mtime_ns,
            )

    def test_source_recording_validation_error_is_reported(self) -> None:
        """Confirm malformed source folders do not look like missing HDF5.

        Inputs: source-shaped recording folder with two real TIFF movies.
        Outputs: the source discovery error reaches the caller.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            _write_source_recording_shape(source_dir)
            (source_dir / "second_movie.tif").touch()

            with self.assertRaisesRegex(ValueError, "raw TIFF movie"):
                resolve_or_convert_recording(source_dir)

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

    def test_remove_roi_label_values_from_layer_clears_only_selected_rois(
        self,
    ) -> None:
        """Confirm ROI deletion edits only requested Labels values.

        Inputs: Labels layer with three ROI values and two selected values.
        Outputs: selected values become background while the other ROI remains.
        """
        layer = _FakeLayer(
            name="rois",
            data=np.array([[1, 2], [3, 2]], dtype=np.int64),
            options={},
        )

        removed_count = remove_roi_label_values_from_layer(layer, (1, 3))

        self.assertEqual(removed_count, 2)
        np.testing.assert_array_equal(
            layer.data,
            np.array([[0, 2], [0, 2]], dtype=np.int64),
        )

    def test_roi_tab_remove_selected_deletes_checked_rois(self) -> None:
        """Confirm the ROI tab can delete selected Labels values.

        Inputs: response plot widget with two plotted ROIs and ROI 2 unchecked.
        Outputs: Remove Selected clears ROI 1 and leaves ROI 2 in the layer.
        """
        _ = QApplication.instance() or QApplication([])
        layer = _FakeLayer(
            name="rois",
            data=np.array([[1, 2], [0, 2]], dtype=np.int64),
            options={},
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_roi_labels_layer(layer)
        response_widget._live_controller.request_update = lambda: None
        response_widget.set_response_plot_data(
            _two_roi_response_plot_data(),
            reset_axes=True,
        )
        response_widget._set_roi_visibility(1, False)
        remove_button = next(
            button
            for button in response_widget.options_widget().findChildren(QPushButton)
            if button.text() == "Remove Selected"
        )

        remove_button.click()

        np.testing.assert_array_equal(
            layer.data,
            np.array([[0, 2], [0, 2]], dtype=np.int64),
        )
        self.assertEqual(response_widget._roi_labels(), ("roi_0002",))
        checkbox_texts = {
            checkbox.text()
            for checkbox in response_widget.options_widget().findChildren(QCheckBox)
        }
        self.assertNotIn("roi_0001", checkbox_texts)
        self.assertIn("roi_0002", checkbox_texts)

    def test_roi_tab_remove_selected_handles_empty_roi_selection(self) -> None:
        """Confirm deleting all selected ROIs leaves a stable empty ROI list.

        Inputs: response plot widget with all plotted ROIs selected.
        Outputs: Remove Selected clears the Labels layer and all ROI checkboxes
        without trying to compute y-axis bounds from empty arrays.
        """
        _ = QApplication.instance() or QApplication([])
        layer = _FakeLayer(
            name="rois",
            data=np.array([[1, 2], [0, 2]], dtype=np.int64),
            options={},
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_roi_labels_layer(layer)
        response_widget._live_controller.request_update = lambda: None
        response_widget.set_response_plot_data(
            _two_roi_response_plot_data(),
            reset_axes=True,
        )
        remove_button = next(
            button
            for button in response_widget.options_widget().findChildren(QPushButton)
            if button.text() == "Remove Selected"
        )

        remove_button.click()

        np.testing.assert_array_equal(
            layer.data,
            np.zeros((2, 2), dtype=np.int64),
        )
        self.assertEqual(response_widget._roi_labels(), ())
        checkbox_texts = {
            checkbox.text()
            for checkbox in response_widget.options_widget().findChildren(QCheckBox)
        }
        self.assertNotIn("roi_0001", checkbox_texts)
        self.assertNotIn("roi_0002", checkbox_texts)

    def test_response_status_clears_stale_roi_list(self) -> None:
        """Confirm invalid live recomputes do not leave stale ROI rows.

        Inputs: response widget with existing plot data.
        Outputs: status-only state clears the old ROI option checkboxes.
        """
        _ = QApplication.instance() or QApplication([])
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(
            _two_roi_response_plot_data(),
            reset_axes=True,
        )

        response_widget.show_response_status("No ROI labels to analyze.")

        checkbox_texts = {
            checkbox.text()
            for checkbox in response_widget.options_widget().findChildren(QCheckBox)
        }
        self.assertNotIn("roi_0001", checkbox_texts)
        self.assertNotIn("roi_0002", checkbox_texts)

    def test_filter_response_plot_data_rois_keeps_matching_rows(self) -> None:
        """Confirm plot data ROI filtering keeps labels and trace rows aligned.

        Inputs: two-ROI plot data and one row index to keep.
        Outputs: every epoch contains only the requested ROI row.
        """
        filtered = filter_response_plot_data_rois(
            _two_roi_response_plot_data(),
            (1,),
        )
        epoch = filtered.epochs[0]

        self.assertEqual(epoch.roi_labels, ("roi_0002",))
        np.testing.assert_array_equal(
            epoch.mean_values,
            np.array([[2.0, 3.0]], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            epoch.sem_values,
            np.zeros((1, 2), dtype=np.float64),
        )

    def test_global_value_bounds_accepts_empty_roi_selection(self) -> None:
        """Confirm empty ROI selection uses stable default y-axis bounds.

        Inputs: two-ROI plot data with no visible ROI rows.
        Outputs: default y-axis bounds instead of a NumPy empty-reduction
        error.
        """
        self.assertEqual(
            global_value_bounds(_two_roi_response_plot_data(), ()), (-1.0, 1.0)
        )

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
            keys=(0, 1),
            visibility={0: True, 1: True},
            on_change=lambda key, visible: calls.append((key, visible)),
        )
        checkboxes = widget.findChildren(QCheckBox)

        checkboxes[1].setChecked(False)

        self.assertEqual(calls, [(1, False)])

    def test_visibility_options_read_initial_state_from_keys(self) -> None:
        """Confirm keyed rows do not read visibility from duplicate labels.

        Inputs: two rows with the same display label and different keys.
        Outputs: each checkbox uses the matching key state.
        """
        _ = QApplication.instance() or QApplication([])
        widget = visibility_options_widget(
            title="ROIs",
            labels=("same roi", "same roi"),
            keys=(0, 1),
            visibility={0: True, 1: False},
            on_change=lambda _key, _visible: None,
        )
        checkboxes = widget.findChildren(QCheckBox)

        self.assertTrue(checkboxes[0].isChecked())
        self.assertFalse(checkboxes[1].isChecked())

    def test_visibility_options_do_not_add_inner_scroll_area(self) -> None:
        """Confirm ROI/Epoch rows rely on the tab scrollbar.

        Inputs: many visibility labels.
        Outputs: option widget has no nested list scroll area.
        """
        _ = QApplication.instance() or QApplication([])
        widget = visibility_options_widget(
            title="ROIs",
            labels=tuple(f"roi_{index}" for index in range(20)),
            visibility={},
            on_change=lambda _key, _visible: None,
        )

        self.assertEqual(widget.findChildren(QScrollArea), [])

    def test_roi_visibility_can_hide_labels_layer_rois(self) -> None:
        """Confirm plot ROI visibility can hide ROI labels in napari.

        Inputs: one Labels layer with two ROI labels and one hidden ROI.
        Outputs: hidden ROI color is dimmed while label pixels remain.
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
        self.assertAlmostEqual(layer.get_color(1)[3], 0.2)
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

        self.assertAlmostEqual(layer.get_color(10)[3], 0.2)
        self.assertEqual(layer.get_color(11)[3], 1.0)
        self.assertEqual(layer.get_color(12)[3], 1.0)
        self.assertFalse(np.array_equal(layer.get_color(11), layer.get_color(12)))

    def test_roi_visibility_layer_uses_stable_keys_for_duplicate_names(self) -> None:
        """Confirm duplicate ROI names can hide distinct label values.

        Inputs: two label values with the same displayed ROI name.
        Outputs: hiding the second key dims only the second label.
        """
        layer = Labels(np.array([[1, 2]], dtype=np.int64))

        apply_roi_visibility_to_labels_layer(
            layer,
            roi_labels=("cell", "cell"),
            visibility={0: True, 1: False},
            keys=(0, 1),
            colors=(QColor("#ff0000"), QColor("#0000ff")),
        )

        self.assertEqual(layer.get_color(1)[3], 1.0)
        self.assertAlmostEqual(layer.get_color(2)[3], 0.2)

    def test_roi_visibility_ignores_malformed_colormap_metadata(self) -> None:
        """Confirm malformed persisted label-colormap metadata is rebuilt.

        Inputs: Labels layer with incomplete twopy colormap metadata.
        Outputs: visibility still applies instead of raising from metadata access.
        """
        layer = Labels(np.array([[1]], dtype=np.int64))
        layer.metadata["twopy_base_label_colormap"] = {"num_colors": "invalid"}

        apply_roi_visibility_to_labels_layer(
            layer,
            roi_labels=("roi_1",),
            visibility={"roi_1": False},
            colors=(QColor("#ff0000"),),
        )

        self.assertAlmostEqual(layer.get_color(1)[3], 0.2)

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

    def test_malformed_display_crop_metadata_is_ignored(self) -> None:
        """Confirm bad crop metadata does not crash label extraction.

        Inputs: display-coordinate labels with incomplete or invalid crop
        metadata.
        Outputs: labels treated as uncropped display data.
        """
        display_labels = np.array([[0, 1], [2, 0]], dtype=np.int64)
        bad_metadata: tuple[dict[str, object], ...] = (
            {
                "twopy_display_spatial_crop": {
                    "axis0_start": 0,
                    "axis1_start": 0,
                    "axis1_stop": 2,
                    "original_shape": (2, 2),
                    "source": "alignment_valid_crop",
                }
            },
            {
                "twopy_display_spatial_crop": {
                    "axis0_start": True,
                    "axis0_stop": 2,
                    "axis1_start": 0,
                    "axis1_stop": 2,
                    "original_shape": (2, 2),
                    "source": "alignment_valid_crop",
                }
            },
        )

        for metadata in bad_metadata:
            with self.subTest(metadata=metadata):
                layer = _FakeLayer(
                    name="rois",
                    data=display_labels,
                    options={"metadata": metadata},
                )

                np.testing.assert_array_equal(
                    movie_labels_from_display_layer(layer),
                    display_labels,
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
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 2), dtype=np.float64),
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

    def test_response_plot_data_avoids_sem_warning_for_single_trial(self) -> None:
        """Confirm one-trial epochs do not emit NumPy SEM warnings.

        Inputs: one trial for one epoch and one ROI.
        Outputs: plot-ready data has zero SEM and no runtime warning.
        """
        dff = RoiDeltaFOverF(
            fluorescence=np.ones((2, 1), dtype=np.float64),
            baseline=np.ones((2, 1), dtype=np.float64),
            values=np.array([[1.0], [3.0]], dtype=np.float64),
            labels=("roi_1",),
            start_frame=0,
            stop_frame=2,
            tau=0.0,
            amplitudes=np.ones(1, dtype=np.float64),
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 1), dtype=np.float64),
            metadata={"method": "test"},
        )
        grouped = group_delta_f_over_f_by_epoch(
            dff,
            (EpochFrameWindow(FrameWindow(0, 0, 2, "odor"), 2, "Odor"),),
            data_rate_hz=2.0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            plot_data = response_plot_data_from_grouped(grouped)

        np.testing.assert_allclose(
            plot_data.epochs[0].sem_values,
            np.zeros((1, 2), dtype=np.float64),
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
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 1), dtype=np.float64),
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

    def test_load_response_plot_data_restores_saved_auto_response_window(self) -> None:
        """Confirm saved analysis reloads the response-window Auto choice.

        Inputs: grouped responses persisted with ``response_window_auto`` true.
        Outputs: loaded plot data hydrates ``ResponseWindowOptions.auto`` true.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "analysis_outputs.h5"
            grouped = replace(
                _tiny_grouped_responses(),
                response_window_auto=True,
            )

            save_analysis_outputs(path, grouped_responses=grouped)

            result = load_response_plot_data(path)

        self.assertIsInstance(result, ResponsePlotData)
        if isinstance(result, ResponsePlotData):
            self.assertEqual(
                result.response_window_options,
                ResponseWindowOptions(
                    auto=True,
                    pre_window_seconds=0.0,
                    post_window_seconds=0.0,
                ),
            )

    def test_load_response_plot_data_restores_saved_baseline_epoch_selection(
        self,
    ) -> None:
        """Confirm saved analysis reloads the baseline epoch selector.

        Inputs: persisted dF/F metadata with a non-default baseline epoch.
        Outputs: loaded plot data hydrates the dF/F Plot-tab selector.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "analysis_outputs.h5"
            save_analysis_outputs(
                path,
                dff=_tiny_dff(
                    {
                        "method": "test",
                        "baseline_epoch_number": 3,
                        "baseline_epoch_name": "Manual baseline",
                        "baseline_sample_seconds": "full",
                        "fit_mode": "log_linear",
                    },
                ),
                grouped_responses=_tiny_grouped_responses(),
            )

            result = load_response_plot_data(path)

        self.assertIsInstance(result, ResponsePlotData)
        if isinstance(result, ResponsePlotData):
            self.assertEqual(
                result.delta_f_over_f_options,
                DeltaFOverFOptions(
                    baseline_epoch_number=3,
                    baseline_epoch_name="Manual baseline",
                    baseline_sample_seconds=None,
                    fit_mode="log_linear",
                    apply_motion_mask=False,
                ),
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
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 1), dtype=np.float64),
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

    def test_epoch_visibility_toggle_is_idempotent_by_row_index(self) -> None:
        """Confirm hiding and showing an epoch restores the same row.

        Inputs: two epoch plots and direct row-index toggles.
        Outputs: the selected epoch set returns to the original row indices.
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
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 1), dtype=np.float64),
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

        response_widget._set_epoch_visibility(0, False)
        response_widget._set_epoch_visibility(0, True)

        self.assertEqual(response_widget._visible_epoch_indices(), (0, 1))

    def test_duplicate_epoch_metadata_does_not_collide_in_visibility(self) -> None:
        """Confirm epoch rows do not share GUI state when metadata matches.

        Inputs: two plot rows with the same epoch number and name.
        Outputs: hiding the second row leaves only the first row visible.
        """
        _ = QApplication.instance() or QApplication([])
        time_seconds = np.array([0.0, 1.0], dtype=np.float64)
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Same",
                    epoch_number=1,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[0.0, 1.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
                EpochResponsePlotData(
                    epoch_name="Same",
                    epoch_number=1,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[2.0, 3.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
            ),
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(plot_data, reset_axes=True)

        response_widget._set_epoch_visibility(1, False)

        self.assertEqual(response_widget._visible_epoch_indices(), (0,))
        self.assertEqual(tuple(response_widget._plot_area.epoch_plot_widgets), (0, 1))

    def test_hidden_epoch_plot_panel_is_hidden_after_layout_refresh(self) -> None:
        """Confirm hidden epoch plots do not remain visible as stale Qt panels.

        Inputs: one gray epoch hidden by default plus two visible epochs.
        Outputs: hiding the last visible row explicitly hides that cached panel.
        """
        _ = QApplication.instance() or QApplication([])
        time_seconds = np.array([0.0, 1.0], dtype=np.float64)
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Gray Interleave",
                    epoch_number=1,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[0.0, 1.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
                EpochResponsePlotData(
                    epoch_name="Odor A",
                    epoch_number=2,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[2.0, 3.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
                EpochResponsePlotData(
                    epoch_name="Odor B",
                    epoch_number=3,
                    roi_labels=("roi_1",),
                    time_seconds=time_seconds,
                    mean_values=np.array([[4.0, 5.0]], dtype=np.float64),
                    sem_values=np.zeros((1, 2), dtype=np.float64),
                ),
            ),
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(plot_data, reset_axes=True)

        response_widget._set_epoch_visibility(2, False)

        self.assertEqual(response_widget._visible_epoch_indices(), (1,))
        self.assertTrue(response_widget._plot_area.epoch_plot_panels[0].isHidden())
        self.assertFalse(response_widget._plot_area.epoch_plot_panels[1].isHidden())
        self.assertTrue(response_widget._plot_area.epoch_plot_panels[2].isHidden())

    def test_roi_visibility_toggle_is_idempotent_by_row_index(self) -> None:
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
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 2), dtype=np.float64),
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

        response_widget._set_roi_visibility(1, False)
        self.assertEqual(response_widget._visible_roi_indices(), (0,))

        response_widget._set_roi_visibility(1, True)
        self.assertEqual(response_widget._visible_roi_indices(), (0, 1))

    def test_roi_visibility_repaints_without_rebuilding_epoch_layout(self) -> None:
        """Confirm ROI toggles do not hide and reshow every epoch panel.

        Inputs: two ROI traces and one visible epoch plot.
        Outputs: hiding one ROI updates plot state without clearing the layout.
        """
        _ = QApplication.instance() or QApplication([])
        plot_data = ResponsePlotData(
            source_path=None,
            epochs=(
                EpochResponsePlotData(
                    epoch_name="Odor",
                    epoch_number=1,
                    roi_labels=("roi_1", "roi_2"),
                    time_seconds=np.array([0.0, 1.0], dtype=np.float64),
                    mean_values=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
                    sem_values=np.zeros((2, 2), dtype=np.float64),
                ),
            ),
        )
        response_widget = cast(Any, create_response_plot_widget(None))
        response_widget.set_response_plot_data(plot_data, reset_axes=True)
        clear_calls: list[str] = []
        response_widget._plot_area.clear_layout_preserving_epoch_cache = lambda: (
            clear_calls.append("clear")
        )

        response_widget._set_roi_visibility(1, False)

        self.assertEqual(clear_calls, [])
        self.assertEqual(response_widget._visible_roi_indices(), (0,))
        self.assertFalse(response_widget._plot_area.epoch_plot_panels[0].isHidden())

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
            baseline_frame_numbers=np.array([0.0]),
            baseline_fluorescence=np.ones((1, 2), dtype=np.float64),
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

        self.assertEqual(global_time_bounds(plot_data, (1,)), (0.0, 1.0))
        self.assertEqual(
            global_value_bounds(plot_data, (0,), (1,)),
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

    def test_epoch_plot_widget_uses_compact_height(self) -> None:
        """Confirm response plots trim unused height below the x-axis.

        Inputs: plot-ready data for one epoch.
        Outputs: widget height preserves the square data area without bottom
        slack.
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

        self.assertEqual(widget.sizeHint().width(), 480)
        self.assertEqual(widget.sizeHint().height(), 432)

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
        self.assertEqual(widget.sizeHint().height(), 272)

    def test_epoch_plot_panel_centers_title(self) -> None:
        """Confirm epoch titles are centered above live response plots.

        Inputs: one titled epoch plot panel.
        Outputs: the title label uses horizontal center alignment.
        """
        _ = QApplication.instance() or QApplication([])
        plot_data = _tiny_response_plot_data()
        plot = EpochPlotWidget(
            plot_data.epochs[0],
            show_sem=True,
            roi_indices=(0,),
            roi_colors=(QColor("#ff0000"),),
            time_min=0.0,
            time_max=1.0,
            value_min=0.0,
            value_max=1.0,
            plot_size=320,
        )

        panel = epoch_plot_panel(title="Epoch 1: Odor", plot=plot)
        title_label = panel.findChild(QLabel)

        self.assertEqual(title_label.text(), "Epoch 1: Odor")
        self.assertTrue(title_label.alignment() & Qt.AlignmentFlag.AlignHCenter)

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
                epoch_indices=(0,),
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
    source_session_dir: Path | None = None,
    stimulus_parameters_json: str = "[]",
) -> Path:
    """Write a tiny converted recording for adapter tests.

    Args:
        root: Temporary directory receiving HDF5 files.
        movie_values: Optional movie array shaped ``(frames, axis0, axis1)``.
        alignment_valid_crop: Optional spatial crop metadata.
        source_session_dir: Optional source recording folder stored in the
            converted recording metadata.
        stimulus_parameters_json: JSON list of stimulus epoch parameter
            dictionaries.

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
        h5_file.attrs["source_session_dir"] = str(source_session_dir or root / "source")
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
        stimulus_group.create_dataset("parameters_json", data=stimulus_parameters_json)
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


def _two_roi_response_plot_data() -> ResponsePlotData:
    """Build tiny response plot data with two ROI rows.

    Args:
        None.

    Returns:
        Plot data whose ROI labels map to Labels values 1 and 2.
    """
    return response_plot_data_from_grouped(_two_roi_grouped_responses())


def _two_roi_grouped_responses() -> GroupedRoiResponses:
    """Build one tiny grouped response object with two ROIs.

    Args:
        None.

    Returns:
        Grouped response data with one epoch and two ROI labels.
    """
    dff = RoiDeltaFOverF(
        fluorescence=np.ones((2, 2), dtype=np.float64),
        baseline=np.ones((2, 2), dtype=np.float64),
        values=np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float64),
        labels=("roi_0001", "roi_0002"),
        start_frame=0,
        stop_frame=2,
        tau=0.0,
        amplitudes=np.ones(2, dtype=np.float64),
        baseline_frame_numbers=np.array([0.0]),
        baseline_fluorescence=np.ones((1, 2), dtype=np.float64),
        metadata={"method": "test"},
    )
    return group_delta_f_over_f_by_epoch(
        dff,
        (EpochFrameWindow(FrameWindow(0, 0, 2, "odor"), 1, "Odor"),),
        data_rate_hz=1.0,
    )


def _tiny_grouped_responses() -> GroupedRoiResponses:
    """Build one tiny grouped response object for napari plot tests.

    Args:
        None.

    Returns:
        Grouped response data with one ROI and one epoch.
    """
    return group_delta_f_over_f_by_epoch(
        _tiny_dff(),
        (EpochFrameWindow(FrameWindow(0, 0, 2, "odor"), 1, "Odor"),),
        data_rate_hz=1.0,
    )


def _tiny_dff(
    metadata: dict[str, str | int | float | bool] | None = None,
) -> RoiDeltaFOverF:
    """Build one tiny dF/F object for napari plot tests.

    Args:
        metadata: Optional dF/F metadata.

    Returns:
        dF/F data with one ROI and two frames.
    """
    return RoiDeltaFOverF(
        fluorescence=np.ones((2, 1), dtype=np.float64),
        baseline=np.ones((2, 1), dtype=np.float64),
        values=np.array([[0.0], [1.0]], dtype=np.float64),
        labels=("roi_1",),
        start_frame=0,
        stop_frame=2,
        tau=0.0,
        amplitudes=np.ones(1, dtype=np.float64),
        baseline_frame_numbers=np.array([0.0]),
        baseline_fluorescence=np.ones((1, 1), dtype=np.float64),
        metadata={"method": "test"} if metadata is None else metadata,
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
