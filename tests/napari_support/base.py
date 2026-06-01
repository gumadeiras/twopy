"""Shared Qt and napari doubles for napari tests.

Inputs: fake viewer state, Qt widgets, and small helper callbacks.
Outputs: reusable objects used by split napari test modules.
"""

# ruff: noqa: F401

import csv
import sqlite3
import unittest
import warnings
from collections.abc import Callable
from concurrent.futures import CancelledError, Future
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from datetime import date
from os import chdir, environ
from pathlib import Path
from threading import Event
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
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QTreeWidgetItem,
    QWidget,
)

import twopy.napari.controls as napari_controls
import twopy.napari.group_matching as group_matching
import twopy.napari.group_matching.roi_assignment as group_matching_roi
import twopy.napari.sidebar as napari_sidebar
import twopy.napari.trial_timeline as trial_timeline
from tests.tempdir import temporary_directory
from twopy import (
    load_manual_fov_group_rows,
    load_manual_roi_match_rows,
    load_roi_set,
    make_manual_fov_group_rows,
    make_roi_set,
    roi_set_to_label_image,
    save_manual_fov_group_rows,
    save_roi_set,
)
from twopy._version import __version__
from twopy.analysis.background_subtraction import BackgroundCorrectedRoiTraces
from twopy.analysis.dff import RoiDeltaFOverF
from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.persistence import save_analysis_outputs
from twopy.analysis.response_maps import (
    EpochResponseMap,
    ResponseMapData,
    ResponseMapOptions,
    load_response_map_data,
)
from twopy.analysis.response_plotting import (
    EpochResponsePlotData,
    ResponsePlotData,
    filter_response_plot_data_rois,
    response_plot_data_from_grouped,
)
from twopy.analysis.response_processing import (
    NormalizationOptions,
    ResponseProcessingOptions,
    RoiCorrelationScores,
    SmoothingOptions,
)
from twopy.analysis.response_window_options import ResponseWindowOptions
from twopy.analysis.responses import GroupedRoiResponses, group_delta_f_over_f_by_epoch
from twopy.analysis.trials import EpochFrameWindow, FrameWindow
from twopy.conversion.types import ConvertedRecording
from twopy.converted import load_converted_recording
from twopy.database.search import ExperimentSearchFilters
from twopy.napari import (
    add_twopy_magicgui_controls,
    open_recording_in_napari,
    roi_label_image_from_layer,
    save_napari_label_rois,
)
from twopy.napari.database_favorites import (
    ExperimentSearchFavorite,
    load_database_search_favorites,
    replace_database_search_favorite,
    save_database_search_favorites,
)
from twopy.napari.database_search import (
    ExperimentFavoriteErrorDialog,
    ExperimentSearchDialog,
    ExperimentSearchErrorDialog,
    RecordingLoadErrorDialog,
)
from twopy.napari.display import (
    display_metadata_for_spatial_crop,
    spatial_crop_from_layer_metadata,
)
from twopy.napari.display_paths import (
    format_output_folder,
    format_twopy_h5_output,
    recording_display_summary,
)
from twopy.napari.group_matching.images import mean_image_roi_overlay_pixmap
from twopy.napari.interactive import LiveResponseController
from twopy.napari.live_analysis import LiveResponseAnalysisCache
from twopy.napari.load_workflow import RecordingLoadFailure, RecordingLoadResult
from twopy.napari.loading import resolve_or_convert_recording
from twopy.napari.movie import resolve_movie_frame_range
from twopy.napari.paths import resolve_launch_recording_path, resolve_recording_paths
from twopy.napari.plotting.data import (
    load_response_plot_data,
    response_plot_min_epoch_duration_seconds,
)
from twopy.napari.plotting.dff_options import DeltaFOverFOptionsWidget
from twopy.napari.plotting.docks import create_response_plot_widget
from twopy.napari.plotting.export import (
    draw_epoch_response_plot,
    draw_response_heatmap,
    draw_roi_contours,
    export_epoch_plots,
    export_response_heatmaps,
    labels_for_recording_image,
    roi_boundary_segments,
)
from twopy.napari.plotting.export_controls import (
    ResponseExportState,
    create_response_export_tab,
)
from twopy.napari.plotting.form_controls import (
    PLOT_CONTROL_WIDTH,
    PLOT_DROPDOWN_WIDTH,
)
from twopy.napari.plotting.label_visibility import apply_roi_visibility_to_labels_layer
from twopy.napari.plotting.normalization_options import NormalizationOptionsWidget
from twopy.napari.plotting.options import (
    plot_display_options_group,
    visibility_options_widget,
)
from twopy.napari.plotting.panels import epoch_plot_panel
from twopy.napari.plotting.processing_options import ResponseProcessingOptionsWidget
from twopy.napari.plotting.response_map_colors import RESPONSE_HEATMAP_COLORMAP
from twopy.napari.plotting.response_map_display import (
    display_response_limit,
    display_response_values,
)
from twopy.napari.plotting.response_map_options import ResponseMapOptionsWidget
from twopy.napari.plotting.response_window_options import ResponseWindowOptionsWidget
from twopy.napari.plotting.roi_generation import (
    RoiGenerationControls,
    RoiGenerationOptions,
    generate_roi_labels,
)
from twopy.napari.plotting.widgets import (
    EpochPlotWidget,
    global_time_bounds,
    global_value_bounds,
    roi_colors_from_layer,
)
from twopy.napari.responses import (
    compute_response_preview,
    response_analysis_request_from_label_image,
)
from twopy.napari.roi import remove_roi_label_values_from_layer
from twopy.napari.sidebar import TWOPY_SIDEBAR_MINIMUM_WIDTH
from twopy.napari.state import write_last_recording_folder
from twopy.napari.text import counted_noun
from twopy.napari.trial_timeline import (
    TRIAL_TIMELINE_DOCK_NAME,
    TrialTimelineController,
    TrialTimelineData,
    TrialTimelineWindow,
    current_trial_text,
    current_trial_window,
    resolve_trial_timeline_data,
)
from twopy.napari.viewer import (
    APPLICATION_TITLE,
    create_viewer,
    interior_random_frame_indices,
)
from twopy.pixel_calibration import PixelCalibrationRow
from twopy.pixel_calibration_profiles import PixelCalibrationProfileMapping
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


class _ChangedSignal(Protocol):
    """Small protocol for magicgui changed signals used in tests."""

    def blocked(self) -> AbstractContextManager[None]:
        """Return a context manager that suppresses change callbacks."""
        ...


class _RecordingFolderControl(Protocol):
    """Small protocol for the recording-folder picker used in tests."""

    value: object
    changed: _ChangedSignal
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
    metadata: object | None = None
    selected_label: int = 0


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


def _combo_texts(combo: QComboBox) -> tuple[str, ...]:
    """Return all visible combo-box labels."""
    return tuple(combo.itemText(index) for index in range(combo.count()))


def _combo_data(combo: QComboBox) -> tuple[object, ...]:
    """Return all combo-box item data."""
    return tuple(combo.itemData(index) for index in range(combo.count()))


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
    layers: SimpleNamespace
    window: _FakeWindow

    def __init__(self) -> None:
        """Create an empty fake viewer.

        Inputs: none.
        Outputs: viewer with no recorded layers.
        """
        self.images: list[_FakeLayer] = []
        self.labels: list[_FakeLayer] = []
        self.layers = SimpleNamespace(selection=SimpleNamespace(active=None))
        self.window = _FakeWindow()
        self.dims = _FakeDims()
        self.text_overlay = _FakeTextOverlay()

    def add_image(self, data: object, *, name: str, **kwargs: object) -> object:
        """Record an image layer request.

        Args:
            data: Image data sent by the adapter.
            name: Layer name.
            kwargs: Napari image options.

        Returns:
            Fake image layer.
        """
        layer = _FakeLayer(
            name=name,
            data=data,
            options=dict(kwargs),
            metadata=kwargs.get("metadata"),
        )
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
        layer = _FakeLayer(
            name=name,
            data=data,
            options=dict(kwargs),
            metadata=kwargs.get("metadata"),
        )
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


class _FakeDimsEvents:
    """Event group with the dims event watched by timeline tools."""

    def __init__(self) -> None:
        """Create a current-step emitter."""
        self.current_step = _FakeEmitter()


class _FakeDims:
    """Small napari dims stand-in for current movie-frame tests."""

    def __init__(self) -> None:
        """Create dims state at frame zero."""
        self.current_step = (0,)
        self.point = (0,)
        self.events = _FakeDimsEvents()

    def set_current_step(self, axis: int, step: int) -> None:
        """Set one current-step axis and emit a dims-change event."""
        values = list(self.current_step)
        while axis >= len(values):
            values.append(0)
        values[axis] = step
        self.current_step = tuple(values)
        self.point = tuple(values)
        self.events.current_step.emit()


@dataclass
class _FakeTextOverlay:
    """Small napari text overlay stand-in."""

    text: str = ""
    visible: bool = False
    position: str = "bottom_right"
    font_size: int = 0
    color: object | None = None
    box: bool = False
    box_color: object | None = None


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


def process_qt_events_until(
    predicate: Callable[[], bool],
    *,
    timeout_seconds: float = 5.0,
) -> None:
    """Process Qt events until an asynchronous GUI action reaches a state.

    Args:
        predicate: Condition that becomes true when the expected GUI state is
            visible.
        timeout_seconds: Maximum wait before failing the test.

    Returns:
        None.

    Raises:
        AssertionError: If the predicate stays false until the timeout.
    """
    app = QApplication.instance() or QApplication([])
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        sleep(0.01)
    app.processEvents()
    raise AssertionError("Qt event condition did not become true")


def _record_loaded_paths(
    loaded_paths: list[tuple[Path, ...]],
    paths: tuple[Path, ...],
) -> RecordingLoadResult:
    """Record database-search paths received by a test callback.

    Args:
        loaded_paths: Mutable list of callback payloads.
        paths: Paths sent by the dialog.

    Returns:
        Structured load result for the dialog.
    """
    loaded_paths.append(paths)
    return RecordingLoadResult(loaded_count=len(paths))


def _tree_leaf_items(item: QTreeWidgetItem) -> tuple[QTreeWidgetItem, ...]:
    """Return all leaf rows below one search-result tree item.

    Args:
        item: Parent item from a database-search result tree.

    Returns:
        Leaf descendants in visual order.
    """
    if item.childCount() == 0:
        return (item,)
    leaves: list[QTreeWidgetItem] = []
    for index in range(item.childCount()):
        child = item.child(index)
        if child is not None:
            leaves.extend(_tree_leaf_items(child))
    return tuple(leaves)


class NapariAdapterTestCase(unittest.TestCase):
    """Base class that isolates napari state-file writes per test."""

    def setUp(self) -> None:
        """Route napari UI state to a temporary file during tests.

        Inputs: none.
        Outputs: isolated state path in the test environment.
        """
        self._state_temp_dir = temporary_directory()
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
