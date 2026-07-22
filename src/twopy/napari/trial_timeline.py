"""Trial timeline HUD and rail for the twopy napari viewer.

Inputs: converted recordings, napari movie layers, and viewer dimension events.
Outputs: a compact bottom timeline plus current trial text in the viewer.

This module owns presentation and viewer lifecycle for trial-structure display.
The scientific timing still comes from ``twopy.analysis``. This file only turns
those frame windows into a lightweight navigation aid while users scroll the
recording stack.
"""

from bisect import bisect_right
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from qtpy.QtCore import QPoint, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from qtpy.QtWidgets import QSizePolicy, QWidget

from twopy.analysis.timing import resolve_recording_timing
from twopy.analysis.trials import EpochFrameWindow, is_baseline_epoch_name
from twopy.converted import RecordingData
from twopy.napari.dims import current_step_index, set_current_step_index
from twopy.napari.protocols import (
    NapariEventEmitter,
    NapariViewer,
    NapariViewerWithDims,
)
from twopy.napari.theme import active_twopy_theme_colors
from twopy.typing_guards import int_or_none, string_key_mapping_or_none

__all__ = [
    "MOVIE_FRAME_START_METADATA_KEY",
    "MOVIE_FRAME_STOP_METADATA_KEY",
    "TRIAL_TIMELINE_DOCK_NAME",
    "TrialTimelineController",
    "TrialTimelineData",
    "TrialTimelineWidget",
    "TrialTimelineWindow",
    "current_trial_text",
    "current_trial_window",
    "movie_frame_metadata",
    "movie_layer_frame_start",
    "movie_layer_frame_stop",
    "resolve_trial_timeline_data",
]

TRIAL_TIMELINE_DOCK_NAME = "twopy trial timeline"
MOVIE_FRAME_START_METADATA_KEY = "twopy_movie_frame_start"
MOVIE_FRAME_STOP_METADATA_KEY = "twopy_movie_frame_stop"
_TIMELINE_HEIGHT = 14
_MINIMUM_TIMELINE_WIDTH = 360
_BASELINE_EPOCH_COLOR = QColor(145, 150, 160)


class _LayerWithDataAndMetadata(Protocol):
    """Small shape of the movie layer needed by the timeline."""

    data: object
    metadata: object


class _ViewerWithTextOverlay(Protocol):
    """Small viewer shape for napari text overlay updates."""

    text_overlay: object


class _TextOverlay(Protocol):
    """Small shape of napari's text overlay object."""

    text: str
    visible: bool
    position: str
    font_size: int
    color: object
    box: bool
    box_color: object


@dataclass(frozen=True)
class _OverlayState:
    """Viewer text-overlay fields to restore after timeline ownership ends."""

    text: str
    visible: bool
    position: str
    font_size: int
    color: object
    box: bool
    box_color: object


@dataclass(frozen=True)
class TrialTimelineWindow:
    """One visible frame window in the recording timeline.

    Inputs: absolute movie frame bounds plus stimulus epoch metadata.
    Outputs: a compact object used by the rail and HUD.
    """

    index: int
    start_frame: int
    stop_frame: int
    epoch_number: int
    epoch_name: str


@dataclass(frozen=True)
class TrialTimelineData:
    """All trial windows needed to render a recording timeline.

    Inputs: movie frame count and stimulus windows.
    Outputs: lookup-ready frame bounds for the current-frame HUD and rail.
    """

    frame_count: int
    windows: tuple[TrialTimelineWindow, ...]
    start_frames: tuple[int, ...]
    stop_frames: tuple[int, ...]


def resolve_trial_timeline_data(recording: RecordingData) -> TrialTimelineData | None:
    """Return timeline data for a recording, or ``None`` when unavailable.

    Args:
        recording: Loaded converted recording.

    Returns:
        Timeline data with one window per contiguous stimulus epoch run, or
        ``None`` if the converted recording cannot produce an auditable
        photodiode-aligned epoch mapping.
    """
    try:
        timing = resolve_recording_timing(recording)
    except ValueError:
        return None
    windows = tuple(
        _timeline_window(index, item) for index, item in enumerate(timing.epoch_windows)
    )
    if len(windows) == 0:
        return None
    return TrialTimelineData(
        frame_count=recording.movie.shape[0],
        windows=windows,
        start_frames=tuple(window.start_frame for window in windows),
        stop_frames=tuple(window.stop_frame for window in windows),
    )


def current_trial_window(
    timeline: TrialTimelineData,
    frame: int,
) -> TrialTimelineWindow | None:
    """Return the window containing one absolute movie frame.

    Args:
        timeline: Timeline data for the active recording.
        frame: Absolute aligned-movie frame index.

    Returns:
        Matching trial window, or ``None`` when the frame is outside all
        stimulus windows.
    """
    index = bisect_right(timeline.start_frames, frame) - 1
    if index < 0 or index >= len(timeline.windows):
        return None
    window = timeline.windows[index]
    if frame >= window.stop_frame:
        return None
    return window


def current_trial_text(timeline: TrialTimelineData, frame: int) -> str:
    """Return concise HUD text for one absolute movie frame.

    Args:
        timeline: Timeline data for the active recording.
        frame: Absolute aligned-movie frame index.

    Returns:
        Human-readable current trial/epoch text.
    """
    window = current_trial_window(timeline, frame)
    if window is None:
        return "No trial"
    return (
        f"Trial {window.index + 1}/{len(timeline.windows)} | "
        f"Epoch {window.epoch_number}: {window.epoch_name}"
    )


def movie_frame_metadata(*, start_frame: int, stop_frame: int) -> dict[str, object]:
    """Return movie-layer metadata for absolute frame mapping.

    Args:
        start_frame: Absolute movie frame shown at displayed stack index zero.
        stop_frame: Exclusive absolute movie frame after the displayed stack.

    Returns:
        Metadata keys stored on the napari movie layer.
    """
    return {
        MOVIE_FRAME_START_METADATA_KEY: int(start_frame),
        MOVIE_FRAME_STOP_METADATA_KEY: int(stop_frame),
    }


def movie_layer_frame_start(movie_layer: object | None) -> int:
    """Return the absolute frame corresponding to displayed stack index zero.

    Args:
        movie_layer: Napari movie layer, or ``None``.

    Returns:
        Absolute frame offset. Missing metadata means the layer starts at frame
        zero, which is true for the normal full-movie loading path.
    """
    value = _movie_layer_metadata_int(movie_layer, MOVIE_FRAME_START_METADATA_KEY)
    return 0 if value is None else max(0, value)


def movie_layer_frame_stop(movie_layer: object | None) -> int | None:
    """Return the exclusive absolute stop frame for the displayed movie layer.

    Args:
        movie_layer: Napari movie layer, or ``None``.

    Returns:
        Exclusive absolute stop frame, or ``None`` when no bounded movie stack
        metadata is available.
    """
    value = _movie_layer_metadata_int(movie_layer, MOVIE_FRAME_STOP_METADATA_KEY)
    return None if value is None else max(0, value)


class TrialTimelineWidget(QWidget):
    """Compact rail showing trial windows across the movie frame stack."""

    frame_selected = Signal(int)

    def __init__(self) -> None:
        """Create an empty timeline rail."""
        super().__init__()
        self._timeline: TrialTimelineData | None = None
        self._current_frame = 0
        self._visible_frame_start = 0
        self._visible_frame_stop: int | None = None
        self.setMinimumHeight(_TIMELINE_HEIGHT)
        self.setMaximumHeight(_TIMELINE_HEIGHT)
        self.setMinimumWidth(_MINIMUM_TIMELINE_WIDTH)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.setMouseTracking(True)

    def set_timeline(self, timeline: TrialTimelineData | None) -> None:
        """Replace the rendered trial windows.

        Args:
            timeline: New timeline data, or ``None`` to clear the rail.

        Returns:
            None.
        """
        self._timeline = timeline
        self.update()

    def set_visible_frame_range(self, start_frame: int, stop_frame: int | None) -> None:
        """Set the movie frames that can be reached by the current stack.

        Args:
            start_frame: Absolute frame shown at stack index zero.
            stop_frame: Exclusive absolute stop frame for the loaded stack, or
                ``None`` when no bounded stack is available.

        Returns:
            None.
        """
        self._visible_frame_start = max(0, int(start_frame))
        self._visible_frame_stop = None if stop_frame is None else max(0, stop_frame)
        self.update()

    def set_current_frame(self, frame: int) -> None:
        """Set the absolute current movie frame shown by the cursor.

        Args:
            frame: Absolute movie frame index.

        Returns:
            None.
        """
        bounded = max(0, int(frame))
        if bounded == self._current_frame:
            return
        self._current_frame = bounded
        self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:  # noqa: N802
        """Paint trial blocks and the current-frame cursor."""
        del a0
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        try:
            self._paint_background(painter)
            timeline = self._timeline
            if timeline is None:
                return
            self._paint_windows(painter, timeline)
            self._paint_outside_visible_range(painter, timeline)
            self._paint_cursor(painter, timeline)
        finally:
            painter.end()

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:  # noqa: N802
        """Seek to the clicked frame on the timeline rail."""
        if a0 is None:
            return
        self._emit_visible_frame(a0.pos())

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:  # noqa: N802
        """Drag-seek while the primary mouse button is held."""
        if a0 is None:
            return
        if a0.buttons() & Qt.MouseButton.LeftButton:
            self._emit_visible_frame(a0.pos())

    def _paint_background(self, painter: QPainter) -> None:
        """Paint the rail background."""
        theme = active_twopy_theme_colors(self.palette())
        painter.fillRect(self.rect(), QColor(theme.window))

    def _paint_windows(
        self,
        painter: QPainter,
        timeline: TrialTimelineData,
    ) -> None:
        """Paint one proportional block per trial window."""
        height = max(1, self.height() - 8)
        y = 4
        for window in timeline.windows:
            start_x = self._x_from_frame(window.start_frame, timeline)
            stop_x = self._x_from_frame(window.stop_frame, timeline)
            rect = QRectF(
                float(start_x),
                float(y),
                max(1.0, float(stop_x - start_x)),
                float(height),
            )
            painter.fillRect(rect, _timeline_window_color(window))

    def _paint_cursor(self, painter: QPainter, timeline: TrialTimelineData) -> None:
        """Paint the current-frame cursor above the trial blocks."""
        x = self._x_from_frame(self._current_frame, timeline)
        theme = active_twopy_theme_colors(self.palette())
        pen = QPen(QColor(theme.text))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(x, 1, x, max(1, self.height() - 2))

    def _paint_outside_visible_range(
        self,
        painter: QPainter,
        timeline: TrialTimelineData,
    ) -> None:
        """Dim timeline regions that are not loaded in the current movie stack."""
        visible_stop = self._visible_frame_stop
        if visible_stop is None:
            return
        theme = active_twopy_theme_colors(self.palette())
        shade = QColor(theme.window)
        shade.setAlpha(180)
        start_x = self._x_from_frame(self._visible_frame_start, timeline)
        stop_x = self._x_from_frame(visible_stop, timeline)
        if start_x > 0:
            painter.fillRect(
                QRectF(0.0, 0.0, float(start_x), float(self.height())),
                shade,
            )
        width = self.width()
        if stop_x < width:
            painter.fillRect(
                QRectF(float(stop_x), 0.0, float(width - stop_x), float(self.height())),
                shade,
            )

    def _x_from_frame(self, frame: int, timeline: TrialTimelineData) -> int:
        """Map an absolute frame to a widget x coordinate."""
        if timeline.frame_count <= 1:
            return 0
        fraction = max(0.0, min(float(frame) / float(timeline.frame_count - 1), 1.0))
        return int(round(fraction * max(1, self.width() - 1)))

    def _frame_from_position(self, point: QPoint) -> int:
        """Map a widget point to an absolute movie frame."""
        timeline = self._timeline
        if timeline is None or self.width() <= 1:
            return 0
        fraction = max(0.0, min(float(point.x()) / float(self.width() - 1), 1.0))
        return int(round(fraction * float(timeline.frame_count - 1)))

    def _emit_visible_frame(self, point: QPoint) -> None:
        """Emit the frame under ``point`` when it is available in the stack."""
        if self._timeline is None:
            return
        frame = self._frame_from_position(point)
        if self._is_frame_visible(frame):
            self.frame_selected.emit(frame)

    def _is_frame_visible(self, frame: int) -> bool:
        """Return whether an absolute frame is loaded in the current stack."""
        if frame < self._visible_frame_start:
            return False
        return self._visible_frame_stop is None or frame < self._visible_frame_stop


class TrialTimelineController:
    """Keep the timeline rail and viewer HUD synchronized with napari dims."""

    def __init__(self, viewer: NapariViewer) -> None:
        """Create a controller for one napari viewer.

        Args:
            viewer: Viewer that owns movie layers, the dims slider, and docks.
        """
        self._viewer = viewer
        self._widget: TrialTimelineWidget | None = None
        self._dock_widget: object | None = None
        self._timeline: TrialTimelineData | None = None
        self._movie_layer: object | None = None
        self._timeline_cache: dict[tuple[Path, Path], TrialTimelineData | None] = {}
        self._overlay_state: _OverlayState | None = None
        self._dims_event: NapariEventEmitter | None = _connect_dims_event(
            viewer,
            self.update_current_frame,
        )

    @property
    def widget(self) -> TrialTimelineWidget:
        """Return the timeline widget owned by this controller."""
        return self._ensure_widget()

    @property
    def dock_widget(self) -> object | None:
        """Return the napari dock widget, if it has been created."""
        return self._dock_widget

    def set_context(
        self,
        recording: RecordingData | None,
        movie_layer: object | None,
    ) -> None:
        """Set the active recording and movie layer for timeline updates.

        Args:
            recording: Selected recording, or ``None`` when nothing is active.
            movie_layer: Movie layer belonging to ``recording``.

        Returns:
            None.
        """
        if recording is None or movie_layer is None:
            self._clear_context()
            return
        timeline = self._timeline_data_for_recording(recording)
        if timeline is None:
            self._clear_context()
            return
        self._movie_layer = movie_layer
        self._timeline = timeline
        self._show_overlay()
        widget = self._ensure_widget()
        widget.set_visible_frame_range(
            movie_layer_frame_start(movie_layer),
            movie_layer_frame_stop(movie_layer),
        )
        widget.set_timeline(timeline)
        self._ensure_dock()
        self._set_dock_visible(True)
        self.update_current_frame()

    def update_current_frame(self, *_event_args: object) -> None:
        """Refresh the rail cursor and HUD from current napari dims state.

        Args:
            _event_args: Optional napari event arguments, ignored.

        Returns:
            None.
        """
        timeline = self._timeline
        if timeline is None:
            return
        frame = self._current_absolute_frame()
        if self._widget is not None:
            self._widget.set_current_frame(frame)
        self._set_overlay_text(current_trial_text(timeline, frame))

    def close(self) -> None:
        """Disconnect napari events owned by this controller."""
        if self._dims_event is not None:
            with suppress(ValueError):
                self._dims_event.disconnect(self.update_current_frame)
            self._dims_event = None
        self._hide_overlay()

    def _clear_context(self) -> None:
        """Clear active recording state and hide owned timeline UI."""
        self._movie_layer = None
        self._timeline = None
        if self._widget is not None:
            self._widget.set_timeline(None)
        self._set_dock_visible(False)
        self._hide_overlay()

    def _ensure_dock(self) -> None:
        """Create the bottom timeline dock when first needed."""
        if self._dock_widget is not None:
            return
        self._dock_widget = self._viewer.window.add_dock_widget(
            self._ensure_widget(),
            name=TRIAL_TIMELINE_DOCK_NAME,
            area="bottom",
        )

    def _ensure_widget(self) -> TrialTimelineWidget:
        """Create the Qt timeline widget lazily when the first timeline exists."""
        if self._widget is None:
            self._widget = TrialTimelineWidget()
            self._widget.frame_selected.connect(self._seek_absolute_frame)
        return self._widget

    def _current_absolute_frame(self) -> int:
        """Return the absolute movie frame represented by current dims."""
        relative_count = _movie_layer_frame_count(self._movie_layer)
        relative_frame = current_step_index(
            self._viewer,
            axis=0,
            step_count=relative_count,
        )
        return movie_layer_frame_start(self._movie_layer) + relative_frame

    def _seek_absolute_frame(self, frame: int) -> None:
        """Move napari's movie stack to one absolute frame."""
        start_frame = movie_layer_frame_start(self._movie_layer)
        relative_count = _movie_layer_frame_count(self._movie_layer)
        relative_frame = max(0, int(frame) - start_frame)
        if relative_count is not None:
            if frame < start_frame or frame >= start_frame + relative_count:
                return
            relative_frame = min(relative_frame, relative_count - 1)
        set_current_step_index(self._viewer, axis=0, step=relative_frame)
        self.update_current_frame()

    def _timeline_data_for_recording(
        self,
        recording: RecordingData,
    ) -> TrialTimelineData | None:
        """Return cached timeline data for one recording."""
        key = (recording.path, recording.movie.path)
        if key not in self._timeline_cache:
            self._timeline_cache[key] = resolve_trial_timeline_data(recording)
        return self._timeline_cache[key]

    def _set_dock_visible(self, visible: bool) -> None:
        """Show or hide the dock when the dock object supports it."""
        if self._dock_widget is None:
            return
        setter = getattr(self._dock_widget, "setVisible", None)
        if callable(setter):
            setter(visible)

    def _set_overlay_text(self, text: str) -> None:
        """Write the current trial text into napari's viewer overlay."""
        overlay = _text_overlay(self._viewer)
        if overlay is None:
            return
        overlay.text = text

    def _show_overlay(self) -> None:
        """Claim and show napari's viewer text overlay."""
        overlay = _text_overlay(self._viewer)
        if overlay is None:
            return
        self._claim_overlay(overlay)
        overlay.visible = True
        overlay.position = "bottom_right"
        overlay.font_size = 10
        overlay.color = "white"
        overlay.box = True
        overlay.box_color = (0.0, 0.0, 0.0, 0.55)

    def _hide_overlay(self) -> None:
        """Restore napari's viewer text overlay after timeline use."""
        overlay = _text_overlay(self._viewer)
        if overlay is None:
            return
        self._restore_overlay(overlay)

    def _claim_overlay(self, overlay: _TextOverlay) -> None:
        """Remember existing overlay state before writing timeline text."""
        if self._overlay_state is not None:
            return
        self._overlay_state = _OverlayState(
            text=overlay.text,
            visible=overlay.visible,
            position=overlay.position,
            font_size=overlay.font_size,
            color=overlay.color,
            box=overlay.box,
            box_color=overlay.box_color,
        )

    def _restore_overlay(self, overlay: _TextOverlay) -> None:
        """Restore overlay fields saved before timeline ownership."""
        state = self._overlay_state
        if state is None:
            overlay.visible = False
            return
        overlay.text = state.text
        overlay.visible = state.visible
        overlay.position = state.position
        overlay.font_size = state.font_size
        overlay.color = state.color
        overlay.box = state.box
        overlay.box_color = state.box_color
        self._overlay_state = None


def _timeline_window(index: int, epoch_window: EpochFrameWindow) -> TrialTimelineWindow:
    """Convert analysis frame-window data into timeline UI data."""
    return TrialTimelineWindow(
        index=index,
        start_frame=epoch_window.window.start_frame,
        stop_frame=epoch_window.window.stop_frame,
        epoch_number=epoch_window.epoch_number,
        epoch_name=epoch_window.epoch_name,
    )


def _timeline_window_color(window: TrialTimelineWindow) -> QColor:
    """Return the rail color for one timeline window."""
    if is_baseline_epoch_name(window.epoch_name):
        return QColor(_BASELINE_EPOCH_COLOR)
    return _condition_epoch_color(window.epoch_number)


def _condition_epoch_color(epoch_number: int) -> QColor:
    """Return a stable high-contrast color for one non-baseline epoch number."""
    palette = (
        QColor(80, 170, 220),
        QColor(238, 169, 73),
        QColor(93, 184, 126),
        QColor(218, 92, 92),
        QColor(174, 135, 232),
        QColor(221, 204, 89),
        QColor(84, 191, 179),
        QColor(229, 129, 172),
    )
    return palette[(max(1, epoch_number) - 1) % len(palette)]


def _movie_layer_frame_count(movie_layer: object | None) -> int | None:
    """Return displayed movie stack length for a movie layer."""
    if movie_layer is None or not hasattr(movie_layer, "data"):
        return None
    data = np.asarray(cast(_LayerWithDataAndMetadata, movie_layer).data)
    if data.ndim != 3:
        return None
    return int(data.shape[0])


def _movie_layer_metadata_int(movie_layer: object | None, key: str) -> int | None:
    """Return one integer metadata value from a movie layer."""
    if movie_layer is None or not hasattr(movie_layer, "metadata"):
        return None
    metadata = string_key_mapping_or_none(
        cast(_LayerWithDataAndMetadata, movie_layer).metadata,
    )
    if metadata is None:
        return None
    return int_or_none(metadata.get(key))


def _text_overlay(viewer: object) -> _TextOverlay | None:
    """Return napari's text overlay object when available."""
    if not hasattr(viewer, "text_overlay"):
        return None
    return cast(_TextOverlay, cast(_ViewerWithTextOverlay, viewer).text_overlay)


def _connect_dims_event(
    viewer: object,
    callback: Callable[..., None],
) -> NapariEventEmitter | None:
    """Connect to napari current-step changes when the event exists."""
    if not hasattr(viewer, "dims"):
        return None
    dims = cast(NapariViewerWithDims, viewer).dims
    events = getattr(dims, "events", None)
    emitter = getattr(events, "current_step", None)
    if emitter is None or not hasattr(emitter, "connect"):
        return None
    event_emitter = cast(NapariEventEmitter, emitter)
    event_emitter.connect(callback)
    return event_emitter
