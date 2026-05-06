"""Live response updates for napari ROI editing.

Inputs: the selected converted recording, its editable Labels layer, and the
response plot widget.
Outputs: debounced plot refreshes as ROI pixels change.

The controller owns GUI orchestration only. It copies Labels pixels on the GUI
thread, then runs the heavier response calculation through the same in-memory
analysis helper used by the manual update button.
"""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from qtpy.QtCore import QTimer

from twopy.converted import RecordingData
from twopy.napari.plotting.data import ResponsePlotData
from twopy.napari.responses import compute_response_plot_data_from_roi_set
from twopy.napari.roi import roi_label_image_from_layer_for_recording
from twopy.roi import make_roi_set_from_label_image

__all__ = ["LiveResponseController"]


class ResponsePlotReceiver(Protocol):
    """Small protocol for the plot widget methods used by live updates."""

    def set_response_plot_data(
        self,
        plot_data: ResponsePlotData,
        *,
        reset_axes: bool,
    ) -> None:
        """Show newly computed plot data.

        Args:
            plot_data: Plot-ready response data.
            reset_axes: Whether plot bounds should return to data-derived
                defaults.

        Returns:
            None.
        """
        ...

    def show_response_status(self, text: str) -> None:
        """Show a user-facing response status message.

        Args:
            text: Status text.

        Returns:
            None.
        """
        ...


class _EventEmitter(Protocol):
    """Small protocol for psygnal event emitters exposed by napari layers."""

    def connect(self, callback: Callable[..., None]) -> object:
        """Connect one callback to the event emitter."""
        ...

    def disconnect(self, callback: Callable[..., None]) -> object:
        """Disconnect one callback from the event emitter."""
        ...


@dataclass(frozen=True)
class _ConnectedEmitter:
    """One napari Labels event connection owned by the controller."""

    emitter: _EventEmitter


class LiveResponseController:
    """Debounced response updater for one active napari Labels layer.

    Inputs: a plot widget plus the selected recording/layer context.
    Outputs: live response plot refreshes after ROI edits.

    The controller keeps only one worker job active. If ROI pixels change while
    a job is running, the running result is discarded when stale and one newer
    job is scheduled. This keeps the UI responsive without building a complex
    queue that would compute obsolete ROIs.
    """

    def __init__(
        self,
        plot_widget: ResponsePlotReceiver,
        *,
        debounce_ms: int = 400,
        run_async: bool = True,
    ) -> None:
        """Create a live updater for one response plot widget.

        Args:
            plot_widget: Widget that receives plot data or status messages.
            debounce_ms: Delay after the latest ROI edit before recomputing.
            run_async: Whether heavy analysis should run in a worker thread.
                Tests can turn this off to verify behavior synchronously.
        """
        self._plot_widget = plot_widget
        self._recording: RecordingData | None = None
        self._roi_labels_layer: object | None = None
        self._connections: list[_ConnectedEmitter] = []
        self._version = 0
        self._active_version: int | None = None
        self._pending_after_active_job = False
        self._future: Future[ResponsePlotData] | None = None
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="twopy-roi-live")
            if run_async
            else None
        )
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._start_latest_job)
        self._poll_timer = QTimer()
        self._poll_timer.setInterval(75)
        self._poll_timer.timeout.connect(self._collect_finished_job)
        self._debounce_ms = debounce_ms

    def set_context(
        self,
        recording: RecordingData | None,
        roi_labels_layer: object | None,
    ) -> None:
        """Set the recording and Labels layer watched by live updates.

        Args:
            recording: Selected converted recording, or ``None``.
            roi_labels_layer: Editable Labels layer for that recording, or
                ``None``.

        Returns:
            None.
        """
        self._disconnect_layer_events()
        self._recording = recording
        self._roi_labels_layer = roi_labels_layer
        self._version += 1
        self._pending_after_active_job = False
        self._connect_layer_events(roi_labels_layer)

    def request_update(self, *_event_args: object) -> None:
        """Schedule a live response update after an ROI edit.

        Args:
            _event_args: Optional napari event arguments. They are ignored
                because the Labels layer itself is the source of truth.

        Returns:
            None.
        """
        self._version += 1
        if self._debounce_ms <= 0:
            self._start_latest_job()
            return
        self._debounce_timer.start(self._debounce_ms)

    def update_now(self) -> None:
        """Compute responses immediately on the GUI thread.

        Args:
            None.

        Returns:
            None.

        The manual update button uses this synchronous path because a direct
        button click should report validation errors immediately. Continuous ROI
        edits use ``request_update`` so heavier work can run in the background.
        """
        self._version += 1
        version = self._version
        try:
            plot_data = self._compute_current_plot_data()
        except ValueError as error:
            if version == self._version:
                self._plot_widget.show_response_status(str(error))
            return
        if version == self._version:
            self._plot_widget.set_response_plot_data(plot_data, reset_axes=True)

    def shutdown(self) -> None:
        """Disconnect layer events and stop background work.

        Args:
            None.

        Returns:
            None.
        """
        self._disconnect_layer_events()
        self._debounce_timer.stop()
        self._poll_timer.stop()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _start_latest_job(self) -> None:
        """Start the newest queued response computation."""
        if self._executor is None:
            self.update_now()
            return
        if self._future is not None:
            self._pending_after_active_job = True
            return

        version = self._version
        try:
            label_image = self._current_label_image()
        except ValueError as error:
            if version == self._version:
                self._plot_widget.show_response_status(str(error))
            return
        if not np.any(label_image > 0):
            if version == self._version:
                self._plot_widget.show_response_status("No ROI labels to analyze.")
            return

        recording = self._require_recording()
        roi_set = make_roi_set_from_label_image(label_image)
        self._active_version = version
        self._future = self._executor.submit(
            compute_response_plot_data_from_roi_set,
            recording,
            roi_set,
            source_path=None,
        )
        self._poll_timer.start()

    def _collect_finished_job(self) -> None:
        """Apply a finished worker result when it is still current."""
        future = self._future
        if future is None or not future.done():
            return

        version = self._active_version
        self._future = None
        self._active_version = None
        self._poll_timer.stop()
        try:
            plot_data = future.result()
        except ValueError as error:
            if version == self._version:
                self._plot_widget.show_response_status(str(error))
        else:
            if version == self._version:
                self._plot_widget.set_response_plot_data(plot_data, reset_axes=True)

        if self._pending_after_active_job or version != self._version:
            self._pending_after_active_job = False
            self._start_latest_job()

    def _compute_current_plot_data(self) -> ResponsePlotData:
        """Return plot data for the current recording and Labels layer."""
        label_image = self._current_label_image()
        if not np.any(label_image > 0):
            msg = "No ROI labels to analyze."
            raise ValueError(msg)
        return compute_response_plot_data_from_roi_set(
            self._require_recording(),
            make_roi_set_from_label_image(label_image),
            source_path=None,
        )

    def _current_label_image(self) -> np.ndarray:
        """Return full-frame ROI labels from the current Labels layer."""
        recording = self._require_recording()
        layer = self._roi_labels_layer
        if layer is None:
            msg = "No ROI Labels layer is available."
            raise ValueError(msg)
        return roi_label_image_from_layer_for_recording(layer, recording)

    def _require_recording(self) -> RecordingData:
        """Return the selected recording or raise a user-facing error."""
        if self._recording is None:
            msg = "No recording loaded."
            raise ValueError(msg)
        return self._recording

    def _connect_layer_events(self, layer: object | None) -> None:
        """Connect Labels data-change events to debounced updates."""
        if layer is None:
            return
        events = getattr(layer, "events", None)
        if events is None:
            return
        for event_name in ("data", "labels_update", "set_data"):
            emitter = getattr(events, event_name, None)
            if emitter is None:
                continue
            connected = cast(_EventEmitter, emitter)
            connected.connect(self.request_update)
            self._connections.append(_ConnectedEmitter(connected))

    def _disconnect_layer_events(self) -> None:
        """Disconnect all Labels event handlers owned by the controller."""
        for connection in self._connections:
            with suppress(RuntimeError, ValueError):
                connection.emitter.disconnect(self.request_update)
        self._connections.clear()
