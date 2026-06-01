"""Live response updates for napari ROI editing.

Inputs: the selected converted recording, its editable Labels layer, and the
response plot widget.
Outputs: debounced plot refreshes as ROI pixels change.

The controller owns GUI orchestration only. It copies Labels pixels on the GUI
thread, then runs the heavier response calculation through the same in-memory
analysis helper used by the manual update button.
"""

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from threading import Event
from typing import Protocol, cast

import numpy as np
from qtpy.QtCore import QTimer

from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.response_plotting import ResponsePlotData
from twopy.analysis.response_processing import ResponseProcessingOptions
from twopy.analysis.response_window_options import ResponseWindowOptions
from twopy.converted import RecordingData
from twopy.napari.live_analysis import LiveResponseAnalysisCache
from twopy.napari.protocols import NapariEventEmitter
from twopy.napari.responses import (
    compute_response_preview,
    response_analysis_request_from_label_image,
)
from twopy.napari.roi import roi_label_image_from_layer_for_recording

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


@dataclass(frozen=True)
class _LiveResponseJob:
    """One cancellable background response computation."""

    version: int
    cancel_event: Event

    def cancel(self) -> None:
        """Mark this job obsolete."""
        self.cancel_event.set()

    def check_cancelled(self) -> None:
        """Raise when this job has been superseded by a newer edit."""
        if self.cancel_event.is_set():
            raise CancelledError()


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
        debounce_ms: int = 200,
        run_async: bool = True,
    ) -> None:
        """Create a live updater for one response plot widget.

        Args:
            plot_widget: Widget that receives plot data or status messages.
            debounce_ms: Delay after the latest ROI edit before recomputing.
            run_async: Whether heavy analysis should run in a worker thread.
                Tests can turn this off to verify behavior synchronously.
        """
        self._plot_widget: ResponsePlotReceiver | None = plot_widget
        self._recording: RecordingData | None = None
        self._roi_labels_layer: object | None = None
        self._delta_f_over_f_options = DeltaFOverFOptions()
        self._response_window_options = ResponseWindowOptions()
        self._response_processing_options = ResponseProcessingOptions()
        self._connections: list[NapariEventEmitter] = []
        self._is_shutdown = False
        self._version = 0
        self._active_version: int | None = None
        self._active_job: _LiveResponseJob | None = None
        self._pending_after_active_job = False
        self._future: Future[ResponsePlotData] | None = None
        self._analysis_cache = LiveResponseAnalysisCache()
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="twopy-roi-live")
            if run_async
            else None
        )
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._start_latest_job)
        self._poll_timer = QTimer()
        self._poll_timer.setInterval(30)
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
        if self._is_shutdown:
            return
        self._cancel_active_job()
        self._recording = recording
        self._roi_labels_layer = roi_labels_layer
        self._analysis_cache = LiveResponseAnalysisCache()
        self._version += 1
        self._pending_after_active_job = False
        self._connect_layer_events(roi_labels_layer)

    def set_response_processing_options(
        self,
        options: ResponseProcessingOptions,
    ) -> None:
        """Store processing options used by the next live response update.

        Args:
            options: GUI-selected response-processing settings.

        Returns:
            None.
        """
        if self._is_shutdown:
            return
        if options == self._response_processing_options:
            return
        self._response_processing_options = options
        self._version += 1

    def set_delta_f_over_f_options(self, options: DeltaFOverFOptions) -> None:
        """Store dF/F options used by the next live response update.

        Args:
            options: GUI-selected dF/F analysis settings.

        Returns:
            None.
        """
        if self._is_shutdown:
            return
        if options == self._delta_f_over_f_options:
            return
        self._delta_f_over_f_options = options
        self._version += 1

    def set_response_window_options(self, options: ResponseWindowOptions) -> None:
        """Store response-window options used by the next live update.

        Args:
            options: GUI-selected automatic/manual response-window settings.

        Returns:
            None.
        """
        if self._is_shutdown:
            return
        if options == self._response_window_options:
            return
        self._response_window_options = options
        self._version += 1

    def request_update(self, *_event_args: object) -> None:
        """Schedule a live response update after an ROI edit.

        Args:
            _event_args: Optional napari event arguments. They are ignored
                because the Labels layer itself is the source of truth.

        Returns:
            None.
        """
        if self._is_shutdown:
            return
        self._version += 1
        self._cancel_active_job()
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
        if self._is_shutdown:
            return
        self._version += 1
        version = self._version
        try:
            plot_data = self._compute_current_plot_data()
        except ValueError as error:
            if version == self._version:
                self._show_response_status(str(error))
            return
        if version == self._version:
            self._set_response_plot_data(plot_data)

    def shutdown(self) -> None:
        """Disconnect layer events and stop background work.

        Args:
            None.

        Returns:
            None.
        """
        if self._is_shutdown:
            return
        self._is_shutdown = True
        self._version += 1
        self._pending_after_active_job = False
        future = self._future
        self._future = None
        self._active_version = None
        self._cancel_active_job()
        if future is not None:
            future.cancel()
        self._disconnect_layer_events()
        self._debounce_timer.stop()
        self._poll_timer.stop()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._recording = None
        self._roi_labels_layer = None
        self._plot_widget = None

    def _start_latest_job(self) -> None:
        """Start the newest queued response computation."""
        if self._is_shutdown:
            return
        if self._executor is None:
            self.update_now()
            return
        if self._future is not None:
            self._pending_after_active_job = True
            self._cancel_active_job()
            return

        version = self._version
        try:
            request = response_analysis_request_from_label_image(
                self._require_recording(),
                self._current_label_image(),
                source_path=None,
                delta_f_over_f_options=self._delta_f_over_f_options,
                response_window_options=self._response_window_options,
                response_processing_options=self._response_processing_options,
            )
        except ValueError as error:
            if version == self._version:
                self._show_response_status(str(error))
            return

        job = _LiveResponseJob(version=version, cancel_event=Event())
        self._active_job = job
        self._active_version = version
        self._future = self._executor.submit(
            self._analysis_cache.compute_response_preview,
            request,
            check_cancelled=job.check_cancelled,
        )
        self._poll_timer.start()

    def _collect_finished_job(self) -> None:
        """Apply a finished worker result when it is still current."""
        if self._is_shutdown:
            return
        future = self._future
        if future is None or not future.done():
            return

        version = self._active_version
        self._future = None
        self._active_version = None
        self._active_job = None
        self._poll_timer.stop()
        try:
            plot_data = future.result()
        except CancelledError:
            pass
        except ValueError as error:
            if version == self._version:
                self._show_response_status(str(error))
        else:
            if version == self._version:
                self._set_response_plot_data(plot_data)

        if self._pending_after_active_job or version != self._version:
            self._pending_after_active_job = False
            if not self._debounce_timer.isActive():
                self._start_latest_job()

    def _cancel_active_job(self) -> None:
        """Ask the current worker job to stop before it finishes stale work."""
        if self._active_job is not None:
            self._active_job.cancel()

    def _compute_current_plot_data(self) -> ResponsePlotData:
        """Return plot data for the current recording and Labels layer."""
        request = response_analysis_request_from_label_image(
            self._require_recording(),
            self._current_label_image(),
            source_path=None,
            delta_f_over_f_options=self._delta_f_over_f_options,
            response_window_options=self._response_window_options,
            response_processing_options=self._response_processing_options,
        )
        return compute_response_preview(request)

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

    def _set_response_plot_data(self, plot_data: ResponsePlotData) -> None:
        """Send computed plot data to the receiver when it is still alive."""
        if self._plot_widget is not None:
            self._plot_widget.set_response_plot_data(plot_data, reset_axes=True)

    def _show_response_status(self, text: str) -> None:
        """Send status text to the receiver when it is still alive."""
        if self._plot_widget is not None:
            self._plot_widget.show_response_status(text)

    def _connect_layer_events(self, layer: object | None) -> None:
        """Connect Labels data-change events to debounced updates."""
        if layer is None:
            return
        events = getattr(layer, "events", None)
        if events is None:
            return
        for event_name in ("paint", "data"):
            emitter = getattr(events, event_name, None)
            if emitter is None:
                continue
            connected = cast(NapariEventEmitter, emitter)
            connected.connect(self.request_update)
            self._connections.append(connected)

    def _disconnect_layer_events(self) -> None:
        """Disconnect all Labels event handlers owned by the controller."""
        for connection in self._connections:
            with suppress(RuntimeError, ValueError):
                connection.disconnect(self.request_update)
        self._connections.clear()
