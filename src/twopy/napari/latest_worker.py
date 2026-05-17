"""Latest-only Qt worker helper for napari widgets.

Inputs: background jobs plus result and error callbacks owned by a widget.
Outputs: debounced, single-worker execution whose stale results are ignored.

This module owns only generic Qt worker lifecycle mechanics. Callers still own
when work should be scheduled, what the work computes, and how results update
domain-specific UI state.
"""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from qtpy.QtCore import QTimer


class LatestWorker[T]:
    """Run only the latest scheduled background job.

    Args:
        thread_name_prefix: Name prefix for the single worker thread.
        on_result: Callback for the latest successful result.
        on_value_error: Callback for latest ``ValueError`` failures.
        on_exception: Callback for latest unexpected failures.
        debounce_ms: Delay after scheduling before work starts.
        poll_ms: Interval used to collect a finished future on the Qt thread.

    The helper keeps at most one active future. If newer work is scheduled while
    a job is running, the active result is ignored when stale and one latest job
    starts after the running job finishes.
    """

    def __init__(
        self,
        *,
        thread_name_prefix: str,
        on_result: Callable[[T], None],
        on_value_error: Callable[[ValueError], None],
        on_exception: Callable[[Exception], None],
        debounce_ms: int = 200,
        poll_ms: int = 30,
    ) -> None:
        """Create a latest-only worker."""
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=thread_name_prefix,
        )
        self._future: Future[T] | None = None
        self._job: Callable[[], T] | None = None
        self._version = 0
        self._active_version: int | None = None
        self._pending_after_active_job = False
        self._is_shutdown = False
        self._on_result = on_result
        self._on_value_error = on_value_error
        self._on_exception = on_exception

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._start_latest_job)
        self._poll_timer = QTimer()
        self._poll_timer.setInterval(poll_ms)
        self._poll_timer.timeout.connect(self.collect_finished_job)
        self._debounce_ms = debounce_ms

    def schedule(self, job: Callable[[], T]) -> None:
        """Schedule a background job after the debounce interval.

        Args:
            job: Zero-argument callable to run in the worker thread.

        Returns:
            None.
        """
        if self._is_shutdown:
            return
        self._version += 1
        self._job = job
        self._debounce_timer.start(self._debounce_ms)

    def invalidate(self) -> None:
        """Discard queued work and ignore any active job result."""
        self._version += 1
        self._pending_after_active_job = False
        self._job = None
        self._debounce_timer.stop()
        if self._future is not None and self._future.cancel():
            self._future = None
            self._active_version = None
            self._poll_timer.stop()

    def shutdown(self) -> None:
        """Stop timers, cancel queued work, and release the worker thread."""
        if self._is_shutdown:
            return
        self._is_shutdown = True
        self.invalidate()
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self._poll_timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def collect_finished_job(self) -> None:
        """Apply one finished worker result when it is still current."""
        if self._is_shutdown:
            return
        future = self._future
        if future is None or not future.done():
            return

        version = self._active_version
        self._future = None
        self._active_version = None
        self._poll_timer.stop()
        try:
            result = future.result()
        except ValueError as error:
            if version == self._version:
                self._on_value_error(error)
        except Exception as error:
            if version == self._version:
                self._on_exception(error)
        else:
            if version == self._version:
                self._on_result(result)

        if self._pending_after_active_job:
            self._pending_after_active_job = False
            if not self._debounce_timer.isActive():
                self._start_latest_job()

    def _start_latest_job(self) -> None:
        """Start the latest queued job if the worker is idle."""
        if self._is_shutdown:
            return
        if self._future is not None:
            self._pending_after_active_job = True
            return
        if self._job is None:
            return
        self._active_version = self._version
        self._future = self._executor.submit(self._job)
        self._poll_timer.start()
