"""Reuse ROI traces while a user edits napari Labels.

Inputs: one response-analysis request from the napari UI.
Outputs: plot-ready response data, reusing movie traces when the recording,
frame range, background method, and ROI masks have not changed.

The cache only avoids repeated movie reads. Each update still reruns dF/F,
response grouping, normalization, and QC through the normal analysis helpers.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twopy.analysis.background_subtraction import (
    BackgroundCorrectedRoiTraces,
    BackgroundCorrectionMethod,
    extract_background_corrected_roi_traces,
)
from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.response_context import (
    ResponseAnalysisContext,
    resolve_response_analysis_context,
)
from twopy.analysis.response_processing import ResponseProcessingOptions
from twopy.analysis.response_window_options import ResponseWindowOptions
from twopy.analysis.workflow import compute_recording_responses_from_traces
from twopy.converted import RecordingData
from twopy.napari.plotting.data import ResponsePlotData
from twopy.napari.responses import (
    ResponseAnalysisRequest,
    response_plot_data_from_computation,
)
from twopy.napari.roi_signatures import roi_mask_signature
from twopy.roi import RoiSet, make_roi_set

__all__ = ["LiveResponseAnalysisCache", "roi_mask_signature"]


@dataclass(frozen=True)
class _TraceContextKey:
    """Cache key for one recording and trace-extraction setup."""

    recording_path: Path
    movie_path: Path
    trace_start: int
    trace_stop: int
    background_method: str


@dataclass(frozen=True)
class _ResponseContextCacheKey:
    """Cache key for resolved response timing and processing context."""

    recording_path: Path
    movie_path: Path
    baseline_epoch_number: int | None
    baseline_epoch_name: str | None
    baseline_mode: str
    response_window_options: ResponseWindowOptions
    response_processing_options: ResponseProcessingOptions


@dataclass(frozen=True)
class _PreparedRoiSet:
    """Current ROI masks plus stable mask signatures."""

    roi_set: RoiSet
    labels: tuple[str, ...]
    signatures: tuple[str, ...]


@dataclass(frozen=True)
class _CachedTrace:
    """One cached ROI trace and the mask signature that produced it."""

    signature: str
    trace: BackgroundCorrectedRoiTraces


class LiveResponseAnalysisCache:
    """Reuse movie traces without reusing finished response results.

    Inputs: the same response-analysis request used by preview and Save Analysis.
    Outputs: plot-ready response data.

    The cache stores only raw/background-corrected ROI traces. Each live update
    still recomputes dF/F, response grouping, normalization, and QC for the full
    current ROI set.
    """

    def __init__(self, *, drop_absent_traces: bool = True) -> None:
        """Create an empty live-analysis trace cache."""
        self._context_key: _TraceContextKey | None = None
        self._traces_by_label: dict[str, _CachedTrace] = {}
        self._contexts_by_key: dict[
            _ResponseContextCacheKey,
            ResponseAnalysisContext,
        ] = {}
        self._drop_absent_traces_enabled = drop_absent_traces

    def clear(self) -> None:
        """Remove all cached ROI traces.

        Args:
            None.

        Returns:
            None.
        """
        self._context_key = None
        self._traces_by_label.clear()
        self._contexts_by_key.clear()

    def compute_response_preview(
        self,
        request: ResponseAnalysisRequest,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ) -> ResponsePlotData:
        """Compute plot data while reusing unchanged ROI traces.

        Args:
            request: Response-analysis request.
            check_cancelled: Optional callback that raises when work is obsolete.

        Returns:
            Plot-ready response data for all current ROIs.
        """
        dff_options = request.delta_f_over_f_options
        context = self._response_context(request)
        current = _prepare_roi_set(request.roi_set)
        context_key = _trace_context_key(request.recording, context, dff_options)
        if context_key != self._context_key:
            self._clear_traces()
            self._context_key = context_key
        if check_cancelled is not None:
            check_cancelled()

        changed_indices = self._changed_trace_indices(current, dff_options)
        if len(changed_indices) > 0:
            self._refresh_traces(
                request.recording,
                current,
                changed_indices=changed_indices,
                context=context,
                background_method=dff_options.background_method,
                check_cancelled=check_cancelled,
            )
        if self._drop_absent_traces_enabled:
            self._drop_absent_traces(current.labels)
        traces = _combine_cached_traces(self._cached_traces(current))
        computation = compute_recording_responses_from_traces(
            request.recording,
            current.roi_set,
            traces,
            context=context,
            baseline_sample_seconds=dff_options.baseline_sample_seconds,
            fit_mode=dff_options.fit_mode,
            apply_motion_mask=dff_options.apply_motion_mask,
            check_cancelled=check_cancelled,
        )
        return response_plot_data_from_computation(request, computation)

    def _clear_traces(self) -> None:
        """Remove cached ROI traces while keeping reusable timing contexts."""
        self._context_key = None
        self._traces_by_label.clear()

    def _response_context(
        self,
        request: ResponseAnalysisRequest,
    ) -> ResponseAnalysisContext:
        """Return cached response-analysis context for one request."""
        dff_options = request.delta_f_over_f_options
        key = _ResponseContextCacheKey(
            recording_path=request.recording.path,
            movie_path=request.recording.movie.path,
            baseline_epoch_number=dff_options.baseline_epoch_number,
            baseline_epoch_name=dff_options.baseline_epoch_name,
            baseline_mode=dff_options.baseline_mode,
            response_window_options=request.response_window_options,
            response_processing_options=request.response_processing_options,
        )
        cached = self._contexts_by_key.get(key)
        if cached is not None:
            return cached

        pre_window_seconds, post_window_seconds = request.response_window_seconds()
        context = resolve_response_analysis_context(
            request.recording,
            baseline_mode=dff_options.baseline_mode,
            baseline_epoch_number=dff_options.baseline_epoch_number,
            baseline_epoch_name=dff_options.baseline_epoch_name,
            response_pre_window_seconds=pre_window_seconds,
            response_post_window_seconds=post_window_seconds,
            response_processing_options=request.response_processing_options,
        )
        self._contexts_by_key[key] = context
        return context

    def _changed_trace_indices(
        self,
        current: _PreparedRoiSet,
        dff_options: DeltaFOverFOptions,
    ) -> tuple[int, ...]:
        """Return ROI indices whose cached traces are absent or stale."""
        stale_indices = self._stale_trace_indices(current)
        if dff_options.background_method == "roi_y_stripe_percentile":
            if len(stale_indices) == 0 and self._cached_labels() == set(current.labels):
                return ()
            return tuple(range(len(current.labels)))
        return stale_indices

    def _cached_labels(self) -> set[str]:
        """Return the ROI labels represented by cached traces."""
        return set(self._traces_by_label)

    def _stale_trace_indices(self, current: _PreparedRoiSet) -> tuple[int, ...]:
        """Return ROI indices missing from the cache or using stale masks."""
        changed: list[int] = []
        for index, (label, signature) in enumerate(
            zip(current.labels, current.signatures, strict=True)
        ):
            cached = self._traces_by_label.get(label)
            if cached is None or cached.signature != signature:
                changed.append(index)
        return tuple(changed)

    def _cached_traces(
        self,
        current: _PreparedRoiSet,
    ) -> tuple[BackgroundCorrectedRoiTraces, ...]:
        """Return cached traces in current ROI order."""
        return tuple(self._traces_by_label[label].trace for label in current.labels)

    def _refresh_traces(
        self,
        recording: RecordingData,
        current: _PreparedRoiSet,
        *,
        changed_indices: tuple[int, ...],
        context: ResponseAnalysisContext,
        background_method: BackgroundCorrectionMethod,
        check_cancelled: Callable[[], None] | None,
    ) -> None:
        """Extract traces for changed ROIs and update the cache."""
        changed_roi_set = _select_roi_set(current.roi_set, changed_indices)
        traces = extract_background_corrected_roi_traces(
            recording,
            changed_roi_set,
            method=background_method,
            start_frame=context.trace_start,
            stop_frame=context.trace_stop,
            spatial_domain="alignment_valid_crop",
            check_cancelled=check_cancelled,
        )
        for output_index, source_index in enumerate(changed_indices):
            label = current.labels[source_index]
            self._traces_by_label[label] = _CachedTrace(
                signature=current.signatures[source_index],
                trace=_single_roi_trace(traces, output_index),
            )

    def _drop_absent_traces(self, labels: tuple[str, ...]) -> None:
        """Remove cache entries for deleted ROI labels."""
        current_labels = set(labels)
        for label in tuple(self._traces_by_label):
            if label not in current_labels:
                del self._traces_by_label[label]


def _trace_context_key(
    recording: RecordingData,
    context: ResponseAnalysisContext,
    dff_options: DeltaFOverFOptions,
) -> _TraceContextKey:
    """Return the cache key for one live update."""
    return _TraceContextKey(
        recording_path=recording.path,
        movie_path=recording.movie.path,
        trace_start=context.trace_start,
        trace_stop=context.trace_stop,
        background_method=dff_options.background_method,
    )


def _prepare_roi_set(roi_set: RoiSet) -> _PreparedRoiSet:
    """Return current ROI masks and stable signatures from a ROI set."""
    return _PreparedRoiSet(
        roi_set=roi_set,
        labels=roi_set.labels,
        signatures=tuple(roi_mask_signature(mask) for mask in roi_set.masks),
    )


def _select_roi_set(roi_set: RoiSet, indices: tuple[int, ...]) -> RoiSet:
    """Return a ROI set containing only selected ROI rows."""
    selected = list(indices)
    return make_roi_set(
        roi_set.masks[selected, :, :],
        labels=tuple(roi_set.labels[index] for index in indices),
    )


def _single_roi_trace(
    traces: BackgroundCorrectedRoiTraces,
    index: int,
) -> BackgroundCorrectedRoiTraces:
    """Return one ROI column as an independent cached trace object."""
    column = slice(index, index + 1)
    return BackgroundCorrectedRoiTraces(
        raw_values=traces.raw_values[:, column].copy(),
        background_values=traces.background_values[:, column].copy(),
        corrected_values=traces.corrected_values[:, column].copy(),
        labels=(traces.labels[index],),
        start_frame=traces.start_frame,
        stop_frame=traces.stop_frame,
        statistic=traces.statistic,
        method=traces.method,
        metadata=dict(traces.metadata),
    )


def _combine_cached_traces(
    traces: tuple[BackgroundCorrectedRoiTraces, ...],
) -> BackgroundCorrectedRoiTraces:
    """Combine one-ROI cached traces into one multi-ROI trace object."""
    if len(traces) == 0:
        msg = "At least one cached ROI trace is required."
        raise ValueError(msg)
    first = traces[0]
    for trace in traces:
        if (
            trace.start_frame != first.start_frame
            or trace.stop_frame != first.stop_frame
        ):
            msg = "Cached traces must share one frame range."
            raise ValueError(msg)
        if trace.method != first.method or trace.statistic != first.statistic:
            msg = "Cached traces must share one extraction method and statistic."
            raise ValueError(msg)
    frame_count = first.stop_frame - first.start_frame
    roi_count = len(traces)
    raw_values = np.empty((frame_count, roi_count), dtype=np.float64)
    background_values = np.empty((frame_count, roi_count), dtype=np.float64)
    corrected_values = np.empty((frame_count, roi_count), dtype=np.float64)
    for index, trace in enumerate(traces):
        raw_values[:, index] = trace.raw_values[:, 0]
        background_values[:, index] = trace.background_values[:, 0]
        corrected_values[:, index] = trace.corrected_values[:, 0]
    labels = tuple(label for trace in traces for label in trace.labels)
    return BackgroundCorrectedRoiTraces(
        raw_values=raw_values,
        background_values=background_values,
        corrected_values=corrected_values,
        labels=labels,
        start_frame=first.start_frame,
        stop_frame=first.stop_frame,
        statistic=first.statistic,
        method=first.method,
        metadata=dict(first.metadata),
    )
