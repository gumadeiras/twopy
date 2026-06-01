"""Cached response previews for Group Matching ROI assignment.

Inputs: selected ROI labels from loaded napari recordings plus preview options.
Outputs: plot-ready one-ROI response data with movie traces reused per recording.

The Group Matching ROI view asks for one selected ROI per recording. This cache
keeps those previews from rereading movie data when only another recording,
plot visibility, smoothing, or normalization changes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from twopy.analysis.response_plotting import ResponsePlotData
from twopy.analysis.response_processing import ResponseProcessingOptions
from twopy.converted import RecordingData
from twopy.napari.responses import (
    ResponseAnalysisRequest,
    response_analysis_request_from_selected_label_image,
)
from twopy.napari.roi_signatures import roi_mask_signature

__all__ = ["SelectedRoiResponseCache"]


@dataclass(frozen=True)
class _SelectedRoiResponseKey:
    """Cache key for one selected Group Matching response preview."""

    recording_path: Path
    movie_path: Path
    source_path: Path
    roi_label: str
    mask_signature: str
    response_processing_options: ResponseProcessingOptions


class _TraceCache(Protocol):
    """Small protocol for the shared live trace cache."""

    def compute_response_preview(
        self,
        request: ResponseAnalysisRequest,
    ) -> ResponsePlotData:
        """Return plot data for one response request."""
        ...


class SelectedRoiResponseCache:
    """Reuse selected-ROI response previews for Group Matching.

    Inputs: one selected ROI label from one loaded recording.
    Outputs: plot-ready response data.

    The cache stores final plot data keyed by the selected mask and processing
    options, and delegates trace reuse to ``LiveResponseAnalysisCache``. Trace
    caches retain absent labels because a user commonly toggles among candidate
    ROIs while matching a field of view.
    """

    def __init__(self) -> None:
        """Create an empty selected-response cache."""
        self._trace_caches: dict[Path, _TraceCache] = {}
        self._responses: dict[_SelectedRoiResponseKey, ResponsePlotData] = {}

    def clear(self) -> None:
        """Drop all cached traces and plot data."""
        self._trace_caches.clear()
        self._responses.clear()

    def retain_recordings(self, recording_paths: tuple[Path, ...]) -> None:
        """Drop cache entries for recordings no longer loaded."""
        keep_paths = {path.expanduser() for path in recording_paths}
        for path in tuple(self._trace_caches):
            if path not in keep_paths:
                del self._trace_caches[path]
        for key in tuple(self._responses):
            if key.recording_path not in keep_paths:
                del self._responses[key]

    def compute_selected_response(
        self,
        recording: RecordingData,
        label_image: npt.ArrayLike,
        *,
        roi_label: str,
        label_value: int,
        source_path: Path,
        response_processing_options: ResponseProcessingOptions,
    ) -> ResponsePlotData:
        """Return plot data for one selected ROI, using cached work if valid."""
        labels = np.asarray(label_image, dtype=np.int64)
        mask = labels == int(label_value)
        key = _SelectedRoiResponseKey(
            recording_path=recording.path.expanduser(),
            movie_path=recording.movie.path.expanduser(),
            source_path=source_path.expanduser(),
            roi_label=roi_label,
            mask_signature=roi_mask_signature(mask),
            response_processing_options=response_processing_options,
        )
        cached = self._responses.get(key)
        if cached is not None:
            return cached

        request = response_analysis_request_from_selected_label_image(
            recording,
            labels,
            label_value=label_value,
            source_path=source_path,
            response_processing_options=response_processing_options,
        )
        trace_cache = self._trace_caches.get(key.recording_path)
        if trace_cache is None:
            trace_cache = _new_trace_cache()
            self._trace_caches[key.recording_path] = trace_cache
        plot_data = trace_cache.compute_response_preview(request)
        self._responses[key] = plot_data
        return plot_data


def _new_trace_cache() -> _TraceCache:
    """Create the shared live trace cache without a napari import cycle."""
    from twopy.napari.live_analysis import LiveResponseAnalysisCache

    return LiveResponseAnalysisCache(drop_absent_traces=False)
