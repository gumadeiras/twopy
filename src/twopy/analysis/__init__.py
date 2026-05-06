"""Analysis helpers for converted two-photon recordings.

Inputs: converted twopy recordings, ROI traces, photodiode-aligned windows, and
analysis parameters.
Outputs: background-corrected traces and trial/frame-window response objects.

This package is GUI-independent and operates only on twopy-owned converted
files or in-memory objects derived from them.
"""

from twopy.analysis.background_subtraction import (
    BackgroundCorrectedRoiTraces,
    BackgroundCorrectionMethod,
    extract_background_corrected_roi_traces,
)
from twopy.analysis.trials import (
    FrameWindow,
    WindowedRoiResponse,
    frame_windows_from_photodiode_alignment,
    make_frame_windows,
    split_traces_by_frame_windows,
)

__all__ = [
    "BackgroundCorrectedRoiTraces",
    "BackgroundCorrectionMethod",
    "extract_background_corrected_roi_traces",
    "frame_windows_from_photodiode_alignment",
    "FrameWindow",
    "make_frame_windows",
    "split_traces_by_frame_windows",
    "WindowedRoiResponse",
]
