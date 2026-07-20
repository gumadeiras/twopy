"""Read-only parity helpers for comparing twopy outputs to prior analyses.

Inputs: optional lab-generated reference outputs.
Outputs: small typed objects for comparison scripts and tests.

Normal twopy analysis should not import this package. Analysis starts from
converted twopy objects. Parity helpers only read prior outputs to check that
the converted-object analysis matches historical results when needed.
"""

from twopy.parity.dff import (
    SavedAnalysisDeltaFOverFComparison,
    compare_saved_analysis_delta_f_over_f,
    saved_analysis_epoch_windows,
)
from twopy.parity.psycho5 import (
    analyze_psycho5_default_recording_responses,
    compute_psycho5_default_recording_responses,
    psycho5_default_interleave_epoch_numbers,
)
from twopy.parity.roi import SavedAnalysisRoiSet, roi_set_from_saved_analysis
from twopy.parity.roi_extraction import (
    psycho5_grid_pixel_size,
    psycho5_grid_roi_label_image,
    psycho5_grid_roi_set,
    psycho5_watershed_image_from_preseg,
    psycho5_watershed_roi_set_from_preseg,
)
from twopy.parity.saved_analysis import (
    SavedAnalysisLastRoi,
    load_saved_analysis_last_roi,
)

__all__ = [
    "SavedAnalysisDeltaFOverFComparison",
    "SavedAnalysisLastRoi",
    "SavedAnalysisRoiSet",
    "analyze_psycho5_default_recording_responses",
    "compare_saved_analysis_delta_f_over_f",
    "compute_psycho5_default_recording_responses",
    "load_saved_analysis_last_roi",
    "psycho5_grid_pixel_size",
    "psycho5_grid_roi_label_image",
    "psycho5_grid_roi_set",
    "psycho5_default_interleave_epoch_numbers",
    "psycho5_watershed_image_from_preseg",
    "psycho5_watershed_roi_set_from_preseg",
    "roi_set_from_saved_analysis",
    "saved_analysis_epoch_windows",
]
