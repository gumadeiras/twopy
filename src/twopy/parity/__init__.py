"""Read-only parity helpers for comparing twopy outputs to prior analyses.

Inputs: optional lab-generated reference outputs.
Outputs: small typed objects for audit scripts and tests.

Normal twopy analysis should not import this package. Analysis starts from
converted twopy objects; parity helpers only read prior outputs to check that
the converted-object analysis matches historical results when needed.
"""

from twopy.parity.dff import (
    SavedAnalysisDeltaFOverFComparison,
    compare_saved_analysis_delta_f_over_f,
    saved_analysis_epoch_windows,
)
from twopy.parity.roi import SavedAnalysisRoiSet, roi_set_from_saved_analysis
from twopy.parity.saved_analysis import (
    SavedAnalysisLastRoi,
    load_saved_analysis_last_roi,
)

__all__ = [
    "SavedAnalysisDeltaFOverFComparison",
    "SavedAnalysisLastRoi",
    "SavedAnalysisRoiSet",
    "compare_saved_analysis_delta_f_over_f",
    "load_saved_analysis_last_roi",
    "roi_set_from_saved_analysis",
    "saved_analysis_epoch_windows",
]
