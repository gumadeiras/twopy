"""Read-only parity helpers for comparing twopy outputs to prior analyses.

Inputs: optional lab-generated reference outputs.
Outputs: small typed objects for audit scripts and tests.

Normal twopy analysis should not import this package. Analysis starts from
converted twopy objects; parity helpers only read prior outputs to check that
the converted-object analysis matches historical results when needed.
"""

from twopy.parity.saved_analysis import (
    SavedAnalysisLastRoi,
    load_saved_analysis_last_roi,
)

__all__ = [
    "SavedAnalysisLastRoi",
    "load_saved_analysis_last_roi",
]
