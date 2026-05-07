"""Typed dF/F analysis settings shared by GUI and workflow callers.

Inputs: user-facing dF/F choices such as background correction and baseline
sampling.
Outputs: one immutable options object that can be passed through GUI adapters
without hiding the scientific workflow parameters.
"""

from dataclasses import dataclass

from twopy.analysis.background_subtraction import BackgroundCorrectionMethod
from twopy.analysis.dff import DeltaFOverFFitMode

__all__ = ["DeltaFOverFOptions"]


@dataclass(frozen=True)
class DeltaFOverFOptions:
    """Options that control ROI dF/F computation before response grouping.

    Inputs: baseline epoch selection, background correction method, baseline
    window sampling, dF/F fit mode, and motion-artifact masking choice.
    Outputs: typed parameters that map directly to
    ``compute_recording_responses`` and ``analyze_recording_responses``.

    The object exists so GUI code can keep dF/F decisions separate from later
    response processing choices such as smoothing and correlation filtering.
    """

    baseline_epoch_number: int = 1
    baseline_epoch_name: str | None = None
    background_method: BackgroundCorrectionMethod = "movie_global_percentile"
    baseline_sample_seconds: float | None = 1.0
    fit_mode: DeltaFOverFFitMode = "direct_bounded_tau"
    apply_motion_mask: bool = True
