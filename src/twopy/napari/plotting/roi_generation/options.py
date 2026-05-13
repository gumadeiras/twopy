"""ROI generation option types shared by UI and actions.

Inputs: none.
Outputs: plain data types for generated ROI templates.
"""

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "RoiGenerationMode",
    "RoiGenerationOptions",
    "RoiGenerationUnits",
]

RoiGenerationMode = Literal["manual", "grid", "watershed"]
RoiGenerationUnits = Literal["pixels", "microns"]


@dataclass(frozen=True)
class RoiGenerationOptions:
    """Requested ROI generation settings from the ROIs tab.

    Args:
        roi_mode: Current ROI interaction mode.
        units: Whether the grid size is specified in pixels or microns.
        grid_size_pixels: Pixel grid width used when ``units`` is ``pixels``.
        micron_grid_size: Physical grid width used when ``units`` is
            ``microns``.
        rig: Calibration rig key.
        calibration_mode: Calibration scanner mode key.
        scanner: Calibration scanner family key.
        zoom: Requested zoom setting.
        allow_extrapolation: Whether calibration may extrapolate outside the
            measured zoom range.
        watershed_min_pixels: Minimum watershed ROI size.
        watershed_smoothing_sigma: Optional Gaussian smoothing sigma before
            watershed seeding.
    """

    roi_mode: RoiGenerationMode
    units: RoiGenerationUnits
    grid_size_pixels: int
    micron_grid_size: float
    rig: str
    calibration_mode: int
    scanner: str
    zoom: float
    allow_extrapolation: bool
    watershed_min_pixels: int
    watershed_smoothing_sigma: float
