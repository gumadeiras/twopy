"""ROI generation option types shared by UI and actions.

Inputs: none.
Outputs: plain data types for generated ROI templates.
"""

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "RoiGenerationOptions",
    "RoiGenerationUnits",
]

RoiGenerationUnits = Literal["pixels", "microns"]


@dataclass(frozen=True)
class RoiGenerationOptions:
    """Requested ROI grid-generation settings from the ROIs tab.

    Args:
        units: Whether the grid size is specified in pixels or microns.
        grid_size_pixels: Pixel grid width used when ``units`` is ``pixels``.
        micron_grid_size: Physical grid width used when ``units`` is
            ``microns``.
        rig: Calibration rig key.
        mode: Calibration scanner mode key.
        scanner: Calibration scanner family key.
        zoom: Requested zoom setting.
        allow_extrapolation: Whether calibration may extrapolate outside the
            measured zoom range.
    """

    units: RoiGenerationUnits
    grid_size_pixels: int
    micron_grid_size: float
    rig: str
    mode: int
    scanner: str
    zoom: float
    allow_extrapolation: bool
