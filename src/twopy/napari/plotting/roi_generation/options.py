"""ROI generation option types shared by UI and actions.

Inputs: none.
Outputs: plain data types for generated ROI templates.
"""

from dataclasses import dataclass, replace
from typing import Literal

from twopy.hdf5_utils import Hdf5Scalar

__all__ = [
    "DEFAULT_GRID_SIZE_PIXELS",
    "DEFAULT_MICRON_GRID_SIZE",
    "DEFAULT_RESPONSE_WATERSHED_CLOSING_RADIUS",
    "DEFAULT_RESPONSE_WATERSHED_FILL_HOLES",
    "DEFAULT_RESPONSE_WATERSHED_MIN_PIXELS",
    "DEFAULT_RESPONSE_WATERSHED_SMOOTHING_SIGMA",
    "DEFAULT_ROI_GENERATION_UNITS",
    "DEFAULT_WATERSHED_MIN_PIXELS",
    "DEFAULT_WATERSHED_SMOOTHING_SIGMA",
    "RoiGenerationMode",
    "RoiGenerationOptions",
    "RoiGenerationUnits",
    "default_roi_generation_options",
    "roi_generation_edited_after_generation",
    "roi_generation_metadata_from_options",
    "roi_generation_options_from_metadata",
]

RoiGenerationMode = Literal["manual", "grid", "watershed", "response_watershed"]
RoiGenerationUnits = Literal["pixels", "microns"]

DEFAULT_ROI_GENERATION_UNITS: RoiGenerationUnits = "pixels"
DEFAULT_GRID_SIZE_PIXELS = 16
DEFAULT_MICRON_GRID_SIZE = 10.0
DEFAULT_WATERSHED_MIN_PIXELS = 1
DEFAULT_WATERSHED_SMOOTHING_SIGMA = 0.0
DEFAULT_RESPONSE_WATERSHED_MIN_PIXELS = 5
DEFAULT_RESPONSE_WATERSHED_SMOOTHING_SIGMA = 0.0
DEFAULT_RESPONSE_WATERSHED_FILL_HOLES = True
DEFAULT_RESPONSE_WATERSHED_CLOSING_RADIUS = 0


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
        response_watershed_min_pixels: Minimum response-watershed ROI size.
        response_watershed_smoothing_sigma: Optional Gaussian smoothing sigma
            before response-watershed seeding.
        response_watershed_fill_holes: Whether response-watershed ROIs fill
            holes inside each connected component.
        response_watershed_closing_radius: Conservative binary closing radius
            before final response-watershed component splitting.
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
    response_watershed_min_pixels: int = DEFAULT_RESPONSE_WATERSHED_MIN_PIXELS
    response_watershed_smoothing_sigma: float = (
        DEFAULT_RESPONSE_WATERSHED_SMOOTHING_SIGMA
    )
    response_watershed_fill_holes: bool = DEFAULT_RESPONSE_WATERSHED_FILL_HOLES
    response_watershed_closing_radius: int = DEFAULT_RESPONSE_WATERSHED_CLOSING_RADIUS


def default_roi_generation_options(
    roi_mode: RoiGenerationMode = "manual",
) -> RoiGenerationOptions:
    """Return the default ROI-generation settings used by the ROIs tab.

    Args:
        roi_mode: ROI mode to use in the returned options.

    Returns:
        Complete generation options with mode-specific defaults.

    These values match the control defaults so callers can build manual
    provenance without reaching into Qt widgets.
    """
    return RoiGenerationOptions(
        roi_mode=roi_mode,
        units=DEFAULT_ROI_GENERATION_UNITS,
        grid_size_pixels=DEFAULT_GRID_SIZE_PIXELS,
        micron_grid_size=DEFAULT_MICRON_GRID_SIZE,
        rig="",
        calibration_mode=0,
        scanner="",
        zoom=1.0,
        allow_extrapolation=True,
        watershed_min_pixels=DEFAULT_WATERSHED_MIN_PIXELS,
        watershed_smoothing_sigma=DEFAULT_WATERSHED_SMOOTHING_SIGMA,
        response_watershed_min_pixels=DEFAULT_RESPONSE_WATERSHED_MIN_PIXELS,
        response_watershed_smoothing_sigma=(DEFAULT_RESPONSE_WATERSHED_SMOOTHING_SIGMA),
        response_watershed_fill_holes=DEFAULT_RESPONSE_WATERSHED_FILL_HOLES,
        response_watershed_closing_radius=(DEFAULT_RESPONSE_WATERSHED_CLOSING_RADIUS),
    )


def roi_generation_metadata_from_options(
    options: RoiGenerationOptions,
    *,
    edited_after_generation: bool = False,
) -> dict[str, Hdf5Scalar]:
    """Return mode-specific HDF5 metadata for generated ROI provenance.

    Args:
        options: ROI-generation settings that produced the current masks.
        edited_after_generation: Whether generated masks were edited by hand
            after creation.

    Returns:
        Scalar HDF5 attributes containing the ROI mode and only the settings
        used by that mode.
    """
    if options.roi_mode == "manual":
        return {"roi_mode": "manual"}
    metadata: dict[str, Hdf5Scalar]
    if options.roi_mode == "watershed":
        metadata = {
            "roi_mode": "watershed",
            "watershed_min_pixels": options.watershed_min_pixels,
            "watershed_smoothing_sigma": options.watershed_smoothing_sigma,
        }
    elif options.roi_mode == "response_watershed":
        metadata = {
            "roi_mode": "response_watershed",
            "response_watershed_min_pixels": options.response_watershed_min_pixels,
            "response_watershed_smoothing_sigma": (
                options.response_watershed_smoothing_sigma
            ),
            "response_watershed_fill_holes": options.response_watershed_fill_holes,
            "response_watershed_closing_radius": (
                options.response_watershed_closing_radius
            ),
        }
    elif options.units == "microns":
        metadata = {
            "roi_mode": "grid",
            "units": "microns",
            "micron_grid_size": options.micron_grid_size,
            "rig": options.rig,
            "calibration_mode": options.calibration_mode,
            "scanner": options.scanner,
            "zoom": options.zoom,
            "allow_extrapolation": options.allow_extrapolation,
        }
    else:
        metadata = {
            "roi_mode": "grid",
            "units": "pixels",
            "grid_size_pixels": options.grid_size_pixels,
        }
    if edited_after_generation:
        metadata["edited_after_generation"] = True
    return metadata


def roi_generation_edited_after_generation(
    metadata: dict[str, Hdf5Scalar] | None,
) -> bool:
    """Return whether saved generated ROI masks were edited after creation.

    Args:
        metadata: Optional scalar attributes read from a ROI HDF5 file.

    Returns:
        ``True`` when metadata explicitly marks post-generation edits.
    """
    if metadata is None:
        return False
    return _metadata_bool(metadata.get("edited_after_generation")) is True


def roi_generation_options_from_metadata(
    metadata: dict[str, Hdf5Scalar] | None,
) -> RoiGenerationOptions | None:
    """Parse saved ROI-generation metadata into complete options.

    Args:
        metadata: Optional scalar attributes read from a ROI HDF5 file.

    Returns:
        Complete ROI-generation options, or ``None`` when metadata is absent or
        malformed.

    Missing mode-specific fields make the whole metadata record unusable. That
    keeps the ROIs tab from showing defaults as if they were saved settings.
    """
    if metadata is None:
        return None
    roi_mode = _metadata_roi_mode(metadata.get("roi_mode"))
    if roi_mode is None:
        return None
    options = default_roi_generation_options(roi_mode)
    if roi_mode == "manual":
        return options
    if roi_mode == "watershed":
        min_pixels = _metadata_int(metadata.get("watershed_min_pixels"), minimum=1)
        smoothing = _metadata_float(
            metadata.get("watershed_smoothing_sigma"),
            minimum=0.0,
        )
        if min_pixels is None or smoothing is None:
            return None
        return replace(
            options,
            watershed_min_pixels=min_pixels,
            watershed_smoothing_sigma=smoothing,
        )
    if roi_mode == "response_watershed":
        min_pixels = _metadata_int(
            metadata.get("response_watershed_min_pixels"),
            minimum=1,
        )
        smoothing = _metadata_float(
            metadata.get("response_watershed_smoothing_sigma"),
            minimum=0.0,
        )
        fill_holes = _metadata_bool(metadata.get("response_watershed_fill_holes"))
        closing = _metadata_int(
            metadata.get("response_watershed_closing_radius"),
            minimum=0,
        )
        if (
            min_pixels is None
            or smoothing is None
            or fill_holes is None
            or closing is None
        ):
            return None
        return replace(
            options,
            response_watershed_min_pixels=min_pixels,
            response_watershed_smoothing_sigma=smoothing,
            response_watershed_fill_holes=fill_holes,
            response_watershed_closing_radius=closing,
        )

    units = _metadata_units(metadata.get("units"))
    if units is None:
        return None
    if units == "pixels":
        grid_size = _metadata_int(metadata.get("grid_size_pixels"), minimum=1)
        if grid_size is None:
            return None
        return replace(options, units="pixels", grid_size_pixels=grid_size)

    micron_grid_size = _metadata_float(metadata.get("micron_grid_size"), minimum=0.001)
    rig = _metadata_str(metadata.get("rig"))
    calibration_mode = _metadata_int(metadata.get("calibration_mode"), minimum=0)
    scanner = _metadata_str(metadata.get("scanner"))
    zoom = _metadata_float(metadata.get("zoom"), minimum=0.001)
    allow_extrapolation = _metadata_bool(metadata.get("allow_extrapolation"))
    if (
        micron_grid_size is None
        or rig is None
        or calibration_mode is None
        or scanner is None
        or zoom is None
        or allow_extrapolation is None
    ):
        return None
    return replace(
        options,
        units="microns",
        micron_grid_size=micron_grid_size,
        rig=rig,
        calibration_mode=calibration_mode,
        scanner=scanner,
        zoom=zoom,
        allow_extrapolation=allow_extrapolation,
    )


def _metadata_roi_mode(value: Hdf5Scalar | None) -> RoiGenerationMode | None:
    if value == "manual":
        return "manual"
    if value == "grid":
        return "grid"
    if value == "watershed":
        return "watershed"
    if value == "response_watershed":
        return "response_watershed"
    return None


def _metadata_units(value: Hdf5Scalar | None) -> RoiGenerationUnits | None:
    if value == "pixels":
        return "pixels"
    if value == "microns":
        return "microns"
    return None


def _metadata_str(value: Hdf5Scalar | None) -> str | None:
    if isinstance(value, str) and value != "":
        return value
    return None


def _metadata_bool(value: Hdf5Scalar | None) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _metadata_int(value: Hdf5Scalar | None, *, minimum: int) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= minimum:
        return value
    return None


def _metadata_float(value: Hdf5Scalar | None, *, minimum: float) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        numeric = float(value)
        if numeric >= minimum:
            return numeric
    return None
