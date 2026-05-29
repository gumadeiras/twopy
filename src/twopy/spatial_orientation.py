"""Persisted spatial orientation markers for twopy-owned arrays.

Inputs: HDF5 groups or datasets that store converted movie-like spatial data.
Outputs: written or validated orientation attributes.

Converted twopy arrays use Python image order matching MATLAB display of the
source data. Persisting that fact beside movie, ROI, and response-map arrays
makes files without the current storage contract fail loudly instead of being
interpreted with guessed pixel geometry.
"""

from pathlib import Path

import h5py

__all__ = [
    "SOURCE_SPATIAL_ORIENTATION",
    "SPATIAL_ORIENTATION_ATTR",
    "SpatialOrientationError",
    "TWOPY_SPATIAL_ORIENTATION",
    "require_twopy_spatial_orientation",
    "write_twopy_spatial_orientation",
]

SPATIAL_ORIENTATION_ATTR = "spatial_orientation"
SOURCE_SPATIAL_ORIENTATION = "alignedMovie_imgFrames_ch1_source_axes"
TWOPY_SPATIAL_ORIENTATION = "matlab_display"


class SpatialOrientationError(ValueError):
    """Raised when persisted spatial data is missing the current orientation.

    Inputs: converted HDF5 metadata read during loading.
    Outputs: a typed failure for files that must be regenerated before use.
    """


def write_twopy_spatial_orientation(h5_object: h5py.Group | h5py.Dataset) -> None:
    """Mark one persisted spatial object as using current twopy orientation.

    Args:
        h5_object: HDF5 group or dataset that stores converted spatial data.

    Returns:
        None.
    """
    h5_object.attrs[SPATIAL_ORIENTATION_ATTR] = TWOPY_SPATIAL_ORIENTATION


def require_twopy_spatial_orientation(
    h5_object: h5py.Group | h5py.Dataset,
    *,
    path: Path,
    label: str,
) -> None:
    """Require one persisted spatial object to use current twopy orientation.

    Args:
        h5_object: HDF5 group or dataset to validate.
        path: File path used in the error message.
        label: Plain-language object name used in the error message.

    Returns:
        None.

    Raises:
        ValueError: If the object is missing the marker or names another
            orientation.
    """
    value = h5_object.attrs.get(SPATIAL_ORIENTATION_ATTR)
    if value == TWOPY_SPATIAL_ORIENTATION:
        return
    msg = (
        f"File {path} has missing or unsupported spatial orientation on "
        f"{label!r}. Regenerate the file from current twopy data so spatial "
        "arrays use Python image order matching MATLAB display."
    )
    raise SpatialOrientationError(msg)
