"""Source-to-twopy spatial axis helpers.

Inputs: source MATLAB movie arrays and source spatial shapes.
Outputs: arrays and shapes in twopy's stored image orientation.

twopy stores converted movies in normal Python image order, with rows before
columns. Python exposes source ``alignedMovie.mat`` spatial axes in the opposite
order from the image shown by MATLAB, so conversion swaps those axes once before
any data becomes twopy-owned.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "orient_source_array_to_twopy",
    "twopy_shape_from_source_movie_shape",
    "twopy_spatial_shape_from_source_spatial_shape",
]


def orient_source_array_to_twopy(
    values: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Return source image or movie data in twopy image axis order.

    Args:
        values: Two-dimensional source image or three-dimensional source movie
            block whose final two axes are source spatial axes.

    Returns:
        View with the final two axes swapped so Python image display matches
        MATLAB display of the source data.
    """
    array = np.asarray(values)
    if array.ndim not in (2, 3):
        msg = f"Expected a 2D image or 3D movie block. Got {array.shape}"
        raise ValueError(msg)
    return np.swapaxes(array, -1, -2)


def twopy_shape_from_source_movie_shape(
    source_shape: Sequence[int],
) -> tuple[int, int, int]:
    """Return converted movie shape for one source movie shape.

    Args:
        source_shape: Source ``alignedMovie.mat`` movie shape.

    Returns:
        ``(frames, y, x)`` shape for stored Python image data.
    """
    if len(source_shape) != 3:
        msg = f"Source movie shape must have 3 dimensions. Got {tuple(source_shape)}"
        raise ValueError(msg)
    return (int(source_shape[0]), int(source_shape[2]), int(source_shape[1]))


def twopy_spatial_shape_from_source_spatial_shape(
    source_spatial_shape: Sequence[int],
) -> tuple[int, int]:
    """Return converted spatial shape for source ``(x, y)`` dimensions.

    Args:
        source_spatial_shape: Two source spatial lengths from
            ``alignedMovie.mat``.

    Returns:
        ``(y, x)`` shape for stored Python images.
    """
    if len(source_spatial_shape) != 2:
        msg = (
            "Source spatial shape must have 2 dimensions. "
            f"Got {tuple(source_spatial_shape)}"
        )
        raise ValueError(msg)
    return (int(source_spatial_shape[1]), int(source_spatial_shape[0]))
