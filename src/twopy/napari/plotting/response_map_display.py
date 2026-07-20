"""Display-only scaling helpers for response heatmaps.

Inputs: persisted response-map values and GUI display toggles.
Outputs: signed response arrays scaled for color rendering.

The analysis module owns response-map computation and persistence. This module
owns only the visual transformation used by the napari heatmap panel and export
figures.

Display scaling clips outliers without changing saved heatmap data. The signed
color range is the 95th percentile of finite absolute response values. With
shared limits enabled, the percentile pool is all loaded epochs. With shared
limits disabled, the pool is only the epoch being drawn. The colorbar still
labels the semantic direction as ``-``, ``0``, ``+`` because the numeric limit is
display-only and may differ between epochs.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from twopy.analysis.response_maps import EpochResponseMap, ResponseMapData

__all__ = [
    "ROBUST_RESPONSE_LIMIT_PERCENTILE",
    "display_response_limit",
    "display_response_values",
    "robust_abs_response_limit",
]

ROBUST_RESPONSE_LIMIT_PERCENTILE = 95.0
_MIN_RESPONSE_LIMIT = 1e-6


def display_response_values(
    map_data: ResponseMapData,
    epoch: EpochResponseMap,
    *,
    shared_limits: bool,
) -> npt.NDArray[np.float64]:
    """Return signed response values ready for color rendering.

    Args:
        map_data: Heatmap data for the loaded recording.
        epoch: Epoch whose response values should be displayed.
        shared_limits: Whether the robust color limit should use all epochs
            instead of the current epoch only.

    Returns:
        Response values clipped to the robust signed display range. The returned
        array remains in persisted response units. Callers pass the matching
        ``display_response_limit`` to the colormap.
    """
    response = epoch.response_values
    scale = _display_response_scale(
        map_data,
        epoch,
        shared_limits=shared_limits,
    )
    return np.clip(response, -scale, scale)


def display_response_limit(
    map_data: ResponseMapData,
    epoch: EpochResponseMap,
    *,
    shared_limits: bool,
) -> float:
    """Return the signed color limit for one displayed response map.

    Args:
        map_data: Heatmap data for the loaded recording.
        epoch: Epoch whose response values should be displayed.
        shared_limits: Whether robust limits use all epochs or only this epoch.

    Returns:
        The robust response limit in persisted response units. Positive and
        negative colors use this same magnitude so zero remains the midpoint.
    """
    return _display_response_scale(
        map_data,
        epoch,
        shared_limits=shared_limits,
    )


def robust_abs_response_limit(
    responses: tuple[npt.NDArray[np.float64], ...],
) -> float:
    """Return the robust signed color limit for response heatmaps.

    Args:
        responses: One or more response arrays contributing to the visual
            scaling pool.

    Returns:
        The nearest-rank 95th percentile of finite absolute response values,
        with a small positive floor for all-zero maps.

    This uses ``numpy.partition`` instead of a full sort so large maps do not
    pay unnecessary sorting cost just to ignore extreme outliers.
    """
    chunks: list[npt.NDArray[np.float64]] = []
    for response in responses:
        finite_abs = np.abs(response[np.isfinite(response)])
        if finite_abs.size > 0:
            chunks.append(finite_abs)
    if len(chunks) == 0:
        return 1.0

    values = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    percentile = ROBUST_RESPONSE_LIMIT_PERCENTILE / 100.0
    rank = min(max(int(np.ceil(percentile * values.size)) - 1, 0), values.size - 1)
    values.partition(rank)
    return max(float(values[rank]), _MIN_RESPONSE_LIMIT)


def _display_response_scale(
    map_data: ResponseMapData,
    epoch: EpochResponseMap,
    *,
    shared_limits: bool,
) -> float:
    """Return the robust absolute response scale used for color rendering."""
    responses = (
        tuple(epoch_data.response_values for epoch_data in map_data.epochs)
        if shared_limits
        else (epoch.response_values,)
    )
    return robust_abs_response_limit(responses)
