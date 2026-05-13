"""Small morphology helpers for generated ROI masks.

Inputs: boolean ROI candidate masks and optional watershed-domain masks.
Outputs: 8-connected, optionally hole-filled boolean components.

These helpers keep shape cleanup separate from extraction/scoring code so ROI
generators can share one definition of connected mask components.
"""

import numpy as np
import numpy.typing as npt
from scipy import ndimage

_EIGHT_CONNECTED = np.ones((3, 3), dtype=np.bool_)


def cleaned_connected_components(
    mask: npt.NDArray[np.bool_],
    *,
    domain_mask: npt.NDArray[np.bool_],
    fill_holes: bool,
    closing_radius: int,
) -> tuple[npt.NDArray[np.bool_], ...]:
    """Return cleaned 8-connected components clipped to a domain mask."""
    cleaned = _closed_candidate_mask(
        mask,
        domain_mask=domain_mask,
        closing_radius=closing_radius,
    )
    components = _connected_components(cleaned)
    if not fill_holes:
        return components
    return tuple(
        _fill_component_holes(component, domain_mask) for component in components
    )


def _closed_candidate_mask(
    mask: npt.NDArray[np.bool_],
    *,
    domain_mask: npt.NDArray[np.bool_],
    closing_radius: int,
) -> npt.NDArray[np.bool_]:
    """Return a closed candidate mask clipped to its allowed domain."""
    if closing_radius == 0:
        return mask
    padded = np.pad(mask, closing_radius, mode="edge")
    closed = ndimage.binary_closing(
        padded,
        structure=_EIGHT_CONNECTED,
        iterations=closing_radius,
    )
    cropped = closed[
        closing_radius : closing_radius + mask.shape[0],
        closing_radius : closing_radius + mask.shape[1],
    ]
    return np.asarray(cropped & domain_mask, dtype=np.bool_)


def _connected_components(
    mask: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.bool_], ...]:
    """Return 8-connected components from one candidate ROI mask."""
    labels, count = ndimage.label(mask, structure=_EIGHT_CONNECTED)
    return tuple(labels == label for label in range(1, count + 1))


def _fill_component_holes(
    mask: npt.NDArray[np.bool_],
    domain_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Fill holes inside one connected ROI component."""
    filled = ndimage.binary_fill_holes(mask)
    return np.asarray(filled & domain_mask, dtype=np.bool_)
