"""Spatial crop contracts for converted two-photon recordings.

Inputs: movie spatial shapes, alignment-derived crop bounds, and ROI masks.
Outputs: typed crop objects and helpers that keep full-frame and cropped
analysis coordinates explicit.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "SpatialCrop",
    "SpatialDomain",
    "full_frame_crop",
]

SpatialDomain = Literal["alignment_valid_crop", "full_frame"]


@dataclass(frozen=True)
class SpatialCrop:
    """A rectangular spatial domain inside a full movie frame.

    Inputs: half-open bounds along the two spatial movie axes plus the original
    full-frame shape.
    Outputs: validated crop metadata and array slicing helpers.

    ``axis0`` and ``axis1`` name NumPy movie axes directly. twopy avoids
    top/left/x/y here because source files and display tools can disagree about
    image orientation, while the array axes are unambiguous.
    """

    axis0_start: int
    axis0_stop: int
    axis1_start: int
    axis1_stop: int
    original_shape: tuple[int, int]
    source: str

    def __post_init__(self) -> None:
        """Validate crop bounds immediately after construction.

        Inputs: dataclass field values.
        Outputs: ``None`` or a clear ``ValueError``.
        """
        axis0_length, axis1_length = self.original_shape
        if (
            self.axis0_start < 0
            or self.axis1_start < 0
            or self.axis0_stop > axis0_length
            or self.axis1_stop > axis1_length
            or self.axis0_start >= self.axis0_stop
            or self.axis1_start >= self.axis1_stop
        ):
            msg = (
                "Invalid spatial crop "
                f"axis0=[{self.axis0_start}, {self.axis0_stop}), "
                f"axis1=[{self.axis1_start}, {self.axis1_stop}) for "
                f"shape {self.original_shape}"
            )
            raise ValueError(msg)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the spatial shape of this crop.

        Inputs: crop bounds.
        Outputs: ``(axis0_length, axis1_length)``.
        """
        return (
            self.axis0_stop - self.axis0_start,
            self.axis1_stop - self.axis1_start,
        )

    @property
    def is_full_frame(self) -> bool:
        """Return whether this crop covers the whole source frame.

        Inputs: crop bounds and original shape.
        Outputs: ``True`` for full-frame domains, otherwise ``False``.
        """
        return (
            self.axis0_start == 0
            and self.axis1_start == 0
            and self.axis0_stop == self.original_shape[0]
            and self.axis1_stop == self.original_shape[1]
        )

    def crop_image(
        self,
        image: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """Return the crop view from a full-frame image.

        Args:
            image: Two-dimensional full-frame image with ``original_shape``.

        Returns:
            View of ``image`` inside this crop.
        """
        if image.shape != self.original_shape:
            msg = f"Image shape {image.shape} does not match {self.original_shape}"
            raise ValueError(msg)
        return image[
            self.axis0_start : self.axis0_stop,
            self.axis1_start : self.axis1_stop,
        ]

    def crop_masks(
        self,
        masks: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.bool_]:
        """Return ROI masks restricted to this crop.

        Args:
            masks: Boolean masks shaped ``(rois, axis0, axis1)`` in full-frame
                movie coordinates.

        Returns:
            Boolean mask view shaped ``(rois, cropped_axis0, cropped_axis1)``.
        """
        if masks.ndim != 3 or masks.shape[1:] != self.original_shape:
            msg = (
                f"ROI mask shape {masks.shape} does not match crop source "
                f"shape {self.original_shape}"
            )
            raise ValueError(msg)
        return masks[
            :,
            self.axis0_start : self.axis0_stop,
            self.axis1_start : self.axis1_stop,
        ]


def full_frame_crop(spatial_shape: tuple[int, int]) -> SpatialCrop:
    """Create a crop object that covers an entire movie frame.

    Args:
        spatial_shape: Full movie spatial shape.

    Returns:
        ``SpatialCrop`` spanning the full frame.
    """
    return SpatialCrop(
        axis0_start=0,
        axis0_stop=spatial_shape[0],
        axis1_start=0,
        axis1_stop=spatial_shape[1],
        original_shape=spatial_shape,
        source="full_frame",
    )
