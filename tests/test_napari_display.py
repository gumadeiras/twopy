"""Pure tests for napari display-crop metadata helpers.

Inputs: Labels data and spatial crop metadata.
Outputs: restored full-frame label images without Qt.
"""

import unittest
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twopy.napari.display import (
    display_metadata_for_spatial_crop,
)
from twopy.napari.roi import roi_label_image_from_layer
from twopy.spatial import SpatialCrop


@dataclass
class _FakeLayer:
    """Small layer double for display-crop tests.

    Args:
        data: Integer Labels data in napari display coordinates.
        metadata: Optional twopy display-crop metadata.

    Returns:
        Layer-shaped object accepted by napari display helpers.
    """

    data: npt.NDArray[np.int64]
    metadata: dict[str, object]


class NapariDisplayTest(unittest.TestCase):
    """Tests pure display-crop metadata helpers."""

    def test_display_labels_round_trip_through_alignment_crop(self) -> None:
        """Confirm cropped napari labels return to full-frame coordinates.

        Inputs: full-frame labels plus an alignment-valid crop.
        Outputs: cropped labels and restored full-frame labels with zeros
        outside the displayed crop.
        """
        crop = SpatialCrop(
            axis0_start=1,
            axis0_stop=3,
            axis1_start=0,
            axis1_stop=3,
            original_shape=(3, 4),
            source="alignment_valid_crop",
        )
        movie_labels = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 2, 0],
                [0, 3, 0, 0],
            ],
            dtype=np.int64,
        )

        display_labels = crop.crop_image(movie_labels)
        layer = _FakeLayer(
            data=display_labels,
            metadata=display_metadata_for_spatial_crop(crop),
        )

        self.assertEqual(display_labels.shape, (2, 3))
        np.testing.assert_array_equal(roi_label_image_from_layer(layer), movie_labels)

    def test_malformed_display_crop_metadata_is_ignored(self) -> None:
        """Confirm bad crop metadata does not crash label extraction.

        Inputs: cropped labels with incomplete or invalid crop
        metadata.
        Outputs: labels treated as uncropped display data.
        """
        display_labels = np.array([[0, 1], [2, 0]], dtype=np.int64)
        bad_metadata: tuple[dict[str, object], ...] = (
            {
                "twopy_display_spatial_crop": {
                    "axis0_start": 0,
                    "axis1_start": 0,
                    "axis1_stop": 2,
                    "original_shape": (2, 2),
                    "source": "alignment_valid_crop",
                }
            },
            {
                "twopy_display_spatial_crop": {
                    "axis0_start": True,
                    "axis0_stop": 2,
                    "axis1_start": 0,
                    "axis1_stop": 2,
                    "original_shape": (2, 2),
                    "source": "alignment_valid_crop",
                }
            },
        )

        for metadata in bad_metadata:
            with self.subTest(metadata=metadata):
                layer = _FakeLayer(data=display_labels, metadata=metadata)

                np.testing.assert_array_equal(
                    roi_label_image_from_layer(layer),
                    display_labels,
                )


if __name__ == "__main__":
    unittest.main()
