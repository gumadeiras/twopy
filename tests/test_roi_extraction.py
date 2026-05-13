"""Tests for GUI-independent ROI discovery.

Inputs: small synthetic images, grid sizes, and accepted-region masks.
Outputs: deterministic ``RoiSet`` objects for downstream trace extraction.
"""

import unittest

import numpy as np

from twopy import (
    RoiExtractionConfig,
    extract_rois_from_image,
    grid_roi_set,
    grid_roi_set_microns,
    grid_size_pixels_from_microns,
    roi_set_to_label_image,
    watershed_roi_set,
)


class RoiExtractionTest(unittest.TestCase):
    """Tests native twopy ROI extraction algorithms."""

    def test_grid_roi_set_tiles_entire_shape_with_edge_cells(self) -> None:
        """Confirm grid extraction covers non-divisible shapes.

        Inputs: a ``3 x 5`` image shape and a two-pixel grid size.
        Outputs: six non-overlapping ROIs including smaller edge cells.
        """
        roi_set = grid_roi_set((3, 5), grid_size_pixels=2)

        self.assertEqual(roi_set.labels, tuple(f"grid_{i:04d}" for i in range(1, 7)))
        self.assertEqual(roi_set.masks.shape, (6, 3, 5))
        np.testing.assert_array_equal(
            roi_set.masks.sum(axis=0),
            np.ones((3, 5), dtype=np.int64),
        )
        np.testing.assert_array_equal(
            roi_set_to_label_image(roi_set),
            np.array(
                [
                    [1, 1, 2, 2, 3],
                    [1, 1, 2, 2, 3],
                    [4, 4, 5, 5, 6],
                ],
            ),
        )

    def test_grid_roi_set_keeps_region_by_roi_center(self) -> None:
        """Confirm region restriction is pure mask filtering.

        Inputs: four grid cells and a region mask covering the right half.
        Outputs: only the two right-side grid ROIs, relabeled compactly.
        """
        region_mask = np.array(
            [
                [False, False, True, True],
                [False, False, True, True],
                [False, False, True, True],
                [False, False, True, True],
            ],
            dtype=np.bool_,
        )

        roi_set = grid_roi_set((4, 4), grid_size_pixels=2, region_mask=region_mask)

        self.assertEqual(roi_set.labels, ("grid_0001", "grid_0002"))
        np.testing.assert_array_equal(
            roi_set_to_label_image(roi_set),
            np.array(
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 2, 2],
                    [0, 0, 2, 2],
                ],
            ),
        )

    def test_grid_roi_set_microns_uses_floor_pixel_conversion(self) -> None:
        """Confirm physical grid sizes become native pixel grid sizes.

        Inputs: a one-micron grid and 0.4 micron-per-pixel calibration.
        Outputs: a two-pixel grid because ``floor(1 / 0.4) == 2``.
        """
        roi_set = grid_roi_set_microns(
            (2, 3),
            micron_grid_size=1.0,
            pixel_size_um=0.4,
        )

        self.assertEqual(grid_size_pixels_from_microns(1.0, 0.4), 2)
        np.testing.assert_array_equal(
            roi_set_to_label_image(roi_set),
            np.array([[1, 1, 2], [1, 1, 2]]),
        )

    def test_watershed_roi_set_splits_two_bright_structures(self) -> None:
        """Confirm watershed extraction finds bright local basins.

        Inputs: one image with two separated peaks.
        Outputs: two ROIs partitioning the full image.
        """
        image = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 1.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

        roi_set = watershed_roi_set(image)

        self.assertEqual(roi_set.labels, ("watershed_0001", "watershed_0002"))
        self.assertEqual(roi_set.masks.shape, (2, 3, 5))
        np.testing.assert_array_equal(
            roi_set.masks.sum(axis=0),
            np.ones(image.shape, dtype=np.int64),
        )
        self.assertTrue(roi_set.masks[0, 1, 1])
        self.assertTrue(roi_set.masks[1, 1, 3])

    def test_watershed_roi_set_filters_small_regions(self) -> None:
        """Confirm watershed extraction can drop undersized basins.

        Inputs: an image with two basins and a minimum pixel threshold.
        Outputs: only basins meeting the size threshold.
        """
        image = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 1.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )
        all_rois = watershed_roi_set(image)
        pixel_counts = all_rois.masks.reshape(all_rois.masks.shape[0], -1).sum(axis=1)
        min_pixels = int(pixel_counts.max())

        filtered = watershed_roi_set(image, min_pixels=min_pixels)

        self.assertLess(filtered.masks.shape[0], all_rois.masks.shape[0])
        self.assertGreaterEqual(filtered.masks.sum(), min_pixels)

    def test_extract_rois_from_image_rejects_region_shape_mismatch(self) -> None:
        """Confirm region masks must share image coordinates.

        Inputs: a two-dimensional image and a mismatched region mask.
        Outputs: clear validation error before extraction.
        """
        config = RoiExtractionConfig(
            method="grid",
            region_mask=np.ones((3, 3), dtype=np.bool_),
        )

        with self.assertRaisesRegex(ValueError, "region mask shape"):
            extract_rois_from_image(np.zeros((2, 2)), config)

    def test_grid_roi_set_rejects_invalid_grid_size(self) -> None:
        """Confirm grid extraction fails loudly on invalid config.

        Inputs: a positive image shape and zero grid size.
        Outputs: clear validation error.
        """
        with self.assertRaisesRegex(ValueError, "grid_size_pixels"):
            grid_roi_set((2, 2), grid_size_pixels=0)


if __name__ == "__main__":
    unittest.main()
