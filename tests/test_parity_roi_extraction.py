"""Tests for psycho5 ROI extraction parity helpers.

Inputs: small synthetic images, MATLAB-style watershed labels, and grid sizes.
Outputs: source-faithful psycho5 label images for comparison with native twopy
ROI extraction.
"""

import unittest

import numpy as np

from twopy import grid_roi_set, roi_set_to_label_image, watershed_roi_set
from twopy.parity import (
    psycho5_grid_pixel_size,
    psycho5_grid_roi_label_image,
    psycho5_grid_roi_set,
    psycho5_watershed_image_from_preseg,
    psycho5_watershed_roi_set_from_preseg,
)


class Psycho5RoiExtractionParityTest(unittest.TestCase):
    """Tests source-faithful psycho5 ROI extraction helpers."""

    def test_psycho5_grid_uses_matlab_distinct_block_order(self) -> None:
        """Confirm psycho5 grid labels scan down columns of blocks first.

        Inputs: a non-divisible image shape and a two-pixel block size.
        Outputs: edge blocks plus MATLAB-style ``im2col(..., 'distinct')``
        ordering.
        """
        label_image = psycho5_grid_roi_label_image((3, 5), 2)

        np.testing.assert_array_equal(
            label_image,
            np.array(
                [
                    [1, 1, 3, 3, 5],
                    [1, 1, 3, 3, 5],
                    [2, 2, 4, 4, 6],
                ],
            ),
        )

    def test_psycho5_grid_roi_set_wraps_grid_labels(self) -> None:
        """Confirm psycho5 grid labels become normal twopy ROI masks.

        Inputs: a ``2 x 3`` image shape and two-pixel block size.
        Outputs: ``RoiSet`` labels preserving psycho5 grid numbering.
        """
        roi_set = psycho5_grid_roi_set((2, 3), 2)

        self.assertEqual(
            roi_set.labels,
            ("psycho5_grid_0001", "psycho5_grid_0002"),
        )
        np.testing.assert_array_equal(
            roi_set_to_label_image(roi_set),
            np.array([[1, 1, 2], [1, 1, 2]]),
        )

    def test_psycho5_grid_pixel_size_uses_source_formula(self) -> None:
        """Confirm micron grid conversion follows psycho5 arithmetic.

        Inputs: micron size, zoom, image width, and lab calibration constant.
        Outputs: floor of the psycho5 pixel conversion.
        """
        self.assertEqual(
            psycho5_grid_pixel_size(
                micron_grid_size=3.0,
                zoom_factor=2.0,
                image_axis1_size=128,
                one_x_pixel_per_micron_256_axis1=1.5,
            ),
            18,
        )

    def test_psycho5_watershed_fills_zero_borders_by_nearest_segment_peak(
        self,
    ) -> None:
        """Confirm psycho5 ``WatershedImage`` border assignment.

        Inputs: an original image and MATLAB watershed labels before border
        fill.
        Outputs: border pixels assigned to nearest segment peak, with distance
        ties going to the lower segment label.
        """
        image = np.array(
            [
                [9.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 8.0],
            ],
        )
        preseg = np.array(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 2],
            ],
        )

        filled = psycho5_watershed_image_from_preseg(image, preseg)

        np.testing.assert_array_equal(
            filled,
            np.array(
                [
                    [1, 1, 1],
                    [1, 1, 2],
                    [1, 2, 2],
                ],
            ),
        )

    def test_psycho5_watershed_roi_set_wraps_filled_labels(self) -> None:
        """Confirm psycho5 watershed labels become normal twopy ROI masks.

        Inputs: an image and MATLAB watershed labels before border fill.
        Outputs: ``RoiSet`` labels preserving psycho5 watershed numbering.
        """
        roi_set = psycho5_watershed_roi_set_from_preseg(
            np.array([[2.0, 0.0], [0.0, 1.0]]),
            np.array([[1, 0], [0, 2]]),
        )

        self.assertEqual(
            roi_set.labels,
            ("psycho5_watershed_0001", "psycho5_watershed_0002"),
        )
        np.testing.assert_array_equal(
            roi_set_to_label_image(roi_set),
            np.array([[1, 1], [1, 2]]),
        )

    def test_native_grid_order_is_not_psycho5_grid_order(self) -> None:
        """Confirm twopy grid order is a native Python-facing choice.

        Inputs: one shape/grid pair.
        Outputs: different label numbering despite identical ROI geometry.
        """
        native_labels = roi_set_to_label_image(grid_roi_set((3, 5), grid_size_pixels=2))
        psycho5_labels = psycho5_grid_roi_label_image((3, 5), 2)

        self.assertFalse(np.array_equal(native_labels, psycho5_labels))
        np.testing.assert_array_equal(native_labels > 0, psycho5_labels > 0)

    def test_native_watershed_is_not_psycho5_watershed_border_fill(self) -> None:
        """Confirm native watershed does not claim MATLAB watershed parity.

        Inputs: the same image used for a psycho5 pre-segmentation comparison.
        Outputs: different labels because twopy uses a local inspectable
        watershed-style flood instead of MATLAB ``watershed(-image, 8)`` plus
        psycho5's nearest-peak border fill.
        """
        image = np.array(
            [
                [9.0, 8.0, 7.0, 6.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 8.0],
            ],
        )
        psycho5_labels = psycho5_watershed_image_from_preseg(
            image,
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2],
                ],
            ),
        )
        native_labels = roi_set_to_label_image(watershed_roi_set(image))

        self.assertFalse(np.array_equal(native_labels, psycho5_labels))


if __name__ == "__main__":
    unittest.main()
