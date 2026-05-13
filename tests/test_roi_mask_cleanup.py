"""Tests for generated ROI mask morphology cleanup.

Inputs: small boolean masks near image and watershed-domain boundaries.
Outputs: cleaned connected components that do not shrink away from edges.
"""

import unittest

import numpy as np

from twopy.roi_mask_cleanup import cleaned_connected_components


class RoiMaskCleanupTest(unittest.TestCase):
    """Tests edge-safe cleanup for generated ROI masks."""

    def test_closing_preserves_image_edge_components(self) -> None:
        """Confirm closing does not pull components away from image borders.

        Inputs: two same-edge segments separated by a small gap.
        Outputs: radius-five closing joins them while keeping top-edge pixels.
        """
        mask = np.zeros((20, 20), dtype=np.bool_)
        mask[0, 5:8] = True
        mask[0, 10:13] = True

        components = cleaned_connected_components(
            mask,
            domain_mask=np.ones_like(mask, dtype=np.bool_),
            fill_holes=False,
            closing_radius=5,
        )
        closed = np.logical_or.reduce(components)

        self.assertEqual(len(components), 1)
        self.assertTrue(np.all(closed[0, 5:13]))
        self.assertFalse(np.any(closed[5:, :]))

    def test_closing_preserves_domain_edge_components(self) -> None:
        """Confirm closing respects watershed-domain boundaries.

        Inputs: two segments on the top edge of an allowed watershed domain.
        Outputs: closing joins them without shrinking off the domain edge.
        """
        domain = np.zeros((20, 20), dtype=np.bool_)
        domain[4:16, 4:16] = True
        mask = np.zeros_like(domain)
        mask[4, 6:8] = True
        mask[4, 10:12] = True

        components = cleaned_connected_components(
            mask,
            domain_mask=domain,
            fill_holes=False,
            closing_radius=4,
        )
        closed = np.logical_or.reduce(components)

        self.assertEqual(len(components), 1)
        self.assertTrue(np.all(closed[4, 6:12]))
        self.assertFalse(np.any(closed[:4, :]))


if __name__ == "__main__":
    unittest.main()
