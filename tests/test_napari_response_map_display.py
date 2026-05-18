"""Pure tests for napari response-map display helpers.

Inputs: response-map arrays and the shared heatmap colormap.
Outputs: display limits and clipped values without importing Qt widgets.
"""

import unittest

import numpy as np

from twopy.analysis.response_maps import EpochResponseMap, ResponseMapData
from twopy.napari.plotting.response_map_colors import RESPONSE_HEATMAP_COLORMAP
from twopy.napari.plotting.response_map_display import (
    display_response_limit,
    display_response_values,
)
from twopy.napari.plotting.response_map_options import ResponseMapOptions
from twopy.spatial import SpatialCrop


class NapariResponseMapDisplayTest(unittest.TestCase):
    """Tests pure response-map display scaling behavior."""

    def test_response_heatmap_colormap_has_black_zero_without_white_band(self) -> None:
        """Confirm the signed heatmap colorbar has a black neutral point.

        Inputs: the shared response heatmap colormap sampled across the full
        signed response range.
        Outputs: zero is black and no sampled color is close to white.
        """
        colors = np.asarray(RESPONSE_HEATMAP_COLORMAP(np.linspace(0.0, 1.0, 257)))
        zero_color = colors[128, :3]
        low_positive_color = colors[154, :3]
        mid_positive_color = colors[192, :3]
        high_positive_color = colors[-1, :3]

        self.assertLess(float(np.max(zero_color)), 0.02)
        self.assertFalse(np.any(np.all(colors[:, :3] > 0.9, axis=1)))
        self.assertLess(float(low_positive_color[0]), float(mid_positive_color[0]))
        self.assertLess(float(mid_positive_color[0]), float(high_positive_color[0]))

    def test_response_heatmap_display_uses_robust_limits(self) -> None:
        """Confirm outliers do not set the heatmap color limit.

        Inputs: one response map with a single extreme pixel.
        Outputs: the visual limit follows the 95th percentile and clips the
        outlier.
        """
        response = np.concatenate(
            (np.linspace(0.01, 0.20, 100, dtype=np.float64), np.array([1.0])),
        ).reshape(101, 1)
        map_data = ResponseMapData(
            mean_image=np.ones((101, 1), dtype=np.float64),
            epochs=(
                EpochResponseMap(
                    epoch_name="Odor",
                    epoch_number=1,
                    response_values=response,
                    trial_count=1,
                ),
            ),
            options=ResponseMapOptions(),
            spatial_crop=SpatialCrop(0, 101, 0, 1, (101, 1), "test"),
            response_scale=1.0,
        )
        epoch = map_data.epochs[0]

        limit = display_response_limit(
            map_data,
            epoch,
            shared_limits=False,
        )
        values = display_response_values(
            map_data,
            epoch,
            shared_limits=False,
        )

        self.assertLess(limit, 0.20)
        self.assertGreater(limit, 0.18)
        self.assertEqual(float(values[-1, 0]), limit)


if __name__ == "__main__":
    unittest.main()
