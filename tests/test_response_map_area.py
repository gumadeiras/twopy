"""Tests for napari response heatmap widget sizing.

Inputs: tiny response-map data and Qt heatmap widgets.
Outputs: widget size hints that keep colorbar space inside the requested size.
"""

import unittest

import numpy as np
from qtpy.QtWidgets import QApplication

from twopy.analysis.response_maps import (
    EpochResponseMap,
    ResponseMapData,
    ResponseMapOptions,
)
from twopy.napari.plotting.docks.response_map_area import (
    _COLORBAR_HEIGHT,
    _MAP_COLORBAR_SPACING,
    ResponseMapArea,
    _heatmap_image_size,
)
from twopy.spatial import SpatialCrop


class ResponseMapAreaTest(unittest.TestCase):
    """Test live response-heatmap widget behavior."""

    def test_heatmap_size_reserves_colorbar_space(self) -> None:
        """Confirm Plot-tab size is the heatmap content budget.

        Inputs: one cached heatmap rendered at two requested sizes.
        Outputs: the image side plus colorbar height and gap equals the
        requested size, so the colorbar cannot make heatmaps larger than the
        selected Plot size.
        """
        _ = QApplication.instance() or QApplication([])
        area = ResponseMapArea("No recording loaded.")
        map_data = _tiny_response_map_data()

        area.render(
            map_data=map_data,
            epoch_indices=(0,),
            map_size=320,
            shared_limits=True,
        )

        image_size = _heatmap_image_size(320)
        self.assertEqual(image_size + _COLORBAR_HEIGHT + _MAP_COLORBAR_SPACING, 320)
        self.assertEqual(area.epoch_map_widgets[0].sizeHint().width(), image_size)
        self.assertEqual(area.epoch_map_widgets[0].sizeHint().height(), image_size)
        self.assertEqual(
            area.epoch_colorbar_widgets[0].sizeHint().width(),
            image_size,
        )
        self.assertEqual(
            area.epoch_colorbar_widgets[0].sizeHint().height(),
            _COLORBAR_HEIGHT,
        )

        area.update_epoch_map_widgets(epoch_indices=(0,), map_size=240)

        image_size = _heatmap_image_size(240)
        self.assertEqual(image_size + _COLORBAR_HEIGHT + _MAP_COLORBAR_SPACING, 240)
        self.assertEqual(area.epoch_map_widgets[0].sizeHint().width(), image_size)
        self.assertEqual(area.epoch_map_widgets[0].sizeHint().height(), image_size)
        self.assertEqual(
            area.epoch_colorbar_widgets[0].sizeHint().width(),
            image_size,
        )


def _tiny_response_map_data() -> ResponseMapData:
    """Return one tiny response map for widget tests.

    Args:
        None.

    Returns:
        Response-map data with one epoch and finite signed responses.
    """
    return ResponseMapData(
        mean_image=np.ones((2, 2), dtype=np.float64),
        epochs=(
            EpochResponseMap(
                epoch_name="Odor",
                epoch_number=1,
                response_values=np.array(
                    [[-0.5, 0.0], [0.25, 0.5]],
                    dtype=np.float64,
                ),
                trial_count=1,
            ),
        ),
        options=ResponseMapOptions(),
        spatial_crop=SpatialCrop(0, 2, 0, 2, (2, 2), "test"),
        response_scale=1.0,
    )
