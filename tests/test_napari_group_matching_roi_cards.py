"""Tests for clickable Group Matching ROI preview cards.

Inputs: fake napari mean-image and Labels layers plus Qt click events.
Outputs: assertions that preview clicks select and clear the card ROI selector.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QColor, QMouseEvent
from qtpy.QtWidgets import QApplication, QComboBox, QLabel

from twopy.napari.group_matching.roi_cards import (
    RoiRecordingCard,
    _roi_label_at_preview_point,
)
from twopy.napari.session import LoadedNapariRecording


class GroupMatchingRoiCardTest(unittest.TestCase):
    """Tests for Group Matching ROI card preview selection."""

    def test_preview_point_uses_aspect_ratio_padding(self) -> None:
        """Confirm click hit-testing ignores centered preview padding.

        Inputs: a wide Labels image rendered into a square preview label.
        Outputs: clicks in the top letterbox padding return no ROI, while a
        click inside the rendered image maps to the correct ROI label.
        """
        labels = np.zeros((10, 20), dtype=np.int64)
        labels[0, 5] = 2
        layer = _FakeLayer(labels)

        self.assertIsNone(
            _roi_label_at_preview_point(
                layer,
                x=55,
                y=54,
                preview_width=220,
                preview_height=220,
                pixmap_width=220,
                pixmap_height=110,
            ),
        )
        self.assertEqual(
            _roi_label_at_preview_point(
                layer,
                x=55,
                y=60,
                preview_width=220,
                preview_height=220,
                pixmap_width=220,
                pixmap_height=110,
            ),
            "roi_0002",
        )

    def test_preview_click_toggles_roi_selector(self) -> None:
        """Confirm clicking the selected ROI clears the card selector.

        Inputs: one ROI card with a single ROI mask in the rendered preview.
        Outputs: first click selects the ROI, second click returns the dropdown
        to ``No ROI``, and background clicks leave selection unchanged.
        """
        app = QApplication.instance() or QApplication([])
        labels = np.zeros((10, 20), dtype=np.int64)
        labels[0, 5] = 2
        callback_count = 0

        def count_selection_change() -> None:
            nonlocal callback_count
            callback_count += 1

        card = RoiRecordingCard(
            recording=_loaded_recording(labels),
            fov_group_id="fov_1",
            selected_roi="",
            trace_color=QColor("#1f77b4"),
            on_selection_changed=count_selection_change,
        )
        card.show()
        app.processEvents()

        selector = card.findChild(QComboBox, "roi_selector")
        image_label = card.findChild(QLabel, "roi_preview_image")
        assert selector is not None
        assert image_label is not None

        _send_left_click(
            image_label,
            x=55,
            y=60,
        )
        self.assertEqual(selector.currentData(), "roi_0002")
        self.assertEqual(callback_count, 1)

        _send_left_click(
            image_label,
            x=55,
            y=60,
        )
        self.assertEqual(selector.currentData(), "")
        self.assertEqual(callback_count, 2)

        _send_left_click(
            image_label,
            x=0,
            y=0,
        )
        self.assertEqual(selector.currentData(), "")
        self.assertEqual(callback_count, 2)


class _FakeLayer:
    """Small napari layer double exposing data for ROI card previews."""

    def __init__(self, data: np.ndarray) -> None:
        """Store fake layer data."""
        self.data = data


def _loaded_recording(labels: np.ndarray) -> LoadedNapariRecording:
    """Return the loaded-recording fields needed by ``RoiRecordingCard``."""
    return cast(
        LoadedNapariRecording,
        SimpleNamespace(
            recording=SimpleNamespace(
                source_session_dir=Path("/tmp/fly/stimulus/2026/01_02/03_04_05"),
            ),
            roi_save_file=Path("/tmp/rois.h5"),
            mean_image_layer=_FakeLayer(np.zeros(labels.shape, dtype=np.float64)),
            movie_layer=None,
            roi_labels_layer=_FakeLayer(labels),
        ),
    )


def _send_left_click(widget: QLabel, *, x: int, y: int) -> None:
    """Send one typed primary-button press to a preview label."""
    point = QPointF(float(x), float(y))
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        point,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.mousePressEvent(event)


if __name__ == "__main__":
    unittest.main()
