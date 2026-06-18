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
from qtpy.QtWidgets import QApplication, QComboBox, QLabel, QPushButton

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

    def test_preview_click_toggles_multiple_rois(self) -> None:
        """Confirm preview clicks toggle independent ROI selections.

        Inputs: one ROI card with two ROI masks in the rendered preview.
        Outputs: clicks add each ROI, then remove only the clicked selected ROI.
        """
        app = QApplication.instance() or QApplication([])
        labels = np.zeros((10, 20), dtype=np.int64)
        labels[0, 5] = 2
        labels[1, 10] = 3
        callback_count = 0

        def count_selection_change() -> None:
            nonlocal callback_count
            callback_count += 1

        card = RoiRecordingCard(
            recording=_loaded_recording(labels),
            fov_group_id="fov_1",
            selected_rois=(),
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
        self.assertEqual(card.selected_roi_labels(), ("roi_0002",))
        self.assertEqual(selector.currentData(), "")
        self.assertEqual(callback_count, 1)

        _send_left_click(
            image_label,
            x=110,
            y=75,
        )
        self.assertEqual(card.selected_roi_labels(), ("roi_0002", "roi_0003"))
        self.assertEqual(callback_count, 2)

        _send_left_click(
            image_label,
            x=55,
            y=60,
        )
        self.assertEqual(card.selected_roi_labels(), ("roi_0003",))
        self.assertEqual(callback_count, 3)

        _send_left_click(
            image_label,
            x=0,
            y=0,
        )
        self.assertEqual(card.selected_roi_labels(), ("roi_0003",))
        self.assertEqual(callback_count, 3)

    def test_selected_roi_chip_removes_roi_and_notifies_parent(self) -> None:
        """Confirm chip removal updates card state and emits one change.

        Inputs: one ROI card restored with two selected ROIs.
        Outputs: clicking the first chip removes only that ROI and calls the
        parent selection-change callback.
        """
        app = QApplication.instance() or QApplication([])
        labels = np.zeros((10, 20), dtype=np.int64)
        labels[0, 5] = 2
        labels[1, 10] = 3
        callback_count = 0

        def count_selection_change() -> None:
            nonlocal callback_count
            callback_count += 1

        card = RoiRecordingCard(
            recording=_loaded_recording(labels),
            fov_group_id="fov_1",
            selected_rois=("roi_0002", "roi_0003"),
            trace_color=QColor("#1f77b4"),
            on_selection_changed=count_selection_change,
        )
        card.show()
        app.processEvents()

        chips = [
            button
            for button in card.findChildren(QPushButton, "selected_roi_chip")
            if button.text() == "ROI 2"
        ]
        self.assertEqual(len(chips), 1)
        self.assertFalse(chips[0].icon().isNull())
        self.assertNotIn("#d62728", chips[0].styleSheet())
        self.assertEqual(chips[0].toolTip(), "Remove ROI 2")
        self.assertEqual(chips[0].accessibleName(), "Remove ROI 2")

        chips[0].click()

        self.assertEqual(card.selected_roi_labels(), ("roi_0003",))
        self.assertEqual(callback_count, 1)

    def test_selected_roi_chips_show_overflow_summary(self) -> None:
        """Confirm crowded cards show a compact overflow count.

        Inputs: one ROI card restored with five selected ROIs.
        Outputs: four removable chips are visible and the fifth ROI is counted
        in a compact overflow label.
        """
        app = QApplication.instance() or QApplication([])
        labels = np.arange(1, 7, dtype=np.int64).reshape(2, 3)
        card = RoiRecordingCard(
            recording=_loaded_recording(labels),
            fov_group_id="fov_1",
            selected_rois=(
                "roi_0001",
                "roi_0002",
                "roi_0003",
                "roi_0004",
                "roi_0005",
            ),
            trace_color=QColor("#1f77b4"),
            on_selection_changed=lambda: None,
        )
        card.show()
        app.processEvents()

        self.assertEqual(len(card.findChildren(QPushButton, "selected_roi_chip")), 4)
        overflow = card.findChild(QLabel, "selected_roi_overflow")
        assert overflow is not None
        self.assertEqual(overflow.text(), "+1 more")


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
