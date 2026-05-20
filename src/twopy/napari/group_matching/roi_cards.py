"""ROI-selection cards for manual group matching.

Inputs: one loaded napari recording, FOV assignment text, ROI Labels data, and
a trace color.
Outputs: a fixed-size card that lets users pick one ROI for the selected FOV group.

The card owns the per-recording selector and image preview. The ROI assignment
view owns saved groups, response previews, and CSV persistence.
"""

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from twopy.napari.group_matching.cards import fov_group_display_id, image_overlay_stack
from twopy.napari.group_matching.images import (
    THUMBNAIL_SIZE,
    mean_image_roi_overlay_pixmap,
)
from twopy.napari.session import LoadedNapariRecording

__all__ = [
    "ROI_CARD_HEIGHT",
    "ROI_CARD_WIDTH",
    "ROI_CONTROL_HEIGHT",
    "RoiRecordingCard",
    "roi_label_display_id",
    "roi_label_value",
    "roi_labels_from_layer_data",
]

ROI_CARD_WIDTH = THUMBNAIL_SIZE + 24
ROI_CARD_HEIGHT = THUMBNAIL_SIZE + 58
ROI_CONTROL_HEIGHT = 28


class _LayerWithData(Protocol):
    """Small protocol for napari layers whose label data can be inspected."""

    data: object


class RoiRecordingCard(QFrame):
    """One visual ROI assignment card for a loaded recording."""

    def __init__(
        self,
        *,
        recording: LoadedNapariRecording,
        fov_group_id: str,
        selected_roi: str,
        trace_color: QColor,
        on_selection_changed: Callable[[], None],
    ) -> None:
        """Create one ROI card."""
        super().__init__()
        self.recording_path = recording.recording.source_session_dir.expanduser()
        self.recording = recording
        self.trace_color = trace_color
        self._image_label = QLabel()
        self._image_label.setObjectName("roi_preview_image")
        fov_id = fov_group_display_id(fov_group_id)
        self._overlay_label = QLabel(f"{self.recording_path.name} - FOV ID: {fov_id}")
        self._roi_selector = QComboBox()
        self._roi_selector.setObjectName("roi_selector")
        self._roi_selector.setMinimumWidth(118)
        self._roi_selector.setFixedHeight(ROI_CONTROL_HEIGHT)
        self._trace_label = QLabel("ROI")
        self._trace_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._trace_label.setFixedSize(42, ROI_CONTROL_HEIGHT)
        self._trace_label.setStyleSheet(_trace_label_style(trace_color))

        self._roi_selector.addItem("No ROI", "")
        for roi_label in roi_labels_from_layer_data(recording.roi_labels_layer):
            self._roi_selector.addItem(roi_label_display_id(roi_label), roi_label)
        if selected_roi:
            index = self._roi_selector.findData(selected_roi)
            if index >= 0:
                self._roi_selector.setCurrentIndex(index)
        self._roi_selector.currentIndexChanged.connect(
            lambda _index: self._refresh_after_selection_change(on_selection_changed),
        )

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        controls.addWidget(self._trace_label)
        controls.addWidget(self._roi_selector, 1)
        controls_widget = QWidget()
        controls_widget.setFixedWidth(THUMBNAIL_SIZE)
        controls_widget.setLayout(controls)

        image_stack = image_overlay_stack(
            self._image_label,
            self._overlay_label,
            size=THUMBNAIL_SIZE,
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(controls_widget, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(image_stack, 0, Qt.AlignmentFlag.AlignHCenter)
        self.setLayout(layout)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("roi_assignment_card")
        self.setFixedSize(ROI_CARD_WIDTH, ROI_CARD_HEIGHT)
        self.refresh_preview()

    def selected_roi_label(self) -> str | None:
        """Return the currently selected ROI label."""
        data = self._roi_selector.currentData()
        if not isinstance(data, str) or data == "":
            return None
        return data

    def set_selected_roi(self, roi_label: str) -> None:
        """Set the selected ROI label without emitting intermediate updates."""
        self._roi_selector.blockSignals(True)
        try:
            index = 0 if roi_label == "" else self._roi_selector.findData(roi_label)
            self._roi_selector.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self._roi_selector.blockSignals(False)
        self.refresh_preview()

    def refresh_preview(self) -> None:
        """Refresh selected-ROI image and response previews."""
        roi_label = self.selected_roi_label()
        pixmap = mean_image_roi_overlay_pixmap(
            mean_image_layer=self.recording.mean_image_layer,
            roi_labels_layer=self.recording.roi_labels_layer,
            roi_label=roi_label,
            selected_color=self.trace_color,
            contrast_percentile=100.0,
        )
        if pixmap is not None:
            self._image_label.setPixmap(pixmap)

    def _refresh_after_selection_change(self, callback: Callable[[], None]) -> None:
        """Refresh the ROI image and shared response plot after selection."""
        self.refresh_preview()
        callback()


def roi_labels_from_layer_data(layer: object | None) -> tuple[str, ...]:
    """Return selectable twopy ROI labels from a napari Labels layer.

    Args:
        layer: Napari Labels layer or a test double exposing integer ``data``.

    Returns:
        Sorted ``roi_####`` labels for positive values present in the label
        image. Zero is background and is omitted.
    """
    if layer is None or not hasattr(layer, "data"):
        return ()
    labels_layer = cast(_LayerWithData, layer)
    label_image = np.asarray(labels_layer.data)
    if label_image.size == 0:
        return ()
    labels = sorted(
        {
            int(label_value)
            for label_value in np.unique(label_image)
            if int(label_value) > 0
        },
    )
    return tuple(f"roi_{label_value:04d}" for label_value in labels)


def _trace_label_style(color: QColor) -> str:
    """Return stylesheet text for one trace-color chip."""
    return (
        "QLabel {"
        f"background-color: {color.name()};"
        "color: white;"
        "font-weight: bold;"
        "padding: 0px 6px;"
        "border-radius: 3px;"
        "}"
    )


def roi_label_value(roi_label: str | None) -> int | None:
    """Return the integer Labels value encoded in ``roi_####`` text."""
    if roi_label is None:
        return None
    suffix = roi_label.removeprefix("roi_")
    return int(suffix) if suffix.isdecimal() else None


def roi_label_display_id(roi_label: str) -> str:
    """Return the compact ROI id shown in the card dropdown."""
    suffix = roi_label.removeprefix("roi_")
    return str(int(suffix)) if suffix.isdecimal() else roi_label
