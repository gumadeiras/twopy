"""ROI-selection cards for manual group matching.

Inputs: one loaded napari recording, FOV assignment text, ROI Labels data, and
a trace color.
Outputs: a fixed-size card that lets users pick ROIs for the selected FOV group.

The card owns the per-recording selector and image preview. The ROI assignment
view owns saved groups, response previews, and CSV persistence.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QIcon, QMouseEvent, QPainter, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from twopy.napari.display_paths import format_recording_minute_label
from twopy.napari.group_matching.cards import fov_group_display_id, image_overlay_stack
from twopy.napari.group_matching.images import (
    THUMBNAIL_SIZE,
    mean_image_roi_overlay_pixmap,
)
from twopy.napari.protocols import NapariLayerWithData
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
ROI_CARD_HEIGHT = THUMBNAIL_SIZE + 92
ROI_CONTROL_HEIGHT = 28
VISIBLE_SELECTED_ROI_CHIPS = 4


class _RoiPreviewLabel(QLabel):
    """Mean-image ROI preview that reports left-click image coordinates."""

    def __init__(self, on_click: Callable[[int, int], None]) -> None:
        """Create a clickable preview label.

        Args:
            on_click: Callback receiving the click position in label pixels.

        Returns:
            None.

        The ROI card uses this widget so image clicks can drive the same
        selector state as the dropdown without adding a second selection model.
        """
        super().__init__()
        self._on_click = on_click
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, ev: QMouseEvent | None) -> None:  # noqa: N802
        """Forward primary-button clicks to the owning ROI card."""
        if ev is not None and ev.button() == Qt.MouseButton.LeftButton:
            position = ev.pos()
            self._on_click(position.x(), position.y())
            ev.accept()
            return
        super().mousePressEvent(ev)


class RoiRecordingCard(QFrame):
    """One visual ROI assignment card for a loaded recording."""

    def __init__(
        self,
        *,
        recording: LoadedNapariRecording,
        fov_group_id: str,
        selected_rois: tuple[str, ...],
        trace_color: QColor,
        on_selection_changed: Callable[[], None],
    ) -> None:
        """Create one ROI card."""
        super().__init__()
        self.recording_path = recording.recording.source_session_dir.expanduser()
        self._recording_label = format_recording_minute_label(self.recording_path)
        self.recording = recording
        self.trace_color = trace_color
        self._on_selection_changed = on_selection_changed
        self._selected_rois: tuple[str, ...] = ()
        self._image_label = _RoiPreviewLabel(self._toggle_roi_at_preview_point)
        self._image_label.setObjectName("roi_preview_image")
        fov_id = fov_group_display_id(fov_group_id)
        self._overlay_label = QLabel(f"{self._recording_label} - FOV ID: {fov_id}")
        self._roi_selector = QComboBox()
        self._roi_selector.setObjectName("roi_selector")
        self._roi_selector.setMinimumWidth(118)
        self._roi_selector.setFixedHeight(ROI_CONTROL_HEIGHT)
        self._trace_label = QLabel("ROI")
        self._trace_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._trace_label.setFixedSize(42, ROI_CONTROL_HEIGHT)
        self._trace_label.setStyleSheet(_trace_label_style(trace_color))

        self._roi_selector.addItem("Add ROI", "")
        for roi_label in roi_labels_from_layer_data(recording.roi_labels_layer):
            self._roi_selector.addItem(roi_label_display_id(roi_label), roi_label)
        self._roi_selector.currentIndexChanged.connect(
            lambda _index: self._handle_selector_change(),
        )

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        controls.addWidget(self._trace_label)
        controls.addWidget(self._roi_selector, 1)
        controls_widget = QWidget()
        controls_widget.setFixedWidth(THUMBNAIL_SIZE)
        controls_widget.setLayout(controls)

        self._chips_widget = QWidget()
        self._chips_layout = QGridLayout()
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setHorizontalSpacing(4)
        self._chips_layout.setVerticalSpacing(4)
        self._chips_widget.setFixedWidth(THUMBNAIL_SIZE)
        self._chips_widget.setLayout(self._chips_layout)

        image_stack = image_overlay_stack(
            self._image_label,
            self._overlay_label,
            size=THUMBNAIL_SIZE,
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(controls_widget, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._chips_widget, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(image_stack, 0, Qt.AlignmentFlag.AlignHCenter)
        self.setLayout(layout)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("roi_assignment_card")
        self.setFixedSize(ROI_CARD_WIDTH, ROI_CARD_HEIGHT)
        self.set_selected_rois(selected_rois)
        self.refresh_preview()

    def selected_roi_labels(self) -> tuple[str, ...]:
        """Return the currently selected ROI labels."""
        return self._selected_rois

    def set_selected_rois(self, roi_labels: tuple[str, ...]) -> None:
        """Set selected ROI labels without emitting intermediate updates."""
        valid_labels = [
            roi_label
            for roi_label in roi_labels
            if self._roi_selector.findData(roi_label) >= 0
        ]
        self._selected_rois = tuple(dict.fromkeys(valid_labels))
        self._reset_selector()
        self._refresh_chips()
        self.refresh_preview()

    def refresh_preview(self) -> None:
        """Refresh selected-ROI image and response previews."""
        pixmap = mean_image_roi_overlay_pixmap(
            mean_image_layer=self.recording.mean_image_layer,
            roi_labels_layer=self.recording.roi_labels_layer,
            roi_labels=self._selected_rois,
            selected_color=self.trace_color,
            contrast_percentile=100.0,
        )
        if pixmap is not None:
            self._image_label.setPixmap(pixmap)

    def _handle_selector_change(self) -> None:
        """Toggle the chosen dropdown ROI and reset the selector."""
        data = self._roi_selector.currentData()
        if not isinstance(data, str) or data == "":
            return
        self._toggle_roi(data)
        self._reset_selector()
        self._refresh_chips()
        self.refresh_preview()
        self._on_selection_changed()

    def _reset_selector(self) -> None:
        """Return the dropdown to its add-ROI prompt."""
        self._roi_selector.blockSignals(True)
        try:
            self._roi_selector.setCurrentIndex(0)
        finally:
            self._roi_selector.blockSignals(False)

    def _toggle_roi(self, roi_label: str) -> None:
        """Add or remove one ROI label from this recording card."""
        if roi_label in self._selected_rois:
            self._selected_rois = tuple(
                selected for selected in self._selected_rois if selected != roi_label
            )
            return
        self._selected_rois = (*self._selected_rois, roi_label)

    def _refresh_chips(self) -> None:
        """Redraw the selected-ROI chips under the dropdown."""
        while self._chips_layout.count() > 0:
            item = self._chips_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if len(self._selected_rois) == 0:
            label = QLabel("No ROIs selected")
            label.setObjectName("selected_roi_empty")
            self._chips_layout.addWidget(label, 0, 0)
            return
        visible_rois = self._selected_rois[:VISIBLE_SELECTED_ROI_CHIPS]
        for index, roi_label in enumerate(visible_rois):
            roi_id = roi_label_display_id(roi_label)
            button = QPushButton(f"ROI {roi_id}")
            button.setObjectName("selected_roi_chip")
            button.setFixedHeight(22)
            button.setIcon(_remove_roi_icon())
            button.setToolTip(f"Remove ROI {roi_id}")
            button.setAccessibleName(f"Remove ROI {roi_id}")
            button.clicked.connect(
                lambda _checked, label=roi_label: self._remove_roi(label),
            )
            self._chips_layout.addWidget(button, index // 2, index % 2)
        hidden_count = len(self._selected_rois) - len(visible_rois)
        if hidden_count > 0:
            label = QLabel(f"+{hidden_count} more")
            label.setObjectName("selected_roi_overflow")
            self._chips_layout.addWidget(label, 2, 0, 1, 2)

    def _remove_roi(self, roi_label: str) -> None:
        """Remove one chip-selected ROI and refresh the card."""
        self._selected_rois = tuple(
            selected for selected in self._selected_rois if selected != roi_label
        )
        self._refresh_chips()
        self.refresh_preview()
        self._on_selection_changed()

    def _toggle_roi_at_preview_point(self, x: int, y: int) -> None:
        """Toggle the ROI under one preview-image click.

        Args:
            x: Horizontal click coordinate in preview-label pixels.
            y: Vertical click coordinate in preview-label pixels.

        Returns:
            None.

        Clicking an unselected ROI adds it. Clicking a selected ROI removes it
        so quick visual review does not require opening the dropdown.
        """
        pixmap = self._image_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        clicked_roi = _roi_label_at_preview_point(
            self.recording.roi_labels_layer,
            x=x,
            y=y,
            preview_width=self._image_label.width(),
            preview_height=self._image_label.height(),
            pixmap_width=pixmap.width(),
            pixmap_height=pixmap.height(),
        )
        if clicked_roi is None:
            return
        self._toggle_roi(clicked_roi)
        self._refresh_chips()
        self.refresh_preview()
        self._on_selection_changed()


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
    labels_layer = cast(NapariLayerWithData, layer)
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


def _roi_label_at_preview_point(
    layer: object | None,
    *,
    x: int,
    y: int,
    preview_width: int,
    preview_height: int,
    pixmap_width: int,
    pixmap_height: int,
) -> str | None:
    """Return the ROI label under one rendered preview coordinate.

    Args:
        layer: Napari Labels layer or test double exposing integer ``data``.
        x: Horizontal click coordinate in preview-label pixels.
        y: Vertical click coordinate in preview-label pixels.
        preview_width: Width of the fixed Qt label displaying the pixmap.
        preview_height: Height of the fixed Qt label displaying the pixmap.
        pixmap_width: Width of the aspect-preserving rendered pixmap.
        pixmap_height: Height of the aspect-preserving rendered pixmap.

    Returns:
        ``roi_####`` label under the click, or ``None`` for background, invalid
        geometry, clicks in letterbox padding, or unavailable label data.

    Group Matching previews scale images with preserved aspect ratio. This
    helper removes the centered padding before reading the underlying Labels
    image, so clicks select the cell mask the user sees rather than a distorted
    thumbnail coordinate.
    """
    label_image = _label_image(layer)
    if label_image is None:
        return None
    image_index = _image_index_at_preview_point(
        x=x,
        y=y,
        preview_width=preview_width,
        preview_height=preview_height,
        pixmap_width=pixmap_width,
        pixmap_height=pixmap_height,
        image_shape=label_image.shape,
    )
    if image_index is None:
        return None
    row_index, column_index = image_index
    try:
        label_value = int(label_image[row_index, column_index])
    except (OverflowError, TypeError, ValueError):
        return None
    return f"roi_{label_value:04d}" if label_value > 0 else None


def _label_image(layer: object | None) -> np.ndarray | None:
    """Return a two-dimensional Labels image from a napari-shaped layer."""
    if layer is None or not hasattr(layer, "data"):
        return None
    label_image = np.asarray(cast(NapariLayerWithData, layer).data)
    if label_image.ndim != 2 or 0 in label_image.shape:
        return None
    return label_image


def _image_index_at_preview_point(
    *,
    x: int,
    y: int,
    preview_width: int,
    preview_height: int,
    pixmap_width: int,
    pixmap_height: int,
    image_shape: tuple[int, ...],
) -> tuple[int, int] | None:
    """Return the image row and column under one preview-label coordinate."""
    if (
        preview_width <= 0
        or preview_height <= 0
        or pixmap_width <= 0
        or pixmap_height <= 0
        or len(image_shape) < 2
    ):
        return None
    image_height, image_width = image_shape[:2]
    if image_width <= 0 or image_height <= 0:
        return None
    x_offset = (preview_width - pixmap_width) / 2.0
    y_offset = (preview_height - pixmap_height) / 2.0
    if (
        x < x_offset
        or y < y_offset
        or x >= x_offset + pixmap_width
        or y >= y_offset + pixmap_height
    ):
        return None
    image_x = int((x - x_offset) * image_width / pixmap_width)
    image_y = int((y - y_offset) * image_height / pixmap_height)
    column_index = min(max(image_x, 0), image_width - 1)
    row_index = min(max(image_y, 0), image_height - 1)
    return row_index, column_index


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


def _remove_roi_icon() -> QIcon:
    """Return a small red remove marker for selected-ROI chips."""
    pixmap = QPixmap(14, 14)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setPen(QColor("#d62728"))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "x")
    painter.end()
    return QIcon(pixmap)


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
