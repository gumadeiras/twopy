"""Response heatmap rendering for the twopy napari dock.

Inputs: response-map data, visible epoch rows, and a shared display size.
Outputs: Qt widgets laid out as one heatmap panel per visible epoch.

This module owns only Qt rendering/cache state. Response-map computation lives
in ``twopy.analysis.response_maps`` and export rendering lives in the export
module.
"""

import numpy as np
import numpy.typing as npt
from qtpy.QtCore import QRectF, QSize, Qt
from qtpy.QtGui import QImage, QPainter, QPaintEvent
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from twopy.analysis.response_maps import EpochResponseMap, ResponseMapData
from twopy.napari.plotting.response_map_colors import RESPONSE_HEATMAP_COLORMAP
from twopy.napari.plotting.response_map_display import (
    display_response_limit,
    display_response_values,
)
from twopy.napari.plotting.widgets import clear_layout

__all__ = ["ResponseMapArea"]

RESPONSE_OVERLAY_ALPHA = 0.42
_COLORBAR_HEIGHT = 34
_MAP_COLORBAR_SPACING = 6


class ResponseMapArea:
    """Manage the scrollable response-heatmap strip.

    Args:
        initial_status: Status shown before maps are available.

    Returns:
        A scroll area plus methods for status and heatmap rendering.

    Heatmap panels cache one rendered Qt image per epoch. Changing computation
    options replaces the cached data; changing epoch visibility or plot size
    reuses the existing widgets. Shared display limits repaint the cached image
    from the same computed ``ResponseMapData`` without recomputing dF/F maps.
    """

    def __init__(self, initial_status: str) -> None:
        """Create an empty response-map area."""
        self.epoch_map_widgets: dict[int, EpochMapWidget] = {}
        self.epoch_colorbar_widgets: dict[int, EpochColorbarWidget] = {}
        self.epoch_map_panels: dict[int, QWidget] = {}
        self._epoch_keys: dict[int, tuple[int, str]] = {}
        self._status_label = QLabel(initial_status)
        self._status_label.setWordWrap(True)
        self._layout = QHBoxLayout()
        self._layout.addWidget(self._status_label)

        content = QWidget()
        content.setLayout(self._layout)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(content)

    def set_status(self, text: str) -> None:
        """Replace heatmaps with one status message."""
        self.clear_epoch_cache()
        clear_layout(self._layout)
        self._status_label = QLabel(text)
        self._status_label.setWordWrap(True)
        self._layout.addWidget(self._status_label)
        self._layout.addStretch(1)

    def clear(self) -> None:
        """Clear all heatmap widgets."""
        self.clear_epoch_cache()
        clear_layout(self._layout)

    def render(
        self,
        *,
        map_data: ResponseMapData,
        epoch_indices: tuple[int, ...],
        map_size: int,
        shared_limits: bool,
    ) -> None:
        """Render visible epoch heatmap panels."""
        self.clear_layout_preserving_epoch_cache()
        self.ensure_epoch_map_cache(
            map_data=map_data,
            map_size=map_size,
            shared_limits=shared_limits,
        )
        self.render_cached_epoch_widgets(
            epoch_indices=epoch_indices,
            map_size=map_size,
        )

    def render_cached_epoch_widgets(
        self,
        *,
        epoch_indices: tuple[int, ...],
        map_size: int,
    ) -> None:
        """Show selected cached heatmap panels and update their display size."""
        self.clear_layout_preserving_epoch_cache()
        for epoch_index in epoch_indices:
            if epoch_index not in self.epoch_map_panels:
                continue
            self._layout.addWidget(self.epoch_map_panels[epoch_index])
            self.epoch_map_panels[epoch_index].show()
        self._layout.addStretch(1)
        self.update_epoch_map_widgets(
            epoch_indices=epoch_indices,
            map_size=map_size,
        )

    def update_epoch_map_widgets(
        self,
        *,
        epoch_indices: tuple[int, ...],
        map_size: int,
    ) -> None:
        """Update display sizing for cached heatmap widgets."""
        image_size = _heatmap_image_size(map_size)
        for epoch_index in epoch_indices:
            if epoch_index not in self.epoch_map_widgets:
                continue
            self.epoch_map_widgets[epoch_index].update_display(map_size=image_size)
            self.epoch_colorbar_widgets[epoch_index].update_display(
                map_size=image_size,
            )

    def has_epoch_widgets(self, epoch_indices: tuple[int, ...]) -> bool:
        """Return whether all requested epoch widgets exist."""
        return all(index in self.epoch_map_widgets for index in epoch_indices)

    def ensure_epoch_map_cache(
        self,
        *,
        map_data: ResponseMapData | None,
        map_size: int,
        shared_limits: bool,
    ) -> None:
        """Create one persistent heatmap widget per loaded epoch."""
        if map_data is None:
            self.clear_epoch_cache()
            return
        image_size = _heatmap_image_size(map_size)
        expected_indices = set(range(len(map_data.epochs)))
        for epoch_index in tuple(self.epoch_map_widgets):
            expected_key = (
                _epoch_key(map_data.epochs[epoch_index])
                if epoch_index in expected_indices
                else None
            )
            if (
                epoch_index not in expected_indices
                or self._epoch_keys.get(epoch_index) != expected_key
            ):
                self.epoch_map_panels[epoch_index].deleteLater()
                del self.epoch_map_panels[epoch_index]
                del self.epoch_map_widgets[epoch_index]
                del self.epoch_colorbar_widgets[epoch_index]
                self._epoch_keys.pop(epoch_index, None)
        for epoch_index, epoch in enumerate(map_data.epochs):
            if epoch_index in self.epoch_map_widgets:
                self.epoch_map_widgets[epoch_index].update_data(
                    map_data,
                    epoch,
                    shared_limits=shared_limits,
                )
                continue
            widget = EpochMapWidget(
                map_data,
                epoch,
                map_size=image_size,
                shared_limits=shared_limits,
            )
            colorbar = EpochColorbarWidget(map_size=image_size)
            self.epoch_map_widgets[epoch_index] = widget
            self.epoch_colorbar_widgets[epoch_index] = colorbar
            self._epoch_keys[epoch_index] = _epoch_key(epoch)
            panel = _epoch_map_panel(
                title=f"Epoch {epoch.epoch_number}: {epoch.epoch_name}",
                map_widget=widget,
                colorbar_widget=colorbar,
            )
            panel.hide()
            self.epoch_map_panels[epoch_index] = panel

    def clear_epoch_cache(self) -> None:
        """Delete cached heatmap panels and widgets."""
        for panel in self.epoch_map_panels.values():
            panel.deleteLater()
        self.epoch_map_widgets.clear()
        self.epoch_colorbar_widgets.clear()
        self.epoch_map_panels.clear()
        self._epoch_keys.clear()

    def clear_layout_preserving_epoch_cache(self) -> None:
        """Clear the strip while keeping cached heatmap panels alive."""
        cached_panels = set(self.epoch_map_panels.values())
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is None:
                continue
            if widget in cached_panels:
                widget.hide()
            else:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()


class EpochMapWidget(QWidget):
    """Custom Qt widget that draws one epoch response heatmap."""

    def __init__(
        self,
        map_data: ResponseMapData,
        epoch: EpochResponseMap,
        *,
        map_size: int,
        shared_limits: bool,
    ) -> None:
        """Create one heatmap widget."""
        super().__init__()
        self._map_size = int(map_size)
        self._image = _rgba_qimage(
            map_data,
            epoch,
            shared_limits=shared_limits,
        )
        self.setFixedSize(self._map_size, self._map_size)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        """Return the preferred square heatmap size."""
        return QSize(self._map_size, self._map_size)

    def update_data(
        self,
        map_data: ResponseMapData,
        epoch: EpochResponseMap,
        *,
        shared_limits: bool,
    ) -> None:
        """Replace computed heatmap data without rebuilding the widget."""
        self._image = _rgba_qimage(
            map_data,
            epoch,
            shared_limits=shared_limits,
        )
        self.update()

    def update_display(self, *, map_size: int) -> None:
        """Update visible heatmap size."""
        self._map_size = int(map_size)
        self.setFixedSize(self._map_size, self._map_size)
        self.updateGeometry()
        self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """Draw the heatmap image."""
        del a0
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        painter.drawImage(QRectF(self.rect()), self._image)
        painter.end()


class EpochColorbarWidget(QWidget):
    """Custom Qt widget that draws the shared signed heatmap colorbar."""

    def __init__(self, *, map_size: int) -> None:
        """Create one horizontal colorbar widget."""
        super().__init__()
        self._map_size = int(map_size)
        self.setFixedSize(self._map_size, _COLORBAR_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        """Return the preferred colorbar size."""
        return QSize(self._map_size, _COLORBAR_HEIGHT)

    def update_display(self, *, map_size: int) -> None:
        """Update visible colorbar width."""
        self._map_size = int(map_size)
        self.setFixedSize(self._map_size, _COLORBAR_HEIGHT)
        self.updateGeometry()
        self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """Draw the signed response colorbar."""
        del a0
        painter = QPainter(self)
        painter.drawImage(
            QRectF(0.0, 0.0, float(self.width()), 12.0), _colorbar_image()
        )
        painter.drawText(0, 30, "-")
        painter.drawText(self.width() // 2 - 4, 30, "0")
        painter.drawText(self.width() - 10, 30, "+")
        painter.end()


def _epoch_map_panel(
    *,
    title: str,
    map_widget: EpochMapWidget,
    colorbar_widget: EpochColorbarWidget,
) -> QWidget:
    """Create one titled heatmap panel."""
    panel = QWidget()
    layout = QVBoxLayout()
    layout.setSpacing(_MAP_COLORBAR_SPACING)
    title_label = QLabel(title)
    title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    layout.addWidget(title_label)
    layout.addWidget(map_widget)
    layout.addWidget(colorbar_widget)
    panel.setLayout(layout)
    return panel


def _heatmap_image_size(map_size: int) -> int:
    """Return the response-map image side for one requested heatmap size.

    The Plot-tab Size control represents the heatmap content budget, matching
    the way response plots include axis-label space inside the requested size.
    The colorbar is part of that content, so the displayed image is smaller
    than the requested budget by the colorbar height and the fixed gap above it.
    """
    return max(1, int(map_size) - _COLORBAR_HEIGHT - _MAP_COLORBAR_SPACING)


def _rgba_qimage(
    map_data: ResponseMapData,
    epoch: EpochResponseMap,
    *,
    shared_limits: bool,
) -> QImage:
    """Return a copied RGBA QImage for one response map.

    The mean image is rendered as robust grayscale first, then the signed
    response overlay is blended over it. Background or masked response pixels
    are treated as zero response for display so the mean image stays visible.
    """
    background = _grayscale_rgba(map_data.mean_image)
    response = display_response_values(
        map_data,
        epoch,
        shared_limits=shared_limits,
    )
    response_limit = display_response_limit(
        map_data,
        epoch,
        shared_limits=shared_limits,
    )
    rgba = _blend_signed_response(
        background,
        response,
        response_limit=response_limit,
    )
    height, width = rgba.shape[:2]
    image_bytes = rgba.tobytes()
    return QImage(
        image_bytes,
        width,
        height,
        rgba.strides[0],
        QImage.Format.Format_RGBA8888,
    ).copy()


def _grayscale_rgba(image: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """Return a normalized grayscale image as opaque RGBA."""
    background = _normalized_gray(image)
    gray = np.asarray(np.clip(background * 255.0, 0.0, 255.0), dtype=np.uint8)
    rgba = np.empty((*background.shape, 4), dtype=np.uint8)
    rgba[:, :, 0] = gray
    rgba[:, :, 1] = gray
    rgba[:, :, 2] = gray
    rgba[:, :, 3] = 255
    return rgba


def _blend_signed_response(
    rgba: npt.NDArray[np.uint8],
    response: npt.NDArray[np.float64],
    *,
    response_limit: float,
) -> npt.NDArray[np.uint8]:
    """Return RGBA with signed response values blended over grayscale.

    ``response_limit`` is the robust absolute color limit. Values at ``-limit``
    use the blue end of the map, values at ``+limit`` use orange, and zero uses
    black. The overlay alpha is fixed here so live rendering and export stay
    visually aligned.
    """
    limit = max(float(response_limit), np.finfo(np.float64).eps)
    filled_response = np.asarray(response, dtype=np.float64).copy()
    filled_response[np.isnan(filled_response)] = 0.0
    filled_response[np.isposinf(filled_response)] = limit
    filled_response[np.isneginf(filled_response)] = -limit
    normalized = np.clip((filled_response / limit + 1.0) / 2.0, 0.0, 1.0)
    colors = np.asarray(
        RESPONSE_HEATMAP_COLORMAP(normalized),
        dtype=float,
    )
    heat = np.asarray(np.clip(colors[:, :, :3] * 255.0, 0.0, 255.0), dtype=np.uint8)
    for channel in range(3):
        rgba[:, :, channel] = np.asarray(
            (1.0 - RESPONSE_OVERLAY_ALPHA) * rgba[:, :, channel]
            + RESPONSE_OVERLAY_ALPHA * heat[:, :, channel],
            dtype=np.uint8,
        )
    return rgba


def _colorbar_image() -> QImage:
    """Return a copied horizontal signed-response colorbar image."""
    values = np.linspace(-1.0, 1.0, 256, dtype=np.float64)[None, :]
    normalized = (values + 1.0) / 2.0
    colors = np.asarray(RESPONSE_HEATMAP_COLORMAP(normalized), dtype=float)
    rgba = np.asarray(np.clip(colors * 255.0, 0.0, 255.0), dtype=np.uint8)
    height, width = rgba.shape[:2]
    image_bytes = rgba.tobytes()
    return QImage(
        image_bytes,
        width,
        height,
        rgba.strides[0],
        QImage.Format.Format_RGBA8888,
    ).copy()


def _normalized_gray(image: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a robustly normalized grayscale image."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.float64)
    low, high = np.percentile(finite, (1.0, 99.5))
    scale = max(float(high - low), np.finfo(np.float64).eps)
    return np.clip((image - low) / scale, 0.0, 1.0)


def _epoch_key(epoch: EpochResponseMap) -> tuple[int, str]:
    """Return the stable identity for one heatmap panel."""
    return (epoch.epoch_number, epoch.epoch_name)
