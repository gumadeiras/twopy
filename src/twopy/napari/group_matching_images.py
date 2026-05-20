"""Image preview helpers for manual group matching.

Inputs: napari-shaped mean-image and Labels layers.
Outputs: Qt pixmaps used by FOV and ROI assignment cards.

This module owns display-only image normalization. It does not read recordings,
write group tables, or make ROI matching decisions.
"""

from typing import Protocol, cast

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QImage, QPainter, QPixmap

__all__ = [
    "mean_image_roi_overlay_pixmap",
    "mean_image_thumbnail_pixmap",
]

THUMBNAIL_SIZE = 220
ROI_PREVIEW_SIZE = 260


class _LayerWithData(Protocol):
    """Small protocol for napari layers whose image data can be previewed."""

    data: object


def mean_image_thumbnail_pixmap(
    layer: object,
    *,
    contrast_percentile: float,
) -> QPixmap | None:
    """Return a Qt pixmap thumbnail for a recording mean-image layer.

    Args:
        layer: Napari Image layer or test double exposing ``data``.
        contrast_percentile: Upper percentile used as the white point.

    Returns:
        ``QPixmap`` containing a grayscale thumbnail, or ``None`` when the
        layer data cannot be represented as a two-dimensional image.
    """
    image = _layer_image(layer)
    if image is None:
        return None
    thumbnail = _uint8_thumbnail(image, contrast_percentile=contrast_percentile)
    return _grayscale_pixmap(thumbnail, size=THUMBNAIL_SIZE)


def mean_image_roi_overlay_pixmap(
    *,
    mean_image_layer: object,
    roi_labels_layer: object | None,
    roi_label: str | None,
    selected_color: QColor,
    contrast_percentile: float,
) -> QPixmap | None:
    """Return a mean-image preview with one selected ROI highlighted.

    Args:
        mean_image_layer: Napari mean-image layer.
        roi_labels_layer: Napari Labels layer for the same displayed crop.
        roi_label: ``roi_####`` label to highlight, or ``None`` for no overlay.
        selected_color: Color used for the selected ROI mask and matching trace.
        contrast_percentile: Upper percentile used as the white point.

    Returns:
        Pixmap containing the grayscale mean image, low-opacity context for all
        ROIs, numeric ROI labels, and a selected-ROI overlay colored to match
        the recording trace, or ``None`` when image data are unavailable.
    """
    image = _layer_image(mean_image_layer)
    if image is None:
        return None
    rgba = _grayscale_rgba(image, contrast_percentile=contrast_percentile)
    label_value = _roi_label_value(roi_label)
    labels = _layer_image(roi_labels_layer)
    roi_centers: tuple[tuple[int, float, float], ...] = ()
    if labels is not None and labels.shape == image.shape:
        integer_labels = labels.astype(np.int64, copy=False)
        roi_centers = _roi_label_centers(integer_labels)
        _blend_roi_masks(
            rgba,
            integer_labels,
            selected_label=label_value,
            selected_color=selected_color,
        )
    pixmap = _rgba_pixmap(rgba, size=ROI_PREVIEW_SIZE)
    _draw_roi_numbers(
        pixmap,
        roi_centers=roi_centers,
        image_shape=image.shape,
        selected_label=label_value,
    )
    return pixmap


def _layer_image(layer: object | None) -> np.ndarray | None:
    """Return a 2D array from a napari-shaped layer."""
    if layer is None or not hasattr(layer, "data"):
        return None
    image = np.asarray(cast(_LayerWithData, layer).data)
    return image if image.ndim == 2 else None


def _uint8_thumbnail(
    image: np.ndarray,
    *,
    contrast_percentile: float,
) -> np.ndarray:
    """Normalize one image to a contiguous uint8 grayscale thumbnail array."""
    values = np.asarray(image, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    minimum = float(np.min(finite_values))
    clipped_percentile = min(max(contrast_percentile, 1.0), 100.0)
    maximum = float(np.percentile(finite_values, clipped_percentile))
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)
    return np.ascontiguousarray(np.round(scaled * 255.0).astype(np.uint8))


def _grayscale_rgba(
    image: np.ndarray,
    *,
    contrast_percentile: float,
) -> np.ndarray:
    """Return a normalized grayscale image as opaque RGBA."""
    gray = _uint8_thumbnail(image, contrast_percentile=contrast_percentile)
    rgba = np.empty((*gray.shape, 4), dtype=np.uint8)
    rgba[:, :, 0] = gray
    rgba[:, :, 1] = gray
    rgba[:, :, 2] = gray
    rgba[:, :, 3] = 255
    return rgba


def _grayscale_pixmap(image: np.ndarray, *, size: int) -> QPixmap:
    """Return a scaled grayscale pixmap."""
    height, width = image.shape
    qimage = QImage(
        image.tobytes(),
        width,
        height,
        width,
        QImage.Format.Format_Grayscale8,
    ).copy()
    return _scaled_pixmap(qimage, size=size)


def _rgba_pixmap(image: np.ndarray, *, size: int) -> QPixmap:
    """Return a scaled RGBA pixmap."""
    height, width = image.shape[:2]
    qimage = QImage(
        image.tobytes(),
        width,
        height,
        image.strides[0],
        QImage.Format.Format_RGBA8888,
    ).copy()
    return _scaled_pixmap(qimage, size=size)


def _blend_roi_masks(
    rgba: np.ndarray,
    labels: np.ndarray,
    *,
    selected_label: int | None,
    selected_color: QColor,
) -> None:
    """Blend all ROI masks into the preview, emphasizing the selected mask."""
    roi_values = tuple(
        int(value)
        for value in np.unique(labels)
        if int(value) > 0 and int(value) != selected_label
    )
    for roi_value in roi_values:
        _blend_mask_color(
            rgba,
            mask=labels == roi_value,
            color=(31, 180, 190),
            alpha=0.28,
        )
    if selected_label is not None:
        selected_mask = labels == selected_label
        rgba[selected_mask, 0] = selected_color.red()
        rgba[selected_mask, 1] = selected_color.green()
        rgba[selected_mask, 2] = selected_color.blue()
        rgba[selected_mask, 3] = 255


def _blend_mask_color(
    rgba: np.ndarray,
    *,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Alpha-blend one color into selected pixels of an opaque RGBA image."""
    if not np.any(mask):
        return
    base = rgba[mask, :3].astype(np.float64, copy=False)
    blended = np.round((1.0 - alpha) * base + alpha * np.asarray(color))
    rgba[mask, :3] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    rgba[mask, 3] = 255


def _roi_label_centers(labels: np.ndarray) -> tuple[tuple[int, float, float], ...]:
    """Return label value and centroid coordinates for positive ROI labels."""
    centers: list[tuple[int, float, float]] = []
    for label_value in sorted(int(value) for value in np.unique(labels) if value > 0):
        y_indices, x_indices = np.nonzero(labels == label_value)
        if y_indices.size == 0:
            continue
        centers.append(
            (
                label_value,
                float(np.mean(x_indices)),
                float(np.mean(y_indices)),
            ),
        )
    return tuple(centers)


def _draw_roi_numbers(
    pixmap: QPixmap,
    *,
    roi_centers: tuple[tuple[int, float, float], ...],
    image_shape: tuple[int, ...],
    selected_label: int | None,
) -> None:
    """Draw ROI numbers at scaled ROI centroids on top of a preview pixmap."""
    if len(roi_centers) == 0 or len(image_shape) < 2:
        return
    height, width = image_shape[:2]
    if height == 0 or width == 0:
        return
    painter = QPainter(pixmap)
    try:
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        x_scale = pixmap.width() / width
        y_scale = pixmap.height() / height
        for label_value, x_center, y_center in roi_centers:
            text = str(label_value)
            x = int(round(x_center * x_scale))
            y = int(round(y_center * y_scale))
            foreground = (
                QColor(255, 246, 230)
                if label_value == selected_label
                else QColor(215, 255, 255)
            )
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(x + 1, y + 1, text)
            painter.setPen(foreground)
            painter.drawText(x, y, text)
    finally:
        painter.end()


def _scaled_pixmap(image: QImage, *, size: int) -> QPixmap:
    """Return a smoothly scaled pixmap with preserved aspect ratio."""
    return QPixmap.fromImage(image).scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _roi_label_value(roi_label: str | None) -> int | None:
    """Return the integer Labels value encoded in ``roi_####`` text."""
    if roi_label is None:
        return None
    suffix = roi_label.removeprefix("roi_")
    return int(suffix) if suffix.isdecimal() else None
