"""Image preview helpers for manual group matching.

Inputs: napari-shaped mean-image and Labels layers.
Outputs: Qt pixmaps used by FOV and ROI assignment cards.

This module owns display-only image normalization. It does not read recordings,
write group tables, or make ROI matching decisions.
"""

from typing import Protocol, cast

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap

__all__ = [
    "mean_image_roi_overlay_pixmap",
    "mean_image_thumbnail_pixmap",
]

THUMBNAIL_SIZE = 180
ROI_PREVIEW_SIZE = 220


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
    contrast_percentile: float,
) -> QPixmap | None:
    """Return a mean-image preview with one selected ROI highlighted.

    Args:
        mean_image_layer: Napari mean-image layer.
        roi_labels_layer: Napari Labels layer for the same displayed crop.
        roi_label: ``roi_####`` label to highlight, or ``None`` for no overlay.
        contrast_percentile: Upper percentile used as the white point.

    Returns:
        Pixmap containing grayscale mean image and an orange selected-ROI
        overlay, or ``None`` when image data are unavailable.
    """
    image = _layer_image(mean_image_layer)
    if image is None:
        return None
    rgba = _grayscale_rgba(image, contrast_percentile=contrast_percentile)
    label_value = _roi_label_value(roi_label)
    labels = _layer_image(roi_labels_layer)
    if label_value is not None and labels is not None and labels.shape == image.shape:
        mask = labels.astype(np.int64, copy=False) == label_value
        rgba[mask, 0] = 255
        rgba[mask, 1] = 136
        rgba[mask, 2] = 0
        rgba[mask, 3] = 255
    return _rgba_pixmap(rgba, size=ROI_PREVIEW_SIZE)


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
