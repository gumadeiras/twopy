"""Synchronize plot ROI visibility with the napari Labels overlay.

Inputs: the current napari Labels layer, ROI labels, ROI visibility state, and
ROI colors.
Outputs: a Labels-layer colormap where hidden ROIs are transparent.

This module only changes how labels are displayed. It never edits the label
image, so hiding an ROI in the plot options cannot delete ROI data.
"""

from typing import Protocol, cast

from qtpy.QtGui import QColor

__all__ = ["apply_roi_visibility_to_labels_layer"]


class _LabelsLayerWithColormap(Protocol):
    """Small protocol for napari Labels layers with settable colormaps."""

    colormap: object


def apply_roi_visibility_to_labels_layer(
    layer: object | None,
    *,
    roi_labels: tuple[str, ...],
    visibility: dict[str, bool],
    colors: tuple[QColor, ...],
) -> None:
    """Apply ROI visibility state to the napari Labels layer display.

    Args:
        layer: Current napari Labels layer, if available.
        roi_labels: ROI names in analysis/plot order.
        visibility: Visibility by ROI name.
        colors: ROI colors in the same order as ``roi_labels``.

    Returns:
        None.

    The Labels layer stores ROI identity as integer labels: ROI index ``0`` maps
    to Labels value ``1``. Hidden ROIs keep their label pixels in ``layer.data``;
    only their display alpha becomes zero.
    """
    if layer is None or not hasattr(layer, "colormap"):
        return

    try:
        from napari.utils.colormaps import direct_colormap
    except ImportError:
        return

    color_dict: dict[int | None, object] = {0: "transparent", None: "transparent"}
    for roi_index, roi_label in enumerate(roi_labels):
        label_value = roi_index + 1
        color = colors[roi_index] if roi_index < len(colors) else QColor("#4cc9f0")
        color_dict[label_value] = _qcolor_rgba(
            color,
            visible=visibility.get(roi_label, True),
        )

    labels_layer = cast(_LabelsLayerWithColormap, layer)
    labels_layer.colormap = direct_colormap(color_dict)


def _qcolor_rgba(color: QColor, *, visible: bool) -> tuple[float, float, float, float]:
    """Return a normalized RGBA tuple for one ROI label.

    Args:
        color: ROI color.
        visible: Whether the ROI should be visible in the Labels overlay.

    Returns:
        RGBA tuple in napari's expected ``0`` to ``1`` float range.
    """
    return (
        color.redF(),
        color.greenF(),
        color.blueF(),
        1.0 if visible else 0.0,
    )
