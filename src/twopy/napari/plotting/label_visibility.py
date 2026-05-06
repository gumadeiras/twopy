"""Synchronize plot ROI visibility with the napari Labels overlay.

Inputs: the current napari Labels layer, ROI labels, ROI visibility state, and
ROI colors.
Outputs: a Labels-layer colormap where hidden ROIs are transparent.

This module only changes how labels are displayed. It never edits the label
image, so hiding an ROI in the plot options cannot delete ROI data.
"""

from typing import Any, Protocol, cast

from qtpy.QtGui import QColor

__all__ = ["apply_roi_visibility_to_labels_layer", "roi_label_value_from_label"]

_LABEL_COLOR_LOOKAHEAD = 4096
_BASE_LABEL_COLORMAP_METADATA_KEY = "twopy_base_label_colormap"


class _LabelsLayerWithColormap(Protocol):
    """Small protocol for napari Labels layers with settable colormaps."""

    colormap: object
    metadata: dict[str, object]


class _LabelColormapLike(Protocol):
    """Small protocol for napari label colormaps used for color sampling."""

    colors: object
    seed: object
    background_value: object


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

    Hidden ROIs keep their label pixels in ``layer.data``; only their display
    alpha becomes zero. Future labels stay drawable because the direct colormap
    contains napari-style colors for a broad range of label values instead of a
    single fallback color.
    """
    if layer is None or not hasattr(layer, "colormap"):
        return

    try:
        from napari.utils.colormaps import direct_colormap
    except ImportError:
        return

    labels_layer = cast(_LabelsLayerWithColormap, layer)
    color_dict = _visible_label_color_dict(
        labels_layer,
        roi_labels=roi_labels,
    )
    for roi_index, roi_label in enumerate(roi_labels):
        label_value = roi_label_value_from_label(
            roi_label,
            fallback=roi_index + 1,
        )
        color = colors[roi_index] if roi_index < len(colors) else QColor("#4cc9f0")
        color_dict[label_value] = _qcolor_rgba(
            color,
            visible=visibility.get(roi_label, True),
        )

    labels_layer.colormap = direct_colormap(color_dict)


def roi_label_value_from_label(label: str, *, fallback: int) -> int:
    """Return the napari integer label represented by one ROI label.

    Args:
        label: ROI label from analysis output, usually ``roi_0001``.
        fallback: One-based ROI position used when the label has no numeric
            suffix.

    Returns:
        Positive integer label value for the napari Labels layer.

    napari lets users draw label ``10`` without drawing labels ``1`` through
    ``9``. ``make_roi_set_from_label_image`` preserves that value in labels like
    ``roi_0010``, so visibility needs to target label ``10`` instead of the
    first plotted ROI position.
    """
    suffix = label.rsplit("_", maxsplit=1)[-1]
    if suffix.isdecimal():
        return int(suffix)
    return fallback


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


def _visible_label_color_dict(
    layer: _LabelsLayerWithColormap,
    *,
    roi_labels: tuple[str, ...],
) -> dict[int | None, object]:
    """Return visible colors for known and future Labels values.

    Args:
        layer: napari Labels layer.
        roi_labels: ROI names in analysis/plot order.

    Returns:
        Mapping from label values to RGBA colors for napari's direct colormap.
    """
    color_count = max(
        _LABEL_COLOR_LOOKAHEAD,
        _max_roi_label_value(roi_labels) + 512,
    )
    color_dict: dict[int | None, object] = {0: "transparent"}
    for label_value in range(1, color_count + 1):
        color_dict[label_value] = _base_label_rgba(layer, label_value)
    color_dict[None] = color_dict[color_count]
    return color_dict


def _max_roi_label_value(roi_labels: tuple[str, ...]) -> int:
    """Return the largest integer label represented by ROI labels.

    Args:
        roi_labels: ROI names in analysis/plot order.

    Returns:
        Maximum positive Labels value, or ``0`` when there are no ROIs.
    """
    if len(roi_labels) == 0:
        return 0
    return max(
        roi_label_value_from_label(roi_label, fallback=index + 1)
        for index, roi_label in enumerate(roi_labels)
    )


def _base_label_rgba(
    layer: _LabelsLayerWithColormap,
    label_value: int,
) -> tuple[float, float, float, float]:
    """Return napari's cyclic label color for one label value.

    Args:
        layer: napari Labels layer.
        label_value: Positive integer label value.

    Returns:
        RGBA tuple in ``0`` to ``1`` float range.
    """
    try:
        from napari.utils.colormaps import label_colormap
    except ImportError:
        return _qcolor_rgba(QColor("#4cc9f0"), visible=True)

    params = _base_label_colormap_params(layer)
    colormap = label_colormap(
        num_colors=params["num_colors"],
        seed=params["seed"],
        background_value=params["background_value"],
    )
    rgba = colormap.map(label_value)
    return (
        float(rgba[0]),
        float(rgba[1]),
        float(rgba[2]),
        float(rgba[3]),
    )


def _base_label_colormap_params(
    layer: _LabelsLayerWithColormap,
) -> dict[str, int | float]:
    """Return stable cyclic-colormap parameters for one Labels layer.

    Args:
        layer: napari Labels layer.

    Returns:
        Plain parameters for reconstructing napari's cyclic label colormap.

    The first call records the original cyclic colormap. Later calls may see a
    direct visibility colormap, so reading from metadata preserves the original
    color cycle while users toggle ROI visibility.
    """
    stored = layer.metadata.get(_BASE_LABEL_COLORMAP_METADATA_KEY)
    if isinstance(stored, dict):
        stored_params = cast(dict[str, object], stored)
        return {
            "num_colors": int(cast(Any, stored_params["num_colors"])),
            "seed": float(cast(Any, stored_params["seed"])),
            "background_value": int(cast(Any, stored_params["background_value"])),
        }

    colormap = cast(_LabelColormapLike, layer.colormap)
    colors = getattr(colormap, "colors", ())
    params = {
        "num_colors": max(len(colors) - 1, 1),
        "seed": float(getattr(colormap, "seed", 0.5)),
        "background_value": int(getattr(colormap, "background_value", 0)),
    }
    layer.metadata[_BASE_LABEL_COLORMAP_METADATA_KEY] = params
    return params
