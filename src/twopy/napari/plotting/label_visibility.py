"""Synchronize plot ROI visibility with the napari Labels overlay.

Inputs: the current napari Labels layer, ROI labels, ROI visibility state, and
ROI colors.
Outputs: a Labels-layer colormap where deselected ROIs are dimmed.

This module only changes how labels are displayed. It never edits the label
image, so hiding an ROI in the plot options cannot delete ROI data.
"""

from collections.abc import Mapping
from typing import Protocol, cast

from qtpy.QtGui import QColor

from twopy.roi import roi_label_value_from_label, roi_label_values_from_labels
from twopy.typing_guards import int_or_none, string_key_mapping_or_none

__all__ = [
    "apply_roi_visibility_to_labels_layer",
    "roi_label_value_from_label",
    "roi_label_values_from_labels",
]

_LABEL_COLOR_LOOKAHEAD = 4096
_BASE_LABEL_COLORMAP_METADATA_KEY = "twopy_base_label_colormap"
_DESELECTED_ROI_ALPHA = 0.2
type VisibilityKey = int | str
type VisibilityState = Mapping[int, bool] | Mapping[str, bool]


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
    visibility: VisibilityState,
    colors: tuple[QColor, ...],
    keys: tuple[VisibilityKey, ...] | None = None,
) -> None:
    """Apply ROI visibility state to the napari Labels layer display.

    Args:
        layer: Current napari Labels layer, if available.
        roi_labels: ROI names in analysis/plot order.
        visibility: Visibility by ROI name, or by ``keys`` when keys are
            provided.
        colors: ROI colors in the same order as ``roi_labels``.
        keys: Optional stable visibility keys matching ``roi_labels``.

    Returns:
        None.

    Deselected ROIs keep their label pixels in ``layer.data``; only their
    display alpha is reduced. Future labels stay drawable because the direct
    colormap contains napari-style colors for a broad range of label values
    instead of a single fallback color.
    """
    if layer is None or not hasattr(layer, "colormap"):
        return

    try:
        from napari.utils.colormaps import direct_colormap
    except ImportError:
        return

    labels_layer = cast(_LabelsLayerWithColormap, layer)
    visibility_lookup = dict(visibility.items())
    color_dict = _visible_label_color_dict(
        labels_layer,
        roi_labels=roi_labels,
    )
    for roi_index, roi_label in enumerate(roi_labels):
        label_value = roi_label_value_from_label(
            roi_label,
            fallback=roi_index + 1,
        )
        key = (
            keys[roi_index] if keys is not None and roi_index < len(keys) else roi_label
        )
        color = colors[roi_index] if roi_index < len(colors) else QColor("#4cc9f0")
        color_dict[label_value] = _qcolor_rgba(color, visibility_lookup.get(key, True))

    labels_layer.colormap = direct_colormap(color_dict)


def _qcolor_rgba(color: QColor, visible: bool) -> tuple[float, float, float, float]:
    """Return a normalized RGBA tuple for one ROI label.

    Args:
        color: ROI color.
        visible: Whether the ROI should be fully visible in the Labels overlay.

    Returns:
        RGBA tuple in napari's expected ``0`` to ``1`` float range.
    """
    return (
        color.redF(),
        color.greenF(),
        color.blueF(),
        1.0 if visible else _DESELECTED_ROI_ALPHA,
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
        return _qcolor_rgba(QColor("#4cc9f0"), True)

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
    stored_params = string_key_mapping_or_none(stored)
    if stored_params is not None:
        params = _parsed_label_colormap_params(stored_params)
        if params is not None:
            return params

    colormap = cast(_LabelColormapLike, layer.colormap)
    colors = getattr(colormap, "colors", ())
    params = {
        "num_colors": max(len(colors) - 1, 1),
        "seed": float(getattr(colormap, "seed", 0.5)),
        "background_value": int(getattr(colormap, "background_value", 0)),
    }
    layer.metadata[_BASE_LABEL_COLORMAP_METADATA_KEY] = params
    return params


def _parsed_label_colormap_params(
    params: dict[str, object],
) -> dict[str, int | float] | None:
    """Return persisted label-colormap parameters when all fields parse."""
    num_colors = int_or_none(params.get("num_colors"))
    seed = _float_or_none(params.get("seed"))
    background_value = int_or_none(params.get("background_value"))
    if num_colors is None or seed is None or background_value is None:
        return None
    return {
        "num_colors": num_colors,
        "seed": seed,
        "background_value": background_value,
    }


def _float_or_none(value: object) -> float | None:
    """Return a float for simple numeric metadata values."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except ValueError:
        return None
