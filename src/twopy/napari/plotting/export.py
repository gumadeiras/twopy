"""Export napari response views as Illustrator-friendly figures.

Inputs: the current napari viewer state, loaded recording metadata, ROI Labels
data, and plot-ready response traces.
Outputs: PDF and PNG files for recording views, ROI overlays, and response
plots.

This module owns figure export only. The live napari widgets decide when to
call it, while this file keeps publication-output choices explicit and easy to
audit.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, cast

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from twopy.converted import RecordingData
from twopy.napari.display import display_image_from_movie_image
from twopy.napari.plotting.data import EpochResponsePlotData, ResponsePlotData

__all__ = [
    "export_epoch_plots",
    "export_epoch_roi_overlay_plots",
    "export_recording_roi_overlay",
    "export_recording_view",
    "export_roi_view",
]

MM_TO_INCH = 1.0 / 25.4
DEFAULT_FORMATS = ("pdf", "png")


class _LayerWithData(Protocol):
    """Small protocol for napari layers that expose array data."""

    data: object


class _LayerList(Protocol):
    """Small protocol for napari layer lists used by export."""

    def __getitem__(self, name: str) -> object:
        """Return a layer by name.

        Args:
            name: Layer name.

        Returns:
            Matching layer object.
        """
        ...

    def __iter__(self) -> Iterator[object]:
        """Return an iterator over layers.

        Args:
            None.

        Returns:
            Layer iterator.
        """
        ...


class _ViewerWithLayers(Protocol):
    """Small protocol for napari viewers that expose a layer list."""

    layers: _LayerList


class _ViewerWithDims(Protocol):
    """Small protocol for napari viewers that expose dimension state."""

    dims: object


def export_recording_view(
    *,
    viewer: object | None,
    recording: RecordingData,
    output_dir: Path,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> tuple[Path, ...]:
    """Save the current movie frame, or the mean image when no movie is loaded.

    Args:
        viewer: Napari viewer used to find the current displayed movie frame.
        recording: Loaded converted recording, used for mean-image fallback.
        output_dir: Directory where exported files should be written.
        formats: File formats to write.

    Returns:
        Paths written to disk.
    """
    image = current_recording_image(viewer=viewer, recording=recording)
    fig, ax = image_figure(image.shape)
    draw_recording_image(ax, image)
    return save_figure_bundle(
        fig,
        output_dir / "recording_view" / "recording_view",
        formats=formats,
    )


def export_roi_view(
    *,
    roi_labels_layer: object | None,
    recording: RecordingData,
    output_dir: Path,
    roi_label_values: tuple[int, ...],
    roi_colors: tuple[str, ...],
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> tuple[Path, ...]:
    """Save current ROI contours without a recording image underneath.

    Args:
        roi_labels_layer: Current napari Labels layer.
        recording: Loaded converted recording, used for empty-label fallback.
        output_dir: Directory where exported files should be written.
        roi_label_values: Integer Labels values in plot ROI order.
        roi_colors: Hex colors in plot ROI order.
        formats: File formats to write.

    Returns:
        Paths written to disk.
    """
    labels = current_roi_labels(
        roi_labels_layer=roi_labels_layer,
        recording=recording,
    )
    fig, ax = image_figure(labels.shape)
    draw_roi_contours(
        ax,
        labels=labels,
        roi_label_values=roi_label_values,
        roi_indices=tuple(range(len(roi_label_values))),
        roi_colors=roi_colors,
    )
    return save_figure_bundle(
        fig, output_dir / "roi_view" / "roi_view", formats=formats
    )


def export_recording_roi_overlay(
    *,
    viewer: object | None,
    recording: RecordingData,
    roi_labels_layer: object | None,
    output_dir: Path,
    roi_label_values: tuple[int, ...],
    roi_colors: tuple[str, ...],
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> tuple[Path, ...]:
    """Save current recording image with all current ROI contours overlaid.

    Args:
        viewer: Napari viewer used to find the current displayed movie frame.
        recording: Loaded converted recording, used for mean-image fallback.
        roi_labels_layer: Current napari Labels layer.
        output_dir: Directory where exported files should be written.
        roi_label_values: Integer Labels values in plot ROI order.
        roi_colors: Hex colors in plot ROI order.
        formats: File formats to write.

    Returns:
        Paths written to disk.
    """
    image = current_recording_image(viewer=viewer, recording=recording)
    labels = current_roi_labels(
        roi_labels_layer=roi_labels_layer,
        recording=recording,
    )
    fig, ax = image_figure(image.shape)
    draw_recording_image(ax, image)
    draw_roi_contours(
        ax,
        labels=labels,
        roi_label_values=roi_label_values,
        roi_indices=tuple(range(len(roi_label_values))),
        roi_colors=roi_colors,
    )
    return save_figure_bundle(
        fig,
        output_dir / "recording_roi_overlay" / "recording_roi_overlay",
        formats=formats,
    )


def export_epoch_plots(
    *,
    plot_data: ResponsePlotData,
    output_dir: Path,
    epoch_keys: tuple[tuple[int, str], ...],
    roi_indices: tuple[int, ...],
    roi_colors: tuple[str, ...],
    show_sem: bool,
    time_bounds: tuple[float, float],
    value_bounds: tuple[float, float],
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> tuple[Path, ...]:
    """Save one editable response plot per visible stimulus epoch.

    Args:
        plot_data: Plot-ready response data.
        output_dir: Directory where exported files should be written.
        epoch_keys: Visible ``(epoch_number, epoch_name)`` pairs.
        roi_indices: Zero-based ROI indices to draw.
        roi_colors: Hex colors in plot ROI order.
        show_sem: Whether to draw SEM bands.
        time_bounds: Shared x-axis bounds.
        value_bounds: Shared y-axis bounds.
        formats: File formats to write.

    Returns:
        Paths written to disk.
    """
    written: list[Path] = []
    for epoch in _visible_epochs(plot_data, epoch_keys):
        fig, ax = plot_figure()
        draw_epoch_response_plot(
            ax,
            epoch=epoch,
            roi_indices=roi_indices,
            roi_colors=roi_colors,
            show_sem=show_sem,
            time_bounds=time_bounds,
            value_bounds=value_bounds,
        )
        ax.set_title(_epoch_title(epoch), fontsize=7)
        written.extend(
            save_figure_bundle(
                fig,
                output_dir / "plots" / safe_stem(_epoch_title(epoch)),
                formats=formats,
            )
        )
    return tuple(written)


def export_epoch_roi_overlay_plots(
    *,
    viewer: object | None,
    recording: RecordingData,
    roi_labels_layer: object | None,
    plot_data: ResponsePlotData,
    output_dir: Path,
    epoch_keys: tuple[tuple[int, str], ...],
    roi_indices: tuple[int, ...],
    roi_label_values: tuple[int, ...],
    roi_colors: tuple[str, ...],
    show_sem: bool,
    time_bounds: tuple[float, float],
    value_bounds: tuple[float, float],
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> tuple[Path, ...]:
    """Save one two-column ROI overlay plus response plot per epoch and ROI.

    Args:
        viewer: Napari viewer used to find the current displayed movie frame.
        recording: Loaded converted recording, used for mean-image fallback.
        roi_labels_layer: Current napari Labels layer.
        plot_data: Plot-ready response data.
        output_dir: Directory where exported files should be written.
        epoch_keys: Visible ``(epoch_number, epoch_name)`` pairs.
        roi_indices: Zero-based ROI indices to draw.
        roi_label_values: Integer Labels values in plot ROI order.
        roi_colors: Hex colors in plot ROI order.
        show_sem: Whether to draw SEM bands.
        time_bounds: Shared x-axis bounds.
        value_bounds: Shared y-axis bounds.
        formats: File formats to write.

    Returns:
        Paths written to disk.
    """
    image = current_recording_image(viewer=viewer, recording=recording)
    labels = current_roi_labels(
        roi_labels_layer=roi_labels_layer,
        recording=recording,
    )
    written: list[Path] = []
    for epoch in _visible_epochs(plot_data, epoch_keys):
        for roi_index in roi_indices:
            fig, image_ax, plot_ax = overlay_plot_figure(image.shape)
            draw_recording_image(image_ax, image)
            draw_roi_contours(
                image_ax,
                labels=labels,
                roi_label_values=roi_label_values,
                roi_indices=(roi_index,),
                roi_colors=roi_colors,
            )
            draw_epoch_response_plot(
                plot_ax,
                epoch=epoch,
                roi_indices=(roi_index,),
                roi_colors=roi_colors,
                show_sem=show_sem,
                time_bounds=time_bounds,
                value_bounds=value_bounds,
            )
            plot_ax.set_title(_epoch_title(epoch), fontsize=7)
            stem = safe_stem(f"{_epoch_title(epoch)} {epoch.roi_labels[roi_index]}")
            written.extend(
                save_figure_bundle(
                    fig,
                    output_dir / "plots_with_rois" / stem,
                    formats=formats,
                )
            )
    return tuple(written)


def current_recording_image(
    *,
    viewer: object | None,
    recording: RecordingData,
) -> npt.NDArray[np.float64]:
    """Return the current display-coordinate recording image.

    Args:
        viewer: Napari viewer that may have an ``aligned movie`` layer.
        recording: Loaded converted recording, used for mean-image fallback.

    Returns:
        Two-dimensional display-coordinate image.
    """
    movie_layer = _layer_by_name(viewer, "aligned movie")
    if movie_layer is not None:
        data = np.asarray(cast(_LayerWithData, movie_layer).data)
        if data.ndim == 3:
            index = _current_movie_layer_index(viewer, data.shape[0])
            return cast(npt.NDArray[np.float64], data[index, :, :])
        if data.ndim == 2:
            return cast(npt.NDArray[np.float64], data)

    mean_image = recording.alignment_valid_crop.crop_image(recording.mean_image)
    return cast(npt.NDArray[np.float64], display_image_from_movie_image(mean_image))


def current_roi_labels(
    *,
    roi_labels_layer: object | None,
    recording: RecordingData,
) -> npt.NDArray[np.int64]:
    """Return current display-coordinate ROI labels.

    Args:
        roi_labels_layer: Current napari Labels layer.
        recording: Loaded converted recording, used for empty-label fallback.

    Returns:
        Two-dimensional display-coordinate integer label image.
    """
    if roi_labels_layer is None:
        crop = recording.alignment_valid_crop
        return np.zeros((crop.shape[1], crop.shape[0]), dtype=np.int64)
    return np.asarray(cast(_LayerWithData, roi_labels_layer).data, dtype=np.int64)


def image_figure(shape: tuple[int, int]) -> tuple[Figure, Axes]:
    """Create a figure sized for one recording or ROI image.

    Args:
        shape: ``(rows, columns)`` image shape.

    Returns:
        Matplotlib figure and image axis.
    """
    rows, columns = shape
    width_mm = max(30.0, min(80.0, 45.0 * columns / max(rows, 1)))
    height_mm = max(30.0, min(80.0, 45.0 * rows / max(columns, 1)))
    fig, ax = _axis_figure(
        axis_w_mm=width_mm,
        axis_h_mm=height_mm,
        left_margin_mm=2.0,
        bottom_margin_mm=2.0,
        right_margin_mm=2.0,
        top_margin_mm=2.0,
    )
    ax.set_axis_off()
    return fig, ax


def plot_figure() -> tuple[Figure, Axes]:
    """Create one fixed-size response plot figure.

    Args:
        None.

    Returns:
        Matplotlib figure and response axis.
    """
    return _axis_figure(axis_w_mm=48.0, axis_h_mm=48.0)


def overlay_plot_figure(shape: tuple[int, int]) -> tuple[Figure, Axes, Axes]:
    """Create a two-column figure with ROI image plus response plot.

    Args:
        shape: ``(rows, columns)`` recording image shape.

    Returns:
        Figure, image axis, and response axis.
    """
    rows, columns = shape
    image_w_mm = max(28.0, min(42.0, 38.0 * columns / max(rows, 1)))
    axis_w_mm = image_w_mm + 8.0 + 48.0
    fig, left_ax = _axis_figure(
        axis_w_mm=axis_w_mm,
        axis_h_mm=48.0,
        left_margin_mm=4.0,
        bottom_margin_mm=12.0,
        right_margin_mm=4.0,
        top_margin_mm=5.0,
    )
    total_w_mm, total_h_mm = _figure_size_mm(fig)
    left_ax.set_position(
        (
            4.0 / total_w_mm,
            12.0 / total_h_mm,
            image_w_mm / total_w_mm,
            48.0 / total_h_mm,
        )
    )
    left_ax.set_axis_off()
    plot_left_mm = 4.0 + image_w_mm + 8.0
    plot_ax = fig.add_axes(
        (
            plot_left_mm / total_w_mm,
            12.0 / total_h_mm,
            48.0 / total_w_mm,
            48.0 / total_h_mm,
        )
    )
    _style_axis(plot_ax)
    return fig, left_ax, plot_ax


def draw_recording_image(ax: Axes, image: npt.NDArray[np.float64]) -> None:
    """Draw one grayscale recording image.

    Args:
        ax: Matplotlib axis to draw into.
        image: Two-dimensional display-coordinate image.

    Returns:
        None.
    """
    ax.imshow(image, cmap="gray", origin="upper", interpolation="nearest")
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(image.shape[0] - 0.5, -0.5)


def draw_roi_contours(
    ax: Axes,
    *,
    labels: npt.NDArray[np.int64],
    roi_label_values: tuple[int, ...],
    roi_indices: tuple[int, ...],
    roi_colors: tuple[str, ...],
) -> None:
    """Draw ROI boundaries as editable vector contours.

    Args:
        ax: Matplotlib axis to draw into.
        labels: Display-coordinate integer Labels image.
        roi_label_values: Integer Labels values in plot ROI order.
        roi_indices: Zero-based ROI indices to draw.
        roi_colors: Hex colors in plot ROI order.

    Returns:
        None.
    """
    for roi_index in roi_indices:
        if roi_index >= len(roi_label_values) or roi_index >= len(roi_colors):
            continue
        mask = labels == roi_label_values[roi_index]
        if not np.any(mask):
            continue
        ax.contour(
            mask.astype(np.float64),
            levels=(0.5,),
            colors=(roi_colors[roi_index],),
            linewidths=0.75,
        )
    ax.set_xlim(-0.5, labels.shape[1] - 0.5)
    ax.set_ylim(labels.shape[0] - 0.5, -0.5)


def draw_epoch_response_plot(
    ax: Axes,
    *,
    epoch: EpochResponsePlotData,
    roi_indices: tuple[int, ...],
    roi_colors: tuple[str, ...],
    show_sem: bool,
    time_bounds: tuple[float, float],
    value_bounds: tuple[float, float],
) -> None:
    """Draw one editable response plot matching the live napari plot.

    Args:
        ax: Matplotlib axis to draw into.
        epoch: Plot-ready data for one stimulus epoch.
        roi_indices: Zero-based ROI indices to draw.
        roi_colors: Hex colors in plot ROI order.
        show_sem: Whether to draw SEM bands.
        time_bounds: Shared x-axis bounds.
        value_bounds: Shared y-axis bounds.

    Returns:
        None.
    """
    for roi_index in roi_indices:
        color = roi_colors[roi_index]
        mean = epoch.mean_values[roi_index]
        sem = epoch.sem_values[roi_index]
        if show_sem:
            ax.fill_between(
                epoch.time_seconds,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.25,
                linewidth=0.0,
                zorder=1,
            )
        ax.plot(
            epoch.time_seconds,
            mean,
            color=color,
            linewidth=0.75,
            zorder=2,
        )
    ax.axhline(0.0, color="#6d7683", linewidth=0.5, linestyle=(0, (6, 4)), zorder=0)
    ax.axvline(0.0, color="#6d7683", linewidth=0.5, linestyle=(0, (6, 4)), zorder=0)
    ax.set_xlim(*time_bounds)
    ax.set_ylim(*value_bounds)
    ax.set_xticks(tuple(np.linspace(time_bounds[0], time_bounds[1], 6)))
    ax.set_yticks(tuple(np.linspace(value_bounds[0], value_bounds[1], 6)))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F")


def save_figure_bundle(
    fig: Figure,
    path_without_suffix: Path,
    *,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> tuple[Path, ...]:
    """Save a figure as PDF and PNG with editable vector-friendly PDF text.

    Args:
        fig: Matplotlib figure to save.
        path_without_suffix: Output path without file extension.
        formats: File suffixes to write.

    Returns:
        Paths written to disk.
    """
    set_export_style()
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in formats:
        output_path = path_without_suffix.with_suffix(f".{suffix}")
        fig.savefig(output_path, format=suffix, dpi=300, transparent=True)
        written.append(output_path)
    return tuple(written)


def set_export_style() -> None:
    """Apply Matplotlib settings that keep exported text and paths editable."""
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Atkinson Hyperlegible",
        "Arial",
        "Helvetica",
        "DejaVu Sans",
    ]
    mpl.rcParams["font.size"] = 7
    mpl.rcParams["axes.labelsize"] = 7
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["figure.facecolor"] = "none"
    mpl.rcParams["axes.facecolor"] = "none"
    mpl.rcParams["savefig.facecolor"] = "none"


def safe_stem(text: str) -> str:
    """Return a filesystem-safe, readable file stem.

    Args:
        text: Human-readable plot title or label.

    Returns:
        Lowercase stem with unsafe characters replaced by underscores.
    """
    characters = [
        character.lower() if character.isalnum() else "_" for character in text.strip()
    ]
    stem = "".join(characters).strip("_")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem or "figure"


def _visible_epochs(
    plot_data: ResponsePlotData,
    epoch_keys: tuple[tuple[int, str], ...],
) -> tuple[EpochResponsePlotData, ...]:
    """Return plot data for selected epochs in saved plot order."""
    selected = set(epoch_keys)
    return tuple(
        epoch
        for epoch in plot_data.epochs
        if (epoch.epoch_number, epoch.epoch_name) in selected
    )


def _epoch_title(epoch: EpochResponsePlotData) -> str:
    """Return a display title for one epoch plot."""
    return f"Epoch {epoch.epoch_number}: {epoch.epoch_name}"


def _axis_figure(
    *,
    axis_w_mm: float,
    axis_h_mm: float,
    left_margin_mm: float = 15.0,
    bottom_margin_mm: float = 12.0,
    right_margin_mm: float = 5.0,
    top_margin_mm: float = 5.0,
) -> tuple[Figure, Axes]:
    """Create a figure with an exact data-axis size.

    Args:
        axis_w_mm: Width of the data axis in millimeters.
        axis_h_mm: Height of the data axis in millimeters.
        left_margin_mm: Left margin for labels.
        bottom_margin_mm: Bottom margin for labels.
        right_margin_mm: Right margin.
        top_margin_mm: Top margin.

    Returns:
        Matplotlib figure and axis.

    Fixed axis dimensions make exported figures predictable in Illustrator.
    Avoiding ``bbox_inches='tight'`` also avoids surprise clipping masks and
    changing axis sizes.
    """
    set_export_style()
    total_w_mm = left_margin_mm + axis_w_mm + right_margin_mm
    total_h_mm = bottom_margin_mm + axis_h_mm + top_margin_mm
    fig = Figure(figsize=(total_w_mm * MM_TO_INCH, total_h_mm * MM_TO_INCH))
    ax = fig.add_axes(
        (
            left_margin_mm / total_w_mm,
            bottom_margin_mm / total_h_mm,
            axis_w_mm / total_w_mm,
            axis_h_mm / total_h_mm,
        )
    )
    _style_axis(ax)
    return fig, ax


def _style_axis(ax: Axes) -> None:
    """Apply simple publication-style axes to one response plot."""
    ax.patch.set_alpha(0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.25)
    ax.spines["bottom"].set_linewidth(0.25)
    ax.tick_params(direction="in", width=0.25, length=1.4, pad=1.0)
    ax.xaxis.labelpad = 1.0
    ax.yaxis.labelpad = 0.0


def _figure_size_mm(fig: Figure) -> tuple[float, float]:
    """Return one figure's size in millimeters."""
    width_in, height_in = fig.get_size_inches()
    return float(width_in / MM_TO_INCH), float(height_in / MM_TO_INCH)


def _layer_by_name(viewer: object | None, name: str) -> object | None:
    """Return a napari layer by name using napari's layer-list API when present."""
    if viewer is None or not hasattr(viewer, "layers"):
        return None
    layers = cast(_ViewerWithLayers, viewer).layers
    try:
        return layers[name]
    except (KeyError, TypeError, IndexError):
        pass
    for layer in layers:
        if getattr(layer, "name", None) == name:
            return layer
    return None


def _current_movie_layer_index(viewer: object | None, frame_count: int) -> int:
    """Return the current movie-layer frame index from napari dims."""
    if viewer is None or not hasattr(viewer, "dims"):
        return 0
    dims = cast(_ViewerWithDims, viewer).dims
    for attribute in ("current_step", "point"):
        value = getattr(dims, attribute, None)
        if value is None:
            continue
        try:
            return max(0, min(int(value[0]), frame_count - 1))
        except (IndexError, TypeError, ValueError):
            continue
    return 0
