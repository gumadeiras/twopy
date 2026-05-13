"""Qt-free ROI generation actions for the napari ROIs tab.

Inputs: converted recording metadata, ROI-generation options, and calibration
rows.
Outputs: movie-coordinate label images plus user-facing status text.

The response dock owns UI lifecycle; this module owns the generation decision
tree so it can be tested without napari widgets.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from twopy.analysis.epoch_mapping import interpolate_stimulus_epochs_to_frame_windows
from twopy.analysis.trials import EpochFrameWindow
from twopy.converted import RecordingData
from twopy.napari.plotting.roi_generation.options import RoiGenerationOptions
from twopy.pixel_calibration import PixelCalibrationRow, resolve_pixel_size_um
from twopy.response_roi_extraction import response_watershed_roi_set
from twopy.roi import roi_set_to_label_image
from twopy.roi_extraction import grid_roi_set, grid_roi_set_microns, watershed_roi_set

__all__ = [
    "GeneratedRoiLabels",
    "generate_roi_labels",
]


@dataclass(frozen=True)
class GeneratedRoiLabels:
    """Generated full-frame ROI labels plus status text.

    Args:
        label_image: Movie-coordinate integer ROI label image.
        status_text: User-facing description of the generated ROIs.
    """

    label_image: npt.NDArray[np.int64]
    status_text: str


def generate_roi_labels(
    recording: RecordingData,
    options: RoiGenerationOptions,
    calibrations: tuple[PixelCalibrationRow, ...],
    *,
    epoch_windows: Sequence[EpochFrameWindow] | None = None,
) -> GeneratedRoiLabels:
    """Generate ROI labels for a converted recording.

    Args:
        recording: Loaded converted recording.
        options: ROI generation options from the ROIs tab.
        calibrations: Pixel calibration rows used for micron-sized grid mode.
        epoch_windows: Optional stimulus windows used by response-watershed
            generation. When omitted, they are resolved from converted
            stimulus/photodiode data.

    Returns:
        Generated labels in movie coordinates plus status text.

    Raises:
        ValueError: If calibration lookup or ROI generation fails.
    """
    if options.roi_mode == "manual":
        msg = "Manual ROI mode does not generate labels."
        raise ValueError(msg)

    if options.roi_mode == "watershed":
        roi_set = watershed_roi_set(
            recording.mean_image,
            min_pixels=options.watershed_min_pixels,
            smoothing_sigma=options.watershed_smoothing_sigma,
        )
        return GeneratedRoiLabels(
            label_image=roi_set_to_label_image(roi_set),
            status_text=f"Created watershed ROIs: {len(roi_set.labels)} ROIs.",
        )

    if options.roi_mode == "response_watershed":
        resolved_epoch_windows = (
            tuple(epoch_windows)
            if epoch_windows is not None
            else interpolate_stimulus_epochs_to_frame_windows(recording).windows
        )
        roi_set = response_watershed_roi_set(
            recording,
            resolved_epoch_windows,
            min_pixels=options.response_watershed_min_pixels,
            score_smoothing_sigma=options.response_watershed_smoothing_sigma,
            fill_holes=options.response_watershed_fill_holes,
            closing_radius=options.response_watershed_closing_radius,
        )
        return GeneratedRoiLabels(
            label_image=roi_set_to_label_image(roi_set),
            status_text=(
                f"Created response watershed ROIs: {len(roi_set.labels)} ROIs."
            ),
        )

    if options.units == "pixels":
        roi_set = grid_roi_set(
            recording.movie.shape[1:],
            grid_size_pixels=options.grid_size_pixels,
        )
        return GeneratedRoiLabels(
            label_image=roi_set_to_label_image(roi_set),
            status_text=f"Created pixel grid ROIs: {options.grid_size_pixels} px.",
        )

    pixel_size = resolve_pixel_size_um(
        calibrations,
        rig=options.rig,
        mode=options.calibration_mode,
        scanner=options.scanner,
        zoom=options.zoom,
        allow_extrapolation=options.allow_extrapolation,
    )
    roi_set = grid_roi_set_microns(
        recording.movie.shape[1:],
        micron_grid_size=options.micron_grid_size,
        pixel_size_um=pixel_size.pixel_size_um,
    )
    return GeneratedRoiLabels(
        label_image=roi_set_to_label_image(roi_set),
        status_text=f"Estimated pixel size: {pixel_size.pixel_size_um:.6g} um/px.",
    )
