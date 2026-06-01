"""Build napari ROI response requests for twopy analysis.

Inputs: a loaded converted recording, ROI labels or masks, and Plot-tab
analysis options.
Outputs: one object that carries those values plus plot-ready response data.

Preview, live update, and Save Analysis all use the same request object. That
keeps plotted previews and saved outputs on the same ROI masks, response
windows, dF/F settings, and processing options.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.response_plotting import (
    ResponsePlotData,
    response_plot_data_from_grouped,
)
from twopy.analysis.response_processing import ResponseProcessingOptions
from twopy.analysis.response_window_options import ResponseWindowOptions
from twopy.analysis.workflow import (
    AnalysisResponseComputation,
    compute_recording_responses,
)
from twopy.converted import RecordingData
from twopy.napari.plotting.data import (
    response_plot_min_epoch_duration_seconds,
    response_plot_window_seconds_for_recording,
)
from twopy.napari.roi import roi_label_image_from_layer_for_recording
from twopy.roi import RoiSet, make_roi_set, make_roi_set_from_label_image

__all__ = [
    "ResponseAnalysisRequest",
    "compute_response_preview",
    "response_analysis_request_from_label_image",
    "response_analysis_request_from_labels",
    "response_analysis_request_from_selected_label_image",
]


@dataclass(frozen=True)
class ResponseAnalysisRequest:
    """One ROI response-analysis request from the napari workflow.

    Args:
        recording: Loaded converted recording shown in napari.
        roi_set: ROI masks in movie coordinates.
        source_path: Optional display path attached to plot data.
        delta_f_over_f_options: dF/F analysis settings.
        response_window_options: response-window settings.
        response_processing_options: smoothing, filtering, normalization, and
            correlation-QC settings.

    The request stores the ROI masks instead of the napari layer. Preview and
    Save Analysis therefore run on the same masks after the layer has been
    checked. Requests built from napari Labels layers use labels such as
    ``roi_0004``. The number comes from the integer label painted in the layer,
    so each drawn ROI has the same name while the user edits its pixels.
    """

    recording: RecordingData
    roi_set: RoiSet
    source_path: Path | None = None
    delta_f_over_f_options: DeltaFOverFOptions = field(
        default_factory=DeltaFOverFOptions
    )
    response_window_options: ResponseWindowOptions = field(
        default_factory=ResponseWindowOptions
    )
    response_processing_options: ResponseProcessingOptions = field(
        default_factory=ResponseProcessingOptions
    )

    def response_window_seconds(self) -> tuple[float, float]:
        """Return pre/post response windows for this recording and options.

        Args:
            None.

        Returns:
            Seconds before and after each stimulus epoch used by response
            analysis.

        The Plot-tab window options can depend on recording timing metadata, so
        the request resolves those seconds beside the recording it carries.
        """
        return response_plot_window_seconds_for_recording(
            self.recording,
            self.response_window_options,
        )


def response_analysis_request_from_labels(
    recording: RecordingData | None,
    roi_labels_layer: object | None,
    *,
    source_path: Path | None = None,
    delta_f_over_f_options: DeltaFOverFOptions | None = None,
    response_window_options: ResponseWindowOptions | None = None,
    response_processing_options: ResponseProcessingOptions | None = None,
) -> ResponseAnalysisRequest:
    """Create a response-analysis request from the active napari Labels layer.

    Args:
        recording: Loaded converted recording, or ``None`` when no recording is
            active.
        roi_labels_layer: Current napari Labels layer.
        source_path: Optional display path attached to plot data.
        delta_f_over_f_options: Optional dF/F settings.
        response_window_options: Optional response-window settings.
        response_processing_options: Optional response-processing settings.

    Returns:
        Response-analysis request ready for preview or save.

    Raises:
        ValueError: If no recording or Labels layer is available, if the layer
            shape is wrong for the recording, or if no ROI labels exist.
    """
    if recording is None:
        msg = "No recording loaded."
        raise ValueError(msg)
    if roi_labels_layer is None:
        msg = "No ROI Labels layer is available."
        raise ValueError(msg)

    label_image = roi_label_image_from_layer_for_recording(roi_labels_layer, recording)
    return response_analysis_request_from_label_image(
        recording,
        label_image,
        source_path=source_path,
        delta_f_over_f_options=delta_f_over_f_options,
        response_window_options=response_window_options,
        response_processing_options=response_processing_options,
    )


def response_analysis_request_from_label_image(
    recording: RecordingData,
    label_image: npt.ArrayLike,
    *,
    source_path: Path | None = None,
    delta_f_over_f_options: DeltaFOverFOptions | None = None,
    response_window_options: ResponseWindowOptions | None = None,
    response_processing_options: ResponseProcessingOptions | None = None,
) -> ResponseAnalysisRequest:
    """Create a response-analysis request from a full-frame label image.

    Args:
        recording: Loaded converted recording.
        label_image: Full-frame integer ROI label image.
        source_path: Optional display path attached to plot data.
        delta_f_over_f_options: Optional dF/F settings.
        response_window_options: Optional response-window settings.
        response_processing_options: Optional response-processing settings.

    Returns:
        Response-analysis request ready for preview or save.

    Raises:
        ValueError: If the label image cannot be converted into a non-empty
        ``RoiSet``.
    """
    return ResponseAnalysisRequest(
        recording=recording,
        roi_set=make_roi_set_from_label_image(label_image),
        source_path=source_path,
        delta_f_over_f_options=delta_f_over_f_options or DeltaFOverFOptions(),
        response_window_options=response_window_options or ResponseWindowOptions(),
        response_processing_options=(
            response_processing_options or ResponseProcessingOptions()
        ),
    )


def response_analysis_request_from_selected_label_image(
    recording: RecordingData,
    label_image: npt.ArrayLike,
    *,
    label_value: int,
    source_path: Path | None = None,
    delta_f_over_f_options: DeltaFOverFOptions | None = None,
    response_window_options: ResponseWindowOptions | None = None,
    response_processing_options: ResponseProcessingOptions | None = None,
) -> ResponseAnalysisRequest:
    """Create a response request for one label while preserving its ROI name.

    Args:
        recording: Loaded converted recording.
        label_image: Full-frame integer ROI label image.
        label_value: Positive integer label value to isolate.
        source_path: Optional display path attached to plot data.
        delta_f_over_f_options: Optional dF/F settings.
        response_window_options: Optional response-window settings.
        response_processing_options: Optional response-processing settings.

    Returns:
        Response-analysis request containing exactly one ROI mask.

    Raises:
        ValueError: If the label value is not positive or no matching pixels are
        present.

    Group-matching previews analyze one selected ROI at a time. Building the
    ``RoiSet`` directly keeps the displayed label stable as ``roi_####``
    instead of turning every selected binary mask into ``roi_0001``.
    """
    if label_value <= 0:
        msg = f"Selected ROI label must be positive; got {label_value}"
        raise ValueError(msg)
    labels = np.asarray(label_image, dtype=np.int64)
    mask = labels == int(label_value)
    if not bool(np.any(mask)):
        msg = f"Selected ROI label {label_value} has no pixels."
        raise ValueError(msg)
    return ResponseAnalysisRequest(
        recording=recording,
        roi_set=make_roi_set(
            mask[np.newaxis, :, :],
            labels=(f"roi_{label_value:04d}",),
        ),
        source_path=source_path,
        delta_f_over_f_options=delta_f_over_f_options or DeltaFOverFOptions(),
        response_window_options=response_window_options or ResponseWindowOptions(),
        response_processing_options=(
            response_processing_options or ResponseProcessingOptions()
        ),
    )


def compute_response_preview(
    request: ResponseAnalysisRequest,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> ResponsePlotData:
    """Compute plot-ready response data for one request without saving files.

    Args:
        request: Response-analysis request.
        check_cancelled: Optional callback that raises when work is obsolete.

    Returns:
        Plot-ready response means and SEMs grouped by stimulus epoch.

    Live ROI editing and manual preview use this same in-memory path. Explicit
    Save Analysis remains a separate action.
    """
    dff_options = request.delta_f_over_f_options
    pre_window_seconds, post_window_seconds = request.response_window_seconds()
    computation = compute_recording_responses(
        request.recording,
        request.roi_set,
        baseline_mode=dff_options.baseline_mode,
        baseline_epoch_number=dff_options.baseline_epoch_number,
        baseline_epoch_name=dff_options.baseline_epoch_name,
        background_method=dff_options.background_method,
        baseline_sample_seconds=dff_options.baseline_sample_seconds,
        fit_mode=dff_options.fit_mode,
        apply_motion_mask=dff_options.apply_motion_mask,
        response_pre_window_seconds=pre_window_seconds,
        response_post_window_seconds=post_window_seconds,
        response_processing_options=request.response_processing_options,
        check_cancelled=check_cancelled,
    )
    return response_plot_data_from_computation(request, computation)


def response_plot_data_from_computation(
    request: ResponseAnalysisRequest,
    computation: AnalysisResponseComputation,
) -> ResponsePlotData:
    """Project core response-analysis results into napari plot data.

    Args:
        request: Original request whose source path and Plot-tab options should
            be attached to the plot.
        computation: Core response-analysis output.

    Returns:
        Plot-ready response means, SEMs, options, and QC metadata.

    Core analysis intentionally does not know about napari plot defaults. This
    function is the small adapter that adds display metadata after scientific
    computation has finished.
    """
    return response_plot_data_from_grouped(
        computation.grouped_responses,
        source_path=request.source_path,
        epoch_windows=computation.epoch_windows,
        delta_f_over_f_options=request.delta_f_over_f_options,
        response_window_options=request.response_window_options,
        response_processing_options=computation.response_processing_options,
        correlation_scores=computation.correlation_scores,
        correlation_window_stop_default_seconds=response_plot_min_epoch_duration_seconds(
            computation.epoch_windows,
            data_rate_hz=computation.grouped_responses.data_rate_hz,
        ),
    )
