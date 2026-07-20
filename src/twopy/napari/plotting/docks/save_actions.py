"""Save Analysis action for response plotting docks.

Inputs: one response-analysis request and selected output paths.
Outputs: persisted ROI HDF5 plus normal twopy analysis response outputs.

This module keeps persistence side effects out of the response plot widget so
the widget only coordinates UI state and displays the result.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from twopy.analysis.response_maps import ResponseMapData, save_response_map_data
from twopy.analysis.workflow import analyze_recording_responses
from twopy.analysis_cache import AnalysisSyncPlan, build_analysis_sync_plan_from_roots
from twopy.filenames import RESPONSE_HEATMAPS_FILENAME
from twopy.hdf5_utils import Hdf5Scalar
from twopy.napari.output_routing import (
    NapariOutputRoute,
    recording_output_route,
    same_output_path,
)
from twopy.napari.plotting.docks.paths import (
    resolved_analysis_path,
    resolved_roi_save_file,
)
from twopy.napari.responses import ResponseAnalysisRequest
from twopy.napari.text import counted_noun
from twopy.roi import save_roi_set


@dataclass(frozen=True)
class SaveAnalysisResult:
    """Result of the Save Analysis action.

    Args:
        roi_output_path: HDF5 path where ROIs were saved.
        analysis_output_path: HDF5 path where response analysis was saved.
        response_heatmap_path: Optional HDF5 path where heatmaps were saved.
        status_text: User-facing Metadata-tab status text.
        sync_plan: Optional background publish-sync plan.

    Returns:
        Immutable result values needed to refresh the dock after saving.
    """

    roi_output_path: Path
    analysis_output_path: Path
    response_heatmap_path: Path | None
    status_text: str
    sync_plan: AnalysisSyncPlan | None


def save_current_roi_analysis(
    request: ResponseAnalysisRequest,
    *,
    roi_save_file: Path | None,
    analysis_path: Path | None,
    output_route: NapariOutputRoute | None = None,
    response_map_data: ResponseMapData | None = None,
    roi_generation_metadata: Mapping[str, Hdf5Scalar] | None = None,
) -> SaveAnalysisResult:
    """Persist current Labels ROIs and run response analysis.

    Args:
        request: Response-analysis request from the current napari state.
        roi_save_file: Explicit ROI output path, if one is loaded.
        analysis_path: Explicit analysis output path, if one is loaded.
        output_route: Local and published output folders for this recording.
        response_map_data: Optional recording-level heatmaps to persist beside
            ROI analysis outputs.
        roi_generation_metadata: Optional ROI-generation provenance written to
            the saved ROI HDF5 file.

    Returns:
        Saved output paths and status text for the Metadata tab.

    Raises:
        ValueError: If analysis settings are invalid for saving.
    """
    route = output_route or recording_output_route(request.recording)
    roi_output_path = resolved_roi_save_file(roi_save_file, request.recording)
    analysis_output_path = resolved_analysis_path(analysis_path, request.recording)
    save_roi_set(
        request.roi_set,
        roi_output_path,
        generation_metadata=roi_generation_metadata,
    )
    dff_options = request.delta_f_over_f_options
    pre_window_seconds, post_window_seconds = request.response_window_seconds()
    run = analyze_recording_responses(
        request.recording.path,
        request.roi_set,
        output_path=analysis_output_path,
        baseline_mode=dff_options.baseline_mode,
        baseline_epoch_number=dff_options.baseline_epoch_number,
        baseline_epoch_name=dff_options.baseline_epoch_name,
        background_method=dff_options.background_method,
        baseline_sample_seconds=dff_options.baseline_sample_seconds,
        fit_mode=dff_options.fit_mode,
        apply_motion_mask=dff_options.apply_motion_mask,
        response_pre_window_seconds=pre_window_seconds,
        response_post_window_seconds=post_window_seconds,
        response_window_auto=request.response_window_options.auto,
        response_processing_options=request.response_processing_options,
    )
    response_heatmap_path = _save_response_heatmaps(request, response_map_data)
    output_folder = str(run.output_path.parent)
    status_text = (
        f"Saved {counted_noun(len(request.roi_set.labels), 'ROI', 'ROIs')} "
        f"locally to {output_folder}"
    )
    sync_plan = build_analysis_sync_plan_from_roots(
        local_root=route.local_root,
        publish_root=route.publish_root,
        local_paths=(
            request.recording.path,
            request.recording.movie.path,
            roi_output_path,
            run.output_path,
            response_heatmap_path,
            run.response_summary_trials_csv_path,
            run.response_summary_grouped_csv_path,
        ),
    )
    status_text = f"{status_text}. {_sync_status_suffix(route, sync_plan)}"
    return SaveAnalysisResult(
        roi_output_path=roi_output_path,
        analysis_output_path=run.output_path,
        response_heatmap_path=response_heatmap_path,
        status_text=status_text,
        sync_plan=sync_plan,
    )


def _sync_status_suffix(
    route: NapariOutputRoute,
    sync_plan: AnalysisSyncPlan | None,
) -> str:
    """Return the save-status suffix for one route and optional sync plan."""
    if sync_plan is not None:
        return f"Syncing to {sync_plan.publish_root}"
    if same_output_path(route.local_root, route.publish_root):
        return f"Saved directly to {route.publish_root}"
    return f"No syncable files under {route.local_root}"


def _save_response_heatmaps(
    request: ResponseAnalysisRequest,
    response_map_data: ResponseMapData | None,
) -> Path | None:
    """Persist recording-level response heatmaps when they are available."""
    if response_map_data is None:
        return None
    path = request.recording.path.parent / RESPONSE_HEATMAPS_FILENAME
    save_response_map_data(path, response_map_data)
    return path
