"""Shared filenames for twopy-owned recording and analysis outputs.

Inputs: none.
Outputs: stable file and directory names used across conversion, analysis, and
napari.

Keeping these user-visible names in one native twopy module prevents conversion,
GUI, cache, and export code from drifting.
"""

__all__ = [
    "ALIGNED_MOVIE_FILENAME",
    "ANALYSIS_OUTPUT_FILENAME",
    "CSV_EXPORTS_DIRNAME",
    "EXPORTS_DIRNAME",
    "PLOT_EXPORTS_DIRNAME",
    "PLOT_WITH_ROIS_EXPORTS_DIRNAME",
    "RESPONSE_HEATMAP_EXPORTS_DIRNAME",
    "RESPONSE_HEATMAPS_FILENAME",
    "RECORDING_DATA_FILENAME",
    "RECORDING_VIEW_EXPORT_STEM",
    "RESPONSE_SUMMARY_GROUPED_FILENAME",
    "RESPONSE_SUMMARY_TRIALS_FILENAME",
    "ROI_FILENAME",
    "ROI_VIEW_EXPORT_STEM",
    "RECORDING_ROI_OVERLAY_EXPORT_STEM",
]

RECORDING_DATA_FILENAME = "recording_data.h5"
ALIGNED_MOVIE_FILENAME = "aligned_movie.h5"
ROI_FILENAME = "rois.h5"
ANALYSIS_OUTPUT_FILENAME = "analysis_outputs.h5"
RESPONSE_HEATMAPS_FILENAME = "response_heatmaps.h5"

EXPORTS_DIRNAME = "exports"
CSV_EXPORTS_DIRNAME = "csvs"
PLOT_EXPORTS_DIRNAME = "plots"
PLOT_WITH_ROIS_EXPORTS_DIRNAME = "plots_with_rois"
RESPONSE_HEATMAP_EXPORTS_DIRNAME = "response_heatmaps"

RESPONSE_SUMMARY_TRIALS_FILENAME = "response_summary_trials.csv"
RESPONSE_SUMMARY_GROUPED_FILENAME = "response_summary_grouped.csv"

RECORDING_VIEW_EXPORT_STEM = "recording_view"
ROI_VIEW_EXPORT_STEM = "roi_view"
RECORDING_ROI_OVERLAY_EXPORT_STEM = "recording_roi_overlay"
