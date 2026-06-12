"""Path resolution and labels for response plotting docks.

Inputs: optional recording state, user-selected analysis output, and ROI output
paths.
Outputs: concrete paths plus compact Metadata-tab label text.

These helpers keep path fallback policy in one place for preview, save, and
status display code.
"""

from pathlib import Path

from twopy.converted import RecordingData
from twopy.filenames import ANALYSIS_OUTPUT_FILENAME, ROI_FILENAME
from twopy.napari.display_paths import (
    format_twopy_h5_output,
    microscope_display_lines,
    recording_display_summary,
)
from twopy.napari.output_routing import NapariOutputRoute
from twopy.napari.paths import DEFAULT_PATH_TEXT
from twopy.napari.plotting.data import default_analysis_output_path
from twopy.napari.plotting.panels import SidebarTextLabel


def resolved_analysis_path(
    analysis_path: Path | None,
    recording: RecordingData | None,
) -> Path:
    """Return the analysis HDF5 path for the selected recording.

    Args:
        analysis_path: Explicit analysis HDF5 path, if one is loaded.
        recording: Optional loaded converted recording.

    Returns:
        Concrete analysis output path used by reload, save, and labels.
    """
    if analysis_path is not None:
        return analysis_path
    if recording is not None:
        return default_analysis_output_path(recording)
    return Path(ANALYSIS_OUTPUT_FILENAME)


def resolved_roi_save_file(
    roi_save_file: Path | None,
    recording: RecordingData | None,
) -> Path:
    """Return the ROI HDF5 path for the selected recording.

    Args:
        roi_save_file: Explicit ROI output path, if one is loaded.
        recording: Optional loaded converted recording.

    Returns:
        Concrete ROI HDF5 path used by Save Analysis.
    """
    if roi_save_file is not None:
        return roi_save_file
    if recording is not None:
        return recording.path.parent / ROI_FILENAME
    return Path(ROI_FILENAME)


def refresh_update_path_labels(
    *,
    recording: RecordingData | None,
    analysis_path: Path | None,
    roi_save_file: Path | None,
    output_route: NapariOutputRoute | None,
    recording_summary_label: SidebarTextLabel,
    microscope_summary_label: SidebarTextLabel,
    analysis_path_label: SidebarTextLabel,
    roi_save_path_label: SidebarTextLabel,
) -> None:
    """Refresh compact recording and output labels in the Metadata tab.

    Args:
        recording: Optional loaded converted recording.
        analysis_path: Explicit analysis output path, if one is loaded.
        roi_save_file: Explicit ROI output path, if one is loaded.
        output_route: Local and published output folders, if known.
        recording_summary_label: Label showing the selected recording summary.
        microscope_summary_label: Label showing selected microscope metadata.
        analysis_path_label: Label showing the analysis output path.
        roi_save_path_label: Label showing the ROI output path.

    Returns:
        None.
    """
    if recording is None:
        recording_summary_label.setText("No recording loaded.")
        microscope_summary_label.setText("No microscope metadata.")
        analysis_path_label.setText(f"Analysis output: {DEFAULT_PATH_TEXT}")
        roi_save_path_label.setText(f"ROI output: {DEFAULT_PATH_TEXT}")
        return

    summary = recording_display_summary(recording)
    recording_summary_label.setText("\n\n".join(summary.lines()))
    microscope_summary_label.setText("\n\n".join(microscope_display_lines(recording)))
    analysis_file = resolved_analysis_path(analysis_path, recording)
    roi_file = resolved_roi_save_file(roi_save_file, recording)
    if output_route is None:
        analysis_path_label.setText(
            f"Analysis output: {format_twopy_h5_output(analysis_file)}"
        )
        roi_save_path_label.setText(f"ROI output: {format_twopy_h5_output(roi_file)}")
        return

    analysis_path_label.setText(
        "Analysis output:\n"
        f"Local: {analysis_file}\n"
        f"Output: {output_route.publish_root / analysis_file.name}"
    )
    roi_save_path_label.setText(
        "ROI output:\n"
        f"Local: {roi_file}\n"
        f"Output: {output_route.publish_root / roi_file.name}"
    )
