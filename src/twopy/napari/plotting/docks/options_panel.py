"""Tabbed options panel construction for response plotting docks.

Inputs: current Plot-tab option values plus callbacks owned by the response
plot widget.
Outputs: a tab widget and the child controls that the response plot widget must
update as recordings and analysis outputs change.

Keeping this construction in one small module makes the main response plot
widget focus on state transitions and plot rendering instead of Qt assembly.
"""

from collections.abc import Callable
from dataclasses import dataclass

from qtpy.QtWidgets import QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget

from twopy.analysis.dff_options import DeltaFOverFOptions
from twopy.analysis.response_maps import ResponseMapOptions
from twopy.analysis.response_processing import (
    NormalizationOptions,
    ResponseProcessingOptions,
)
from twopy.analysis.response_window_options import ResponseWindowOptions
from twopy.napari.paths import DEFAULT_PATH_TEXT
from twopy.napari.plotting.dff_options import DeltaFOverFOptionsWidget
from twopy.napari.plotting.export_controls import (
    ResponseExportState,
    create_response_export_tab,
)
from twopy.napari.plotting.normalization_options import NormalizationOptionsWidget
from twopy.napari.plotting.panels import response_metadata_tab, scrolling_tab
from twopy.napari.plotting.processing_options import ResponseProcessingOptionsWidget
from twopy.napari.plotting.response_map_options import ResponseMapOptionsWidget
from twopy.napari.plotting.response_window_options import ResponseWindowOptionsWidget
from twopy.napari.plotting.roi_generation import (
    RoiGenerationControls,
    RoiGenerationOptions,
)
from twopy.pixel_calibration import PixelCalibrationRow
from twopy.pixel_calibration_profiles import PixelCalibrationProfileMapping


@dataclass(frozen=True)
class ResponseOptionsPanel:
    """Hold the response options tabs and the controls updated by the widget.

    Args:
        tabs: Root tab widget docked as the response options panel.
        recording_summary_label: Metadata-tab label showing the loaded recording.
        microscope_summary_label: Metadata-tab label showing microscope metadata.
        analysis_path_label: Metadata-tab label showing the analysis output path.
        roi_save_path_label: Metadata-tab label showing the ROI output path.
        update_status_label: Metadata-tab label showing save or processing status.
        reload_saved_button: Load-tab button that reloads persisted response
            outputs.
        plot_options_layout: Plot-tab layout containing static and dynamic
            processing controls.
        plot_display_options_layout: Plot-tab layout rebuilt when axis defaults
            change.
        roi_options_layout: ROIs-tab layout rebuilt from loaded plot data.
        roi_generation_widget: ROIs-tab widget for manual/generated ROI modes.
        epoch_options_layout: Epochs-tab layout rebuilt from loaded plot data.
        processing_options_widget: Plot-tab response processing controls.
        response_window_options_widget: Plot-tab response-window controls.
        response_map_options_widget: Plot-tab response-map controls.
        delta_f_over_f_options_widget: Plot-tab dF/F controls.
        normalization_options_widget: Plot-tab normalization controls.

    Returns:
        Immutable container for widgets and layouts owned by the options panel.
    """

    tabs: QTabWidget
    recording_summary_label: QLabel
    microscope_summary_label: QLabel
    analysis_path_label: QLabel
    roi_save_path_label: QLabel
    update_status_label: QLabel
    reload_saved_button: QPushButton
    plot_options_layout: QVBoxLayout
    plot_display_options_layout: QVBoxLayout
    roi_options_layout: QVBoxLayout
    roi_generation_widget: RoiGenerationControls
    epoch_options_layout: QVBoxLayout
    processing_options_widget: ResponseProcessingOptionsWidget
    response_window_options_widget: ResponseWindowOptionsWidget
    response_map_options_widget: ResponseMapOptionsWidget
    delta_f_over_f_options_widget: DeltaFOverFOptionsWidget
    normalization_options_widget: NormalizationOptionsWidget


def create_response_options_panel(
    *,
    response_processing_options: ResponseProcessingOptions,
    response_window_options: ResponseWindowOptions,
    response_map_options: ResponseMapOptions,
    delta_f_over_f_options: DeltaFOverFOptions,
    on_response_processing_change: Callable[[ResponseProcessingOptions], None],
    on_response_window_change: Callable[[ResponseWindowOptions], None],
    on_response_map_change: Callable[[ResponseMapOptions], None],
    on_response_map_shared_limits_change: Callable[[bool], None],
    on_delta_f_over_f_change: Callable[[DeltaFOverFOptions], None],
    on_normalization_change: Callable[[NormalizationOptions], None],
    on_reload_saved: Callable[[], None],
    on_save_analysis: Callable[[], None],
    on_create_generated_rois: Callable[[RoiGenerationOptions], None],
    export_state: Callable[[], ResponseExportState],
    pixel_calibrations: tuple[PixelCalibrationRow, ...],
    pixel_calibration_profile_mappings: tuple[PixelCalibrationProfileMapping, ...],
) -> ResponseOptionsPanel:
    """Create the tabbed response options panel.

    Args:
        response_processing_options: Initial response processing settings.
        response_window_options: Initial response-window settings.
        response_map_options: Initial response-map settings.
        delta_f_over_f_options: Initial dF/F settings.
        on_response_processing_change: Callback for processing control edits.
        on_response_window_change: Callback for response-window control edits.
        on_response_map_change: Callback for response-map control edits.
        on_response_map_shared_limits_change: Callback for heatmap display
            scaling edits.
        on_delta_f_over_f_change: Callback for dF/F control edits.
        on_normalization_change: Callback for normalization control edits.
        on_reload_saved: Callback for the Load-tab reload button.
        on_save_analysis: Callback for the Save ROIs + analysis button.
        on_create_generated_rois: Callback for generated-ROI actions.
        export_state: Callback that supplies current export state.
        pixel_calibrations: Calibration rows for micron-sized grid controls.
        pixel_calibration_profile_mappings: ScanImage config mappings used to
            prefill calibration choices when metadata identifies one group.

    Returns:
        Options panel with root tabs plus child controls needed by the owner.
    """
    tabs = QTabWidget()
    reload_saved_button = QPushButton("Reload saved analysis")
    reload_saved_button.clicked.connect(on_reload_saved)
    save_analysis_button = QPushButton("Save ROIs + analysis")
    save_analysis_button.clicked.connect(on_save_analysis)

    recording_summary_label = QLabel("No recording loaded.")
    recording_summary_label.setWordWrap(True)
    microscope_summary_label = QLabel("No microscope metadata.")
    microscope_summary_label.setWordWrap(True)
    analysis_path_label = QLabel(f"Analysis output: {DEFAULT_PATH_TEXT}")
    analysis_path_label.setWordWrap(True)
    roi_save_path_label = QLabel(f"ROI output: {DEFAULT_PATH_TEXT}")
    roi_save_path_label.setWordWrap(True)
    update_status_label = QLabel("")
    update_status_label.setWordWrap(True)

    plot_options_layout = QVBoxLayout()
    plot_options_layout.setSpacing(6)
    plot_display_options_layout = QVBoxLayout()
    plot_display_options_layout.setContentsMargins(0, 0, 0, 0)
    plot_display_options_widget = QWidget()
    plot_display_options_widget.setLayout(plot_display_options_layout)

    processing_options_widget = ResponseProcessingOptionsWidget(
        response_processing_options,
        on_change=on_response_processing_change,
    )
    response_window_options_widget = ResponseWindowOptionsWidget(
        response_window_options,
        on_change=on_response_window_change,
    )
    response_map_options_widget = ResponseMapOptionsWidget(
        response_map_options,
        on_change=on_response_map_change,
        on_shared_limits_change=on_response_map_shared_limits_change,
    )
    delta_f_over_f_options_widget = DeltaFOverFOptionsWidget(
        delta_f_over_f_options,
        on_change=on_delta_f_over_f_change,
    )
    normalization_options_widget = NormalizationOptionsWidget(
        response_processing_options.normalization,
        on_change=on_normalization_change,
    )
    plot_options_layout.addWidget(plot_display_options_widget)
    plot_options_layout.addWidget(response_window_options_widget)
    plot_options_layout.addWidget(response_map_options_widget)
    plot_options_layout.addWidget(delta_f_over_f_options_widget)
    plot_options_layout.addWidget(normalization_options_widget)
    plot_options_layout.addWidget(processing_options_widget)
    plot_options_layout.addStretch(1)

    roi_tab_layout = QVBoxLayout()
    roi_generation_widget = RoiGenerationControls(
        pixel_calibrations,
        pixel_calibration_profile_mappings,
        on_generate=on_create_generated_rois,
    )
    roi_options_layout = QVBoxLayout()
    roi_tab_layout.addWidget(roi_generation_widget)
    roi_tab_layout.addLayout(roi_options_layout)
    epoch_options_layout = QVBoxLayout()
    tabs.addTab(
        response_metadata_tab(
            recording_summary_label=recording_summary_label,
            microscope_summary_label=microscope_summary_label,
            analysis_output_label=analysis_path_label,
            roi_output_label=roi_save_path_label,
        ),
        "Metadata",
    )
    tabs.addTab(scrolling_tab(plot_options_layout), "Plot")
    tabs.addTab(scrolling_tab(roi_tab_layout), "ROIs")
    tabs.addTab(scrolling_tab(epoch_options_layout), "Epochs")
    tabs.addTab(
        create_response_export_tab(
            export_state,
            save_analysis_button=save_analysis_button,
        ),
        "Export",
    )

    return ResponseOptionsPanel(
        tabs=tabs,
        recording_summary_label=recording_summary_label,
        microscope_summary_label=microscope_summary_label,
        analysis_path_label=analysis_path_label,
        roi_save_path_label=roi_save_path_label,
        update_status_label=update_status_label,
        reload_saved_button=reload_saved_button,
        plot_options_layout=plot_options_layout,
        plot_display_options_layout=plot_display_options_layout,
        roi_options_layout=roi_options_layout,
        roi_generation_widget=roi_generation_widget,
        epoch_options_layout=epoch_options_layout,
        processing_options_widget=processing_options_widget,
        response_window_options_widget=response_window_options_widget,
        response_map_options_widget=response_map_options_widget,
        delta_f_over_f_options_widget=delta_f_over_f_options_widget,
        normalization_options_widget=normalization_options_widget,
    )
