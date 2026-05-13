"""Small Qt panel builders for response plotting widgets.

Inputs: already-created buttons, labels, layouts, and plot widgets.
Outputs: composed Qt widgets used by the napari response docks.

This module owns generic panel layout only. It does not load data, compute
responses, or export figures.
"""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from twopy.napari.plotting.widgets import EpochPlotWidget

__all__ = [
    "epoch_plot_panel",
    "response_metadata_tab",
    "scrolling_tab",
]


def epoch_plot_panel(*, title: str, plot: EpochPlotWidget) -> QWidget:
    """Create one titled plot panel for the horizontal response strip.

    Args:
        title: Plot title.
        plot: Epoch response plot widget.

    Returns:
        Qt widget containing the title and plot.
    """
    panel = QWidget()
    layout = QVBoxLayout()
    title_label = QLabel(title)
    title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    layout.addWidget(title_label)
    layout.addWidget(plot)
    panel.setLayout(layout)
    return panel


def response_metadata_tab(
    *,
    recording_summary_label: QLabel,
    analysis_output_label: QLabel,
    roi_output_label: QLabel,
    status_label: QLabel,
) -> QWidget:
    """Create the tab that shows selected-recording metadata.

    Args:
        recording_summary_label: Label showing root, genotype, stimulus, and
            recording time for the selected recording.
        analysis_output_label: Label showing the analysis output path.
        roi_output_label: Label showing the ROI output path.
        status_label: Label showing the latest save or reload status.

    Returns:
        Scrollable Qt widget for the Metadata tab.
    """
    layout = QVBoxLayout()
    layout.addWidget(recording_summary_label)
    layout.addWidget(analysis_output_label)
    layout.addWidget(roi_output_label)
    layout.addWidget(status_label)
    layout.addStretch(1)
    return scrolling_tab(layout)


def scrolling_tab(layout: QVBoxLayout) -> QScrollArea:
    """Create a scrollable options tab around one layout.

    Args:
        layout: Layout that callers will populate with controls.

    Returns:
        Scroll area suitable for a tab body.
    """
    content = QWidget()
    content.setLayout(layout)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    return scroll
