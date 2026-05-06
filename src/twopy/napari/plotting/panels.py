"""Small Qt panel builders for response plotting widgets.

Inputs: already-created buttons, labels, layouts, and plot widgets.
Outputs: composed Qt widgets used by the napari response docks.

This module owns generic panel layout only. It does not load data, compute
responses, or export figures.
"""

from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from twopy.napari.plotting.widgets import EpochPlotWidget

__all__ = [
    "epoch_plot_panel",
    "plot_size_widget",
    "response_update_tab",
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
    layout.addWidget(QLabel(title))
    layout.addWidget(plot)
    panel.setLayout(layout)
    return panel


def response_update_tab(
    *,
    refresh_button: QPushButton,
    update_from_rois_button: QPushButton,
    analysis_path_label: QLabel,
) -> QWidget:
    """Create the tab that owns response recompute/reload actions.

    Args:
        refresh_button: Button that reloads persisted response outputs.
        update_from_rois_button: Button that computes responses from current
            Labels ROIs.
        analysis_path_label: Label showing the analysis output path.

    Returns:
        Qt widget for the Update tab.
    """
    tab = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(refresh_button)
    layout.addWidget(update_from_rois_button)
    layout.addWidget(analysis_path_label)
    layout.addStretch(1)
    tab.setLayout(layout)
    return tab


def plot_size_widget(spin_box: QSpinBox) -> QWidget:
    """Create the compact plot-size control row.

    Args:
        spin_box: Numeric plot-size input.

    Returns:
        Qt widget containing the label and spin box.
    """
    widget = QWidget()
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Plot size"))
    layout.addWidget(spin_box)
    widget.setLayout(layout)
    return widget


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
