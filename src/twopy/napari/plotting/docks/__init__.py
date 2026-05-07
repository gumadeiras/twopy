"""Response plotting dock package for the twopy napari adapter.

Inputs: napari viewer state, converted recordings, and current ROI Labels
layers.
Outputs: docked response plots and tabbed response-plot options.

The package exposes the same high-level dock API while keeping factories,
widget coordination, and option-panel construction in separate modules.
"""

from twopy.napari.plotting.docks.api import (
    add_twopy_response_options_widget,
    add_twopy_response_plot_widget,
    create_response_plot_widget,
    refresh_response_plot_widget,
)

__all__ = [
    "add_twopy_response_options_widget",
    "add_twopy_response_plot_widget",
    "create_response_plot_widget",
    "refresh_response_plot_widget",
]
