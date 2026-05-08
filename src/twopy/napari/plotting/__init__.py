"""Response plotting package for the twopy napari adapter.

Inputs: converted recordings, analysis outputs, ROI Labels layers, and plot
options.
Outputs: response plot dock widgets plus small data and option helpers.

This package keeps response plotting scoped under one napari subpackage. It
does not own conversion, ROI storage, or scientific analysis logic.
"""

from twopy.napari.plotting.docks import (
    add_twopy_response_plot_widget,
    create_twopy_response_options_widget,
    refresh_response_plot_widget,
)

__all__ = [
    "add_twopy_response_plot_widget",
    "create_twopy_response_options_widget",
    "refresh_response_plot_widget",
]
