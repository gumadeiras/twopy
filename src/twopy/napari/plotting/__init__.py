"""Response plotting package for the twopy napari adapter.

Inputs: converted recordings, analysis outputs, ROI Labels layers, and plot
options.
Outputs: response plot dock widgets plus small data and option helpers.

This package keeps response plotting scoped under one napari subpackage. It
does not own conversion, ROI storage, or scientific analysis logic.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

_LAZY_EXPORTS = {
    "add_twopy_response_plot_widget",
    "create_twopy_response_options_widget",
    "refresh_response_plot_widget",
}


def __getattr__(name: str) -> object:
    """Load response dock helpers only when callers ask for them.

    Args:
        name: Attribute requested from this package.

    Returns:
        The exported response dock helper.

    Keeping the dock import lazy lets plot-data and widget submodules import this
    package without also importing the full response dock and figure-export stack.
    """
    if name not in _LAZY_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module = import_module("twopy.napari.plotting.docks")
    value = getattr(module, name)
    globals()[name] = value
    return value
