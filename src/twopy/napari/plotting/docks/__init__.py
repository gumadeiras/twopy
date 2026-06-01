"""Response plotting dock package for the twopy napari adapter.

Inputs: napari viewer state, converted recordings, and current ROI Labels
layers.
Outputs: docked response plots and tabbed response-plot options.

The package exposes the same high-level dock API while keeping factories,
widget coordination, and option-panel construction in separate modules.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twopy.napari.plotting.docks.api import (
        add_twopy_response_plot_widget,
        create_response_plot_widget,
        create_twopy_response_options_widget,
        refresh_response_plot_widget,
    )

__all__ = [
    "add_twopy_response_plot_widget",
    "create_response_plot_widget",
    "create_twopy_response_options_widget",
    "refresh_response_plot_widget",
]

_LAZY_EXPORTS = {
    "add_twopy_response_plot_widget",
    "create_response_plot_widget",
    "create_twopy_response_options_widget",
    "refresh_response_plot_widget",
}


def __getattr__(name: str) -> object:
    """Load response dock API helpers only when callers ask for them.

    Args:
        name: Attribute requested from this package.

    Returns:
        The exported dock API helper.
    """
    if name not in _LAZY_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module = import_module("twopy.napari.plotting.docks.api")
    value = getattr(module, name)
    globals()[name] = value
    return value
