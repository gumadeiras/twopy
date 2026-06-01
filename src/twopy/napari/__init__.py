"""Napari adapter for twopy.

Inputs: converted twopy recordings, ROI labels, and optional launch settings.
Outputs: napari viewers, image layers, Labels ROI layers, and small magicgui
controls.

The package keeps GUI code scoped by responsibility. Core conversion,
database, ROI, and analysis modules stay independent so a future napari plugin
can wrap these helpers instead of owning scientific logic.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twopy.napari.controls import NapariSidebarWidgets, add_twopy_magicgui_controls
    from twopy.napari.launcher import launch_napari, main
    from twopy.napari.roi import roi_label_image_from_layer, save_napari_label_rois
    from twopy.napari.types import NapariRecordingView
    from twopy.napari.viewer import open_recording_in_napari

__all__ = [
    "NapariRecordingView",
    "NapariSidebarWidgets",
    "add_twopy_magicgui_controls",
    "launch_napari",
    "main",
    "open_recording_in_napari",
    "roi_label_image_from_layer",
    "save_napari_label_rois",
]

_LAZY_EXPORTS = {
    "NapariRecordingView": ("twopy.napari.types", "NapariRecordingView"),
    "NapariSidebarWidgets": ("twopy.napari.controls", "NapariSidebarWidgets"),
    "add_twopy_magicgui_controls": (
        "twopy.napari.controls",
        "add_twopy_magicgui_controls",
    ),
    "launch_napari": ("twopy.napari.launcher", "launch_napari"),
    "main": ("twopy.napari.launcher", "main"),
    "open_recording_in_napari": (
        "twopy.napari.viewer",
        "open_recording_in_napari",
    ),
    "roi_label_image_from_layer": (
        "twopy.napari.roi",
        "roi_label_image_from_layer",
    ),
    "save_napari_label_rois": ("twopy.napari.roi", "save_napari_label_rois"),
}


def __getattr__(name: str) -> object:
    """Load napari helpers only when callers ask for them.

    Args:
        name: Attribute requested from this package.

    Returns:
        The exported napari helper or type.

    Importing a napari submodule for tests or scripting should not also import
    the full viewer/control stack. Attribute access keeps the public package API
    intact while making the heavier GUI modules explicit.
    """
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from error
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
