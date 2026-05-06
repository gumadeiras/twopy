"""Napari adapter for twopy.

Inputs: converted twopy recordings, ROI labels, and optional launch settings.
Outputs: napari viewers, image layers, Labels ROI layers, and small magicgui
controls.

The package keeps GUI code scoped by responsibility. Core conversion,
database, ROI, and analysis modules stay independent so a future napari plugin
can wrap these helpers instead of owning scientific logic.
"""

from twopy.napari.controls import add_twopy_magicgui_controls
from twopy.napari.launcher import launch_napari, main
from twopy.napari.roi import roi_label_image_from_layer, save_napari_label_rois
from twopy.napari.types import NapariRecordingView
from twopy.napari.viewer import open_recording_in_napari

__all__ = [
    "NapariRecordingView",
    "add_twopy_magicgui_controls",
    "launch_napari",
    "main",
    "open_recording_in_napari",
    "roi_label_image_from_layer",
    "save_napari_label_rois",
]
