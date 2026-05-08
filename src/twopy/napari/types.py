"""Typed data objects returned by the twopy napari adapter.

Inputs: loaded recordings and napari layer/widget objects.
Outputs: dataclasses that let scripts inspect the GUI objects twopy created.
"""

from dataclasses import dataclass

from twopy.converted import RecordingData


@dataclass(frozen=True)
class NapariRecordingView:
    """Objects created when a converted recording is opened in napari.

    Inputs: a converted recording path and optional ROI set.
    Outputs: viewer, loaded recording, and the layer objects added to the
    viewer.

    Keeping these references makes script-driven napari sessions auditable:
    callers can inspect which recording was loaded and where each layer came
    from.
    """

    viewer: object
    recording: RecordingData
    mean_image_layer: object
    movie_layer: object | None
    roi_labels_layer: object | None
    load_widget: object | None
    loaded_recordings_widget: object | None
    twopy_sidebar_widget: object | None
    twopy_sidebar_dock_widget: object | None
    response_plot_widget: object | None
    response_plot_dock_widget: object | None
    response_options_widget: object | None
