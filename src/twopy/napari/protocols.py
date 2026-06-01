"""Tiny napari protocols used by tests and GUI adapter code.

Inputs: napari viewer, window, and layer objects.
Outputs: structural types for the small subset of napari behavior twopy uses.

These protocols keep the adapter testable without importing or starting napari
in unit tests.
"""

from collections.abc import Callable
from typing import Protocol


class NapariWindow(Protocol):
    """Small protocol for adding dock widgets to a napari viewer.

    Inputs: widget objects.
    Outputs: dock widgets owned by napari.
    """

    def add_dock_widget(
        self,
        widget: object,
        *,
        name: str,
        area: str,
    ) -> object:
        """Add a dock widget to the viewer window.

        Args:
            widget: Widget object to dock.
            name: Dock title shown by napari.
            area: Napari dock area name.

        Returns:
            The dock widget created by napari.
        """
        ...


class NapariViewer(Protocol):
    """Small viewer protocol used to keep tests independent from napari.

    Inputs: image or label layer data.
    Outputs: napari layer objects.
    """

    @property
    def window(self) -> NapariWindow:
        """Return the viewer window used to add dock widgets.

        Inputs: viewer object.
        Outputs: napari window object.
        """
        ...

    def add_image(self, data: object, *, name: str, **kwargs: object) -> object:
        """Add an image layer to the viewer.

        Args:
            data: Array-like image data.
            name: Layer name shown in napari.
            kwargs: napari image-layer options.

        Returns:
            The layer object created by napari.
        """
        ...

    def add_labels(self, data: object, *, name: str, **kwargs: object) -> object:
        """Add a labels layer to the viewer.

        Args:
            data: Two-dimensional integer label image.
            name: Layer name shown in napari.
            kwargs: napari label-layer options.

        Returns:
            The layer object created by napari.
        """
        ...


class NapariLayerWithData(Protocol):
    """Small protocol for napari layers that expose array-like data.

    Inputs: a napari layer object.
    Outputs: access to its ``data`` attribute.
    """

    data: object


class NapariEventEmitter(Protocol):
    """Small protocol for psygnal event emitters exposed by napari objects.

    Inputs: callback functions owned by GUI controllers.
    Outputs: connection or disconnection side effects on the napari emitter.

    This lets controllers subscribe to napari events without importing napari
    or psygnal during unit tests.
    """

    def connect(self, callback: Callable[..., None]) -> object:
        """Connect one callback to the event emitter.

        Args:
            callback: Function called by the event emitter.

        Returns:
            Emitter-specific connection result.
        """
        ...

    def disconnect(self, callback: Callable[..., None]) -> object:
        """Disconnect one callback from the event emitter.

        Args:
            callback: Function previously connected to the event emitter.

        Returns:
            Emitter-specific disconnection result.
        """
        ...


class NapariViewerWithDims(Protocol):
    """Small protocol for napari viewers that expose dimension state.

    Inputs: napari viewer-like object.
    Outputs: access to the ``dims`` state used for frame-aware tools.
    """

    dims: object
