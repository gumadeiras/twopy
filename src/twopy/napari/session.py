"""Loaded-recording session helpers for the twopy napari adapter.

Inputs: loaded converted recordings and the napari layers created for them.
Outputs: a small Qt list panel plus helper functions for selecting and
unloading recordings.

This module owns GUI session bookkeeping only. It does not load files or compute
responses; it records which viewer layers belong to which loaded recording so
the controls can switch context cleanly.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from twopy.converted import RecordingData
from twopy.napari.plotting import refresh_response_plot_widget
from twopy.napari.protocols import NapariViewer
from twopy.napari.types import NapariRecordingView

__all__ = [
    "LoadedNapariRecording",
    "LoadedRecordingsPanel",
    "record_loaded_view",
    "remove_loaded_recording_layers",
    "render_loaded_recordings_panel",
    "select_loaded_recording",
    "set_loaded_recording_visibility",
    "unload_all_loaded_recordings",
    "unload_loaded_recording",
]


class NapariSessionState(Protocol):
    """State fields required by loaded-recording session helpers."""

    viewer: NapariViewer
    roi_labels_layer: object | None
    roi_save_file: Path
    recording: RecordingData | None
    response_plot_widget: object | None
    trial_timeline_controller: object | None
    defer_timeline_updates: bool
    loaded_recordings: list["LoadedNapariRecording"]
    selected_recording_index: int | None
    loaded_recordings_panel: "LoadedRecordingsPanel | None"
    group_matching_panel: object | None
    recording_required_widgets: list[object]


class _LayerWithVisibility(Protocol):
    """Small protocol for napari layers whose visibility can be toggled."""

    visible: bool


class _TrialTimelineController(Protocol):
    """Small protocol for updating the selected recording timeline."""

    def set_context(
        self,
        recording: RecordingData | None,
        movie_layer: object | None,
    ) -> None:
        """Set the active recording and movie layer."""
        ...


@dataclass(frozen=True)
class LoadedNapariRecording:
    """One recording loaded into a shared napari viewer.

    Inputs: converted recording object plus the layers created for that
    recording.
    Outputs: one auditable record that links napari layers and ROI save default
    to the converted recording that owns them.
    """

    recording: RecordingData
    roi_save_file: Path
    mean_image_layer: object
    movie_layer: object | None
    roi_labels_layer: object | None

    def layers(self) -> tuple[object, ...]:
        """Return all napari layers owned by this loaded recording.

        Args:
            None.

        Returns:
            Tuple containing the mean image, optional movie, and optional ROI
            Labels layer.
        """
        layers: list[object] = [self.mean_image_layer]
        if self.movie_layer is not None:
            layers.append(self.movie_layer)
        if self.roi_labels_layer is not None:
            layers.append(self.roi_labels_layer)
        return tuple(layers)


class LoadedRecordingsPanel(QWidget):
    """Qt panel listing recordings loaded in the current napari session."""

    def __init__(
        self,
        *,
        on_select: Callable[[int | None], None],
        on_unload: Callable[[int], None],
        on_unload_all: Callable[[], None],
    ) -> None:
        """Create an initially empty loaded-recordings panel.

        Args:
            on_select: Callback receiving the selected row, or ``None`` when
                no recording is selected.
            on_unload: Callback receiving the row to unload.
            on_unload_all: Callback unloading every loaded recording.
        """
        super().__init__()
        self._on_select = on_select
        self._on_unload = on_unload
        self._on_unload_all = on_unload_all
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._select_row)
        unload_button = QPushButton("Unload selected")
        unload_button.clicked.connect(self._unload_selected)
        unload_all_button = QPushButton("Unload all")
        unload_all_button.clicked.connect(self._unload_all)

        button_layout = QHBoxLayout()
        button_layout.addWidget(unload_button)
        button_layout.addWidget(unload_all_button)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Loaded Recordings"))
        layout.addWidget(self._list)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def set_recordings(
        self,
        recordings: tuple[LoadedNapariRecording, ...],
        *,
        selected_index: int | None,
    ) -> None:
        """Replace the list contents without changing analysis state.

        Args:
            recordings: Loaded recordings to show.
            selected_index: Row that should be visually selected.

        Returns:
            None.
        """
        self._list.blockSignals(True)
        try:
            self._list.clear()
            for recording in recordings:
                self._list.addItem(
                    str(recording.recording.source_session_dir.expanduser())
                )
            if selected_index is None:
                self._list.setCurrentRow(-1)
            else:
                self._list.setCurrentRow(selected_index)
        finally:
            self._list.blockSignals(False)

    def selected_index(self) -> int | None:
        """Return the current selected row.

        Args:
            None.

        Returns:
            Selected row, or ``None`` when no row is selected.
        """
        row = self._list.currentRow()
        return row if row >= 0 else None

    def _select_row(self, row: int) -> None:
        """Forward list selection changes to the control state."""
        self._on_select(row if row >= 0 else None)

    def _unload_selected(self) -> None:
        """Ask the controls to unload the currently selected recording."""
        selected = self.selected_index()
        if selected is not None:
            self._on_unload(selected)

    def _unload_all(self) -> None:
        """Ask the controls to unload every currently loaded recording."""
        self._on_unload_all()


def remove_loaded_recording_layers(
    viewer: object,
    loaded_recording: LoadedNapariRecording,
) -> None:
    """Remove all viewer layers owned by one loaded recording.

    Args:
        viewer: napari viewer that owns the layers.
        loaded_recording: Loaded recording whose layers should be removed.

    Returns:
        None.

    Real napari exposes a ``viewer.layers`` collection. Tests use small fake
    image/label lists, so the helper supports both shapes without moving layer
    ownership into the tests.
    """
    for layer in loaded_recording.layers():
        _remove_layer(viewer, layer)


def record_loaded_view(
    state: NapariSessionState,
    *,
    view: NapariRecordingView,
    roi_save_file: Path,
    replace_selected: bool,
) -> None:
    """Register a loaded recording view and make it the active context.

    Args:
        state: Mutable napari session state.
        view: Viewer/layer objects returned by ``open_recording_in_napari``.
        roi_save_file: Default ROI file for this recording.
        replace_selected: Whether to replace the selected loaded recording
            instead of appending a new list entry.

    Returns:
        None.
    """
    loaded = LoadedNapariRecording(
        recording=view.recording,
        roi_save_file=roi_save_file,
        mean_image_layer=view.mean_image_layer,
        movie_layer=view.movie_layer,
        roi_labels_layer=view.roi_labels_layer,
    )
    if replace_selected and state.selected_recording_index is not None:
        index = state.selected_recording_index
        remove_loaded_recording_layers(state.viewer, state.loaded_recordings[index])
        state.loaded_recordings[index] = loaded
    else:
        state.loaded_recordings.append(loaded)
        index = len(state.loaded_recordings) - 1
    select_loaded_recording(state, index)


def select_loaded_recording(
    state: NapariSessionState,
    index: int | None,
) -> None:
    """Make one loaded recording the active save/plot context.

    Args:
        state: Mutable napari session state.
        index: Loaded-recording list index, or ``None`` to clear selection.

    Returns:
        None.
    """
    if index is None or index < 0 or index >= len(state.loaded_recordings):
        state.selected_recording_index = None
        state.recording = None
        state.roi_labels_layer = None
        set_loaded_recording_visibility(state.loaded_recordings, selected_index=None)
        refresh_response_plot_widget(
            state.response_plot_widget,
            recording=None,
            roi_labels_layer=None,
            roi_save_file=None,
        )
        if not state.defer_timeline_updates:
            refresh_trial_timeline_controller(
                state.trial_timeline_controller,
                recording=None,
                movie_layer=None,
            )
        render_loaded_recordings_panel(state)
        return

    selected = state.loaded_recordings[index]
    state.selected_recording_index = index
    state.recording = selected.recording
    state.roi_labels_layer = selected.roi_labels_layer
    state.roi_save_file = selected.roi_save_file
    set_loaded_recording_visibility(state.loaded_recordings, selected_index=index)
    refresh_response_plot_widget(
        state.response_plot_widget,
        recording=selected.recording,
        roi_labels_layer=selected.roi_labels_layer,
        roi_save_file=selected.roi_save_file,
    )
    if not state.defer_timeline_updates:
        refresh_trial_timeline_controller(
            state.trial_timeline_controller,
            recording=selected.recording,
            movie_layer=selected.movie_layer,
        )
    render_loaded_recordings_panel(state)


def refresh_trial_timeline_controller(
    controller: object | None,
    *,
    recording: RecordingData | None,
    movie_layer: object | None,
) -> None:
    """Refresh a trial timeline controller when one exists.

    Args:
        controller: Optional timeline controller.
        recording: Selected recording, or ``None``.
        movie_layer: Selected recording's movie layer, or ``None``.

    Returns:
        None.
    """
    if controller is None:
        return
    cast(_TrialTimelineController, controller).set_context(recording, movie_layer)


def unload_loaded_recording(state: NapariSessionState, index: int) -> None:
    """Unload one recording and remove its layers from napari.

    Args:
        state: Mutable napari session state.
        index: Loaded-recording list index to remove.

    Returns:
        None.
    """
    if index < 0 or index >= len(state.loaded_recordings):
        return
    removed = state.loaded_recordings.pop(index)
    remove_loaded_recording_layers(state.viewer, removed)
    if len(state.loaded_recordings) == 0:
        select_loaded_recording(state, None)
        return
    next_index = min(index, len(state.loaded_recordings) - 1)
    select_loaded_recording(state, next_index)


def unload_all_loaded_recordings(state: NapariSessionState) -> None:
    """Unload every recording and remove all owned layers from napari.

    Args:
        state: Mutable napari session state.

    Returns:
        None.
    """
    if len(state.loaded_recordings) == 0:
        return
    for loaded_recording in state.loaded_recordings:
        remove_loaded_recording_layers(state.viewer, loaded_recording)
    state.loaded_recordings.clear()
    select_loaded_recording(state, None)


def render_loaded_recordings_panel(state: NapariSessionState) -> None:
    """Sync the loaded-recordings Qt panel with session state.

    Args:
        state: Mutable napari session state.

    Returns:
        None.
    """
    if state.loaded_recordings_panel is not None:
        state.loaded_recordings_panel.set_recordings(
            tuple(state.loaded_recordings),
            selected_index=state.selected_recording_index,
        )
    refresh = getattr(state.group_matching_panel, "refresh", None)
    if callable(refresh):
        refresh()
    has_recordings = len(state.loaded_recordings) > 0
    for widget in state.recording_required_widgets:
        set_enabled = getattr(widget, "setEnabled", None)
        if callable(set_enabled):
            set_enabled(has_recordings)


def set_loaded_recording_visibility(
    loaded_recordings: list[LoadedNapariRecording],
    *,
    selected_index: int | None,
) -> None:
    """Show selected recording layers and hide layers from other recordings.

    Args:
        loaded_recordings: Recordings currently loaded in the viewer.
        selected_index: Recording index that should be visible, or ``None`` to
            hide all loaded recording layers.

    Returns:
        None.
    """
    for index, loaded_recording in enumerate(loaded_recordings):
        visible = selected_index is not None and index == selected_index
        for layer in loaded_recording.layers():
            if hasattr(layer, "visible"):
                cast(_LayerWithVisibility, layer).visible = visible


def _remove_layer(viewer: object, layer: object) -> None:
    """Remove one layer from napari or from test doubles.

    Args:
        viewer: Real or fake napari viewer.
        layer: Layer object to remove.

    Returns:
        None.
    """
    layer_collection = getattr(viewer, "layers", None)
    if layer_collection is not None:
        try:
            layer_collection.remove(layer)
            return
        except (AttributeError, TypeError, ValueError):
            pass

    for collection_name in ("images", "labels"):
        collection = getattr(viewer, collection_name, None)
        if isinstance(collection, list) and layer in collection:
            collection.remove(layer)
            return
