"""Shared protocols for manual group-matching widgets.

Inputs: shared napari GUI state.
Outputs: structural types used by FOV and ROI assignment views.

These protocols keep the staged group-matching widgets decoupled from the full
napari control state while preserving typed access to loaded recordings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from twopy.napari.session import LoadedNapariRecording

__all__ = ["GroupMatchingState"]


class GroupMatchingState(Protocol):
    """Shared napari control-state fields needed by group matching.

    Inputs: the larger napari GUI state object.
    Outputs: the loaded recordings shown by FOV and ROI matching views.
    """

    loaded_recordings: list[LoadedNapariRecording]
