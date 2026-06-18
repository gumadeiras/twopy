"""Empty-viewer message helpers for the twopy napari adapter.

Inputs: napari viewer objects that may expose welcome and text overlays.
Outputs: a twopy-owned message while no recording is loaded.

The center viewer belongs to napari, but twopy owns the empty recording
workflow. These helpers hide napari's generic welcome overlay and use napari's
public text overlay for the small twopy-specific launch message.
"""

from typing import Protocol, cast

from twopy._version import __version__

__all__ = [
    "EMPTY_VIEWER_MESSAGE",
    "hide_empty_viewer_message",
    "show_empty_viewer_message",
]

EMPTY_VIEWER_MESSAGE = (
    f"twopy {__version__}\n\n"
    "Getting Started\n\n"
    "Search or load recordings manually\n"
    "using the Load tab on the right."
)


class _WelcomeScreen(Protocol):
    """Small shape of napari's welcome overlay."""

    visible: bool


class _ViewerWithWelcomeScreen(Protocol):
    """Small viewer shape for napari welcome-screen updates."""

    welcome_screen: object


class _ViewerWithTextOverlay(Protocol):
    """Small viewer shape for napari text overlay updates."""

    text_overlay: object


class _TextOverlay(Protocol):
    """Small shape of napari's canvas text overlay."""

    text: str
    visible: bool
    position: str
    font_size: int
    color: object
    box: bool
    box_color: object


def show_empty_viewer_message(viewer: object) -> None:
    """Show twopy's empty-recording message on a napari viewer.

    Args:
        viewer: Napari viewer or test double.

    Returns:
        None.

    This replaces napari's generic welcome screen only while twopy has no
    loaded recording. The message tells the user what the empty canvas means in
    the twopy workflow.
    """
    welcome_screen = _welcome_screen(viewer)
    if welcome_screen is not None:
        welcome_screen.visible = False

    overlay = _text_overlay(viewer)
    if overlay is None:
        return
    overlay.text = EMPTY_VIEWER_MESSAGE
    overlay.visible = True
    overlay.position = "top_center"
    overlay.font_size = 13
    overlay.color = "white"
    overlay.box = True
    overlay.box_color = (0.0, 0.0, 0.0, 0.65)


def hide_empty_viewer_message(viewer: object) -> None:
    """Hide twopy's empty-recording message when real content owns the viewer.

    Args:
        viewer: Napari viewer or test double.

    Returns:
        None.

    The helper clears only the exact twopy empty message. Trial HUD text and any
    other overlay owner are left untouched.
    """
    overlay = _text_overlay(viewer)
    if overlay is None or overlay.text != EMPTY_VIEWER_MESSAGE:
        return
    overlay.text = ""
    overlay.visible = False


def _welcome_screen(viewer: object) -> _WelcomeScreen | None:
    """Return napari's welcome screen overlay when the viewer exposes one."""
    if not hasattr(viewer, "welcome_screen"):
        return None
    return cast(_WelcomeScreen, cast(_ViewerWithWelcomeScreen, viewer).welcome_screen)


def _text_overlay(viewer: object) -> _TextOverlay | None:
    """Return napari's text overlay when the viewer exposes one."""
    if not hasattr(viewer, "text_overlay"):
        return None
    return cast(_TextOverlay, cast(_ViewerWithTextOverlay, viewer).text_overlay)
