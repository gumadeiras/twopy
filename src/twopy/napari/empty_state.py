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
    "refresh_empty_viewer_message",
    "show_empty_viewer_message",
]

_EMPTY_VIEWER_GUIDANCE = (
    "Getting Started\n\n"
    "Search or load recordings manually\n"
    "using the Load tab on the right."
)
_BASE_EMPTY_VIEWER_MESSAGE = f"twopy {__version__}\n\n{_EMPTY_VIEWER_GUIDANCE}"
_UPDATE_NOTICE_ATTRIBUTE = "_twopy_empty_viewer_update_notice"
EMPTY_VIEWER_MESSAGE = _BASE_EMPTY_VIEWER_MESSAGE


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


def show_empty_viewer_message(
    viewer: object, *, update_notice: str | None = None
) -> None:
    """Show twopy's empty-recording message on a napari viewer.

    Args:
        viewer: Napari viewer or test double.
        update_notice: Optional compact notice shown when a newer twopy release is
            known.

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
    resolved_notice = update_notice or _stored_update_notice(viewer)
    overlay.text = empty_viewer_message(update_notice=resolved_notice)
    overlay.visible = True
    overlay.position = "top_center"
    overlay.font_size = 18
    overlay.color = None
    overlay.box = False
    overlay.box_color = (0.0, 0.0, 0.0, 0.0)


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
    if overlay is None or not _is_empty_viewer_message(overlay.text):
        return
    overlay.text = ""
    overlay.visible = False


def refresh_empty_viewer_message(
    viewer: object,
    *,
    update_notice: str | None,
) -> None:
    """Update twopy's empty-recording message when it already owns the overlay.

    Args:
        viewer: Napari viewer or test double.
        update_notice: Optional compact notice shown when a newer twopy release is
            known.

    Returns:
        None.
    """
    setattr(viewer, _UPDATE_NOTICE_ATTRIBUTE, update_notice)
    overlay = _text_overlay(viewer)
    if overlay is None or not _is_empty_viewer_message(overlay.text):
        return
    overlay.text = empty_viewer_message(update_notice=update_notice)


def empty_viewer_message(*, update_notice: str | None = None) -> str:
    """Return the empty-viewer text for the current update state."""
    if not update_notice:
        return _BASE_EMPTY_VIEWER_MESSAGE
    return f"twopy {__version__}\n{update_notice}\n\n{_EMPTY_VIEWER_GUIDANCE}"


def _stored_update_notice(viewer: object) -> str | None:
    """Return the compact update notice stored on one viewer."""
    notice = getattr(viewer, _UPDATE_NOTICE_ATTRIBUTE, None)
    return notice if isinstance(notice, str) else None


def _is_empty_viewer_message(text: str) -> bool:
    """Return whether text is owned by twopy's empty-recording overlay."""
    return text == EMPTY_VIEWER_MESSAGE or (
        text.startswith(f"twopy {__version__}\n")
        and text.endswith(_EMPTY_VIEWER_GUIDANCE)
    )


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
