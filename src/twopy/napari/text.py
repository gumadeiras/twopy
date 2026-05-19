"""Small text-formatting helpers for napari status messages and fields.

Inputs: counts, noun labels, and Qt text fields.
Outputs: short human-readable phrases and consistently styled placeholders.

Keeping this tiny helper in napari scope avoids repeating plural rules across
separate dock widgets.
"""

from qtpy.QtWidgets import QLineEdit

__all__ = ["configure_placeholder", "counted_noun"]


def configure_placeholder(line_edit: QLineEdit, text: str) -> None:
    """Apply the shared napari placeholder style to one text field.

    Args:
        line_edit: Text field that should show a placeholder hint while empty.
        text: Placeholder text without the trailing ellipsis.

    Returns:
        None.

    The placeholder is shown as italic hint text, while entered user text
    returns to the normal font. The visible placeholder always ends in ``...``.
    """
    line_edit.setPlaceholderText(_placeholder_text(text))
    line_edit.textChanged.connect(
        lambda value, widget=line_edit: _set_hint_font(
            widget,
            is_hint_visible=value == "",
        ),
    )
    _set_hint_font(line_edit, is_hint_visible=line_edit.text() == "")


def counted_noun(count: int, singular: str, plural: str | None = None) -> str:
    """Return a count plus the correct singular or plural noun.

    Args:
        count: Number being displayed.
        singular: Noun to use when ``count`` is one.
        plural: Optional plural noun. When omitted, ``s`` is appended to
            ``singular``.

    Returns:
        Text such as ``"1 ROI"`` or ``"2 files"``.
    """
    noun = singular if count == 1 else plural or f"{singular}s"
    return f"{count} {noun}"


def _placeholder_text(text: str) -> str:
    """Return placeholder text with exactly one trailing ellipsis."""
    normalized = text.strip().removesuffix("...").rstrip()
    return f"{normalized}..."


def _set_hint_font(line_edit: QLineEdit, *, is_hint_visible: bool) -> None:
    """Italicize empty fields so placeholder hints read as hints."""
    font = line_edit.font()
    font.setItalic(is_hint_visible)
    line_edit.setFont(font)
