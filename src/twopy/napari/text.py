"""Small text-formatting helpers for napari status messages.

Inputs: counts and noun labels.
Outputs: short human-readable phrases for GUI status text.

Keeping this tiny helper in napari scope avoids repeating plural rules across
separate dock widgets.
"""

__all__ = ["counted_noun"]


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
