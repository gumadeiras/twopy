"""Runtime guards for narrowing untyped boundary values.

Inputs: objects decoded from HDF5, JSON, YAML, Qt, napari, or similar external
libraries.
Outputs: plain Python values whose runtime shape has been checked before type
narrowing.

These helpers keep ``typing.cast`` localized to the point where runtime
validation proves a value's contract.
"""

from collections.abc import Collection
from typing import SupportsInt, cast

__all__ = [
    "int_or_none",
    "int_tuple_or_none",
    "string_key_mapping_or_none",
    "require_string_choice",
    "require_string_key_mapping",
    "require_string_key_mapping_values",
]


def int_or_none(value: object) -> int | None:
    """Return an integer value when conversion is unambiguous.

    Args:
        value: Candidate integer-like value from external metadata.

    Returns:
        Plain ``int`` or ``None`` when conversion fails.

    Boolean values are rejected even though ``bool`` subclasses ``int`` because
    crop bounds and shapes should not silently treat flags as coordinates.
    """
    if isinstance(value, bool):
        return None
    if not isinstance(value, str | bytes | bytearray) and not hasattr(value, "__int__"):
        return None
    try:
        return int(cast(str | bytes | bytearray | SupportsInt, value))
    except (TypeError, ValueError):
        return None


def int_tuple_or_none(value: object, *, length: int) -> tuple[int, ...] | None:
    """Return a fixed-length tuple of integers when every item parses.

    Args:
        value: Candidate sequence of integer-like values.
        length: Required tuple length.

    Returns:
        Integer tuple or ``None`` when the value is not a matching sequence.
    """
    if not isinstance(value, tuple | list) or len(value) != length:
        return None

    parsed: list[int] = []
    for item in value:
        parsed_item = int_or_none(item)
        if parsed_item is None:
            return None
        parsed.append(parsed_item)
    return tuple(parsed)


def string_key_mapping_or_none(value: object) -> dict[str, object] | None:
    """Return a dictionary when every key is text.

    Args:
        value: Candidate mapping-like object from external metadata.

    Returns:
        Copy of the dictionary with string keys, or ``None`` when invalid.
    """
    if not isinstance(value, dict):
        return None

    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def require_string_key_mapping(value: object, *, name: str) -> dict[str, object]:
    """Return a dictionary after validating that every key is text.

    Args:
        value: Candidate dictionary from an external boundary.
        name: Human-readable value name for error messages.

    Returns:
        Copy of the dictionary with string keys.

    Raises:
        ValueError: If ``value`` is not a dictionary with string keys.
    """
    result = string_key_mapping_or_none(value)
    if result is None:
        msg = f"{name} must be a dictionary with string keys"
        raise ValueError(msg)
    return result


def require_string_key_mapping_values(
    value: object,
    *,
    name: str,
) -> dict[str, dict[str, object]]:
    """Return a dictionary whose values are string-key dictionaries.

    Args:
        value: Candidate nested dictionary from an external boundary.
        name: Human-readable value name for error messages.

    Returns:
        Nested dictionary copied after validating both mapping levels.

    Raises:
        ValueError: If the top-level value or any nested value is malformed.
    """
    outer = require_string_key_mapping(value, name=name)
    result: dict[str, dict[str, object]] = {}
    for key, item in outer.items():
        result[key] = require_string_key_mapping(
            item,
            name=f"{name}[{key!r}]",
        )
    return result


def require_string_choice[TextChoice: str](
    value: object,
    *,
    name: str,
    allowed: Collection[TextChoice],
) -> TextChoice:
    """Return a text value after checking it is one of the allowed choices.

    Args:
        value: Candidate text value from an external boundary.
        name: Human-readable value name for error messages.
        allowed: Valid string values.

    Returns:
        The selected allowed value narrowed to the literal type of ``allowed``.

    Raises:
        ValueError: If ``value`` is not one of the allowed strings.
    """
    text = str(value)
    if text in allowed:
        return cast(TextChoice, text)

    allowed_text = ", ".join(repr(item) for item in sorted(allowed))
    msg = f"{name} must be one of {allowed_text}; got {text!r}"
    raise ValueError(msg)
