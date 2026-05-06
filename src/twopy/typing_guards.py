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

import numpy as np
import numpy.typing as npt

__all__ = [
    "int_or_none",
    "int_tuple_or_none",
    "require_bool_array",
    "require_float64_array",
    "require_int64_array",
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


def require_float64_array(
    value: object,
    *,
    name: str,
    ndim: int | None = None,
) -> npt.NDArray[np.float64]:
    """Return a NumPy array after validating float64 dtype and rank.

    Args:
        value: Candidate array value from an external boundary.
        name: Human-readable dataset name for error messages.
        ndim: Optional required number of dimensions.

    Returns:
        ``float64`` NumPy array.

    Raises:
        ValueError: If dtype or rank does not match.
    """
    array = np.asarray(value)
    _require_array_contract(array, name=name, dtype=np.dtype(np.float64), ndim=ndim)
    return cast(npt.NDArray[np.float64], array)


def require_int64_array(
    value: object,
    *,
    name: str,
    ndim: int | None = None,
) -> npt.NDArray[np.int64]:
    """Return a NumPy array after validating int64 dtype and rank.

    Args:
        value: Candidate array value from an external boundary.
        name: Human-readable dataset name for error messages.
        ndim: Optional required number of dimensions.

    Returns:
        ``int64`` NumPy array.

    Raises:
        ValueError: If dtype or rank does not match.
    """
    array = np.asarray(value)
    _require_array_contract(array, name=name, dtype=np.dtype(np.int64), ndim=ndim)
    return cast(npt.NDArray[np.int64], array)


def require_bool_array(
    value: object,
    *,
    name: str,
    ndim: int | None = None,
) -> npt.NDArray[np.bool_]:
    """Return a NumPy array after validating bool dtype and rank.

    Args:
        value: Candidate array value from an external boundary.
        name: Human-readable dataset name for error messages.
        ndim: Optional required number of dimensions.

    Returns:
        Boolean NumPy array.

    Raises:
        ValueError: If dtype or rank does not match.
    """
    array = np.asarray(value)
    _require_array_contract(array, name=name, dtype=np.dtype(np.bool_), ndim=ndim)
    return cast(npt.NDArray[np.bool_], array)


def _require_array_contract(
    array: npt.NDArray[np.generic],
    *,
    name: str,
    dtype: np.dtype[np.generic],
    ndim: int | None,
) -> None:
    """Validate the dtype and rank expected for a persisted array."""
    if array.dtype != dtype:
        msg = f"{name} must have dtype {dtype.name}; got {array.dtype}"
        raise ValueError(msg)
    if ndim is not None and array.ndim != ndim:
        msg = f"{name} must have {ndim} dimension(s); got shape {array.shape}"
        raise ValueError(msg)
