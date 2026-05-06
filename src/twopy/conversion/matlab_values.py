"""Convert MATLAB-loaded values into plain Python values.

Inputs: objects returned by SciPy MATLAB loading.
Outputs: scalars, tuples, dictionaries, and JSON-ready values.

The conversion layer keeps MATLAB-specific object handling out of analysis code.
"""

from pathlib import Path
from typing import Protocol, cast

import numpy as np
import scipy.io


class _MatlabStruct(Protocol):
    """Small protocol for SciPy MATLAB structs.

    Inputs: SciPy ``mat_struct`` objects.
    Outputs: field names used by conversion helpers.
    """

    _fieldnames: list[str] | tuple[str, ...]


def load_scipy_variable(path: Path, name: str) -> object:
    """Load one variable from an older MATLAB file through SciPy.

    Args:
        path: MATLAB file path.
        name: Variable name to load.

    Returns:
        Loaded MATLAB variable as a Python/NumPy object.

    Raises:
        KeyError: If the variable is absent.
    """
    loaded = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    if name not in loaded:
        msg = f"Missing variable {name!r} in {path}"
        raise KeyError(msg)
    return cast(object, loaded[name])


def nested_matlab_field(root: object, dotted_name: str) -> object:
    """Read a dotted field path from a MATLAB struct.

    Args:
        root: MATLAB struct-like object.
        dotted_name: Field path such as ``acq.frameRate``.

    Returns:
        Python value for the nested field.
    """
    value = root
    for part in dotted_name.split("."):
        value = getattr(value, part)
    return matlab_value_to_python(value)


def matlab_struct_to_dict(value: object) -> dict[str, object]:
    """Convert a MATLAB struct-like object to a plain dictionary.

    Args:
        value: Object returned by SciPy for a MATLAB struct.

    Returns:
        Dictionary of field names to converted values.
    """
    struct = cast(_MatlabStruct, value)
    fields = tuple(struct._fieldnames)
    return {field: matlab_value_to_python(getattr(struct, field)) for field in fields}


def matlab_value_to_python(value: object) -> object:
    """Convert small MATLAB values into plain Python-friendly values.

    Args:
        value: MATLAB value loaded by SciPy.

    Returns:
        Scalars as Python scalars, struct arrays as tuples, and numeric arrays as
        NumPy arrays.
    """
    if hasattr(value, "_fieldnames"):
        return matlab_struct_to_dict(value)

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ()
        if value.dtype == object:
            return tuple(matlab_value_to_python(item) for item in np.ravel(value))
        if value.size == 1:
            return matlab_value_to_python(np.ravel(value)[0])
        return value

    if isinstance(value, np.generic):
        return value.item()

    return value


def json_ready(value: object) -> object:
    """Convert Python/NumPy values into JSON-compatible objects.

    Args:
        value: Converted MATLAB value or Python scalar.

    Returns:
        JSON-compatible nested scalars, lists, and dictionaries.
    """
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=object).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
