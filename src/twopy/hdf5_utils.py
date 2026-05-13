"""Small HDF5 value helpers shared by twopy persistence boundaries.

Inputs: raw h5py dataset values and attributes.
Outputs: plain Python strings, scalars, lists, and UTF-8 string datasets.

These helpers keep HDF5 encoding decisions visible and consistent without
hiding the higher-level file schema owned by each caller.
"""

from typing import Literal, overload

import h5py
import numpy as np

__all__ = [
    "Hdf5Scalar",
    "decode_hdf5_string",
    "plain_hdf5_attr_value",
    "read_string_dataset",
    "write_string_dataset",
]

type Hdf5Scalar = str | int | float | bool


def decode_hdf5_string(value: object) -> str:
    """Decode one HDF5 string-like scalar.

    Args:
        value: Raw value read from h5py.

    Returns:
        Text decoded as UTF-8 when h5py returns bytes, otherwise ``str(value)``.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return str(value)


def read_string_dataset(
    group: h5py.Group | h5py.File,
    name: str,
) -> tuple[str, ...]:
    """Read a one-dimensional UTF-8 HDF5 string dataset.

    Args:
        group: HDF5 group or file containing the dataset.
        name: Dataset name or path.

    Returns:
        Tuple of decoded strings.
    """
    return tuple(decode_hdf5_string(value) for value in group[name][()])


def write_string_dataset(
    group: h5py.Group | h5py.File,
    name: str,
    values: tuple[str, ...],
) -> None:
    """Write text values as a UTF-8 HDF5 dataset.

    Args:
        group: HDF5 group or file that receives the dataset.
        name: Dataset name.
        values: Text values to persist.

    Returns:
        None.
    """
    group.create_dataset(
        name,
        data=np.asarray(values, dtype=h5py.string_dtype("utf-8")),
    )


@overload
def plain_hdf5_attr_value(
    value: object,
    *,
    allow_array: Literal[True],
    scalar_only: bool = False,
) -> object: ...


@overload
def plain_hdf5_attr_value(
    value: object,
    *,
    allow_array: bool = False,
    scalar_only: Literal[True] = True,
) -> Hdf5Scalar: ...


@overload
def plain_hdf5_attr_value(
    value: object,
    *,
    allow_array: Literal[False] = False,
    scalar_only: Literal[False] = False,
) -> object: ...


def plain_hdf5_attr_value(
    value: object,
    *,
    allow_array: bool = False,
    scalar_only: bool = False,
) -> object:
    """Convert a small HDF5 attribute value into plain Python data.

    Args:
        value: Raw h5py attribute value.
        allow_array: Whether array attributes should become Python lists.
        scalar_only: Whether unsupported values should be coerced to text.

    Returns:
        A decoded string, Python scalar, list, or original value depending on
        the requested boundary contract.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer | int):
        return int(value)
    if isinstance(value, np.floating | float):
        return float(value)
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        if isinstance(item, str | int | float | bool):
            return item
        return str(item) if scalar_only else item
    if isinstance(value, np.ndarray):
        if allow_array:
            return np.asarray(value, dtype=object).tolist()
        return str(value) if scalar_only else value
    if scalar_only and not isinstance(value, str | int | float | bool):
        return str(value)
    return value
