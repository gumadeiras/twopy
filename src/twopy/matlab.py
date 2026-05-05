"""Inspect MATLAB files used by two-photon microscope recordings.

Inputs: MATLAB ``.mat`` files from a recording or prior lab analysis.
Outputs: typed summaries of variables, shapes, dtypes, and storage format.

This module is intentionally an inspection layer first. It lets twopy understand
what MATLAB files contain before later code converts selected variables into
analysis-specific Python objects.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import h5py
import scipy.io

__all__ = [
    "MatlabHdf5Group",
    "MatlabFileSummary",
    "MatlabLoadedFile",
    "MatlabVariableSummary",
    "inspect_mat_file",
    "load_mat_file",
]

MatlabFileFormat = Literal["hdf5-mat", "scipy-mat"]


@dataclass(frozen=True)
class MatlabVariableSummary:
    """Human-readable summary of one MATLAB variable.

    Inputs: one variable loaded from or discovered inside a MATLAB file.
    Outputs: name, Python-facing type, shape, dtype, and whether it is a group.

    This object keeps inspection results small and auditable; large arrays stay
    on disk unless later conversion code explicitly asks to load them.
    """

    name: str
    python_type: str
    shape: tuple[int, ...] | None
    dtype: str | None
    is_group: bool


@dataclass(frozen=True)
class MatlabFileSummary:
    """Summary of one MATLAB file.

    Inputs: path to a MATLAB file.
    Outputs: detected storage format plus summaries of visible variables.

    The summary is designed for logs and GUI inspection, not numerical analysis.
    """

    path: Path
    file_format: MatlabFileFormat
    variables: tuple[MatlabVariableSummary, ...]


@dataclass(frozen=True)
class MatlabHdf5Group:
    """Reference-like summary for an HDF5 group inside a MATLAB file.

    Inputs: group name and its child names.
    Outputs: a small Python object that says the group exists without loading it.

    HDF5 groups can contain nested references and large data. Keeping them as a
    summary avoids pretending every MATLAB value is a simple array.
    """

    name: str
    child_names: tuple[str, ...]


@dataclass(frozen=True)
class MatlabLoadedFile:
    """Loaded MATLAB variables as Python objects.

    Inputs: MATLAB file path and optional variable names to load.
    Outputs: a mapping from variable names to Python objects.

    For older MAT files, SciPy returns Python and NumPy objects. For HDF5-backed
    MAT files, datasets are loaded when requested and groups are summarized.
    """

    path: Path
    file_format: MatlabFileFormat
    variables: dict[str, object]


def inspect_mat_file(path: Path) -> MatlabFileSummary:
    """Inspect one MATLAB file without loading more data than needed.

    Args:
        path: MATLAB ``.mat`` file to inspect.

    Returns:
        A summary describing the file format and visible variables.

    Raises:
        FileNotFoundError: If the path does not exist as a file.
        ValueError: If neither h5py nor SciPy can read the file.

    h5py is tried first because HDF5-backed MAT files can be inspected lazily.
    Older MAT files then fall back to SciPy.
    """
    mat_path = path.expanduser()
    if not mat_path.is_file():
        msg = f"Missing MATLAB file: {mat_path}"
        raise FileNotFoundError(msg)

    try:
        return _inspect_hdf5_mat_file(mat_path)
    except OSError:
        return _inspect_scipy_mat_file(mat_path)


def load_mat_file(
    path: Path,
    variable_names: Iterable[str] | None = None,
) -> MatlabLoadedFile:
    """Load variables from one MATLAB file into Python objects.

    Args:
        path: MATLAB ``.mat`` file to read.
        variable_names: Optional variable names to load. ``None`` means load all
            visible top-level variables.

    Returns:
        Loaded variables keyed by MATLAB variable name.

    Raises:
        FileNotFoundError: If the path does not exist as a file.
        KeyError: If a requested variable is absent.
        ValueError: If neither h5py nor SciPy can read the file.

    Use explicit ``variable_names`` for large HDF5-backed files, such as aligned
    movies, so callers control memory use.
    """
    mat_path = path.expanduser()
    if not mat_path.is_file():
        msg = f"Missing MATLAB file: {mat_path}"
        raise FileNotFoundError(msg)

    wanted = tuple(variable_names) if variable_names is not None else None

    try:
        return _load_hdf5_mat_file(mat_path, wanted)
    except OSError:
        return _load_scipy_mat_file(mat_path, wanted)


def _inspect_hdf5_mat_file(path: Path) -> MatlabFileSummary:
    """Inspect an HDF5-backed MATLAB file with lazy dataset metadata.

    Args:
        path: MATLAB file that h5py can open.

    Returns:
        A summary of top-level HDF5 groups and datasets.

    HDF5 stores array shape and dtype in metadata, so this path avoids loading
    large movie arrays while still showing what is inside.
    """
    variables: list[MatlabVariableSummary] = []

    with h5py.File(path, "r") as mat_file:
        for name, item in sorted(mat_file.items()):
            if isinstance(item, h5py.Dataset):
                variables.append(
                    MatlabVariableSummary(
                        name=name,
                        python_type="h5py.Dataset",
                        shape=tuple(int(length) for length in item.shape),
                        dtype=str(item.dtype),
                        is_group=False,
                    ),
                )
            else:
                variables.append(
                    MatlabVariableSummary(
                        name=name,
                        python_type=type(item).__name__,
                        shape=None,
                        dtype=None,
                        is_group=True,
                    ),
                )

    return MatlabFileSummary(
        path=path, file_format="hdf5-mat", variables=tuple(variables)
    )


def _load_hdf5_mat_file(
    path: Path,
    variable_names: tuple[str, ...] | None,
) -> MatlabLoadedFile:
    """Load requested top-level variables from an HDF5-backed MATLAB file.

    Args:
        path: MATLAB file that h5py can open.
        variable_names: Variable names to load, or ``None`` for all top-level
            entries.

    Returns:
        Loaded dataset values and group summaries.

    HDF5 datasets are read with ``[()]`` because that is h5py's direct way to
    materialize one dataset as a Python/NumPy object.
    """
    variables: dict[str, object] = {}

    with h5py.File(path, "r") as mat_file:
        names = variable_names if variable_names is not None else tuple(mat_file.keys())
        _ensure_hdf5_names_exist(mat_file, names)

        for name in names:
            item = mat_file[name]
            if isinstance(item, h5py.Dataset):
                variables[name] = item[()]
            else:
                variables[name] = MatlabHdf5Group(
                    name=name,
                    child_names=tuple(sorted(str(child) for child in item)),
                )

    return MatlabLoadedFile(path=path, file_format="hdf5-mat", variables=variables)


def _inspect_scipy_mat_file(path: Path) -> MatlabFileSummary:
    """Inspect an older MATLAB file through SciPy.

    Args:
        path: MATLAB file that SciPy can load.

    Returns:
        A summary of non-private MATLAB variables.

    SciPy loads older MAT files into Python objects. We use it only for files
    that h5py cannot open, which avoids accidentally loading large HDF5 arrays.
    """
    try:
        loaded = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=False)
    except (NotImplementedError, OSError, ValueError) as error:
        msg = f"Could not inspect MATLAB file {path}: {error}"
        raise ValueError(msg) from error

    variables = tuple(
        _summarize_loaded_variable(name, value)
        for name, value in sorted(loaded.items())
        if not name.startswith("__")
    )

    return MatlabFileSummary(path=path, file_format="scipy-mat", variables=variables)


def _load_scipy_mat_file(
    path: Path,
    variable_names: tuple[str, ...] | None,
) -> MatlabLoadedFile:
    """Load variables from an older MATLAB file through SciPy.

    Args:
        path: MATLAB file that SciPy can read.
        variable_names: Variable names to keep, or ``None`` for all visible
            variables.

    Returns:
        Loaded variables keyed by MATLAB variable name.

    SciPy loads the file into memory. That is acceptable for the small metadata
    files observed so far, but large files should use HDF5-backed loading.
    """
    try:
        loaded = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=False)
    except (NotImplementedError, OSError, ValueError) as error:
        msg = f"Could not load MATLAB file {path}: {error}"
        raise ValueError(msg) from error

    visible = {
        str(name): cast(object, value)
        for name, value in loaded.items()
        if not name.startswith("__")
    }

    if variable_names is None:
        return MatlabLoadedFile(path=path, file_format="scipy-mat", variables=visible)

    missing = tuple(name for name in variable_names if name not in visible)
    if missing:
        msg = f"Missing MATLAB variable(s) in {path}: {', '.join(missing)}"
        raise KeyError(msg)

    return MatlabLoadedFile(
        path=path,
        file_format="scipy-mat",
        variables={name: visible[name] for name in variable_names},
    )


def _summarize_loaded_variable(name: str, value: object) -> MatlabVariableSummary:
    """Summarize one SciPy-loaded MATLAB variable.

    Args:
        name: MATLAB variable name.
        value: Python object returned by ``scipy.io.loadmat``.

    Returns:
        Shape and dtype information when available, plus the Python type name.

    This function uses structural checks instead of assuming every MATLAB value
    is a NumPy array. MATLAB structs, cells, and scalars can arrive differently.
    """
    shape = _shape_tuple(value)
    dtype = _dtype_name(value)

    return MatlabVariableSummary(
        name=name,
        python_type=type(value).__name__,
        shape=shape,
        dtype=dtype,
        is_group=False,
    )


def _shape_tuple(value: object) -> tuple[int, ...] | None:
    """Return an object's shape as a tuple of ints when it has one.

    Args:
        value: Python object from a MATLAB loader.

    Returns:
        The object's shape or ``None`` when shape metadata is absent.

    The conversion to plain ``int`` keeps summaries stable across NumPy scalar
    integer implementations.
    """
    shape = getattr(value, "shape", None)
    if not isinstance(shape, tuple):
        return None
    return tuple(int(length) for length in shape)


def _dtype_name(value: object) -> str | None:
    """Return an object's dtype name when it has one.

    Args:
        value: Python object from a MATLAB loader.

    Returns:
        String dtype name or ``None`` when dtype metadata is absent.

    The result is text because the inspection layer is for display and logging.
    """
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(cast(object, dtype))


def _ensure_hdf5_names_exist(mat_file: h5py.File, names: tuple[str, ...]) -> None:
    """Raise a clear error when requested HDF5 variables are absent.

    Args:
        mat_file: Open HDF5-backed MATLAB file.
        names: Requested top-level variable names.

    Returns:
        None when every requested name exists.

    Raises:
        KeyError: If one or more names are absent.
    """
    missing = tuple(name for name in names if name not in mat_file)
    if missing:
        msg = f"Missing MATLAB variable(s) in {mat_file.filename}: {', '.join(missing)}"
        raise KeyError(msg)
