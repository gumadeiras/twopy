"""Read prior lab saved-analysis outputs for parity checks.

Inputs: optional ``savedAnalysis/*.mat`` files produced by prior analysis.
Outputs: typed NumPy arrays that tests or audit scripts can compare to twopy
outputs.

This module is read-only. It does not make source MATLAB files part of the
twopy analysis path; it exists only to compare twopy-owned converted analysis
against previous saved outputs when those files are available.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt

__all__ = ["SavedAnalysisLastRoi", "load_saved_analysis_last_roi"]


@dataclass(frozen=True)
class SavedAnalysisLastRoi:
    """Selected fields from one saved ``lastRoi`` object.

    Inputs: one HDF5-backed MATLAB ``savedAnalysis`` file.
    Outputs: ROI masks, epoch labels, epoch windows, and saved ROI traces.

    These fields are enough to build numeric parity checks without coupling
    normal twopy analysis to MATLAB-owned files.
    """

    path: Path
    roi_mask_initial: npt.NDArray[np.float64]
    epoch_list: npt.NDArray[np.int64]
    epoch_start_times: tuple[npt.NDArray[np.int64], ...]
    epoch_durations: tuple[npt.NDArray[np.int64], ...]
    time_by_rois_initial: tuple[npt.NDArray[np.float64], ...]


def load_saved_analysis_last_roi(path: Path) -> SavedAnalysisLastRoi:
    """Load selected ``lastRoi`` outputs from a saved-analysis MAT file.

    Args:
        path: Path to an HDF5-backed MATLAB saved-analysis file.

    Returns:
        ``SavedAnalysisLastRoi`` with arrays ready for parity checks.

    Raises:
        ValueError: If the file does not contain the expected ``lastRoi``
            fields.

    MATLAB v7.3 files store cell contents as HDF5 object references. This reader
    resolves those references but keeps the output small and explicit.
    """
    input_path = path.expanduser()
    with h5py.File(input_path, "r") as h5_file:
        if "lastRoi" not in h5_file:
            msg = f"Saved analysis file has no 'lastRoi' group: {input_path}"
            raise ValueError(msg)
        roi_group = h5_file["lastRoi"]
        return SavedAnalysisLastRoi(
            path=input_path,
            roi_mask_initial=cast(
                npt.NDArray[np.float64],
                roi_group["roiMaskInitial"][()],
            ),
            epoch_list=np.ravel(roi_group["epochList"][()]).astype(np.int64),
            epoch_start_times=_read_int_cell_arrays(
                h5_file,
                roi_group,
                "epochStartTimes",
            ),
            epoch_durations=_read_int_cell_arrays(
                h5_file,
                roi_group,
                "epochDurations",
            ),
            time_by_rois_initial=_read_trace_cell_arrays(
                h5_file,
                roi_group,
                "timeByRoisInitial",
            ),
        )


def _read_int_cell_arrays(
    h5_file: h5py.File,
    group: h5py.Group,
    dataset_name: str,
) -> tuple[npt.NDArray[np.int64], ...]:
    """Read a MATLAB cell array of integer vectors.

    Args:
        h5_file: Open HDF5 file.
        group: Group containing the cell dataset.
        dataset_name: Name of the cell dataset.

    Returns:
        Tuple of one-dimensional integer arrays.
    """
    arrays = _read_cell_arrays(h5_file, group, dataset_name)
    return tuple(np.ravel(array).astype(np.int64) for array in arrays)


def _read_trace_cell_arrays(
    h5_file: h5py.File,
    group: h5py.Group,
    dataset_name: str,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Read a MATLAB cell array of ROI trace matrices.

    Args:
        h5_file: Open HDF5 file.
        group: Group containing the cell dataset.
        dataset_name: Name of the cell dataset.

    Returns:
        Tuple of trace arrays shaped ``(frames, rois)`` when the saved matrix is
        two-dimensional.
    """
    arrays = _read_cell_arrays(h5_file, group, dataset_name)
    return tuple(_matlab_trace_matrix_to_frames_by_rois(array) for array in arrays)


def _read_cell_arrays(
    h5_file: h5py.File,
    group: h5py.Group,
    dataset_name: str,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Resolve a MATLAB cell array stored as HDF5 references.

    Args:
        h5_file: Open HDF5 file.
        group: Group containing the cell dataset.
        dataset_name: Name of the cell dataset.

    Returns:
        Tuple of referenced numeric arrays.
    """
    if dataset_name not in group:
        msg = f"Saved analysis lastRoi is missing {dataset_name!r}"
        raise ValueError(msg)
    references = group[dataset_name][()]
    arrays: list[npt.NDArray[np.float64]] = []
    for reference in np.ravel(references):
        if not isinstance(reference, h5py.Reference) or not reference:
            continue
        arrays.append(cast(npt.NDArray[np.float64], h5_file[reference][()]))
    return tuple(arrays)


def _matlab_trace_matrix_to_frames_by_rois(
    array: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return saved trace matrices as ``(frames, rois)``.

    Args:
        array: Referenced MATLAB trace matrix.

    Returns:
        Trace matrix with frame axis first.

    MATLAB writes column-major arrays into HDF5 with dimensions that appear
    reversed to h5py. Saved ROI traces are usually ``rois x frames`` when read
    directly, so this helper transposes two-dimensional trace matrices into the
    twopy convention.
    """
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 2:
        return values.T
    return values
