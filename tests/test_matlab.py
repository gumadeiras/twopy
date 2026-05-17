"""Tests for MATLAB file inspection.

Inputs: small temporary MATLAB files written through SciPy and h5py.
Outputs: assertions that twopy reports variable names, shapes, and formats.
"""

import unittest
from pathlib import Path

import h5py
import scipy.io
from tests.tempdir import temporary_directory

from twopy.matlab import MatlabHdf5Group, inspect_mat_file, load_mat_file


class InspectMatFileTest(unittest.TestCase):
    """Tests the MATLAB inspection layer used before conversion."""

    def test_inspects_scipy_mat_file(self) -> None:
        """Confirm older MATLAB files expose variable summaries.

        Inputs: a temporary SciPy-written MAT file.
        Outputs: assertions for file format, variable name, shape, and dtype.
        """
        with temporary_directory() as temp_dir:
            mat_path = Path(temp_dir) / "metadata.mat"
            scipy.io.savemat(mat_path, {"frame_times": [[0.0, 1.0, 2.0]]})

            summary = inspect_mat_file(mat_path)

            variable = summary.variables[0]
            self.assertEqual(summary.file_format, "scipy-mat")
            self.assertEqual(variable.name, "frame_times")
            self.assertEqual(variable.shape, (1, 3))
            self.assertIn("float", variable.dtype or "")

    def test_inspects_hdf5_mat_file_without_loading_data(self) -> None:
        """Confirm HDF5-backed MAT files expose lazy dataset metadata.

        Inputs: a temporary HDF5 file with a dataset shaped like movie data.
        Outputs: assertions for HDF5 format, shape, dtype, and dataset type.
        """
        with temporary_directory() as temp_dir:
            mat_path = Path(temp_dir) / "alignedMovie.mat"
            with h5py.File(mat_path, "w") as mat_file:
                mat_file.create_dataset(
                    "aligned_movie", shape=(3, 4, 5), dtype="uint16"
                )

            summary = inspect_mat_file(mat_path)

            variable = summary.variables[0]
            self.assertEqual(summary.file_format, "hdf5-mat")
            self.assertEqual(variable.name, "aligned_movie")
            self.assertEqual(variable.shape, (3, 4, 5))
            self.assertEqual(variable.dtype, "uint16")

    def test_loads_requested_scipy_variables(self) -> None:
        """Confirm older MATLAB variables can be loaded by name.

        Inputs: a temporary SciPy-written MAT file.
        Outputs: an assertion that only the requested variable is returned.
        """
        with temporary_directory() as temp_dir:
            mat_path = Path(temp_dir) / "metadata.mat"
            scipy.io.savemat(mat_path, {"frame_times": [[0.0]], "other": [[1.0]]})

            loaded = load_mat_file(mat_path, variable_names=("frame_times",))

            self.assertEqual(tuple(loaded.variables), ("frame_times",))

    def test_loads_requested_hdf5_datasets_and_summarizes_groups(self) -> None:
        """Confirm HDF5 datasets load while groups remain summarized.

        Inputs: a temporary HDF5 MAT-like file with one dataset and one group.
        Outputs: assertions that dataset and group handling stay explicit.
        """
        with temporary_directory() as temp_dir:
            mat_path = Path(temp_dir) / "alignedMovie.mat"
            with h5py.File(mat_path, "w") as mat_file:
                mat_file.create_dataset("aligned_movie", data=[[1, 2, 3]])
                group = mat_file.create_group("metadata")
                group.attrs["source"] = "test"

            loaded = load_mat_file(
                mat_path,
                variable_names=("aligned_movie", "metadata"),
            )

            self.assertIn("aligned_movie", loaded.variables)
            self.assertIsInstance(loaded.variables["metadata"], MatlabHdf5Group)


if __name__ == "__main__":
    unittest.main()
