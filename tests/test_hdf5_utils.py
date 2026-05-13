"""Tests for shared HDF5 attribute and string helpers.

Inputs: temporary HDF5 files with string datasets and mixed attribute values.
Outputs: assertions that helpers preserve twopy's persistence-boundary
semantics.
"""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from twopy.hdf5_utils import (
    decode_hdf5_string,
    plain_hdf5_attr_value,
    read_string_dataset,
    write_string_dataset,
)


class Hdf5UtilsTest(unittest.TestCase):
    """Tests UTF-8 string datasets and plain HDF5 attributes."""

    def test_round_trips_utf8_string_dataset(self) -> None:
        """Confirm string datasets round-trip as Python strings.

        Inputs: a temporary HDF5 file and text label values.
        Outputs: decoded Python strings in original order.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "strings.h5"
            with h5py.File(path, "w") as h5_file:
                write_string_dataset(h5_file, "labels", ("roi_1", "cafe"))

            with h5py.File(path, "r") as h5_file:
                self.assertEqual(
                    read_string_dataset(h5_file, "labels"),
                    ("roi_1", "cafe"),
                )

    def test_decodes_bytes_and_numpy_string_scalars(self) -> None:
        """Confirm HDF5 string scalar variants decode consistently.

        Inputs: bytes, NumPy bytes, and plain text values.
        Outputs: Python ``str`` values.
        """
        self.assertEqual(decode_hdf5_string(b"roi_1"), "roi_1")
        self.assertEqual(decode_hdf5_string(np.bytes_(b"roi_2")), "roi_2")
        self.assertEqual(decode_hdf5_string("roi_3"), "roi_3")

    def test_plain_attr_value_preserves_arrays_when_allowed(self) -> None:
        """Confirm converted-recording metadata can keep array attributes.

        Inputs: NumPy scalar, byte string, and ndarray attribute values.
        Outputs: plain scalars and list values.
        """
        self.assertEqual(plain_hdf5_attr_value(np.int64(3)), 3)
        self.assertEqual(plain_hdf5_attr_value(b"twopy"), "twopy")
        self.assertEqual(
            plain_hdf5_attr_value(np.array([1, 2]), allow_array=True),
            [1, 2],
        )

    def test_plain_attr_value_scalar_only_coerces_unknowns_to_text(self) -> None:
        """Confirm analysis metadata reads unsupported attrs as strings.

        Inputs: array and object values at scalar-only persistence boundaries.
        Outputs: plain scalar-compatible values.
        """
        self.assertEqual(
            plain_hdf5_attr_value(np.array([1, 2]), scalar_only=True),
            "[1 2]",
        )
        self.assertEqual(
            plain_hdf5_attr_value({"method": "manual"}, scalar_only=True),
            "{'method': 'manual'}",
        )


if __name__ == "__main__":
    unittest.main()
