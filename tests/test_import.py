"""Import tests for the twopy package.

Inputs: the installed or source-tree twopy package.
Outputs: unittest assertions that confirm basic package metadata is available.
"""

import unittest

import twopy


class ImportTest(unittest.TestCase):
    """Tests that the package can be imported by analysis code."""

    def test_version_is_available(self) -> None:
        """Confirm the package exposes a version string.

        Inputs: imported twopy module.
        Outputs: a passing assertion when the version matches package metadata.
        """
        self.assertEqual(twopy.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
