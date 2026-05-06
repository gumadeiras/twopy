"""Tests for package metadata that exposes twopy as an application.

Inputs: project metadata from ``pyproject.toml``.
Outputs: assertions that command-line entry points stay wired to the launcher.
"""

import tomllib
import unittest
from pathlib import Path


class PackagingTest(unittest.TestCase):
    """Tests package metadata that is easy to regress accidentally."""

    def test_twopy_console_script_launches_napari(self) -> None:
        """Confirm the ``twopy`` terminal command points at the launcher.

        Inputs: repository ``pyproject.toml``.
        Outputs: expected console-script target.
        """
        pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
        with pyproject_path.open("rb") as pyproject_file:
            pyproject = tomllib.load(pyproject_file)

        self.assertEqual(
            pyproject["project"]["scripts"]["twopy"],
            "twopy.napari:main",
        )


if __name__ == "__main__":
    unittest.main()
