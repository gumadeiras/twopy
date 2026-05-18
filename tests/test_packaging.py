"""Tests for package metadata that exposes twopy as an application.

Inputs: project metadata from ``pyproject.toml``.
Outputs: assertions that command-line entry points stay wired to the launcher.
"""

import tomllib
import unittest
from contextlib import redirect_stdout
from importlib.metadata import version
from io import StringIO
from pathlib import Path

from twopy.napari.launcher import parse_launch_args


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

    def test_twopy_cli_prints_version_aliases(self) -> None:
        """Confirm version flags report package metadata.

        Inputs: long and short launcher version flags.
        Outputs: version text and a clean parser exit for each alias.
        """
        for flag in ("--version", "-v"):
            with self.subTest(flag=flag):
                output = StringIO()
                with (
                    self.assertRaises(SystemExit) as exit_context,
                    redirect_stdout(output),
                ):
                    parse_launch_args([flag])

                self.assertEqual(exit_context.exception.code, 0)
                self.assertEqual(output.getvalue().strip(), f"twopy {version('twopy')}")


if __name__ == "__main__":
    unittest.main()
