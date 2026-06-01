"""Import tests for the twopy package.

Inputs: the installed or source-tree twopy package.
Outputs: unittest assertions that confirm basic package metadata is available.
"""

import subprocess
import sys
import unittest
from importlib.metadata import version

import twopy


class ImportTest(unittest.TestCase):
    """Tests that the package can be imported by analysis code."""

    def test_version_is_available(self) -> None:
        """Confirm the package exposes a version string.

        Inputs: imported twopy module.
        Outputs: a passing assertion when the version matches package metadata.
        """
        self.assertEqual(twopy.__version__, version("twopy"))

    def test_top_level_import_leaves_gui_modules_lazy(self) -> None:
        """Confirm script imports do not import napari, Qt, or matplotlib.

        Inputs: a fresh Python process that imports ``twopy``.
        Outputs: a passing assertion when GUI modules are still absent.
        """
        script = (
            "import sys; import twopy; "
            "blocked = [name for name in sys.modules "
            "if name == 'twopy.napari' or name.startswith('twopy.napari.') "
            "or name == 'qtpy' or name.startswith('qtpy.') "
            "or name == 'napari' or name.startswith('napari.') "
            "or name == 'matplotlib' or name.startswith('matplotlib.')]; "
            "sys.stdout.write('\\n'.join(blocked))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
