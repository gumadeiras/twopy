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

    def test_top_level_all_exports_resolve(self) -> None:
        """Confirm every public top-level name resolves at runtime.

        Inputs: imported twopy module.
        Outputs: a passing assertion when every ``__all__`` name is importable.
        """
        missing = [name for name in twopy.__all__ if not hasattr(twopy, name)]

        self.assertEqual(missing, [])

    def test_top_level_lazy_exports_are_listed_by_dir(self) -> None:
        """Confirm introspection can see lazy top-level names before access.

        Inputs: imported twopy module.
        Outputs: a passing assertion when public lazy names appear in ``dir``.
        """
        names = dir(twopy)

        self.assertIn("__version__", names)
        self.assertIn("make_roi_set", names)

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

    def test_plotting_package_import_leaves_dock_widget_lazy(self) -> None:
        """Confirm plotting package import does not import heavy dock modules.

        Inputs: a fresh Python process that imports ``twopy.napari.plotting``.
        Outputs: a passing assertion when figure export, Qt, and dock widget
        modules are still absent.
        """
        script = (
            "import sys; import twopy.napari.plotting; "
            "blocked = [name for name in sys.modules "
            "if name == 'twopy.napari.plotting.docks.response_plot_widget' "
            "or name.startswith('matplotlib') "
            "or name == 'qtpy' or name.startswith('qtpy.')]; "
            "sys.stdout.write('\\n'.join(blocked))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout, "")

    def test_plotting_barrel_exports_are_available_without_widget_import(self) -> None:
        """Confirm plotting dock helpers remain importable from package barrels.

        Inputs: a fresh Python process that imports response dock helper names.
        Outputs: a passing assertion when helper imports succeed without loading
        the concrete response widget.
        """
        script = (
            "import sys; "
            "from twopy.napari.plotting import add_twopy_response_plot_widget; "
            "from twopy.napari.plotting.docks import create_response_plot_widget; "
            "assert callable(add_twopy_response_plot_widget); "
            "assert callable(create_response_plot_widget); "
            "blocked = [name for name in sys.modules "
            "if name == 'twopy.napari.plotting.docks.response_plot_widget' "
            "or name.startswith('matplotlib') "
            "or name == 'qtpy' or name.startswith('qtpy.')]; "
            "sys.stdout.write('\\n'.join(blocked))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout, "")

    def test_napari_barrel_exports_remain_available(self) -> None:
        """Confirm napari package exports still resolve through lazy access.

        Inputs: a fresh Python process that imports public napari helpers.
        Outputs: a passing assertion when the exported helper is callable.
        """
        script = (
            "from twopy.napari import open_recording_in_napari; "
            "assert callable(open_recording_in_napari)"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
