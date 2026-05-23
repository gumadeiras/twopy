"""Tests for the developer function-inventory script.

Inputs: temporary source and test files with direct calls, method calls, and
docstrings.
Outputs: assertions that inventory metrics stay stable enough for audits.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import ModuleType


def load_inventory_module() -> ModuleType:
    """Load the inventory script as a test module.

    Args:
        None.

    Returns:
        Loaded ``scripts/function_inventory.py`` module.

    The script is not a package module, so tests import it from its repository
    path instead of adding a production dependency.
    """
    script_path = Path(__file__).parents[1] / "scripts" / "function_inventory.py"
    spec = importlib.util.spec_from_file_location("function_inventory", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load function_inventory script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FunctionInventoryTest(unittest.TestCase):
    """Tests inventory counts used for refactor planning."""

    def test_inventory_counts_code_docs_uses_and_direct_tests(self) -> None:
        """Confirm function rows include size, fan-in, and direct test counts.

        Inputs: one temporary package function, one class method, and one test.
        Outputs: resolved counts for code lines, docstrings, calls, and tests.
        """
        inventory = load_inventory_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "src" / "twopy"
            test_dir = root / "tests"
            source_dir.mkdir(parents=True)
            test_dir.mkdir()
            (source_dir / "__init__.py").write_text(
                textwrap.dedent(
                    '''
                    """Sample public package."""

                    from twopy.sample import add_one

                    __all__ = ["add_one"]
                    '''
                ),
                encoding="utf-8",
            )
            (source_dir / "sample.py").write_text(
                textwrap.dedent(
                    '''
                    """Sample package."""

                    __all__ = ["add_one", "Counter"]

                    def add_one(value: int) -> int:
                        """Add one.

                        Used by scripts.
                        """
                        return value + 1

                    class Counter:
                        """Small counter."""

                        def _private_factory(self) -> object:
                            """Return an object with an internal helper."""

                            def callback() -> None:
                                return None

                            return callback

                        def bump(self, value: int) -> int:
                            """Increase a value."""
                            return add_one(value)

                        def twice(self, value: int) -> int:
                            """Call through self."""
                            return self.bump(self.bump(value))
                    '''
                ),
                encoding="utf-8",
            )
            (test_dir / "test_sample.py").write_text(
                textwrap.dedent(
                    '''
                    """Sample tests."""

                    import unittest

                    from twopy import add_one
                    from twopy.sample import Counter


                    class SampleTest(unittest.TestCase):
                        """Tests sample helpers."""

                        def test_add_one(self) -> None:
                            """Check direct function use."""
                            self.assertEqual(add_one(1), 2)

                        def test_counter(self) -> None:
                            """Check direct method use."""
                            self.assertEqual(Counter().twice(1), 3)
                    '''
                ),
                encoding="utf-8",
            )
            (test_dir / "helpers.py").write_text(
                textwrap.dedent(
                    '''
                    """Shared test helper re-exports."""

                    from twopy import add_one
                    '''
                ),
                encoding="utf-8",
            )
            (test_dir / "test_reexport.py").write_text(
                textwrap.dedent(
                    '''
                    """Tests that call a public function through test helpers."""

                    import unittest

                    from tests.helpers import add_one


                    class ReexportTest(unittest.TestCase):
                        """Tests helper-mediated function calls."""

                        def test_helper_reexport(self) -> None:
                            """Check direct use through test support."""
                            self.assertEqual(add_one(2), 3)
                    '''
                ),
                encoding="utf-8",
            )

            metrics = inventory.build_inventory(
                root=root,
                source_dirs=(source_dir,),
                test_dir=test_dir,
            )
            rows = {
                row["function"]: row
                for row in csv.DictReader(
                    inventory.render_csv_for_tests(metrics).splitlines(),
                )
            }

        add_one = rows["twopy.sample.add_one"]
        self.assertEqual(add_one["code_lines"], "2")
        self.assertEqual(add_one["docstring_lines"], "2")
        self.assertEqual(add_one["direct_call_site_count"], "3")
        self.assertEqual(add_one["direct_test_function_count"], "2")
        self.assertEqual(add_one["api_surface"], "exported_api")
        self.assertEqual(add_one["domain"], "core")

        bump = rows["twopy.sample.Counter.bump"]
        self.assertEqual(bump["kind"], "method")
        self.assertEqual(bump["direct_call_site_count"], "1")
        self.assertEqual(bump["direct_test_function_count"], "0")
        self.assertEqual(bump["api_surface"], "exported_api")

        twice = rows["twopy.sample.Counter.twice"]
        self.assertEqual(twice["direct_test_function_count"], "1")

        callback = rows["twopy.sample.Counter._private_factory.callback"]
        self.assertEqual(callback["api_surface"], "private")


if __name__ == "__main__":
    unittest.main()
