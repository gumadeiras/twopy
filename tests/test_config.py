"""Tests for twopy YAML configuration loading.

Inputs: temporary YAML config files.
Outputs: assertions that valid paths load and invalid configs fail clearly.
"""

import tempfile
import unittest
from pathlib import Path

from twopy.config import load_config


class LoadConfigTest(unittest.TestCase):
    """Tests the machine-local config loader."""

    def test_loads_database_and_data_paths(self) -> None:
        """Confirm expected config keys become typed paths.

        Inputs: a temporary YAML file with ``database_path`` and ``data_path``.
        Outputs: assertions that both values load as ``Path`` objects.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                "database_path: /Volumes/magic/clarklab/_Database\n"
                "data_path: /Volumes/magic/clarklab/2p_microscope_data\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                config.database_path, Path("/Volumes/magic/clarklab/_Database")
            )
            self.assertEqual(
                config.data_path,
                Path("/Volumes/magic/clarklab/2p_microscope_data"),
            )

    def test_missing_required_path_raises_clear_error(self) -> None:
        """Confirm incomplete config files fail before analysis code runs.

        Inputs: a temporary YAML file missing ``data_path``.
        Outputs: an assertion that the error names the missing key.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                "database_path: /Volumes/magic/clarklab/_Database\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "data_path"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
