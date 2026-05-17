"""Tests for twopy YAML configuration loading.

Inputs: temporary YAML config files.
Outputs: assertions that valid paths load and invalid configs fail clearly.
"""

import unittest
from pathlib import Path

from tests.tempdir import temporary_directory

from twopy.config import (
    DEFAULT_ANALYSIS_CACHE_DIR,
    load_config,
    resolve_analysis_cache_dir,
    resolve_analysis_output_dir,
    resolve_analysis_work_dir,
)
from twopy.pixel_calibration import DEFAULT_PIXEL_CALIBRATION_PATH


class LoadConfigTest(unittest.TestCase):
    """Tests the machine-local config loader."""

    def test_loads_database_and_data_paths(self) -> None:
        """Confirm expected config keys become typed paths.

        Inputs: a temporary YAML file with ``database_path`` and ``data_path``.
        Outputs: assertions that both values load as ``Path`` objects.
        """
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                "database_path: /Volumes/magic/clarklab/_Database\n"
                "data_path: /Volumes/magic/clarklab/2p_microscope_data\n"
                "database_access: copy\n"
                "analysis_output: source\n",
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
            self.assertEqual(config.database_access, "copy")
            self.assertEqual(config.analysis_output, "source")
            self.assertTrue(config.analysis_caching)
            self.assertEqual(config.analysis_cache_dir, DEFAULT_ANALYSIS_CACHE_DIR)
            self.assertEqual(
                config.pixel_calibration_path,
                DEFAULT_PIXEL_CALIBRATION_PATH,
            )

    def test_database_access_defaults_to_copy(self) -> None:
        """Confirm missing DB access uses fast local-copy mode.

        Inputs: a temporary YAML file without ``database_access``.
        Outputs: an assertion that the loaded mode is ``copy``.
        """
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                "database_path: /Volumes/magic/clarklab/_Database\n"
                "data_path: /Volumes/magic/clarklab/2p_microscope_data\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.database_access, "copy")
            self.assertTrue(config.analysis_caching)

    def test_loads_analysis_cache_settings(self) -> None:
        """Confirm optional analysis cache settings override defaults.

        Inputs: a temporary YAML file with explicit cache settings.
        Outputs: assertions that cache policy and directory are parsed.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yml"
            cache_dir = root / "cache"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {root / 'data'}\n"
                "database_access: copy\n"
                "analysis_output: source\n"
                "analysis_caching: false\n"
                f"analysis_cache_dir: {cache_dir}\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertFalse(config.analysis_caching)
            self.assertEqual(config.analysis_cache_dir, cache_dir)

    def test_loads_alternative_pixel_calibration_path(self) -> None:
        """Confirm config can point at an alternative calibration registry.

        Inputs: a temporary YAML file with ``pixel_calibration_path``.
        Outputs: loaded override path for scripts that need custom calibration.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yml"
            calibration_path = root / "calibrations" / "pixel_size.csv"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {root / 'data'}\n"
                "database_access: copy\n"
                "analysis_output: source\n"
                f"pixel_calibration_path: {calibration_path}\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.pixel_calibration_path, calibration_path)

    def test_invalid_analysis_caching_raises_clear_error(self) -> None:
        """Confirm analysis cache policy must be a YAML boolean.

        Inputs: a temporary YAML file with an invalid cache policy.
        Outputs: an assertion that the error names the invalid key.
        """
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                "database_path: /Volumes/magic/clarklab/_Database\n"
                "data_path: /Volumes/magic/clarklab/2p_microscope_data\n"
                "database_access: copy\n"
                "analysis_output: source\n"
                "analysis_caching: maybe\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "analysis_caching"):
                load_config(config_path)

    def test_invalid_database_access_raises_clear_error(self) -> None:
        """Confirm unsupported DB access modes fail during config loading.

        Inputs: a temporary YAML file with an invalid ``database_access``.
        Outputs: an assertion that the error names the invalid key.
        """
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                "database_path: /Volumes/magic/clarklab/_Database\n"
                "data_path: /Volumes/magic/clarklab/2p_microscope_data\n"
                "database_access: network\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "database_access"):
                load_config(config_path)

    def test_missing_analysis_output_raises_clear_error(self) -> None:
        """Confirm output routing is explicit in config.

        Inputs: a temporary YAML file without ``analysis_output``.
        Outputs: an assertion that the error names the missing key.
        """
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                "database_path: /Volumes/magic/clarklab/_Database\n"
                "data_path: /Volumes/magic/clarklab/2p_microscope_data\n"
                "database_access: copy\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "analysis_output"):
                load_config(config_path)

    def test_resolves_source_analysis_output_dir(self) -> None:
        """Confirm ``source`` writes inside the recording folder.

        Inputs: a config using ``analysis_output: source`` and a recording path.
        Outputs: the expected ``twopy`` folder path.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yml"
            recording_dir = root / "data" / "fly" / "stim" / "2023" / "10_17"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {root / 'data'}\n"
                "database_access: copy\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                resolve_analysis_output_dir(config, recording_dir),
                recording_dir / "twopy",
            )

    def test_resolves_mirrored_analysis_output_dir(self) -> None:
        """Confirm path output mirrors the recording structure under data root.

        Inputs: a config with an output root and a recording under ``data_path``.
        Outputs: the mirrored output directory.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "analysis"
            recording_dir = root / "data" / "fly" / "stim" / "2023" / "10_17"
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {root / 'data'}\n"
                "database_access: copy\n"
                f"analysis_output: {output_root}\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                resolve_analysis_output_dir(config, recording_dir),
                output_root / "fly" / "stim" / "2023" / "10_17",
            )

    def test_resolves_cached_analysis_work_dir(self) -> None:
        """Confirm enabled analysis caching uses the local cache for work.

        Inputs: a config with cache enabled and a recording under ``data_path``.
        Outputs: work directory under the configured local cache root.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            cache_root = root / "cache"
            recording_dir = data_root / "fly" / "stim" / "2023" / "10_17"
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {data_root}\n"
                "database_access: copy\n"
                "analysis_output: source\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {cache_root}\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            expected = cache_root / "fly" / "stim" / "2023" / "10_17"
            self.assertEqual(
                resolve_analysis_cache_dir(config, recording_dir), expected
            )
            self.assertEqual(resolve_analysis_work_dir(config, recording_dir), expected)

    def test_resolves_external_analysis_cache_dir(self) -> None:
        """Confirm cache routing accepts recordings outside ``data_path``.

        Inputs: a configured cache root and an external recording path.
        Outputs: stable external cache path under ``_external``.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            cache_root = root / "cache"
            external_recording = root / "external" / "10_17"
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                f"data_path: {root / 'data'}\n"
                "database_access: copy\n"
                "analysis_output: source\n"
                "analysis_caching: true\n"
                f"analysis_cache_dir: {cache_root}\n",
                encoding="utf-8",
            )

            config = load_config(config_path)
            cache_dir = resolve_analysis_cache_dir(config, external_recording)

            self.assertEqual(cache_dir.parent.parent, cache_root)
            self.assertEqual(cache_dir.parent.name, "_external")
            self.assertTrue(cache_dir.name.startswith("10_17-"))

    def test_missing_required_path_raises_clear_error(self) -> None:
        """Confirm incomplete config files fail before analysis code runs.

        Inputs: a temporary YAML file missing ``data_path``.
        Outputs: an assertion that the error names the missing key.
        """
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                "database_path: /Volumes/magic/clarklab/_Database\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "data_path"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
