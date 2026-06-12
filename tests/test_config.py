"""Tests for twopy YAML configuration loading.

Inputs: temporary YAML config files.
Outputs: assertions that valid paths load and invalid configs fail clearly.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

from tests.tempdir import temporary_directory
from twopy.config import (
    CONFIG_ENV_VAR,
    DEFAULT_ANALYSIS_CACHE_DIR,
    config_template_text,
    data_path_match,
    data_path_recording_candidates,
    load_config,
    preferred_config_path,
    resolve_analysis_cache_dir,
    resolve_analysis_output_dir,
    resolve_analysis_work_dir,
    resolve_config_path,
    resolve_data_recording_path,
    write_config_template,
)
from twopy.pixel_calibration import DEFAULT_PIXEL_CALIBRATION_PATH


class LoadConfigTest(unittest.TestCase):
    """Tests the config loader for this machine."""

    def test_loads_database_and_data_paths(self) -> None:
        """Confirm expected config keys become typed paths.

        Inputs: a temporary YAML file with ``database_path`` and ``data_paths``.
        Outputs: assertions that both values load as ``Path`` objects.
        """
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                "database_path: /Volumes/magic/clarklab/_Database\n"
                "data_paths:\n"
                "  - /Volumes/magic/clarklab/2p_microscope_data\n"
                "database_access: copy\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                config.database_path, Path("/Volumes/magic/clarklab/_Database")
            )
            self.assertEqual(
                config.data_paths,
                (Path("/Volumes/magic/clarklab/2p_microscope_data"),),
            )
            self.assertEqual(config.database_access, "copy")
            self.assertEqual(config.analysis_output, "source")
            self.assertTrue(config.analysis_caching)
            self.assertEqual(config.analysis_cache_dir, DEFAULT_ANALYSIS_CACHE_DIR)
            self.assertEqual(
                config.pixel_calibration_path,
                DEFAULT_PIXEL_CALIBRATION_PATH,
            )
            self.assertEqual(config.custom_workflow_paths, ())

    def test_load_config_uses_env_path_for_default_search(self) -> None:
        """Confirm ``TWOPY_CONFIG`` points default loading at one file.

        Inputs: an environment override and a valid temporary config file.
        Outputs: loaded paths from the override file.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "env-config.yml"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                "data_paths:\n"
                f"  - {root / 'data'}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {CONFIG_ENV_VAR: str(config_path)}):
                self.assertEqual(resolve_config_path(), config_path)
                config = load_config()

            self.assertEqual(config.database_path, root / "db")
            self.assertEqual(config.data_paths, (root / "data",))

    def test_write_config_template_creates_editable_yaml(self) -> None:
        """Confirm setup writes the packaged example config.

        Inputs: a temporary destination path.
        Outputs: a parseable config template at that path.
        """
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"

            written_path = write_config_template(config_path)

            self.assertEqual(written_path, config_path)
            self.assertEqual(
                config_path.read_text(encoding="utf-8"),
                config_template_text(),
            )
            config = load_config(config_path)
            self.assertEqual(config.analysis_output, "source")

    def test_write_config_template_refuses_existing_file(self) -> None:
        """Confirm setup does not overwrite config files by default."""
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text("database_path: /tmp/db\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_config_template(config_path)

    def test_preferred_config_path_uses_env_path(self) -> None:
        """Confirm setup honors an explicit config destination."""
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "custom.yml"

            with patch.dict("os.environ", {CONFIG_ENV_VAR: str(config_path)}):
                self.assertEqual(preferred_config_path(), config_path)

    def test_missing_env_config_reports_env_path(self) -> None:
        """Confirm a missing environment override fails loudly."""
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "missing.yml"

            with (
                patch.dict("os.environ", {CONFIG_ENV_VAR: str(config_path)}),
                self.assertRaisesRegex(FileNotFoundError, CONFIG_ENV_VAR),
            ):
                load_config()

    def test_default_search_prefers_local_config_over_user_config(self) -> None:
        """Confirm source checkout config wins over the user config file."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            local_path = root / "local.yml"
            user_path = root / "user.yml"
            local_path.write_text("database_path: /local/db\n", encoding="utf-8")
            user_path.write_text("database_path: /user/db\n", encoding="utf-8")

            with (
                patch.dict("os.environ", {}, clear=True),
                patch("twopy.config.DEFAULT_CONFIG_PATH", local_path),
                patch("twopy.config.user_config_path", return_value=user_path),
            ):
                self.assertEqual(resolve_config_path(), local_path)

    def test_default_search_uses_user_config_when_local_is_missing(self) -> None:
        """Confirm installed users get the user config path."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            local_path = root / "missing-local.yml"
            user_path = root / "user.yml"
            user_path.write_text("database_path: /user/db\n", encoding="utf-8")

            with (
                patch.dict("os.environ", {}, clear=True),
                patch("twopy.config.DEFAULT_CONFIG_PATH", local_path),
                patch("twopy.config.user_config_path", return_value=user_path),
            ):
                self.assertEqual(resolve_config_path(), user_path)

    def test_database_access_defaults_to_copy(self) -> None:
        """Confirm missing DB access uses fast local-copy mode.

        Inputs: a temporary YAML file without ``database_access``.
        Outputs: an assertion that the loaded mode is ``copy``.
        """
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                "database_path: /Volumes/magic/clarklab/_Database\n"
                "data_paths:\n"
                "  - /Volumes/magic/clarklab/2p_microscope_data\n"
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
                "data_paths:\n"
                f"  - {root / 'data'}\n"
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
                "data_paths:\n"
                f"  - {root / 'data'}\n"
                "database_access: copy\n"
                "analysis_output: source\n"
                f"pixel_calibration_path: {calibration_path}\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.pixel_calibration_path, calibration_path)

    def test_loads_custom_workflow_paths(self) -> None:
        """Confirm config can list local custom workflow files or folders."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            first_path = root / "workflows"
            second_path = root / "dsi.py"
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                "data_paths:\n"
                f"  - {root / 'data'}\n"
                "database_access: copy\n"
                "analysis_output: source\n"
                "custom_workflow_paths:\n"
                f"  - {first_path}\n"
                f"  - {second_path}\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.custom_workflow_paths, (first_path, second_path))

    def test_invalid_custom_workflow_paths_raise_clear_error(self) -> None:
        """Confirm custom workflow paths must be listed as strings."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                "data_paths:\n"
                f"  - {root / 'data'}\n"
                "database_access: copy\n"
                "analysis_output: source\n"
                "custom_workflow_paths:\n"
                "  - 5\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "custom_workflow_paths"):
                load_config(config_path)

    def test_invalid_analysis_caching_raises_clear_error(self) -> None:
        """Confirm analysis cache policy must be a YAML boolean.

        Inputs: a temporary YAML file with an invalid cache policy.
        Outputs: an assertion that the error names the invalid key.
        """
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                "database_path: /Volumes/magic/clarklab/_Database\n"
                "data_paths:\n"
                "  - /Volumes/magic/clarklab/2p_microscope_data\n"
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
                "data_paths:\n"
                "  - /Volumes/magic/clarklab/2p_microscope_data\n"
                "database_access: network\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "database_access"):
                load_config(config_path)

    def test_missing_analysis_output_raises_clear_error(self) -> None:
        """Confirm the output path is explicit in config.

        Inputs: a temporary YAML file without ``analysis_output``.
        Outputs: an assertion that the error names the missing key.
        """
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                "database_path: /Volumes/magic/clarklab/_Database\n"
                "data_paths:\n"
                "  - /Volumes/magic/clarklab/2p_microscope_data\n"
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
                "data_paths:\n"
                f"  - {root / 'data'}\n"
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

        Inputs: a config with an output root and a recording under ``data_paths``.
        Outputs: the mirrored output directory.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "analysis"
            recording_dir = root / "data" / "fly" / "stim" / "2023" / "10_17"
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                "data_paths:\n"
                f"  - {root / 'data'}\n"
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

        Inputs: a config with cache enabled and a recording under ``data_paths``.
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
                "data_paths:\n"
                f"  - {data_root}\n"
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
        """Confirm cache paths support recordings outside ``data_paths``.

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
                "data_paths:\n"
                f"  - {root / 'data'}\n"
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

        Inputs: a temporary YAML file missing ``data_paths``.
        Outputs: an assertion that the error names the missing key.
        """
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                "database_path: /Volumes/magic/clarklab/_Database\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "data_paths"):
                load_config(config_path)

    def test_resolves_database_recording_from_first_available_data_path(self) -> None:
        """Confirm ordered data roots select the first existing recording copy."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            first_root = root / "offline"
            second_root = root / "mounted"
            relative_parts = ("fly", "stim", "2023", "10_17", "10_02_49")
            (second_root.joinpath(*relative_parts)).mkdir(parents=True)
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                "data_paths:\n"
                f"  - {first_root}\n"
                f"  - {second_root}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                resolve_data_recording_path(config, relative_parts),
                second_root.joinpath(*relative_parts),
            )

    def test_missing_database_recording_uses_first_data_path_candidate(self) -> None:
        """Confirm missing recordings preserve a concrete first-root path."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            first_root = root / "offline"
            second_root = root / "mounted"
            relative_parts = ("fly", "stim", "2023", "10_17", "10_02_49")
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                "data_paths:\n"
                f"  - {first_root}\n"
                f"  - {second_root}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                resolve_data_recording_path(config, relative_parts),
                first_root.joinpath(*relative_parts),
            )

    def test_data_path_recording_candidates_lists_mirrored_roots(self) -> None:
        """Confirm diagnostics can show every configured recording mirror."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            first_root = root / "data"
            second_root = root / "archive"
            recording_dir = first_root / "fly" / "stim" / "2023" / "10_17"
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                "data_paths:\n"
                f"  - {first_root}\n"
                f"  - {second_root}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                data_path_recording_candidates(config, recording_dir),
                (
                    first_root / "fly" / "stim" / "2023" / "10_17",
                    second_root / "fly" / "stim" / "2023" / "10_17",
                ),
            )

    def test_data_path_match_uses_first_matching_root(self) -> None:
        """Confirm mirrored outputs use the earliest configured matching root."""
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            first_root = root / "data"
            second_root = first_root / "nested"
            recording_dir = second_root / "fly" / "stim" / "2023" / "10_17"
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {root / 'db'}\n"
                "data_paths:\n"
                f"  - {first_root}\n"
                f"  - {second_root}\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                data_path_match(config, recording_dir),
                (first_root, Path("nested") / "fly" / "stim" / "2023" / "10_17"),
            )


if __name__ == "__main__":
    unittest.main()
