"""Tests for package metadata that exposes twopy as an application.

Inputs: project metadata from ``pyproject.toml``.
Outputs: assertions that command-line entry points stay wired to the launcher.
"""

import tomllib
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tests.tempdir import temporary_directory
from twopy.config import CONFIG_ENV_VAR, config_template_text
from twopy.napari.launcher import (
    _ensure_config_for_launch,
    _run_config_command,
    main,
    parse_launch_args,
)


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

    def test_repository_has_one_packaged_config_template(self) -> None:
        """Confirm the packaged config example is the canonical template."""
        repo_root = Path(__file__).parents[1]

        self.assertFalse((repo_root / "config.example.yml").exists())
        self.assertTrue((repo_root / "src" / "twopy" / "config.example.yml").is_file())
        self.assertIn("database_path:", config_template_text())

    def test_twopy_napari_icon_is_packaged(self) -> None:
        """Confirm the app icon used at launch is included in package data."""
        from twopy.napari.viewer import APPLICATION_ICON_SIZES

        repo_root = Path(__file__).parents[1]
        pyproject_path = repo_root / "pyproject.toml"
        with pyproject_path.open("rb") as pyproject_file:
            pyproject = tomllib.load(pyproject_file)

        asset_root = repo_root / "src" / "twopy" / "napari" / "assets"
        self.assertTrue((asset_root / "twopy-app-icon.png").is_file())
        self.assertTrue((asset_root / "twopy-app-icon.icns").is_file())
        for size in APPLICATION_ICON_SIZES:
            with self.subTest(size=size):
                self.assertTrue((asset_root / f"twopy-app-icon-{size}.png").is_file())
        package_data = pyproject["tool"]["setuptools"]["package-data"]["twopy"]
        self.assertIn(
            "napari/assets/*.png",
            package_data,
        )
        self.assertIn(
            "napari/assets/*.icns",
            package_data,
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

    def test_twopy_config_command_parses_setup(self) -> None:
        """Confirm ``twopy config setup`` reaches the config command path."""
        args = parse_launch_args(["config", "setup"])

        self.assertEqual(args.command, "config")
        self.assertEqual(args.config_command, "setup")
        self.assertFalse(args.force)

    def test_recording_path_still_parses_as_launcher_argument(self) -> None:
        """Confirm config command dispatch does not consume recording paths."""
        args = parse_launch_args(["/tmp/recording_data.h5"])

        self.assertIsNone(args.command)
        self.assertEqual(args.recording_data_path, Path("/tmp/recording_data.h5"))

    def test_twopy_config_prints_active_config(self) -> None:
        """Confirm ``twopy config`` validates and shows the selected file."""
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(_valid_config_text(Path(temp_dir)), encoding="utf-8")
            args = parse_launch_args(["config"])
            output = StringIO()

            with (
                patch.dict("os.environ", {CONFIG_ENV_VAR: str(config_path)}),
                redirect_stdout(output),
            ):
                _run_config_command(args)

            text = output.getvalue()
            self.assertIn(str(config_path), text)
            self.assertIn("status: valid", text)
            self.assertIn("database_path:", text)

    def test_twopy_config_exits_for_invalid_config(self) -> None:
        """Confirm ``twopy config`` fails loudly when the file is invalid."""
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                _config_missing_analysis_output_text(Path(temp_dir)),
                encoding="utf-8",
            )
            args = parse_launch_args(["config"])
            error_output = StringIO()

            with (
                patch.dict("os.environ", {CONFIG_ENV_VAR: str(config_path)}),
                redirect_stderr(error_output),
                self.assertRaises(SystemExit) as exit_context,
            ):
                _run_config_command(args)

            self.assertEqual(exit_context.exception.code, 2)
            self.assertIn("Invalid twopy config", error_output.getvalue())

    def test_twopy_config_reports_missing_config_setup_path(self) -> None:
        """Confirm ``twopy config`` gives setup guidance when missing."""
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "missing.yml"
            args = parse_launch_args(["config"])
            output = StringIO()

            with (
                patch.dict("os.environ", {CONFIG_ENV_VAR: str(config_path)}),
                redirect_stdout(output),
            ):
                _run_config_command(args)

            text = output.getvalue()
            self.assertIn("No twopy config file found.", text)
            self.assertIn(str(config_path), text)
            self.assertIn("twopy config setup", text)

    def test_first_launch_creates_config_template_and_stops(self) -> None:
        """Confirm first launch creates a template before opening napari."""
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            output = StringIO()

            with (
                patch.dict("os.environ", {CONFIG_ENV_VAR: str(config_path)}),
                redirect_stdout(output),
            ):
                should_launch = _ensure_config_for_launch()

            self.assertFalse(should_launch)
            self.assertTrue(config_path.is_file())
            self.assertIn(str(config_path), output.getvalue())

    def test_explicit_launch_path_still_creates_config_template_and_stops(self) -> None:
        """Confirm explicit data paths do not bypass first-launch setup."""
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            args = Namespace(
                command=None,
                recording_data_path=Path("/tmp/recording_data.h5"),
                roi_file_to_load=None,
                roi_save_file=None,
                movie_start=0,
                movie_end=None,
                no_movie=False,
            )
            output = StringIO()

            with (
                patch.dict("os.environ", {CONFIG_ENV_VAR: str(config_path)}),
                patch("twopy.napari.launcher.parse_launch_args", return_value=args),
                patch("twopy.napari.launcher.launch_napari") as launch,
                redirect_stdout(output),
            ):
                main()

            self.assertTrue(config_path.is_file())
            self.assertIn(str(config_path), output.getvalue())
            launch.assert_not_called()

    def test_launch_continues_with_valid_config(self) -> None:
        """Confirm launch validation accepts a complete config."""
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(_valid_config_text(Path(temp_dir)), encoding="utf-8")

            with patch.dict("os.environ", {CONFIG_ENV_VAR: str(config_path)}):
                self.assertTrue(_ensure_config_for_launch())

    def test_launch_exits_for_invalid_config(self) -> None:
        """Confirm launch stops before napari when config is invalid."""
        with temporary_directory() as temp_dir:
            config_path = Path(temp_dir) / "config.yml"
            config_path.write_text(
                _config_missing_analysis_output_text(Path(temp_dir)),
                encoding="utf-8",
            )
            error_output = StringIO()

            with (
                patch.dict("os.environ", {CONFIG_ENV_VAR: str(config_path)}),
                redirect_stderr(error_output),
                self.assertRaises(SystemExit) as exit_context,
            ):
                _ensure_config_for_launch()

            self.assertEqual(exit_context.exception.code, 2)
            self.assertIn("analysis_output", error_output.getvalue())


def _valid_config_text(root: Path) -> str:
    """Return a complete config file for launcher tests."""
    return (
        f"database_path: {root / 'db'}\n"
        "data_paths:\n"
        f"  - {root / 'data'}\n"
        "analysis_output: source\n"
    )


def _config_missing_analysis_output_text(root: Path) -> str:
    """Return a config that fails only because analysis output is missing."""
    return f"database_path: {root / 'db'}\ndata_paths:\n  - {root / 'data'}\n"


if __name__ == "__main__":
    unittest.main()
