"""Tests for the public script-friendly twopy API.

Inputs: temporary config and SQLite database files.
Outputs: assertions that scripts can call one simple function to find recordings.
"""

import sqlite3
import unittest
from pathlib import Path

from tests.tempdir import temporary_directory

from twopy import find_recordings


class PublicApiTest(unittest.TestCase):
    """Tests script-level twopy entrypoints."""

    def test_find_recordings_uses_configured_database_path(self) -> None:
        """Confirm the public API loads config and applies filters.

        Inputs: a temporary config file pointing at a temporary database folder.
        Outputs: an assertion that a filtered recording is returned.
        """
        with temporary_directory() as temp_dir:
            root = Path(temp_dir)
            database_dir = root / "db"
            data_dir = root / "data"
            database_dir.mkdir()
            data_dir.mkdir()
            self._write_database(database_dir / "experimentLog.db")
            config_path = root / "config.yml"
            config_path.write_text(
                f"database_path: {database_dir}\n"
                "data_paths:\n"
                f"  - {data_dir}\n"
                "database_access: copy\n"
                "analysis_output: source\n",
                encoding="utf-8",
            )

            recordings = find_recordings(
                year=2023,
                month=10,
                day=17,
                genotype="gh146",
                stimulus="combo",
                sensor="g6f",
                cell_type="ALPN",
                hemisphere="left",
                person="Gustavo",
                database_cache_dir=root / "db_cache",
                config_path=config_path,
            )

            self.assertEqual(len(recordings), 1)
            self.assertEqual(recordings[0].stimulus_presentation_id, 20005)
            self.assertTrue((root / "db_cache" / "experimentLog.db").is_file())

    def _write_database(self, path: Path) -> None:
        """Create the smallest SQLite DB needed by the public API test.

        Args:
            path: SQLite database path to create.

        Returns:
            None. The function writes one fly and one stimulus row.
        """
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                """
                CREATE TABLE fly (
                    flyId INTEGER PRIMARY KEY,
                    genotype TEXT,
                    cellType TEXT,
                    fluorescentProtein TEXT,
                    eye TEXT,
                    surgeon TEXT
                )
                """,
            )
            connection.execute(
                """
                CREATE TABLE stimulusPresentation (
                    stimulusPresentationId INTEGER PRIMARY KEY,
                    fly INT,
                    relativeDataPath TEXT,
                    stimulusFunction TEXT,
                    date TEXT,
                    dataQuality INT
                )
                """,
            )
            connection.execute(
                """
                INSERT INTO fly
                    (flyId, genotype, cellType, fluorescentProtein, eye, surgeon)
                VALUES
                    (10923, 'gh146gal4', 'ALPN', 'g6f', 'left', 'Gustavo')
                """,
            )
            connection.execute(
                """
                INSERT INTO stimulusPresentation
                    (
                        stimulusPresentationId,
                        fly,
                        relativeDataPath,
                        stimulusFunction,
                        date,
                        dataQuality
                    )
                VALUES
                    (
                        20005,
                        10923,
                        'genotype\\stimulus\\2023\\10_17\\10_02_49',
                        'combo_stim_singles',
                        '2023-10-17 10:03:14',
                        1
                    )
                """,
            )
            connection.commit()
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
