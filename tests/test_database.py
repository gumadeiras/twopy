"""Tests for schema-agnostic experiment database queries.

Inputs: temporary SQLite database files shaped like lab DB files.
Outputs: assertions that table discovery and filtering behave predictably.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from twopy.config import TwopyConfig
from twopy.database import (
    ExperimentSearchFilters,
    build_experiment_search_tree,
    find_experiments,
    find_recording_search_results,
    find_recordings,
    find_stimulus_presentations,
    normalize_experiment_date_filter,
    read_database_catalog,
    recording_path_for_database_experiment,
    recording_paths_for_database_experiments,
)


class DatabaseQueryTest(unittest.TestCase):
    """Tests the Clark lab database query layer."""

    def test_catalog_finds_known_database_files(self) -> None:
        """Confirm only known SQLite files are treated as databases.

        Inputs: a temporary database directory with ``experimentLog.db``.
        Outputs: an assertion that the catalog reports that DB file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            database_dir = Path(temp_dir)
            self._write_experiment_log(database_dir / "experimentLog.db")
            (database_dir / "._experimentLog.db").write_text("resource fork")

            catalog = read_database_catalog(database_dir)

            self.assertEqual(catalog.sqlite_files, (database_dir / "experimentLog.db",))

    def test_find_experiments_with_exact_and_contains_filters(self) -> None:
        """Confirm named column filters return matching experiment rows.

        Inputs: a temporary experiment log table with two rows.
        Outputs: an assertion that exact and substring filters find one row.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            database_dir = Path(temp_dir)
            self._write_experiment_log(database_dir / "experimentLog.db")

            records = find_experiments(
                database_dir,
                table_names=("experiments",),
                exact_filters={"genotype": "gh146"},
                contains_filters={"stimulus": "combo"},
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].values["recording_path"], "2023/10_17/10_02_49")

    def test_find_experiments_searches_text_across_columns(self) -> None:
        """Confirm broad search can find text without naming a column.

        Inputs: a temporary experiment log table.
        Outputs: an assertion that search text matches the stimulus column.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            database_dir = Path(temp_dir)
            self._write_experiment_log(database_dir / "experimentLog.db")

            records = find_experiments(database_dir, search_text="odor")

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].values["genotype"], "orco")

    def test_unknown_filter_column_raises_clear_error(self) -> None:
        """Confirm invalid filters fail before SQL execution.

        Inputs: a temporary experiment log table and an unknown column filter.
        Outputs: an assertion that the error names the missing column.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            database_dir = Path(temp_dir)
            self._write_experiment_log(database_dir / "experimentLog.db")

            with self.assertRaisesRegex(ValueError, "missing_column"):
                find_experiments(
                    database_dir,
                    table_names=("experiments",),
                    exact_filters={"missing_column": "value"},
                )

    def test_find_stimulus_presentations_joins_fly_metadata(self) -> None:
        """Confirm modeled experiment search joins stimulus and fly tables.

        Inputs: a temporary DB with Clark-lab-style stimulus and fly tables.
        Outputs: an assertion that the returned experiment includes fly metadata.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            database_dir = Path(temp_dir)
            self._write_clark_style_log(database_dir / "experimentLog.db")

            experiments = find_stimulus_presentations(
                database_dir,
                path_contains="10_02_49",
                genotype_contains="gh146",
            )

            self.assertEqual(len(experiments), 1)
            self.assertEqual(experiments[0].stimulus_presentation_id, 20005)
            self.assertEqual(experiments[0].fly_id, 10923)
            self.assertEqual(experiments[0].genotype, "gh146gal4")
            self.assertEqual(experiments[0].hemisphere, "left")
            self.assertEqual(experiments[0].person, "Gustavo")

    def test_find_stimulus_presentations_filters_research_fields(self) -> None:
        """Confirm modeled search supports user-facing experiment filters.

        Inputs: a temporary Clark-lab-style DB row.
        Outputs: an assertion that date, stimulus, hemisphere, and person match.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            database_dir = Path(temp_dir)
            self._write_clark_style_log(database_dir / "experimentLog.db")

            experiments = find_stimulus_presentations(
                database_dir,
                date_year=2023,
                date_month=10,
                date_day=17,
                stimulus_contains="combo",
                sensor_contains="g6f",
                cell_type_contains="ALPN",
                hemisphere="left",
                person_contains="Gus",
            )

            self.assertEqual(len(experiments), 1)
            self.assertEqual(
                experiments[0].relative_data_path,
                "genotype\\stimulus\\2023\\10_17\\10_02_49",
            )

    def test_find_stimulus_presentations_filters_date_text(self) -> None:
        """Confirm database search can filter by user-entered date text.

        Inputs: a temporary Clark-lab-style DB row.
        Outputs: date substring filters include matching rows and exclude
        non-matching rows.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            database_dir = Path(temp_dir)
            self._write_clark_style_log(database_dir / "experimentLog.db")

            experiments = find_stimulus_presentations(
                database_dir,
                date_contains="2023-10-17",
            )
            missing = find_stimulus_presentations(
                database_dir,
                date_contains="2023-10-18",
            )

            self.assertEqual(len(experiments), 1)
            self.assertEqual(missing, ())

    def test_recording_search_normalizes_date_separator_variants(self) -> None:
        """Confirm GUI-style date filters accept common separator variants.

        Inputs: a temporary Clark-lab-style DB and date-like filter strings.
        Outputs: each ``YYYY<separator>MM<separator>DD`` variant finds the same
        recording as the canonical ``YYYY-MM-DD`` filter.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_dir = root / "db"
            data_dir = root / "data"
            database_dir.mkdir()
            data_dir.mkdir()
            self._write_clark_style_log(database_dir / "experimentLog.db")
            config = TwopyConfig(
                database_path=database_dir,
                data_path=data_dir,
                database_access="direct",
                analysis_output="source",
            )

            for value in (
                "2023-10-17",
                "2023/10/17",
                "2023\\10\\17",
                "2023.10.17",
                "2023_10_17",
            ):
                with self.subTest(value=value):
                    self.assertEqual(
                        normalize_experiment_date_filter(value),
                        "2023-10-17",
                    )
                    experiments = find_recording_search_results(
                        config,
                        ExperimentSearchFilters(date=value),
                    )

                    self.assertEqual(len(experiments), 1)
                    self.assertEqual(experiments[0].stimulus_presentation_id, 20005)

    def test_recording_search_tree_groups_loadable_hierarchy(self) -> None:
        """Confirm search results group by the Load-tab browse hierarchy.

        Inputs: a configured temporary DB and data root.
        Outputs: user, cell type, sensor, stimulus, date, and time nodes that
        all retain the experiments beneath them.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_dir = root / "db"
            data_dir = root / "data"
            database_dir.mkdir()
            data_dir.mkdir()
            self._write_clark_style_log(database_dir / "experimentLog.db")
            self._write_clark_style_log(database_dir / "experimentInitLog.db")
            config = TwopyConfig(
                database_path=database_dir,
                data_path=data_dir,
                database_access="direct",
                analysis_output="source",
            )

            experiments = find_recording_search_results(
                config,
                ExperimentSearchFilters(user="Gus", date="2023-10-17"),
            )
            tree = build_experiment_search_tree(experiments)

            self.assertEqual(len(experiments), 1)
            user = tree.children[0]
            cell_type = user.children[0]
            sensor = cell_type.children[0]
            stimulus = sensor.children[0]
            date = stimulus.children[0]
            time = date.children[0]
            self.assertEqual(user.label, "Gustavo")
            self.assertEqual(cell_type.label, "ALPN")
            self.assertEqual(sensor.label, "g6f")
            self.assertEqual(stimulus.label, "combo_stim_singles")
            self.assertEqual(date.label, "2023-10-17")
            self.assertEqual(time.label, "10:02:49")
            self.assertEqual(time.experiments, experiments)
            self.assertEqual(user.experiments, experiments)
            self.assertEqual(
                recording_path_for_database_experiment(config, experiments[0]),
                data_dir / "genotype" / "stimulus" / "2023" / "10_17" / "10_02_49",
            )
            self.assertEqual(
                recording_paths_for_database_experiments(config, experiments),
                (data_dir / "genotype" / "stimulus" / "2023" / "10_17" / "10_02_49",),
            )

    def test_recording_search_uses_enough_rows_for_hierarchy_counts(self) -> None:
        """Confirm GUI search counts are not distorted by early query limits.

        Inputs: a temporary DB with more than 1000 Gustavo recordings and one
        later LN6 recording.
        Outputs: the search tree includes the later LN6 result instead of
        silently stopping at the first 1000 rows.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_dir = root / "db"
            data_dir = root / "data"
            database_dir.mkdir()
            data_dir.mkdir()
            self._write_large_clark_style_log(database_dir / "experimentLog.db")
            config = TwopyConfig(
                database_path=database_dir,
                data_path=data_dir,
                database_access="direct",
                analysis_output="source",
            )

            experiments = find_recording_search_results(
                config,
                ExperimentSearchFilters(user="Gus"),
            )
            tree = build_experiment_search_tree(experiments)
            user = tree.children[0]
            counts = {child.label: len(child.experiments) for child in user.children}

            self.assertEqual(len(experiments), 1002)
            self.assertEqual(counts["ALPN"], 1001)
            self.assertEqual(counts["LN6"], 1)

    def test_invalid_month_filter_raises_clear_error(self) -> None:
        """Confirm impossible month filters fail before SQL execution.

        Inputs: a temporary Clark-lab-style DB and month 13.
        Outputs: an assertion that range validation names the month filter.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            database_dir = Path(temp_dir)
            self._write_clark_style_log(database_dir / "experimentLog.db")

            with self.assertRaisesRegex(ValueError, "month"):
                find_stimulus_presentations(database_dir, date_month=13)

    def test_find_recordings_can_query_local_copies(self) -> None:
        """Confirm copied database access reuses a local DB cache.

        Inputs: a temporary Clark-lab-style DB and cache directory.
        Outputs: assertions that querying through a cache finds the experiment
        and writes a manifest.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_dir = root / "db"
            cache_dir = root / "cache"
            database_dir.mkdir()
            self._write_clark_style_log(database_dir / "experimentLog.db")

            experiments = find_recordings(
                database_dir,
                genotype="gh146",
                sensor="g6f",
                cell_type="ALPN",
                database_access="copy",
                cache_dir=cache_dir,
            )

            self.assertEqual(len(experiments), 1)
            self.assertTrue((cache_dir / "experimentLog.db").is_file())
            self.assertTrue((cache_dir / "experimentLog.db.manifest.json").is_file())

    def _write_experiment_log(self, path: Path) -> None:
        """Create a small SQLite experiment log fixture.

        Args:
            path: SQLite file path to create.

        Returns:
            None. The function writes a tiny table to disk.

        The fixture uses plausible experiment fields without claiming that the
        real lab schema has these exact column names.
        """
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                """
                CREATE TABLE experiments (
                    id INTEGER PRIMARY KEY,
                    genotype TEXT NOT NULL,
                    stimulus TEXT NOT NULL,
                    recording_path TEXT NOT NULL
                )
                """,
            )
            connection.executemany(
                """
                INSERT INTO experiments (genotype, stimulus, recording_path)
                VALUES (?, ?, ?)
                """,
                [
                    ("gh146", "combo_stim_singles", "2023/10_17/10_02_49"),
                    ("orco", "odor_panel", "2023/10_18/11_00_00"),
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def _write_clark_style_log(self, path: Path) -> None:
        """Create a small DB using observed Clark lab table names.

        Args:
            path: SQLite file path to create.

        Returns:
            None. The function writes stimulus and fly tables to disk.

        This fixture matches only the columns twopy currently queries.
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

    def _write_large_clark_style_log(self, path: Path) -> None:
        """Create a Clark-style DB with enough rows to exercise search limits.

        Args:
            path: SQLite database path to create.

        Returns:
            None. The function writes 1001 ALPN rows followed by one LN6 row.
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
            connection.executemany(
                """
                INSERT INTO fly
                    (flyId, genotype, cellType, fluorescentProtein, eye, surgeon)
                VALUES
                    (?, 'gh146gal4', ?, 'g6f', 'left', 'Gustavo')
                """,
                [(1, "ALPN"), (2, "LN6")],
            )
            rows: list[tuple[int, int, str, str, str, int]] = []
            for index in range(1001):
                hour = index // 3600
                minute = (index // 60) % 60
                second = index % 60
                time_path = f"{hour:02d}_{minute:02d}_{second:02d}"
                rows.append(
                    (
                        index + 1,
                        1,
                        f"genotype\\stimulus\\2023\\10_17\\{time_path}",
                        "combo_stim_singles",
                        f"2023-10-17 {time_path.replace('_', ':')}",
                        1,
                    )
                )
            rows.append(
                (
                    5000,
                    2,
                    "ln6_genotype\\stimulus\\2023\\10_18\\12_00_00",
                    "ln6_stimulus",
                    "2023-10-18 12:00:03",
                    1,
                )
            )
            connection.executemany(
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
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.commit()
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
