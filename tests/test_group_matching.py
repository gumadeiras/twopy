"""Tests for manual group ROI match persistence.

Inputs: small in-memory match decisions and temporary CSV files.
Outputs: validated manual ROI match rows that downstream group analysis can
load without napari.
"""

import tempfile
import unittest
from pathlib import Path

from twopy import (
    append_manual_roi_match_rows,
    load_manual_fov_group_rows,
    load_manual_roi_match_rows,
    make_manual_fov_group_rows,
    make_manual_roi_match_rows,
    next_group_cell_id,
    save_manual_fov_group_rows,
    save_manual_roi_match_rows,
)
from twopy.analysis.group_matching import ManualRoiMatchRow


class GroupMatchingTests(unittest.TestCase):
    """Unit tests for manual ROI match CSV helpers."""

    def test_manual_fov_group_rows_round_trip_csv(self) -> None:
        """Confirm recording-to-FOV decisions are saved as CSV rows.

        Inputs: two recordings assigned to one visual FOV group.
        Outputs: loaded rows preserve paths and FOV labels.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fov_groups.csv"
            rows = make_manual_fov_group_rows(
                {
                    Path("/recordings/a"): "fov_1",
                    Path("/recordings/b"): "fov_1",
                },
            )

            save_manual_fov_group_rows(rows, path)
            loaded = load_manual_fov_group_rows(path)

        self.assertEqual(loaded, rows)

    def test_manual_roi_match_rows_round_trip_csv(self) -> None:
        """Confirm match decisions are saved as inspectable CSV rows.

        Inputs: two recordings assigned to one cell group.
        Outputs: the loaded rows preserve paths, ROI labels, status, and group.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "roi_matches.csv"
            rows = make_manual_roi_match_rows(
                {
                    Path("/recordings/a"): "roi_0002",
                    Path("/recordings/b"): "roi_0010",
                },
                group_cell_id=3,
                fov_group_id="fov_1",
            )

            save_manual_roi_match_rows(rows, path)
            loaded = load_manual_roi_match_rows(path)

        self.assertEqual(loaded, rows)

    def test_append_manual_roi_match_rows_assigns_next_group_id(self) -> None:
        """Confirm appended decisions can continue the group id sequence.

        Inputs: one existing matched group and one unmatched reviewed ROI.
        Outputs: appended rows produce a next group id of three.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "roi_matches.csv"
            first_rows = make_manual_roi_match_rows(
                {Path("/recordings/a"): "roi_0001"},
                group_cell_id=1,
                status="unmatched",
            )
            save_manual_roi_match_rows(first_rows, path)
            second_rows = make_manual_roi_match_rows(
                {Path("/recordings/b"): "roi_0004"},
                group_cell_id=next_group_cell_id(first_rows),
                status="unmatched",
            )

            all_rows = append_manual_roi_match_rows(path, second_rows)

        self.assertEqual(len(all_rows), 2)
        self.assertEqual(all_rows[1].group_cell_id, 2)
        self.assertEqual(all_rows[1].status, "unmatched")
        self.assertEqual(next_group_cell_id(all_rows), 3)

    def test_invalid_manual_roi_match_rows_fail_loudly(self) -> None:
        """Confirm malformed match decisions are rejected.

        Inputs: an invalid empty ROI label.
        Outputs: a clear ``ValueError`` before any CSV is written.
        """
        with self.assertRaisesRegex(ValueError, "roi_label"):
            make_manual_roi_match_rows(
                {Path("/recordings/a"): ""},
                group_cell_id=1,
            )

    def test_invalid_manual_fov_group_rows_fail_loudly(self) -> None:
        """Confirm empty FOV assignments are rejected.

        Inputs: an invalid empty FOV group id.
        Outputs: a clear ``ValueError`` before any CSV is written.
        """
        with self.assertRaisesRegex(ValueError, "FOV"):
            make_manual_fov_group_rows({Path("/recordings/a"): ""})

    def test_invalid_loaded_status_fails_loudly(self) -> None:
        """Confirm unknown CSV statuses are rejected on load.

        Inputs: a CSV row with a status outside the documented literals.
        Outputs: a clear ``ValueError``.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "roi_matches.csv"
            path.write_text(
                "fov_group_id,group_cell_id,recording_path,roi_label,status,note\n"
                "fov_1,1,/recordings/a,roi_0001,auto,\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "status"):
                load_manual_roi_match_rows(path)

    def test_blank_loaded_recording_path_fails_loudly(self) -> None:
        """Confirm blank CSV paths are rejected instead of becoming cwd.

        Inputs: one ROI match CSV row whose ``recording_path`` field is blank.
        Outputs: a clear ``ValueError`` during load.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "roi_matches.csv"
            path.write_text(
                "fov_group_id,group_cell_id,recording_path,roi_label,status,note\n"
                "fov_1,1,,roi_0001,matched,\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "recording_path"):
                load_manual_roi_match_rows(path)

    def test_public_row_type_has_expected_fields(self) -> None:
        """Confirm the row dataclass remains script-friendly.

        Inputs: a direct row construction.
        Outputs: plain attributes for notebooks and downstream analysis code.
        """
        row = ManualRoiMatchRow(
            fov_group_id="fov_1",
            group_cell_id=1,
            recording_path=Path("/recordings/a"),
            roi_label="roi_0001",
            status="matched",
        )

        self.assertEqual(row.recording_path, Path("/recordings/a"))
        self.assertEqual(row.status, "matched")


if __name__ == "__main__":
    unittest.main()
