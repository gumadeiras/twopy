"""Tests for manual group ROI match persistence.

Inputs: small in-memory match decisions and temporary CSV files.
Outputs: validated manual ROI match rows that downstream group analysis can
load without napari.
"""

import unittest
from pathlib import Path

from tests.tempdir import temporary_directory

from twopy import (
    add_manual_roi_match_group,
    append_manual_roi_match_rows,
    load_manual_fov_group_rows,
    load_manual_roi_match_rows,
    make_manual_fov_group_rows,
    make_manual_roi_match_rows,
    matched_manual_roi_groups,
    next_group_cell_id,
    remove_manual_roi_match_group,
    replace_manual_roi_match_group,
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
        with temporary_directory() as temp_dir:
            path = Path(temp_dir) / "fov_groups.csv"
            rows = make_manual_fov_group_rows(
                {
                    Path("/recordings/a"): "fov_1",
                    Path("/recordings/b"): "fov_1",
                },
                notes={
                    Path("/recordings/a"): "sharp overlap",
                    Path("/recordings/b"): "slight drift",
                },
            )

            save_manual_fov_group_rows(rows, path)
            loaded = load_manual_fov_group_rows(path)

        self.assertEqual(loaded, rows)
        self.assertEqual(
            {row.recording_path.name: row.note for row in loaded},
            {"a": "sharp overlap", "b": "slight drift"},
        )

    def test_manual_roi_match_rows_round_trip_csv(self) -> None:
        """Confirm match decisions are saved as inspectable CSV rows.

        Inputs: two recordings assigned to one cell group.
        Outputs: the loaded rows preserve paths, ROI labels, status, and group.
        """
        with temporary_directory() as temp_dir:
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
        with temporary_directory() as temp_dir:
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

    def test_add_manual_roi_match_group_assigns_next_id(self) -> None:
        """Confirm new matched groups are appended with the next table id.

        Inputs: one existing group and a new FOV-scoped ROI selection.
        Outputs: appended rows use the next id and shared note.
        """
        first = make_manual_roi_match_rows(
            {Path("/recordings/a"): "roi_0001"},
            group_cell_id=1,
            fov_group_id="fov_1",
            note="first",
        )

        rows, group_cell_id = add_manual_roi_match_group(
            first,
            {
                Path("/recordings/a"): "roi_0002",
                Path("/recordings/b"): "roi_0004",
            },
            fov_group_id="fov_1",
            note="second",
        )

        self.assertEqual(group_cell_id, 2)
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {row.roi_label for row in rows if row.group_cell_id == 2},
            {"roi_0002", "roi_0004"},
        )
        self.assertEqual(
            {row.note for row in rows if row.group_cell_id == 2},
            {"second"},
        )

    def test_matched_manual_roi_groups_filters_fov_and_status(self) -> None:
        """Confirm group listing ignores other FOVs and reviewed-only rows.

        Inputs: matched, unmatched, and different-FOV rows.
        Outputs: only matched rows in the selected FOV are grouped.
        """
        rows = (
            *make_manual_roi_match_rows(
                {Path("/recordings/a"): "roi_0001"},
                group_cell_id=1,
                fov_group_id="fov_1",
                note="same",
            ),
            *make_manual_roi_match_rows(
                {Path("/recordings/b"): "roi_0002"},
                group_cell_id=2,
                fov_group_id="fov_1",
                status="unmatched",
            ),
            *make_manual_roi_match_rows(
                {Path("/recordings/c"): "roi_0003"},
                group_cell_id=3,
                fov_group_id="fov_2",
            ),
        )

        groups = matched_manual_roi_groups(rows, fov_group_id="fov_1")

        self.assertEqual(tuple(group.group_cell_id for group in groups), (1,))
        self.assertEqual(groups[0].note, "same")
        self.assertEqual(
            {row.recording_path: row.roi_label for row in groups[0].rows},
            {Path("/recordings/a"): "roi_0001"},
        )

    def test_replace_manual_roi_match_group_preserves_unrelated_rows(self) -> None:
        """Confirm replacement touches only one matched group in one FOV.

        Inputs: target rows plus same-id rows in another FOV and non-matched
        review rows.
        Outputs: only the selected matched group is replaced.
        """
        target = make_manual_roi_match_rows(
            {
                Path("/recordings/a"): "roi_0001",
                Path("/recordings/b"): "roi_0002",
            },
            group_cell_id=1,
            fov_group_id="fov_1",
            note="old",
        )
        other_fov = make_manual_roi_match_rows(
            {Path("/recordings/c"): "roi_0003"},
            group_cell_id=1,
            fov_group_id="fov_2",
            note="other",
        )
        reviewed = make_manual_roi_match_rows(
            {Path("/recordings/d"): "roi_0004"},
            group_cell_id=1,
            fov_group_id="fov_1",
            status="unmatched",
            note="reviewed",
        )

        rows = replace_manual_roi_match_group(
            (*target, *other_fov, *reviewed),
            {Path("/recordings/a"): "roi_0005"},
            fov_group_id="fov_1",
            group_cell_id=1,
            note="new",
        )

        self.assertEqual(len(rows), 3)
        self.assertIn(other_fov[0], rows)
        self.assertIn(reviewed[0], rows)
        replacement = rows[-1]
        self.assertEqual(replacement.recording_path, Path("/recordings/a"))
        self.assertEqual(replacement.roi_label, "roi_0005")
        self.assertEqual(replacement.note, "new")

    def test_remove_manual_roi_match_group_preserves_unrelated_rows(self) -> None:
        """Confirm removal is scoped to one matched FOV group.

        Inputs: target and unrelated rows.
        Outputs: selected matched rows are removed and unrelated rows remain.
        """
        target = make_manual_roi_match_rows(
            {Path("/recordings/a"): "roi_0001"},
            group_cell_id=1,
            fov_group_id="fov_1",
        )
        unrelated = make_manual_roi_match_rows(
            {Path("/recordings/b"): "roi_0002"},
            group_cell_id=1,
            fov_group_id="fov_2",
        )

        rows = remove_manual_roi_match_group(
            (*target, *unrelated),
            fov_group_id="fov_1",
            group_cell_id=1,
        )

        self.assertEqual(rows, unrelated)

    def test_mixed_group_notes_have_no_shared_note(self) -> None:
        """Confirm differing row notes do not collapse into a misleading value.

        Inputs: one matched group with different notes per row.
        Outputs: group note is empty.
        """
        rows = (
            ManualRoiMatchRow(
                "fov_1",
                1,
                Path("/recordings/a"),
                "roi_0001",
                "matched",
                "a",
            ),
            ManualRoiMatchRow(
                "fov_1",
                1,
                Path("/recordings/b"),
                "roi_0002",
                "matched",
                "b",
            ),
        )

        groups = matched_manual_roi_groups(rows, fov_group_id="fov_1")

        self.assertEqual(groups[0].note, "")

    def test_manual_roi_match_group_helpers_reject_blank_fov(self) -> None:
        """Confirm FOV-scoped group helpers cannot create unlistable groups.

        Inputs: valid ROI selections with a blank FOV id.
        Outputs: clear validation errors before rows are created or queried.
        """
        rows = make_manual_roi_match_rows(
            {Path("/recordings/a"): "roi_0001"},
            group_cell_id=1,
            fov_group_id="fov_1",
        )
        recording_rois = {Path("/recordings/a"): "roi_0002"}

        with self.assertRaisesRegex(ValueError, "FOV"):
            matched_manual_roi_groups(rows, fov_group_id="")
        with self.assertRaisesRegex(ValueError, "FOV"):
            add_manual_roi_match_group(rows, recording_rois, fov_group_id="")
        with self.assertRaisesRegex(ValueError, "FOV"):
            replace_manual_roi_match_group(
                rows,
                recording_rois,
                fov_group_id="",
                group_cell_id=1,
            )
        with self.assertRaisesRegex(ValueError, "FOV"):
            remove_manual_roi_match_group(
                rows,
                fov_group_id="",
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
        with temporary_directory() as temp_dir:
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
        with temporary_directory() as temp_dir:
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
