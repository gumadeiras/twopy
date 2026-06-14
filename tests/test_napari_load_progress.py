"""Tests for the napari recording-load progress dialog.

Inputs: long recording paths and a Qt application.
Outputs: assertions that the batch queue and active recording summary stay
readable for many selected recordings.
"""

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPalette

from tests.napari_support import Path, QApplication, unittest
from twopy.napari.display_paths import (
    format_recording_queue_path,
    recording_path_display_summary,
)
from twopy.napari.load_progress import RecordingLoadProgressDialog, _style_sheet


class RecordingLoadProgressDialogTest(unittest.TestCase):
    """Tests for the asynchronous recording-load progress dialog."""

    def test_shared_path_helpers_format_source_recordings(self) -> None:
        """Confirm load-progress labels use the shared path parser.

        Inputs: one source recording path in the lab folder layout.
        Outputs: compact queue label plus root/genotype/stimulus/date fields.
        """
        path = _recording_paths(count=1)[0]

        summary = recording_path_display_summary(path)

        self.assertEqual(format_recording_queue_path(path), ".../2026/06_12/18_42_11")
        self.assertEqual(summary.root, Path("/Volumes/DataNAS_2/2p_microscope_data"))
        self.assertEqual(
            summary.genotype,
            "w_w;uasg6f_+;orcolexa,lexaopcschrimsontdtomato_orcogal4",
        )
        self.assertEqual(
            summary.stimulus,
            "bars_singles_stim=5s_bars=3x400ms_dt=200ms_int=20",
        )
        self.assertEqual(summary.date, "2026/06_12/18_42_11")

    def test_shared_path_helpers_ignore_converted_file_suffixes(self) -> None:
        """Confirm converted HDF5 paths still label the source recording.

        Inputs: a converted ``recording_data.h5`` path below a source recording.
        Outputs: the queue label stops at ``YYYY/MM_DD/HH_MM_SS``.
        """
        path = _recording_paths(count=1)[0] / "twopy" / "recording_data.h5"

        self.assertEqual(format_recording_queue_path(path), ".../2026/06_12/18_42_11")

    def test_queue_shows_compact_recording_paths(self) -> None:
        """Confirm queue rows use the final three source folders.

        Inputs: a 20-recording batch with microscope output paths.
        Outputs: each row keeps state separate from a compact recording label.
        """
        _ = QApplication.instance() or QApplication([])
        paths = _recording_paths(count=20)
        dialog = RecordingLoadProgressDialog(
            paths=paths,
            on_cancel_pending=lambda: None,
        )

        dialog.update_progress(
            completed_count=5,
            total_count=20,
            current_path=paths[5],
            active_index=5,
            phase="Reading recording data...",
        )

        active_row = dialog._queue_rows[5]
        self.assertEqual(active_row._state.text(), "Reading")
        self.assertEqual(
            active_row._recording.text(),
            ".../2026/06_12/19_05_43",
        )
        self.assertEqual(
            active_row.frame.objectName(),
            "recording_load_queue_row_active",
        )
        self.assertEqual(dialog._progress.format(), "6 of 20 - reading recording data")

    def test_queue_header_count_reserves_width_at_right_edge(self) -> None:
        """Confirm selected-count text has room at the queue-panel edge.

        Inputs: a double-digit recording count.
        Outputs: the count label reserves enough width and right-aligns inside
        the queue panel.
        """
        _ = QApplication.instance() or QApplication([])
        dialog = RecordingLoadProgressDialog(
            paths=_recording_paths(count=20),
            on_cancel_pending=lambda: None,
        )

        self.assertEqual(dialog._selected_count.text(), "20 selected")
        self.assertGreaterEqual(
            dialog._selected_count.minimumWidth(),
            dialog._selected_count.sizeHint().width(),
        )
        self.assertEqual(
            dialog._selected_count.alignment(),
            Qt.AlignmentFlag.AlignRight,
        )

    def test_queue_scrollbar_is_reserved_for_large_batches(self) -> None:
        """Confirm large queues expose a visible vertical scrollbar.

        Inputs: a 20-recording batch.
        Outputs: the queue scroll area always reserves a vertical scrollbar and
        suppresses horizontal scrolling.
        """
        _ = QApplication.instance() or QApplication([])
        dialog = RecordingLoadProgressDialog(
            paths=_recording_paths(count=20),
            on_cancel_pending=lambda: None,
        )

        self.assertEqual(
            dialog._queue_scroll.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn,
        )
        self.assertEqual(
            dialog._queue_scroll.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertEqual(
            dialog._queue_scroll.layoutDirection(),
            Qt.LayoutDirection.LeftToRight,
        )

    def test_dialog_minimum_width_fits_both_panels(self) -> None:
        """Confirm panel minimum widths cannot overlap at default size.

        Inputs: a many-recording dialog at its default size.
        Outputs: the queue and active panels fit side by side within the
        dialog's minimum width.
        """
        _ = QApplication.instance() or QApplication([])
        dialog = RecordingLoadProgressDialog(
            paths=_recording_paths(count=20),
            on_cancel_pending=lambda: None,
        )
        layout = dialog.layout()
        assert layout is not None
        margins = layout.contentsMargins()

        available_width = (
            dialog.minimumWidth() - margins.left() - margins.right() - layout.spacing()
        )

        self.assertGreaterEqual(available_width, 1120)

    def test_active_path_uses_metadata_style_fields(self) -> None:
        """Confirm the active recording path is split like the Metadata tab.

        Inputs: one source recording under root/genotype/stimulus/date/time.
        Outputs: active labels show root, genotype, stimulus, and date/time.
        """
        _ = QApplication.instance() or QApplication([])
        paths = _recording_paths(count=1)
        dialog = RecordingLoadProgressDialog(
            paths=paths,
            on_cancel_pending=lambda: None,
        )

        dialog.update_progress(
            completed_count=0,
            total_count=1,
            current_path=paths[0],
            active_index=0,
            phase="Preparing recording data...",
        )

        self.assertEqual(
            dialog._root_value.text(),
            "/Volumes/DataNAS_2/2p_microscope_data",
        )
        self.assertEqual(
            dialog._genotype_value.text(),
            "w_w;uasg6f_+;orcolexa,lexaopcschrimsontdtomato_orcogal4",
        )
        self.assertEqual(
            dialog._stimulus_value.text(),
            "bars_singles_stim=5s_bars=3x400ms_dt=200ms_int=20",
        )
        self.assertEqual(dialog._date_value.text(), "2026/06_12/18_42_11")

    def test_active_source_path_can_differ_from_queued_path(self) -> None:
        """Confirm resolved source identity drives the active path fields.

        Inputs: one queued converted output path plus its resolved source path.
        Outputs: active fields show the source while the queued row stays active.
        """
        _ = QApplication.instance() or QApplication([])
        source_path = _recording_paths(count=1)[0]
        queued_path = (
            Path("/analysis-cache")
            / source_path.relative_to("/Volumes/DataNAS_2/2p_microscope_data")
            / "recording_data.h5"
        )
        dialog = RecordingLoadProgressDialog(
            paths=(queued_path,),
            on_cancel_pending=lambda: None,
        )

        dialog.update_progress(
            completed_count=0,
            total_count=1,
            current_path=source_path,
            active_index=0,
            phase="Reading recording data...",
        )

        self.assertEqual(dialog._queue_rows[0]._state.text(), "Reading")
        self.assertEqual(
            dialog._root_value.text(),
            "/Volumes/DataNAS_2/2p_microscope_data",
        )
        self.assertEqual(dialog._date_value.text(), "2026/06_12/18_42_11")

    def test_failed_paths_keep_failed_queue_state(self) -> None:
        """Confirm failed recordings do not appear loaded after completion.

        Inputs: three queued recordings with the first path marked failed.
        Outputs: the failed row shows ``Failed`` while later completed rows show
        normal loaded state.
        """
        _ = QApplication.instance() or QApplication([])
        paths = _recording_paths(count=3)
        dialog = RecordingLoadProgressDialog(
            paths=paths,
            on_cancel_pending=lambda: None,
        )

        dialog.update_progress(
            completed_count=2,
            total_count=3,
            current_path=paths[2],
            active_index=2,
            phase="Adding recording layers...",
            failed_indexes=(0,),
        )

        self.assertEqual(dialog._queue_rows[0]._state.text(), "Failed")
        self.assertEqual(dialog._queue_rows[1]._state.text(), "Loaded")
        self.assertEqual(dialog._queue_rows[2]._state.text(), "Adding")

    def test_duplicate_paths_keep_independent_failure_state(self) -> None:
        """Confirm duplicate selected paths do not share failed row state.

        Inputs: two identical queued paths with only the first row marked failed.
        Outputs: the second row can remain active instead of inheriting failure.
        """
        _ = QApplication.instance() or QApplication([])
        path = _recording_paths(count=1)[0]
        dialog = RecordingLoadProgressDialog(
            paths=(path, path),
            on_cancel_pending=lambda: None,
        )

        dialog.update_progress(
            completed_count=1,
            total_count=2,
            current_path=path,
            active_index=1,
            phase="Reading recording data...",
            failed_indexes=(0,),
        )

        self.assertEqual(dialog._queue_rows[0]._state.text(), "Failed")
        self.assertEqual(dialog._queue_rows[1]._state.text(), "Reading")

    def test_stylesheet_ignores_bright_alternate_base_in_dark_theme(self) -> None:
        """Confirm napari dark theme does not produce light-gray panels.

        Inputs: Qt palette values matching napari's dark mode on macOS.
        Outputs: stylesheet uses dark Window/Base-derived surfaces and does not
        use the bright AlternateBase color.
        """
        _ = QApplication.instance() or QApplication([])
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#323232"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#181818"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#989898"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#374f6d"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))

        stylesheet = _style_sheet(palette)

        self.assertIn("#323232", stylesheet)
        self.assertNotIn("#989898", stylesheet)


def _recording_paths(*, count: int) -> tuple[Path, ...]:
    """Return realistic microscope paths for progress-dialog tests.

    Args:
        count: Number of paths to return.

    Returns:
        Tuple of source recording paths.
    """
    base = Path(
        "/Volumes/DataNAS_2/2p_microscope_data/"
        "w_w;uasg6f_+;orcolexa,lexaopcschrimsontdtomato_orcogal4/"
        "bars_singles_stim=5s_bars=3x400ms_dt=200ms_int=20/2026/06_12",
    )
    times = (
        "18_42_11",
        "18_47_36",
        "18_52_09",
        "18_58_44",
        "19_03_14",
        "19_05_43",
        "19_09_21",
        "19_12_08",
        "19_16_30",
        "19_19_55",
        "19_24_18",
        "19_28_02",
        "19_31_47",
        "19_36_09",
        "19_40_26",
        "19_44_51",
        "19_49_12",
        "19_53_38",
        "19_57_04",
        "20_01_29",
    )
    return tuple(base / time_text for time_text in times[:count])


if __name__ == "__main__":
    unittest.main()
