"""Tests for napari status text and placeholder helpers.

Inputs: counts, noun labels, and Qt line edits.
Outputs: short status phrases and consistently styled placeholder hints.
"""

import unittest

from qtpy.QtWidgets import QApplication, QLineEdit

from twopy.napari.text import configure_placeholder, counted_noun


class NapariTextTest(unittest.TestCase):
    """Tests text helpers used by the napari adapter."""

    def test_counted_noun_formats_status_text(self) -> None:
        """Confirm napari status text uses normal singular and plural words.

        Inputs: singular and plural counts.
        Outputs: phrases with full singular or plural nouns.
        """
        self.assertEqual(counted_noun(1, "file"), "1 file")
        self.assertEqual(counted_noun(2, "file"), "2 files")
        self.assertEqual(counted_noun(1, "ROI", "ROIs"), "1 ROI")
        self.assertEqual(counted_noun(2, "ROI", "ROIs"), "2 ROIs")

    def test_configure_placeholder_formats_and_styles_empty_fields(self) -> None:
        """Confirm placeholder hints use shared ellipsis and italic styling.

        Inputs: one empty Qt line edit.
        Outputs: placeholder text ends in ``...`` and only the empty field is
        italicized.
        """
        _ = QApplication.instance() or QApplication([])
        line_edit = QLineEdit()

        configure_placeholder(line_edit, "Optional note")

        self.assertEqual(line_edit.placeholderText(), "Optional note...")
        self.assertTrue(line_edit.font().italic())
        line_edit.setText("reviewed")
        self.assertFalse(line_edit.font().italic())
        line_edit.clear()
        self.assertTrue(line_edit.font().italic())


if __name__ == "__main__":
    unittest.main()
